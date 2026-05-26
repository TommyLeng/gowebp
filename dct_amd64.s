// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// func fTransform(src []int16, ref []int16, out []int16)
//
// Computes the 4×4 forward DCT of (src - ref), storing 16 int16 coefficients
// into out. Both passes fully vectorised with SSE2: horizontal pass uses
// PMADDWD for rotation terms; vertical pass processes all 4 columns in
// parallel using PMADDWD with int16 narrowing (no PMULLD required).
//
// Stack convention (FP-based, NOSPLIT with $64 locals):
//   src_base+0(FP), ref_base+24(FP), out_base+48(FP)
//
// IMPORTANT: amd64 ABI register conventions:
//   R14 = goroutine pointer (g) — MUST NOT be used as scratch
//   R15 = m's processor (p)    — avoid as scratch
//   Safe GPRs: AX, BX, CX, DX, SI, DI, R8, R9, R10, R11, R12, R13
//
// Both passes use only XMM registers (X0..X15); no GPR scratch is needed
// for the vertical pass. R14 (goroutine pointer) is never touched.
//
// Horizontal pass algorithm (all 4 rows processed in parallel):
//   1. Load src/ref as 2×128-bit (each holds 2 rows of 4 int16) and PSUBW.
//   2. Two-level PUNPCKLWD/PUNPCKHWD interleave acts as a 4×4 int16 transpose,
//      so we end up with each diff column held in a 64-bit half of an XMM:
//        X0 = [d0_col | d1_col] (low|high 64 bits), X1 = [d2_col | d3_col]
//      where d_k_col = [diff at row 0..3, column k] as 4×int16.
//   3. PSHUFD $0x4E on X1 swaps its 64-bit halves to produce [d3 | d2].
//   4. PADDW / PSUBW give [a0 | a1] and [a3 | a2] in one shot.
//   5. tmp[0] / tmp[2]: (a0±a1)*8 via PSHUFD + PADDW/PSUBW + PSLLW $3,
//      then sign-extend the low 64-bit int16x4 to int32x4
//      (PUNPCKLWD self, then PSRAD $16) and store to stack.
//   6. tmp[1] / tmp[3]: interleave a2/a3 (or a3/a2) into int16 pairs, then
//      PMADDWD by a constant [2217, ±5352, ...] vector to get the int32x4
//      result of (a2*2217 + a3*5352) / (a3*2217 - a2*5352).
//      Add the bias (1812 / 937) broadcast and PSRAD $9.
//
// Tmp layout on stack (different from the scalar code's row-major layout —
// we lay tmp out as 4 contiguous int32x4 vectors of "type k across rows"):
//   SP[ 0..15] = t0_vec = [h_t0_r0, h_t0_r1, h_t0_r2, h_t0_r3]  (int32)
//   SP[16..31] = t1_vec = [h_t1_r0, h_t1_r1, h_t1_r2, h_t1_r3]
//   SP[32..47] = t2_vec = [h_t2_r0, h_t2_r1, h_t2_r2, h_t2_r3]
//   SP[48..63] = t3_vec = [h_t3_r0, h_t3_r1, h_t3_r2, h_t3_r3]
// The vertical pass therefore reads SP[i*16 + {0,4,8,12}] for column i.

// Rotation constants for PMADDWD.
//
// PMADDWD takes pairs of int16 (X[2k], X[2k+1]) and computes
//   out_int32[k] = X[2k]*Y[2k] + X[2k+1]*Y[2k+1]
//
// For tmp[1] we need a2*2217 + a3*5352 → the interleaved input is
//   [a2_r0, a3_r0, a2_r1, a3_r1, ...] and the constant is [2217, 5352] × 4.
//
// For tmp[3] we need a3*2217 - a2*5352 → the interleaved input is
//   [a3_r0, a2_r0, a3_r1, a2_r1, ...] and the constant is [2217, -5352] × 4.
DATA  ·rot_k1<>+0x00(SB)/2, $2217
DATA  ·rot_k1<>+0x02(SB)/2, $5352
DATA  ·rot_k1<>+0x04(SB)/2, $2217
DATA  ·rot_k1<>+0x06(SB)/2, $5352
DATA  ·rot_k1<>+0x08(SB)/2, $2217
DATA  ·rot_k1<>+0x0a(SB)/2, $5352
DATA  ·rot_k1<>+0x0c(SB)/2, $2217
DATA  ·rot_k1<>+0x0e(SB)/2, $5352
GLOBL ·rot_k1<>(SB), (NOPTR+RODATA), $16

DATA  ·rot_k3<>+0x00(SB)/2, $2217
DATA  ·rot_k3<>+0x02(SB)/2, $-5352
DATA  ·rot_k3<>+0x04(SB)/2, $2217
DATA  ·rot_k3<>+0x06(SB)/2, $-5352
DATA  ·rot_k3<>+0x08(SB)/2, $2217
DATA  ·rot_k3<>+0x0a(SB)/2, $-5352
DATA  ·rot_k3<>+0x0c(SB)/2, $2217
DATA  ·rot_k3<>+0x0e(SB)/2, $-5352
GLOBL ·rot_k3<>(SB), (NOPTR+RODATA), $16

DATA  ·bias1812<>+0x00(SB)/4, $1812
DATA  ·bias1812<>+0x04(SB)/4, $1812
DATA  ·bias1812<>+0x08(SB)/4, $1812
DATA  ·bias1812<>+0x0c(SB)/4, $1812
GLOBL ·bias1812<>(SB), (NOPTR+RODATA), $16

DATA  ·bias937<>+0x00(SB)/4, $937
DATA  ·bias937<>+0x04(SB)/4, $937
DATA  ·bias937<>+0x08(SB)/4, $937
DATA  ·bias937<>+0x0c(SB)/4, $937
GLOBL ·bias937<>(SB), (NOPTR+RODATA), $16

// Vertical-pass bias constants (int32×4 broadcast).
DATA  ·vbias7<>+0x00(SB)/4, $7
DATA  ·vbias7<>+0x04(SB)/4, $7
DATA  ·vbias7<>+0x08(SB)/4, $7
DATA  ·vbias7<>+0x0c(SB)/4, $7
GLOBL ·vbias7<>(SB), (NOPTR+RODATA), $16

DATA  ·vbias12000<>+0x00(SB)/4, $12000
DATA  ·vbias12000<>+0x04(SB)/4, $12000
DATA  ·vbias12000<>+0x08(SB)/4, $12000
DATA  ·vbias12000<>+0x0c(SB)/4, $12000
GLOBL ·vbias12000<>(SB), (NOPTR+RODATA), $16

DATA  ·vbias51000<>+0x00(SB)/4, $51000
DATA  ·vbias51000<>+0x04(SB)/4, $51000
DATA  ·vbias51000<>+0x08(SB)/4, $51000
DATA  ·vbias51000<>+0x0c(SB)/4, $51000
GLOBL ·vbias51000<>(SB), (NOPTR+RODATA), $16

TEXT ·fTransform(SB),NOSPLIT,$64-72
	// ── LOAD POINTERS ───────────────────────────────────────────────────────
	MOVQ    src_base+0(FP), AX     // AX = src.ptr
	MOVQ    ref_base+24(FP), DI    // DI = ref.ptr
	MOVQ    out_base+48(FP), SI    // SI = out.ptr (kept across passes)

	// ── LOAD 16 int16 FROM src AND ref; SUBTRACT ────────────────────────────
	MOVOU   (AX), X0               // X0 = src rows 0,1 (8 int16)
	MOVOU   16(AX), X1              // X1 = src rows 2,3
	MOVOU   (DI), X2                // X2 = ref rows 0,1
	MOVOU   16(DI), X3              // X3 = ref rows 2,3
	PSUBW   X2, X0                  // X0 = diff rows 0,1
	PSUBW   X3, X1                  // X1 = diff rows 2,3

	// ── TRANSPOSE 4×4 int16 (two-level interleave) ──────────────────────────
	// Goal: per-column vectors held in 64-bit halves of two XMMs.
	// Note: Go assembler mnemonics use the "B/W/L/Q" suffix family, so
	//   PUNPCKLWD → PUNPCKLWL, PUNPCKHWD → PUNPCKHWL, PMADDWD → PMADDWL,
	//   PADDD → PADDL, PSRAD → PSRAL.
	MOVO    X0, X4
	PUNPCKLWL X1, X0                // X0 = [d_r0c0,d_r2c0,d_r0c1,d_r2c1,d_r0c2,d_r2c2,d_r0c3,d_r2c3]
	PUNPCKHWL X1, X4                // X4 = [d_r1c0,d_r3c0,d_r1c1,d_r3c1,d_r1c2,d_r3c2,d_r1c3,d_r3c3]
	MOVO    X0, X1
	PUNPCKLWL X4, X0                // X0 = [d0_col | d1_col]  (col0 in low 64, col1 in high 64)
	PUNPCKHWL X4, X1                // X1 = [d2_col | d3_col]

	// ── BUTTERFLY (4 rows in parallel via 64-bit halves) ────────────────────
	// Want: a0=d0+d3, a1=d1+d2, a2=d1-d2, a3=d0-d3
	PSHUFD  $0x4E, X1, X2           // X2 = [d3_col | d2_col]  (swap 64-bit halves)
	MOVO    X0, X3                  // X3 = [d0 | d1]
	PADDW   X2, X0                  // X0 = [d0+d3 | d1+d2] = [a0 | a1]
	PSUBW   X2, X3                  // X3 = [d0-d3 | d1-d2] = [a3 | a2]

	// ── tmp[0] / tmp[2] : (a0+a1)*8 / (a0-a1)*8 ─────────────────────────────
	// X0 = [a0 | a1].  Use PSHUFD $0x4E to swap halves: X4 = [a1 | a0].
	PSHUFD  $0x4E, X0, X4
	MOVO    X0, X5
	PADDW   X4, X5                  // X5 = [a0+a1 | a0+a1]  (4 int16 in low half)
	PSLLW   $3, X5                  // ×8 (fits int16: |a0+a1|*8 ≤ 16320)
	MOVO    X5, X6
	PUNPCKLWL X6, X6                // sign-extend prep: [t0r0,t0r0,t0r1,t0r1,...]
	PSRAL   $16, X6                 // X6 = t0_vec int32x4 (sign-extended)
	MOVOU   X6, 0(SP)               // store t0_vec at SP[0..15]

	MOVO    X0, X5
	PSUBW   X4, X5                  // X5 = [a0-a1 | a0-a1]
	PSLLW   $3, X5                  // ×8
	MOVO    X5, X6
	PUNPCKLWL X6, X6
	PSRAL   $16, X6                 // X6 = t2_vec int32x4
	MOVOU   X6, 32(SP)              // store t2_vec at SP[32..47]

	// ── tmp[1] : (a2*2217 + a3*5352 + 1812) >> 9 ────────────────────────────
	// X3 = [a3_col | a2_col].  Swap halves to get [a2_col | a3_col] in X7.
	PSHUFD  $0x4E, X3, X7           // X7 = [a2 | a3]
	MOVO    X7, X8
	// Interleave low 64-bit halves: lo(X8)=a2, lo(X3)=a3 → result lanes
	//   X8 = [a2_r0,a3_r0, a2_r1,a3_r1, a2_r2,a3_r2, a2_r3,a3_r3]
	PUNPCKLWL X3, X8
	PMADDWL ·rot_k1<>(SB), X8       // X8 = int32x4 of (a2*2217 + a3*5352)
	PADDL   ·bias1812<>(SB), X8     // + 1812
	PSRAL   $9, X8                  // >> 9 (arithmetic)
	MOVOU   X8, 16(SP)              // store t1_vec at SP[16..31]

	// ── tmp[3] : (a3*2217 - a2*5352 + 937) >> 9 ─────────────────────────────
	// We need pairs [a3, a2].  X3 low = a3, X7 low = a2.
	// PUNPCKLWL X7, X3 → X3 = [a3,a2 interleaved]
	PUNPCKLWL X7, X3                // X3 = [a3_r0,a2_r0, a3_r1,a2_r1, a3_r2,a2_r2, a3_r3,a2_r3]
	PMADDWL ·rot_k3<>(SB), X3       // X3 = int32x4 of (a3*2217 - a2*5352)
	PADDL   ·bias937<>(SB), X3      // + 937
	PSRAL   $9, X3                  // >> 9
	MOVOU   X3, 48(SP)              // store t3_vec at SP[48..63]

	// ── VERTICAL PASS (SSE2, fully vectorised) ──────────────────────────────
	// Lane layout: lane i holds column i (i=0..3) for all 4 output rows.
	//
	// Stack layout (from horizontal pass):
	//   SP[ 0..15] = t0_vec = [t0_r0, t0_r1, t0_r2, t0_r3]  (int32×4)
	//   SP[16..31] = t1_vec = [t1_r0, t1_r1, t1_r2, t1_r3]
	//   SP[32..47] = t2_vec = [t2_r0, t2_r1, t2_r2, t2_r3]
	//   SP[48..63] = t3_vec = [t3_r0, t3_r1, t3_r2, t3_r3]
	//
	// Vertical algorithm (vectorised over all 4 columns simultaneously):
	//   a0 = t0 + t3,  a1 = t1 + t2,  a2 = t1 - t2,  a3 = t0 - t3
	//   out[ 0..3]  = (a0+a1+7) >> 4             [int32 → int16]
	//   out[ 4..7]  = ((a2*2217+a3*5352+12000)>>16) + (a3!=0)
	//   out[ 8..11] = (a0-a1+7) >> 4
	//   out[12..15] = (a3*2217-a2*5352+51000) >> 16
	//
	// For the rotation terms, a2 and a3 are bounded by the horizontal >>9
	// shift and fit in int16.  We narrow them via PACKSSLW, interleave with
	// PUNPCKLWL, then run PMADDWL exactly as the horizontal pass does.
	//
	// Plan9 mnemonic notes (same as horizontal pass):
	//   PADDL=PADDD, PSUBL=PSUBD, PSRAL=PSRAD, PSRLL=PSRLD
	//   PCMPEQL=PCMPEQD, PUNPCKLWL=PUNPCKLWD, PMADDWL=PMADDWD
	//   PACKSSLW=PACKSSDW (pack int32→int16), PANDN=PANDN
	//
	// PACKSSLW Plan9: "PACKSSLW src, dst" → Intel PACKSSDW(dst,src)
	//   → dst = [sat16(dst[0..3]) | sat16(src[0..3])]
	//   So "PACKSSLW X14(zero), Xr" → Xr = [sat16(Xr[0..3]) | zeros]
	//   → int16×4 values in the low 64 bits of Xr.
	//
	// PUNPCKLWL Plan9: "PUNPCKLWL src, dst" → Intel PUNPCKLWD(dst,src)
	//   → dst = [dst[0],src[0], dst[1],src[1], dst[2],src[2], dst[3],src[3]]
	//
	// Register map:
	//   X0 = t0_vec, X1 = t1_vec, X2 = t2_vec, X3 = t3_vec  (int32×4)
	//   X4 = a0,  X5 = a1,  X6 = a2,  X7 = a3              (int32×4)
	//   X8 = out[0..3]  intermediate, then reused as a2_int16_packed
	//   X9 = out[8..11] intermediate, then reused as a3_int16_packed
	//   X10 = PMADDWL result for out[4..7]
	//   X11 = PMADDWL result for out[12..15]
	//   X12 = a3 saved copy for branchless correction
	//   X13 = bias constants (reused for each step)
	//   X14 = zero register (PXOR once, stays zero throughout)
	//   X15 = scratch for narrowing/MOVQ stores

	// ── Step 1: Load t0..t3 from stack ──────────────────────────────────────
	MOVOU    0(SP), X0              // X0 = t0_vec (int32×4)
	MOVOU   16(SP), X1              // X1 = t1_vec
	MOVOU   32(SP), X2              // X2 = t2_vec
	MOVOU   48(SP), X3              // X3 = t3_vec

	// ── Step 2: Butterfly ───────────────────────────────────────────────────
	MOVO    X0, X4
	PADDL   X3, X4                  // X4 = a0 = t0 + t3
	MOVO    X1, X5
	PADDL   X2, X5                  // X5 = a1 = t1 + t2
	MOVO    X1, X6
	PSUBL   X2, X6                  // X6 = a2 = t1 - t2
	MOVO    X0, X7
	PSUBL   X3, X7                  // X7 = a3 = t0 - t3

	// ── Step 3: out[0..3] = (a0+a1+7) >> 4 → X8 ────────────────────────────
	MOVOU   ·vbias7<>(SB), X13     // X13 = [7, 7, 7, 7] (int32×4)
	MOVO    X4, X8
	PADDL   X5, X8                  // a0 + a1
	PADDL   X13, X8                 // + 7
	PSRAL   $4, X8                  // >> 4  →  X8 = out[0..3] int32×4

	// ── Step 4: out[8..11] = (a0-a1+7) >> 4 → X9 ───────────────────────────
	MOVO    X4, X9
	PSUBL   X5, X9                  // a0 - a1
	PADDL   X13, X9                 // + 7
	PSRAL   $4, X9                  // >> 4  →  X9 = out[8..11] int32×4

	// ── Step 5: out[4..7] = ((a2*2217 + a3*5352 + 12000) >> 16) + (a3!=0) ──
	// Save a3 (int32×4) before we overwrite registers used for int16 packing.
	MOVO    X7, X12                 // X12 = a3 saved copy

	// Establish zero register (used by PACKSSLW to produce int16 in low 64b).
	PXOR    X14, X14                // X14 = 0

	// Narrow a2 (X6) and a3 (X7) from int32×4 → int16×4 (low 64 bits).
	// "PACKSSLW X14, X6": Plan9 → Intel PACKSSDW(X6, X14)
	//   → X6 = [sat16(X6[0..3]) | sat16(X14[0..3])]
	//   → X6 low 64b = [a2[0],a2[1],a2[2],a2[3]] as int16×4
	PACKSSLW X14, X6               // X6 = a2_int16×4 in low 64b
	PACKSSLW X14, X7               // X7 = a3_int16×4 in low 64b

	// Interleave: [a2[0],a3[0], a2[1],a3[1], a2[2],a3[2], a2[3],a3[3]]
	// "PUNPCKLWL X7, X6": Plan9 → Intel PUNPCKLWD(X6, X7)
	//   → X6 = [X6[0],X7[0], X6[1],X7[1], X6[2],X7[2], X6[3],X7[3]]
	//   → X6 = [a2[0],a3[0], a2[1],a3[1], ...] ✓
	PUNPCKLWL X7, X6               // X6 = [a2,a3 interleaved] int16×8

	// PMADDWL with rot_k1 = [2217,5352,...]: result = a2*2217 + a3*5352
	MOVO    X6, X10
	PMADDWL ·rot_k1<>(SB), X10    // X10 = int32×4 of (a2*2217 + a3*5352)
	MOVOU   ·vbias12000<>(SB), X13
	PADDL   X13, X10               // + 12000
	PSRAL   $16, X10               // >> 16

	// Branchless (a3 != 0) → +1 correction.
	// Strategy: X15 = (a3==0) mask via PCMPEQL, then invert via PANDN with all-ones.
	// "PANDN src, dst": Plan9 → Intel PANDN(dst,src) = ~dst & src.
	// We want: X15 = ~(a3==0).  Use all-ones constant in X13 as src.
	MOVO    X12, X15               // X15 = a3 (copy from saved X12)
	PCMPEQL X14, X15               // X15 = 0xFFFFFFFF where a3[i]==0, 0 elsewhere
	MOVO    X14, X13
	PCMPEQL X13, X13               // X13 = 0xFFFFFFFF all lanes (all-ones)
	PANDN   X13, X15               // X15 = ~X15 & X13 = ~(a3==0) = (a3!=0) mask
	PSRLL   $31, X15               // X15 = 1 where a3!=0, 0 where a3==0
	PADDL   X15, X10               // X10 = out[4..7] int32×4

	// ── Step 6: out[12..15] = (a3*2217 - a2*5352 + 51000) >> 16 ────────────
	// Need interleaved [a3[0],a2[0], a3[1],a2[1], ...] for rot_k3=[2217,-5352,...].
	// Repack a3 (from X12) and a2: but X6 is already clobbered by PUNPCKLWL.
	// We need fresh int16 packs.  Re-derive from original int32 vectors.
	// But X6 and X7 are clobbered.  We saved a3 in X12 (int32×4); re-pack it.
	// For a2, we need to re-derive: a2 = t1 - t2 = X1 - X2 (still in X1, X2? No:
	//   X1 and X2 were consumed by the butterfly and may be clobbered via PSUBL).
	// Actually: PSUBL X2, X6 was "X6 = X1 - X2" where X1 was in X6 copy.
	// X1 and X2 are still intact (we only wrote to X4,X5,X6,X7 in the butterfly).
	// So a2 = X6_original, but X6 was overwritten by PACKSSLW + PUNPCKLWL.
	// Re-compute a2 from X1 and X2 which are still intact:
	MOVO    X1, X6                 // X6 = t1_vec (reloaded from still-intact X1)
	PSUBL   X2, X6                 // X6 = a2 (re-computed; X1, X2 unchanged)

	// Now pack a3 (X12 = int32×4) and fresh a2 (X6 = int32×4) → int16.
	MOVO    X12, X11               // X11 = a3 (int32×4)
	PACKSSLW X14, X11              // X11 = a3_int16×4 in low 64b
	PACKSSLW X14, X6               // X6  = a2_int16×4 in low 64b

	// Interleave: [a3[0],a2[0], a3[1],a2[1], ...]
	// "PUNPCKLWL X6, X11": Plan9 → Intel PUNPCKLWD(X11,X6)
	//   → X11 = [X11[0],X6[0], X11[1],X6[1], ...] = [a3[0],a2[0], ...] ✓
	PUNPCKLWL X6, X11              // X11 = [a3,a2 interleaved] int16×8

	// PMADDWL with rot_k3 = [2217,-5352,...]: X11 = a3*2217 - a2*5352
	PMADDWL ·rot_k3<>(SB), X11    // X11 = int32×4 of (a3*2217 - a2*5352)
	MOVOU   ·vbias51000<>(SB), X13
	PADDL   X13, X11               // + 51000
	PSRAL   $16, X11               // X11 = out[12..15] int32×4

	// ── Step 7: Narrow int32→int16 and store ────────────────────────────────
	// "PACKSSLW X14(zero), Xr": Plan9 → Intel PACKSSDW(Xr, X14)
	//   → Xr = [sat16(Xr[0..3]) | sat16(X14[0..3])] = [Xr_packed | zeros]
	//   → low 64 bits of Xr hold the 4 int16 output values.
	// MOVQ Xr, mem stores the low 64 bits (8 bytes = 4 int16).
	PACKSSLW X14, X8               // X8  = out[0..3]  int16×4 in low 64b
	PACKSSLW X14, X10              // X10 = out[4..7]  int16×4 in low 64b
	PACKSSLW X14, X9               // X9  = out[8..11] int16×4 in low 64b
	PACKSSLW X14, X11              // X11 = out[12..15] int16×4 in low 64b

	MOVQ    X8,   0(SI)            // out[0..3]
	MOVQ    X10,  8(SI)            // out[4..7]
	MOVQ    X9,  16(SI)            // out[8..11]
	MOVQ    X11, 24(SI)            // out[12..15]

	RET

// func fTransformWHT(in []int16, out []int16)
//
// Computes the 4×4 Walsh-Hadamard Transform on the 16 DC coefficients.
// Pass 1 in int16 (input is 12b → tmp is 14b, fits int16).
// Pass 2 widens to int32 (because b0..b3 reach 16-bit values that don't fit
// signed int16), shifts >>1, then narrows via PACKSSDW (values fit, so the
// saturation is a no-op).
//
// Stack convention (FP-based, NOSPLIT $0):
//   in_base+0(FP), out_base+24(FP)
//
// Layout (4×4 row-major):
//   in[0..3]   = row 0
//   in[4..7]   = row 1
//   in[8..11]  = row 2
//   in[12..15] = row 3
//
// Strategy:
//   1. Load 16 int16 = 32 bytes as 2 XMM (X0=rows0&1, X1=rows2&3).
//   2. 4×4 int16 transpose via PUNPCKLWD/PUNPCKHWD to obtain c01 = (col0|col1)
//      and c23 = (col2|col3) packed across 8 lanes.
//   3. Pass 1 across columns (4 in parallel):
//        a0a1 = c01 + c23 = (a0|a1)
//        a3a2 = c01 - c23 = (a3|a2)
//      then horizontal butterfly within each 64-bit half to get
//      tmp_p0..tmp_p3 (col-vectors of tmp, each 4 int16 in low 64).
//   4. Transpose 4 col-vectors → 4 row-vectors of tmp via PUNPCKLWD + PUNPCKLDQ.
//   5. Widen each row-vector to int32x4 via PUNPCKLWD-with-self + PSRAD #16.
//   6. Pass 2 in int32: a0..a3 then b0..b3 = (a0+a1, a3+a2, a3-a2, a0-a1).
//   7. PSRAD #1 to get >>1; PACKSSDW narrows pairs (b0,b1) and (b2,b3) to int16x8.
//   8. Store 2 × MOVOU = 32 bytes.
//
// Register map:
//   AX = in.ptr
//   DI = out.ptr
//   X0 = rows 0,1 then a0a1 (pass 1 packed) then... reused
//   X1 = rows 2,3 then a3a2 (pass 1 packed) then... reused
//   X2..X5  = transpose / pass-1 scratch
//   X6..X9  = tmp col-vectors (p0..p3), then row-vectors after re-transpose
//   X10..X13 = pass-2 row-vectors as int32x4
//   X14     = a/b scratch
//   X15     = b3 (final)

TEXT ·fTransformWHT(SB),NOSPLIT,$0-48
	MOVQ    in_base+0(FP), AX
	MOVQ    out_base+24(FP), DI

	// ── LOAD 16 int16 = 32 bytes ────────────────────────────────────────────
	MOVOU    0(AX), X0   // X0 = rows 0,1 = (r0[0..3], r1[0..3]) as 8 int16
	MOVOU   16(AX), X1   // X1 = rows 2,3 = (r2[0..3], r3[0..3])

	// ── 4×4 int16 TRANSPOSE: rows → (col0|col1), (col2|col3) ────────────────
	// Goal: c01 = (c0[0..3] | c1[0..3]), c23 = (c2[0..3] | c3[0..3])
	// where ck[r] = M[r][k] (i.e., col k value of row r).
	//
	// Step 1: interleave rows 0 with 2 and rows 1 with 3.
	//   t0 = PUNPCKLWD(X0, X1)
	//     PUNPCKLWD interleaves the LOW 64 bits: src0.lane[0..3] with src1.lane[0..3].
	//     X0 low 64 = r0 = (r0[0..3]); X1 low 64 = r2 = (r2[0..3]).
	//     t0 = (r0[0], r2[0], r0[1], r2[1], r0[2], r2[2], r0[3], r2[3])
	//   t1 = PUNPCKHWD(X0, X1)
	//     X0 high 64 = r1; X1 high 64 = r3.
	//     t1 = (r1[0], r3[0], r1[1], r3[1], r1[2], r3[2], r1[3], r3[3])
	MOVO    X0, X2
	PUNPCKLWL X1, X2                // X2 = t0 = (r0,r2 interleaved at 16-bit)
	MOVO    X0, X3
	PUNPCKHWL X1, X3                // X3 = t1 = (r1,r3 interleaved)

	// Step 2: interleave t0 with t1 to get c01 and c23.
	//   c01 = PUNPCKLWD(t0, t1)
	//     Low 64: (t0[0], t1[0], t0[1], t1[1]) = (r0[0], r1[0], r2[0], r3[0]) = c0
	//     High 64: (t0[2], t1[2], t0[3], t1[3]) = (r0[1], r1[1], r2[1], r3[1]) = c1
	//   c23 = PUNPCKHWD(t0, t1)
	//     Low 64: (t0[4], t1[4], t0[5], t1[5]) = (r0[2], r1[2], r2[2], r3[2]) = c2
	//     High 64: (r0[3], r1[3], r2[3], r3[3]) = c3
	MOVO    X2, X4
	PUNPCKLWL X3, X4                // X4 = c01 = (c0 | c1)
	MOVO    X2, X5
	PUNPCKHWL X3, X5                // X5 = c23 = (c2 | c3)

	// ── PASS 1 (int16, 4 columns in parallel, packed in 8-lane XMMs) ────────
	// a0a1 = c01 + c23 = (c0+c2 | c1+c3) = (a0 | a1)
	// a3a2 = c01 - c23 = (c0-c2 | c1-c3) = (a3 | a2)
	MOVO    X4, X0
	PADDW   X5, X0                  // X0 = a0a1
	MOVO    X4, X1
	PSUBW   X5, X1                  // X1 = a3a2

	// We need 4 col-vectors of tmp: p0, p1, p2, p3 (4 int16 each in low 64).
	// p0 = a0+a1, p1 = a3+a2, p2 = a3-a2, p3 = a0-a1.
	//
	// X0 = (a0 | a1). PSHUFD $0x4E swaps 64-bit halves → (a1 | a0).
	// (X0 + swap(X0)) low 64 = a0+a1 = p0.
	// (X0 - swap(X0)) low 64 = a0-a1 = p3.
	PSHUFD  $0x4E, X0, X2           // X2 = (a1 | a0)
	MOVO    X0, X6
	PADDW   X2, X6                  // X6 low 64 = a0+a1 = p0 ; high = a1+a0 = p0 (also)
	MOVO    X0, X9
	PSUBW   X2, X9                  // X9 low 64 = a0-a1 = p3

	// X1 = (a3 | a2). swap = (a2 | a3).
	// (X1 + swap) low 64 = a3+a2 = p1.
	// (X1 - swap) low 64 = a3-a2 = p2.
	PSHUFD  $0x4E, X1, X3
	MOVO    X1, X7
	PADDW   X3, X7                  // X7 low 64 = p1
	MOVO    X1, X8
	PSUBW   X3, X8                  // X8 low 64 = p2

	// X6=p0, X7=p1, X8=p2, X9=p3 — col-vectors of tmp (each in low 64 = 4 int16).

	// ── TRANSPOSE 4 col-vectors → 4 row-vectors ─────────────────────────────
	// q01 = PUNPCKLWD(p0, p1) = (p0[0], p1[0], p0[1], p1[1], p0[2], p1[2], p0[3], p1[3])
	// q23 = PUNPCKLWD(p2, p3) = (p2[0], p3[0], p2[1], p3[1], p2[2], p3[2], p2[3], p3[3])
	PUNPCKLWL X7, X6                // X6 = q01
	PUNPCKLWL X9, X8                // X8 = q23

	// row01 = PUNPCKLDQ(q01, q23):
	//   Low 64: q01 dword 0 (= (p0[0], p1[0])) + q23 dword 0 (= (p2[0], p3[0]))
	//           = (p0[0], p1[0], p2[0], p3[0]) = ROW 0
	//   High 64: q01 dword 1 + q23 dword 1 = (p0[1], p1[1], p2[1], p3[1]) = ROW 1
	// row23 = PUNPCKHDQ(q01, q23):
	//   Low 64: q01 dword 2 + q23 dword 2 = (p0[2], p1[2], p2[2], p3[2]) = ROW 2
	//   High 64: q01 dword 3 + q23 dword 3 = (p0[3], p1[3], p2[3], p3[3]) = ROW 3
	MOVO    X6, X2
	PUNPCKLLQ X8, X2                // X2 = (ROW 0 | ROW 1)
	MOVO    X6, X3
	PUNPCKHLQ X8, X3                // X3 = (ROW 2 | ROW 3)

	// ── WIDEN TO INT32 ─────────────────────────────────────────────────────
	// PUNPCKLWD(X2, X2) duplicates each int16 lane: (a, a, b, b, c, c, d, d).
	// As int32 view: lane k = (a_k << 16) | a_k (where a_k is uint16).
	// PSRAD #16 arithmetic-shifts each int32 lane right by 16 → sign-extends
	// the int16 to int32. So PUNPCKLWD-with-self + PSRAD #16 = sign-extend
	// low half (lanes 0..3 = ROW 0) to int32x4.
	// PUNPCKHWD-with-self + PSRAD #16 = sign-extend high half (ROW 1) to int32x4.
	MOVO    X2, X10                 // row 0
	PUNPCKLWL X10, X10
	PSRAL   $16, X10                // X10 = row0 as int32x4
	MOVO    X2, X11                 // row 1
	PUNPCKHWL X11, X11
	PSRAL   $16, X11                // X11 = row1 as int32x4
	MOVO    X3, X12                 // row 2
	PUNPCKLWL X12, X12
	PSRAL   $16, X12                // X12 = row2 as int32x4
	MOVO    X3, X13                 // row 3
	PUNPCKHWL X13, X13
	PSRAL   $16, X13                // X13 = row3 as int32x4

	// ── PASS 2 in int32 ────────────────────────────────────────────────────
	// a0 = row0 + row2, a1 = row1 + row3, a2 = row1 - row3, a3 = row0 - row2
	MOVO    X10, X4
	PADDL   X12, X4                 // X4 = a0
	MOVO    X11, X5
	PADDL   X13, X5                 // X5 = a1
	MOVO    X11, X14
	PSUBL   X13, X14                // X14 = a2
	MOVO    X10, X15
	PSUBL   X12, X15                // X15 = a3

	// b0 = (a0+a1)>>1, b1 = (a3+a2)>>1, b2 = (a3-a2)>>1, b3 = (a0-a1)>>1
	MOVO    X4, X10                 // X10 = a0
	PADDL   X5, X10                 // X10 = a0 + a1
	MOVO    X15, X11                // X11 = a3
	PADDL   X14, X11                // X11 = a3 + a2
	MOVO    X15, X12                // X12 = a3
	PSUBL   X14, X12                // X12 = a3 - a2
	MOVO    X4, X13                 // X13 = a0
	PSUBL   X5, X13                 // X13 = a0 - a1

	PSRAL   $1, X10                 // X10 = b0
	PSRAL   $1, X11                 // X11 = b1
	PSRAL   $1, X12                 // X12 = b2
	PSRAL   $1, X13                 // X13 = b3

	// ── NARROW int32 → int16 with saturation (values fit, no clipping) ─────
	// PACKSSDW packs 2 int32x4 → int16x8.
	// Plan9 "PACKSSLW src, dst" = Intel PACKSSDW(dst, src):
	//   dst = (sat16(dst[0..3]), sat16(src[0..3]))
	// So PACKSSLW X11, X10 → X10 = (b0[0..3], b1[0..3]) as int16x8.
	// We want out[0..7] = b0 then b1, so this is correct.
	PACKSSLW X11, X10               // X10 = (b0 int16 | b1 int16)
	PACKSSLW X13, X12               // X12 = (b2 int16 | b3 int16)

	// ── STORE 32 bytes ─────────────────────────────────────────────────────
	MOVOU   X10, 0(DI)               // out[0..7]
	MOVOU   X12, 16(DI)              // out[8..15]

	RET
