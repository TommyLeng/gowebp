// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// func fTransform(src []int16, ref []int16, out []int16)
//
// Computes the 4×4 forward DCT of (src - ref), storing 16 int16 coefficients
// into out. Horizontal pass fully vectorised with SSE2 (PMADDWD for the
// rotation terms); vertical pass uses scalar 32-bit integer math.
//
// Stack convention (FP-based, NOSPLIT with $64 locals):
//   src_base+0(FP), ref_base+24(FP), out_base+48(FP)
//
// IMPORTANT: amd64 ABI register conventions:
//   R14 = goroutine pointer (g) — MUST NOT be used as scratch
//   R15 = m's processor (p)    — avoid as scratch
//   Safe GPRs: AX, BX, CX, DX, SI, DI, R8, R9, R10, R11, R12, R13
//
// The horizontal pass is done entirely in XMM registers (X0..X11), so R14 is
// untouched. The scalar vertical pass uses only AX, BX, CX, DX, SI, R8..R13.
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

	// ── VERTICAL PASS (scalar 32-bit) ───────────────────────────────────────
	// Reads tmp on stack column-by-column: for column i (0..3), the 4 values
	//   val0..val3 live at SP[i*16 + {0,4,8,12}].
	//
	// out[ 0+i] = int16((a0+a1+7) >> 4)
	// out[ 4+i] = int16(((a2*2217 + a3*5352 + 12000) >> 16) + (a3!=0 ? 1 : 0))
	// out[ 8+i] = int16((a0-a1+7) >> 4)
	// out[12+i] = int16((a3*2217 - a2*5352 + 51000) >> 16)
	//
	// Registers: SI=out (preserved), CX=i (loop counter).
	//            Scratch: AX, BX, DX, DI, R8..R13.
	//            R14, R15 are NEVER touched.

	// Set up base of tmp on stack into DI so we can use indexed addressing
	// without relying on `(SP)(index*N)` syntax.
	LEAQ    0(SP), DI               // DI = &tmp[0]

	XORQ    CX, CX                  // CX = i = 0
vloop:
	// Compute byte offset i*16 into the tmp buffer.
	MOVQ    CX, BX
	SHLQ    $4, BX                  // BX = i*16

	// Load 4 int32 values: val0,val1,val2,val3 = tmp[i*4 + 0..3].
	MOVL    (DI)(BX*1), R8          // R8  = val0  (h_t0_r? — actually h_ti_r0)
	MOVL    4(DI)(BX*1), R9         // R9  = val1
	MOVL    8(DI)(BX*1), R10        // R10 = val2
	MOVL    12(DI)(BX*1), R11       // R11 = val3

	// a0 = val0 + val3
	MOVL    R8, BX
	ADDL    R11, BX                 // BX = a0  (int32)
	// a1 = val1 + val2
	MOVL    R9, DX
	ADDL    R10, DX                 // DX = a1
	// a2 = val1 - val2   (R9 clobbered to hold a2)
	SUBL    R10, R9                 // R9 = a2
	// a3 = val0 - val3   (R8 clobbered to hold a3)
	SUBL    R11, R8                 // R8 = a3

	// out[0+i] = int16((a0 + a1 + 7) >> 4)
	MOVL    BX, AX
	ADDL    DX, AX
	ADDL    $7, AX
	SARL    $4, AX
	MOVW    AX, (SI)(CX*2)          // out[0+i] is at SI + i*2

	// out[8+i] = int16((a0 - a1 + 7) >> 4)
	MOVL    BX, AX
	SUBL    DX, AX
	ADDL    $7, AX
	SARL    $4, AX
	MOVW    AX, 16(SI)(CX*2)        // out[8+i] is at SI + 8*2 + i*2

	// out[4+i] = int16(((a2*2217 + a3*5352 + 12000) >> 16) + (a3!=0?1:0))
	MOVL    R9, AX
	IMULL   $2217, AX               // AX = a2 * 2217
	MOVL    R8, R12
	IMULL   $5352, R12              // R12 = a3 * 5352
	ADDL    R12, AX                 // AX = a2*2217 + a3*5352
	ADDL    $12000, AX
	SARL    $16, AX                 // AX = >> 16
	XORL    R12, R12                // extra = 0
	TESTL   R8, R8                  // a3 != 0?
	SETNE   R12                     // R12 (low byte) = (a3 != 0) ? 1 : 0
	MOVBLZX R12, R12                // zero-extend low byte to int32
	ADDL    R12, AX
	MOVW    AX, 8(SI)(CX*2)         // out[4+i] is at SI + 4*2 + i*2

	// out[12+i] = int16((a3*2217 - a2*5352 + 51000) >> 16)
	MOVL    R8, AX
	IMULL   $2217, AX               // AX = a3 * 2217
	MOVL    R9, R12
	IMULL   $5352, R12              // R12 = a2 * 5352
	SUBL    R12, AX                 // AX = a3*2217 - a2*5352
	ADDL    $51000, AX
	SARL    $16, AX
	MOVW    AX, 24(SI)(CX*2)        // out[12+i] is at SI + 12*2 + i*2

	INCQ    CX
	CMPQ    CX, $4
	JL      vloop

	RET
