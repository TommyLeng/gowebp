// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// func iTransform4x4(coeffs []int16, pred []int16, out []int16)
//
// Computes the inverse 4x4 DCT, adds pred, clamps to [0,255], writes int16 out.
// Entirely vectorised with SSE2 — both passes process all 4 columns / rows in
// parallel, working in signed int16 throughout (no 32-bit intermediates).
//
// Stack convention (FP-based, NOSPLIT $0):
//   coeffs_base+0(FP), pred_base+24(FP), out_base+48(FP)
//
// IMPORTANT: amd64 ABI register conventions:
//   R14 = goroutine pointer (g) — MUST NOT be used as scratch
//   R15 = m's processor (p)    — avoid as scratch
//   Safe GPRs: AX, BX, CX, DX, SI, DI, R8, R9, R10, R11, R12, R13
//
// Algorithm (from libwebp src/dsp/dec_sse2.c Transform_SSE2):
//   K1 = 85627 = 65536 + 20091 → k1 = +20091 (int16)
//   K2 = 35468 = 65536 - 30068 → k2 = -30068 (int16, = 0x8A4C signed)
//   MUL(x, K1) = PMULHW(x, k1) + x    (since (x*(k1+65536))>>16 = PMULHW(x,k1)+x)
//   MUL(x, K2) = PMULHW(x, k2) + x    (same trick)
//
//   Vertical pass (each XMM = 4 int16 across columns, in0..in3 = rows 0..3):
//     a  = in0 + in2
//     b  = in0 - in2
//     c  = MUL(in1, K2) - MUL(in3, K1)
//          = [PMULHW(in1,k2) + in1] - [PMULHW(in3,k1) + in3]
//          = PMULHW(in1,k2) - PMULHW(in3,k1) + (in1 - in3)
//     d  = MUL(in1, K1) + MUL(in3, K2)
//          = PMULHW(in1,k1) + in1 + PMULHW(in3,k2) + in3
//     t0 = a + d
//     t1 = b + c
//     t2 = b - c
//     t3 = a - d
//
//   4×4 int16 transpose (VP8Transpose_2_4x4_16b, but for single transform):
//     Round 1: PUNPCKLWL — interleave 16-bit lanes
//     Round 2: PUNPCKLLQ/PUNPCKHLQ — interleave 32-bit lanes
//     Round 3: PUNPCKLQDQ/PUNPCKHQDQ — interleave 64-bit lanes
//
//   Horizontal pass (same butterfly structure as vertical):
//     dc = T0 + 4
//     a  = dc + T2
//     b  = dc - T2
//     c  = MUL(T1, K2) - MUL(T3, K1)
//     d  = MUL(T1, K1) + MUL(T3, K2)
//     res0 = (a + d) >> 3
//     res1 = (b + c) >> 3
//     res2 = (b - c) >> 3
//     res3 = (a - d) >> 3
//
//   Second transpose (same), then add pred, clamp to [0,255] via PACKUSWB,
//   zero-extend back to int16 via PUNPCKLBW, store.
//
// Register map:
//   AX  = coeffs.ptr (freed after load)
//   SI  = pred.ptr
//   DI  = out.ptr
//   X12 = k1 = 20091 broadcast to all 8 int16 lanes
//   X13 = k2 = -30068 broadcast to all 8 int16 lanes
//   X14 = four = 4 broadcast to all 8 int16 lanes
//   X15 = zero (all zeros; stays zero throughout)
//   X0..X3   = in0..in3 (coefficient rows), then t0..t3 after vertical pass
//   X4..X11  = scratch for butterfly, transpose, horizontal pass
//
// Plan9 operand-order reminder (AT&T / GAS): OP src, dst
//   PADDW  X_a, X_b  → b += a
//   PSUBW  X_a, X_b  → b -= a
//   PMULHW X_a, X_b  → b = PMULHW(b, a)  (high 16 bits of signed 16×16)
//   PSRAW  $n, X_b   → b >>= n  (arithmetic, per int16 lane)
//   PACKUSWB X_a, X_b → Intel PACKUSWB(b,a): low8=saturate(b[0..3]), hi8=saturate(a[0..3])
//   PUNPCKLBW X_a, X_b → Intel PUNPCKLBW(b,a): interleave low bytes of b and a
//   PUNPCKLWL X_a, X_b → Intel PUNPCKLWD(b,a): [b0,a0,b1,a1,b2,a2,b3,a3]
//   PUNPCKHWL X_a, X_b → Intel PUNPCKHWD(b,a): [b4,a4,b5,a5,b6,a6,b7,a7]
//   PUNPCKLLQ X_a, X_b → Intel PUNPCKLDQ(b,a): [b[0..31],a[0..31],b[32..63],a[32..63]]
//   PUNPCKHLQ X_a, X_b → Intel PUNPCKHDQ(b,a): [b[64..95],a[64..95],b[96..127],a[96..127]]
//   PUNPCKLQDQ X_a, X_b → Intel PUNPCKLQDQ(b,a): [lo64(b), lo64(a)]
//   PUNPCKHQDQ X_a, X_b → Intel PUNPCKHQDQ(b,a): [hi64(b), hi64(a)]

// Constant data tables (broadcast int16 values to all 8 lanes).
DATA ·k1_i16<>+0x00(SB)/2, $20091
DATA ·k1_i16<>+0x02(SB)/2, $20091
DATA ·k1_i16<>+0x04(SB)/2, $20091
DATA ·k1_i16<>+0x06(SB)/2, $20091
DATA ·k1_i16<>+0x08(SB)/2, $20091
DATA ·k1_i16<>+0x0a(SB)/2, $20091
DATA ·k1_i16<>+0x0c(SB)/2, $20091
DATA ·k1_i16<>+0x0e(SB)/2, $20091
GLOBL ·k1_i16<>(SB), (NOPTR+RODATA), $16

DATA ·k2_i16<>+0x00(SB)/2, $-30068
DATA ·k2_i16<>+0x02(SB)/2, $-30068
DATA ·k2_i16<>+0x04(SB)/2, $-30068
DATA ·k2_i16<>+0x06(SB)/2, $-30068
DATA ·k2_i16<>+0x08(SB)/2, $-30068
DATA ·k2_i16<>+0x0a(SB)/2, $-30068
DATA ·k2_i16<>+0x0c(SB)/2, $-30068
DATA ·k2_i16<>+0x0e(SB)/2, $-30068
GLOBL ·k2_i16<>(SB), (NOPTR+RODATA), $16

DATA ·four_i16<>+0x00(SB)/2, $4
DATA ·four_i16<>+0x02(SB)/2, $4
DATA ·four_i16<>+0x04(SB)/2, $4
DATA ·four_i16<>+0x06(SB)/2, $4
DATA ·four_i16<>+0x08(SB)/2, $4
DATA ·four_i16<>+0x0a(SB)/2, $4
DATA ·four_i16<>+0x0c(SB)/2, $4
DATA ·four_i16<>+0x0e(SB)/2, $4
GLOBL ·four_i16<>(SB), (NOPTR+RODATA), $16

// ── MACRO: compute c and d from in1 and in3 ──────────────────────────────────
// On entry:  X_in1 = in1, X_in3 = in3, X12 = k1, X13 = k2
// Scratch:   X_tmp (any register you supply)
// On exit:   X_c (= c = MUL(in1,K2) - MUL(in3,K1)), X_d (= d = MUL(in1,K1) + MUL(in3,K2))
//
// Inlined directly in the code (macro would require #define, easier to inline).

TEXT ·iTransform4x4(SB),NOSPLIT,$0-72
	MOVQ    coeffs_base+0(FP), AX
	MOVQ    pred_base+24(FP), SI
	MOVQ    out_base+48(FP), DI

	// ── LOAD CONSTANTS ──────────────────────────────────────────────────────
	MOVOU   ·k1_i16<>(SB), X12        // X12 = k1 = 20091 × 8
	MOVOU   ·k2_i16<>(SB), X13        // X13 = k2 = -30068 × 8
	MOVOU   ·four_i16<>(SB), X14      // X14 = 4 × 8
	PXOR    X15, X15                   // X15 = 0 (all zeros, stays zero)

	// ── LOAD 16 int16 COEFFICIENTS AS 4 ROWS ────────────────────────────────
	// Each row is 4 int16 = 8 bytes.  MOVQ loads 8 bytes into the LOW 64 bits
	// of an XMM register; the high 64 bits are zeroed.
	MOVQ    0(AX), X0                  // X0 = row0 [c0,c1,c2,c3, 0,0,0,0]
	MOVQ    8(AX), X1                  // X1 = row1
	MOVQ    16(AX), X2                 // X2 = row2
	MOVQ    24(AX), X3                 // X3 = row3

	// ── VERTICAL PASS ───────────────────────────────────────────────────────
	// a = in0 + in2  →  X4
	// b = in0 - in2  →  X5
	MOVO    X0, X4
	PADDW   X2, X4                     // X4 = a = in0 + in2
	MOVO    X0, X5
	PSUBW   X2, X5                     // X5 = b = in0 - in2

	// c = MUL(in1, K2) - MUL(in3, K1)
	//   = [PMULHW(in1,k2) + in1] - [PMULHW(in3,k1) + in3]
	//   = PMULHW(in1,k2) - PMULHW(in3,k1) + (in1 - in3)
	MOVO    X1, X6
	PMULHW  X13, X6                    // X6 = PMULHW(in1, k2)
	MOVO    X3, X7
	PMULHW  X12, X7                    // X7 = PMULHW(in3, k1)
	MOVO    X1, X8
	PSUBW   X3, X8                     // X8 = in1 - in3
	PSUBW   X7, X6                     // X6 = PMULHW(in1,k2) - PMULHW(in3,k1)
	PADDW   X8, X6                     // X6 = c

	// d = MUL(in1, K1) + MUL(in3, K2)
	//   = PMULHW(in1,k1) + in1 + PMULHW(in3,k2) + in3
	MOVO    X1, X7
	PMULHW  X12, X7                    // X7 = PMULHW(in1, k1)
	PADDW   X1, X7                     // X7 = PMULHW(in1,k1) + in1  [= MUL(in1,K1)]
	MOVO    X3, X8
	PMULHW  X13, X8                    // X8 = PMULHW(in3, k2)
	PADDW   X3, X8                     // X8 = PMULHW(in3,k2) + in3  [= MUL(in3,K2)]
	PADDW   X8, X7                     // X7 = d = MUL(in1,K1) + MUL(in3,K2)

	// t0 = a + d  →  X0 (reuse)
	// t1 = b + c  →  X1
	// t2 = b - c  →  X2
	// t3 = a - d  →  X3
	MOVO    X4, X0
	PADDW   X7, X0                     // X0 = t0 = a + d
	MOVO    X5, X1
	PADDW   X6, X1                     // X1 = t1 = b + c
	MOVO    X5, X2
	PSUBW   X6, X2                     // X2 = t2 = b - c
	MOVO    X4, X3
	PSUBW   X7, X3                     // X3 = t3 = a - d

	// ── TRANSPOSE 4×4 int16 ─────────────────────────────────────────────────
	// Input:  X0=[t0c0,t0c1,t0c2,t0c3,0,0,0,0], X1=[t1c0,...], X2, X3
	// Output: X0=[T0r0,T0r1,T0r2,T0r3,0,0,0,0] = col0 of t, X1=col1, X2=col2, X3=col3
	//
	// Round 1: interleave 16-bit lanes of adjacent row pairs
	//   r01 = PUNPCKLWL(t0, t1) = [t0[0],t1[0],t0[1],t1[1],t0[2],t1[2],t0[3],t1[3]]
	//   r23 = PUNPCKLWL(t2, t3) = [t2[0],t3[0],t2[1],t3[1],t2[2],t3[2],t2[3],t3[3]]
	MOVO    X0, X4                     // X4 = t0 (save)
	MOVO    X2, X5                     // X5 = t2 (save)
	PUNPCKLWL X1, X0                   // X0 = r01 = [t0[0],t1[0],t0[1],t1[1],...]
	PUNPCKLWL X3, X2                   // X2 = r23 = [t2[0],t3[0],t2[1],t3[1],...]
	PUNPCKHWL X1, X4                   // X4 = r01_hi (all zeros since inputs have 0 in hi64)
	PUNPCKHWL X3, X5                   // X5 = r23_hi (all zeros)

	// Round 2: interleave 32-bit dword pairs
	//   t10 = PUNPCKLLQ(r01, r23) = [r01[0..31],r23[0..31],r01[32..63],r23[32..63]]
	//       = [t0[0],t1[0],t2[0],t3[0], t0[1],t1[1],t2[1],t3[1]]  (col0 low, col1 high)
	//   t11 = PUNPCKLLQ(r01_hi, r23_hi) = [0,0,0,0,...]  (all zeros)
	//   t12 = PUNPCKHLQ(r01, r23) = [r01[64..95],r23[64..95],r01[96..127],r23[96..127]]
	//       = [t0[2],t1[2],t2[2],t3[2], t0[3],t1[3],t2[3],t3[3]]  (col2 low, col3 high)
	//   t13 = PUNPCKHLQ(r01_hi, r23_hi) = zeros
	MOVO    X0, X6                     // X6 = r01 (save for hi)
	MOVO    X2, X7                     // X7 = r23 (save for hi)
	PUNPCKLLQ X2, X0                   // X0 = t10 = [col0 in lo64 | col1 in hi64]
	PUNPCKHLQ X7, X6                   // X6 = t12 = [col2 in lo64 | col3 in hi64]
	// X4, X5 are all zeros → t11, t13 not needed (just use X15)

	// Round 3: interleave 64-bit qword pairs to get final columns
	//   out0 = PUNPCKLQDQ(t10, t11) = [lo64(t10), lo64(t11)] = [col0, 0,0,0,0]
	//   out1 = PUNPCKHQDQ(t10, t11) = [hi64(t10), hi64(t11)] = [col1, 0,0,0,0]
	//   out2 = PUNPCKLQDQ(t12, t13) = [col2, 0,0,0,0]
	//   out3 = PUNPCKHQDQ(t12, t13) = [col3, 0,0,0,0]
	MOVO    X0, X1                     // X1 = t10 (save hi64 for col1)
	PUNPCKLQDQ X15, X0                 // X0 = [lo64(t10), lo64(zero)] = [col0, 0,0,0,0]
	PUNPCKHQDQ X15, X1                 // X1 = [hi64(t10), hi64(zero)] = [col1, 0,0,0,0]
	MOVO    X6, X2                     // X2 = t12 (save hi64 for col3)
	PUNPCKLQDQ X15, X6                 // X6 = [col2, 0,0,0,0]
	PUNPCKHQDQ X15, X2                 // X2 = [col3, 0,0,0,0]
	MOVO    X6, X3                     // X3 = col2 (in lo64)
	// Now: X0=T0=[col0], X1=T1=[col1], X3=T2=[col2], X2=T3=[col3]  (all in lo64)

	// ── HORIZONTAL PASS ─────────────────────────────────────────────────────
	// T0..T3 are now the 4 "columns" (each with 4 int16 = one value per orig row)
	// dc = T0 + 4
	MOVO    X0, X4
	PADDW   X14, X4                    // X4 = dc = T0 + 4

	// a = dc + T2  →  X5
	// b = dc - T2  →  X6
	MOVO    X4, X5
	PADDW   X3, X5                     // X5 = a = dc + T2
	MOVO    X4, X6
	PSUBW   X3, X6                     // X6 = b = dc - T2

	// c = MUL(T1, K2) - MUL(T3, K1)
	MOVO    X1, X7
	PMULHW  X13, X7                    // X7 = PMULHW(T1, k2)
	MOVO    X2, X8
	PMULHW  X12, X8                    // X8 = PMULHW(T3, k1)
	MOVO    X1, X9
	PSUBW   X2, X9                     // X9 = T1 - T3
	PSUBW   X8, X7                     // X7 = PMULHW(T1,k2) - PMULHW(T3,k1)
	PADDW   X9, X7                     // X7 = c

	// d = MUL(T1, K1) + MUL(T3, K2)
	MOVO    X1, X8
	PMULHW  X12, X8                    // X8 = PMULHW(T1, k1)
	PADDW   X1, X8                     // X8 = MUL(T1, K1)
	MOVO    X2, X9
	PMULHW  X13, X9                    // X9 = PMULHW(T3, k2)
	PADDW   X2, X9                     // X9 = MUL(T3, K2)
	PADDW   X9, X8                     // X8 = d

	// res0 = (a+d) >> 3,  res1 = (b+c) >> 3,  res2 = (b-c) >> 3,  res3 = (a-d) >> 3
	MOVO    X5, X0
	PADDW   X8, X0                     // X0 = a + d
	MOVO    X6, X1
	PADDW   X7, X1                     // X1 = b + c
	MOVO    X6, X2
	PSUBW   X7, X2                     // X2 = b - c
	MOVO    X5, X3
	PSUBW   X8, X3                     // X3 = a - d
	PSRAW   $3, X0                     // X0 = res0
	PSRAW   $3, X1                     // X1 = res1
	PSRAW   $3, X2                     // X2 = res2
	PSRAW   $3, X3                     // X3 = res3

	// ── SECOND TRANSPOSE: per-col-result → per-row-result ────────────────────
	// Same pattern as first transpose.
	// Input:  X0=[res0[r0,r1,r2,r3],0,0,0,0], X1=res1, X2=res2, X3=res3
	// Output: X0=[out_row0], X1=[out_row1], X2=[out_row2], X3=[out_row3]
	//
	// Round 1
	MOVO    X0, X4
	MOVO    X2, X5
	PUNPCKLWL X1, X0                   // X0 = [res0[0],res1[0],res0[1],res1[1],...]
	PUNPCKLWL X3, X2                   // X2 = [res2[0],res3[0],res2[1],res3[1],...]
	PUNPCKHWL X1, X4                   // X4 = zeros
	PUNPCKHWL X3, X5                   // X5 = zeros

	// Round 2
	MOVO    X0, X6
	MOVO    X2, X7
	PUNPCKLLQ X2, X0                   // X0 = [row0 in lo64 | row1 in hi64]
	PUNPCKHLQ X7, X6                   // X6 = [row2 in lo64 | row3 in hi64]

	// Round 3
	MOVO    X0, X1
	PUNPCKLQDQ X15, X0                 // X0 = out_row0 in lo64
	PUNPCKHQDQ X15, X1                 // X1 = out_row1 in lo64
	MOVO    X6, X2
	PUNPCKLQDQ X15, X6                 // X6 = out_row2 in lo64
	PUNPCKHQDQ X15, X2                 // X2 = out_row3 in lo64
	MOVO    X6, X3                     // X3 = out_row2

	// Now: X0=out_row0, X1=out_row1, X3=out_row2, X2=out_row3  (all in lo64)

	// ── ADD PRED, CLAMP [0,255], STORE AS int16 ──────────────────────────────
	// pred is int16[16] in row-major order; each row is 4 int16 = 8 bytes.
	// Strategy:
	//   1. Load 8 bytes of pred into lo64 of a temp XMM (MOVQ).
	//   2. Add to the result row (PADDW).
	//   3. Clamp [0,255]:
	//      a. PACKUSWB X15, X_row → saturates int16 to uint8, 4 bytes in lo32.
	//         (Plan9: PACKUSWB X15, X_row → Intel PACKUSWB(X_row, X15):
	//          result = [pack(X_row[0..3]), pack(X15[0..3])] = [b0,b1,b2,b3, 0,0,0,0, ...]
	//          in the low 8 bytes)
	//      b. PUNPCKLBW X15, X_row → zero-extends low 8 bytes to 16-bit:
	//         result = [b0,0,b1,0,b2,0,b3,0, ...] = 4 int16 in [0,255]
	//   4. MOVQ X_row, (out + j*8) — store 8 bytes (4 int16).

	// Row 0
	MOVQ    0(SI), X4                  // X4 = pred row 0
	PADDW   X4, X0                     // X0 = out_row0 + pred_row0
	PACKUSWB X15, X0                   // X0 = [b0,b1,b2,b3,0,...] as 8 bytes (clamped uint8)
	PUNPCKLBW X15, X0                  // X0 = [b0,0,b1,0,b2,0,b3,0,...] = 4 int16
	MOVQ    X0, 0(DI)

	// Row 1
	MOVQ    8(SI), X4                  // X4 = pred row 1
	PADDW   X4, X1
	PACKUSWB X15, X1
	PUNPCKLBW X15, X1
	MOVQ    X1, 8(DI)

	// Row 2
	MOVQ    16(SI), X4                 // X4 = pred row 2
	PADDW   X4, X3
	PACKUSWB X15, X3
	PUNPCKLBW X15, X3
	MOVQ    X3, 16(DI)

	// Row 3
	MOVQ    24(SI), X4                 // X4 = pred row 3
	PADDW   X4, X2
	PACKUSWB X15, X2
	PUNPCKLBW X15, X2
	MOVQ    X2, 24(DI)

	RET
