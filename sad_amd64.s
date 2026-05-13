// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// func sad4x4(src, pred []int16) int64
//
// Computes sum of |src[i] - pred[i]| for i in 0..15 using SSE2.
//
// Go FP-based calling convention:
//   src_base+0(FP)   = src.ptr
//   pred_base+24(FP) = pred.ptr
//   ret+48(FP)       = return value (int64)
//
// SSE2 strategy:
//   1. Load 16 int16 from src  → X0 (8H) and X1 (8H)
//   2. Load 16 int16 from pred → X2 (8H) and X3 (8H)
//   3. diff = PSUBW(src, pred)
//   4. abs(diff) via PMAXSW(diff, PSUBW(zero, diff))
//   5. Sum using PADDW, then horizontal sum via zero-extending to int32
//   6. Sum pairs via PSLLDQ shifts

TEXT ·sad4x4(SB),NOSPLIT,$0-56
	MOVQ    src_base+0(FP), AX
	MOVQ    pred_base+24(FP), DI

	// Load src[0..7] → X0, src[8..15] → X1
	MOVOU   (AX), X0
	MOVOU   16(AX), X1

	// Load pred[0..7] → X2, pred[8..15] → X3
	MOVOU   (DI), X2
	MOVOU   16(DI), X3

	// diff = src - pred (int16 signed subtraction)
	PSUBW   X2, X0   // X0 = src[0..7] - pred[0..7]
	PSUBW   X3, X1   // X1 = src[8..15] - pred[8..15]

	// abs(diff): abs(x) = max(x, -x) using PMAXSW
	PXOR    X2, X2
	PXOR    X3, X3
	PSUBW   X0, X2   // X2 = -X0
	PMAXSW  X2, X0   // X0 = max(X0, -X0) = abs
	PXOR    X3, X3
	PSUBW   X1, X3   // X3 = -X1
	PMAXSW  X3, X1   // X1 = abs(X1)

	// Sum X0 and X1 (8 abs-diffs each, sum ≤ 2040 per half → no int16 overflow)
	PADDW   X1, X0   // X0 = 8 abs-diff pairs (each 16-bit sum ≤ 4080)

	// Horizontal sum of 8 int16 → int64
	// Zero-extend int16 to int32 using PUNPCKLWL/PUNPCKHWL with zero register
	PXOR    X2, X2
	MOVO    X0, X3
	// Unpack low 4 int16 to int32
	PUNPCKLWL X2, X0  // X0 = [s0(32),s1(32),s2(32),s3(32)]
	PUNPCKHWL X2, X3  // X3 = [s4(32),s5(32),s6(32),s7(32)]
	PADDD   X3, X0   // X0 = pairwise sum (4 int32)

	// Sum 4 int32 → 1 int64
	MOVO    X0, X1
	PSRLDQ  $8, X1   // X1 = [s2+s6, s3+s7, 0, 0] (shift right 8 bytes)
	PADDD   X1, X0   // X0[0..1] = partial sum int32
	MOVO    X0, X1
	PSRLDQ  $4, X1
	PADDD   X1, X0   // X0[0] = total sum as int32

	// Extract to int64 (zero-extended)
	MOVL    X0, AX
	MOVLQZX AX, AX
	MOVQ    AX, ret+48(FP)

	RET
