// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// SSE2 strategy:
//
//   PSUBW    — int16 lane-wise subtraction (signed wraparound; safe here
//              because diffs are in [-255, +255])
//   PMADDWD  — pairwise int16×int16 → int32 plus horizontal add of pairs.
//              For squaring with same operands, gives sum-of-squares
//              packed into 4 int32 lanes.
//   PADDD    — int32 accumulate.
//
//   ssd4x4  needs no loop (16 int16 fits in two xmm).
//   ssd16x16 loops 16 passes × 16 int16 = 256.
//
// Overflow:
//   Max diff² = 255² = 65025. Per PMADDWD lane: 2×65025 = 130050.
//   ssd4x4 final sum ≤ 16×65025 = 1,040,400  (well within int32).
//   ssd16x16 final sum ≤ 256×65025 = 16,646,400 (still in int32).

// func ssd4x4(src, pred []int16) int64
TEXT ·ssd4x4(SB),NOSPLIT,$0-56
	MOVQ    src_base+0(FP), AX
	MOVQ    pred_base+24(FP), DI

	// Load src[0..7] → X0, src[8..15] → X1
	MOVOU   (AX), X0
	MOVOU   16(AX), X1

	// Load pred[0..7] → X2, pred[8..15] → X3
	MOVOU   (DI), X2
	MOVOU   16(DI), X3

	// diff = src - pred (int16 subtraction)
	PSUBW   X2, X0
	PSUBW   X3, X1

	// PMADDWD X0, X0 : lane i32[k] = diff16[2k]*diff16[2k] + diff16[2k+1]*diff16[2k+1]
	//   yielding 4 int32 lanes, each holding the sum of 2 squared int16 diffs.
	PMADDWD X0, X0
	PMADDWD X1, X1

	// X0 = X0 + X1 → 4 int32 lanes containing 4 squared-pair sums
	PADDD   X1, X0

	// Horizontal sum of 4 int32 → 1 int32 in X0[0]
	MOVO    X0, X1
	PSRLDQ  $8, X1      // X1 = [lane2, lane3, 0, 0]
	PADDD   X1, X0      // X0[0] = lane0+lane2, X0[1] = lane1+lane3
	MOVO    X0, X1
	PSRLDQ  $4, X1      // X1 = [lane1+lane3, 0, 0, 0]
	PADDD   X1, X0      // X0[0] = total

	// Extract int32, zero-extend to int64 (result is non-negative)
	MOVL    X0, AX
	MOVLQZX AX, AX
	MOVQ    AX, ret+48(FP)
	RET

// func ssd16x16(src, pred []int16) int64
//
// 256 int16 = 512 bytes total. Process 16 int16 (32 bytes) per pass × 16 passes.
TEXT ·ssd16x16(SB),NOSPLIT,$0-56
	MOVQ    src_base+0(FP), AX
	MOVQ    pred_base+24(FP), DI

	// Zero accumulator X4
	PXOR    X4, X4

	MOVQ    $16, CX         // 16 passes × 32 bytes = 512 bytes
loop16x16:
	// Load 16 int16 from src (X0+X1) and 16 from pred (X2+X3)
	MOVOU   (AX), X0
	MOVOU   16(AX), X1
	MOVOU   (DI), X2
	MOVOU   16(DI), X3

	ADDQ    $32, AX
	ADDQ    $32, DI

	// diff
	PSUBW   X2, X0
	PSUBW   X3, X1

	// Pairwise square + horizontal pair-sum → int32 lanes
	PMADDWD X0, X0
	PMADDWD X1, X1

	// Accumulate
	PADDD   X0, X4
	PADDD   X1, X4

	SUBQ    $1, CX
	JNE     loop16x16

	// Horizontal sum of X4 (4 int32 lanes) → int32 in X4[0]
	MOVO    X4, X0
	PSRLDQ  $8, X0
	PADDD   X0, X4
	MOVO    X4, X0
	PSRLDQ  $4, X0
	PADDD   X0, X4

	// Extract, zero-extend, store
	MOVL    X4, AX
	MOVLQZX AX, AX
	MOVQ    AX, ret+48(FP)
	RET
