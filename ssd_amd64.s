// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// SSE2 strategy for SSD:
//
//   For diff values d in [-255, +255], d^2 fits in uint16 (max 65025 < 65536).
//   PMULLW gives the low 16 bits of d*d (signed 16×16→low16), which equals
//   the correct uint16 value since the product is always non-negative and < 65536.
//   PUNPCKLWL/PUNPCKHWL with a zero register zero-extends uint16 to uint32.
//   PADDD accumulates the int32 squared sums.
//
// Overflow analysis:
//   ssd4x4:   max sum = 16 × 65025 = 1,040,400  < 2^31  (fits int32)
//   ssd16x16: max sum = 256 × 65025 = 16,646,400 < 2^31  (fits int32)

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

	// diff = src - pred (int16 signed subtraction)
	PSUBW   X2, X0    // X0 = d[0..7]
	PSUBW   X3, X1    // X1 = d[8..15]

	// Square: PMULLW keeps low 16 bits of d*d.
	// Since |d| <= 255, d^2 <= 65025 < 65536, so the low 16 bits are correct.
	MOVO    X0, X2
	PMULLW  X2, X2    // X2 = d[0..7]^2  as uint16
	MOVO    X1, X3
	PMULLW  X3, X3    // X3 = d[8..15]^2 as uint16

	// Zero-extend uint16 squares to uint32 and accumulate into X4.
	PXOR    X4, X4    // accumulator = 0
	PXOR    X5, X5    // zero register (stays zero throughout)

	MOVO    X2, X6
	PUNPCKLWL X5, X6  // X6 = uint32(d[0..3]^2)   [low  4 lanes of X2]
	PADDD   X6, X4

	MOVO    X2, X6
	PUNPCKHWL X5, X6  // X6 = uint32(d[4..7]^2)   [high 4 lanes of X2]
	PADDD   X6, X4

	MOVO    X3, X6
	PUNPCKLWL X5, X6  // X6 = uint32(d[8..11]^2)  [low  4 lanes of X3]
	PADDD   X6, X4

	MOVO    X3, X6
	PUNPCKHWL X5, X6  // X6 = uint32(d[12..15]^2) [high 4 lanes of X3]
	PADDD   X6, X4

	// Horizontal sum of 4 int32 in X4 → single int32 in X4[0]
	MOVO    X4, X0
	PSRLDQ  $8, X0    // X0 = [X4[2], X4[3], 0, 0]
	PADDD   X0, X4    // X4[0] += X4[2], X4[1] += X4[3]
	MOVO    X4, X0
	PSRLDQ  $4, X0    // X0 = [X4[1], 0, 0, 0]
	PADDD   X0, X4    // X4[0] = total

	// Extract int32, zero-extend to int64, store result
	MOVL    X4, AX
	MOVLQZX AX, AX
	MOVQ    AX, ret+48(FP)
	RET

// func ssd16x16(src, pred []int16) int64
//
// 256 int16 = 512 bytes. Process 16 int16 (32 bytes) per pass × 16 passes.
TEXT ·ssd16x16(SB),NOSPLIT,$0-56
	MOVQ    src_base+0(FP), AX
	MOVQ    pred_base+24(FP), DI

	PXOR    X4, X4    // accumulator = 0
	PXOR    X5, X5    // zero register

	MOVQ    $16, CX
loop16x16:
	// Load 16 int16 from src and pred (32 bytes = two XMM each)
	MOVOU   (AX), X0
	MOVOU   16(AX), X1
	MOVOU   (DI), X2
	MOVOU   16(DI), X3
	ADDQ    $32, AX
	ADDQ    $32, DI

	// diff
	PSUBW   X2, X0    // X0 = d[0..7]
	PSUBW   X3, X1    // X1 = d[8..15]

	// Square via PMULLW (low 16 bits = correct uint16 value since d^2 <= 65025)
	MOVO    X0, X2
	PMULLW  X2, X2    // X2 = d[0..7]^2
	MOVO    X1, X3
	PMULLW  X3, X3    // X3 = d[8..15]^2

	// Zero-extend to uint32 and accumulate
	MOVO    X2, X6
	PUNPCKLWL X5, X6
	PADDD   X6, X4

	MOVO    X2, X6
	PUNPCKHWL X5, X6
	PADDD   X6, X4

	MOVO    X3, X6
	PUNPCKLWL X5, X6
	PADDD   X6, X4

	MOVO    X3, X6
	PUNPCKHWL X5, X6
	PADDD   X6, X4

	SUBQ    $1, CX
	JNE     loop16x16

	// Horizontal sum of 4 int32 in X4
	MOVO    X4, X0
	PSRLDQ  $8, X0
	PADDD   X0, X4
	MOVO    X4, X0
	PSRLDQ  $4, X0
	PADDD   X0, X4

	MOVL    X4, AX
	MOVLQZX AX, AX
	MOVQ    AX, ret+48(FP)
	RET
