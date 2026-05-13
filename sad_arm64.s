// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// func sad4x4(src, pred []int16) int64
//
// Computes sum of |src[i] - pred[i]| for i in 0..15 (16 int16 values = 32 bytes).
//
// Go register-based ABI (arm64, since Go 1.17):
//   R0 = src ptr
//   R1 = src len  (not used, assumed ≥ 16)
//   R2 = src cap  (not used)
//   R3 = pred ptr
//   R4 = pred len (not used)
//   R5 = pred cap (not used)
//   R0 (return) = int64 result
//
// NEON strategy:
//   1. Load 16×int16 (32 bytes) from src  → V0.H8 + V1.H8
//   2. Load 16×int16 (32 bytes) from pred → V2.H8 + V3.H8
//   3. UABD V4.8H, V0.8H, V2.8H  — unsigned abs diff (safe for 0-255 values)
//   4. UABD V5.8H, V1.8H, V3.8H
//   5. UADDLV V4(S-scalar), V4.8H — sum 8×uint16 → uint32
//   6. UADDLV V5(S-scalar), V5.8H
//   7. Extract S[0] from each, add, return
//
// Note: UABD requires the destination register to be clean before use (hardware
// limitation on some lanes); we zero V4/V5 with VEOR before writing.
// Note: UADDLV on .8H produces a 32-bit result in S[0] (not D[0]).

TEXT ·sad4x4(SB),NOSPLIT,$0-56
	// Load pointers from stack (FP-based calling convention)
	MOVD    src_base+0(FP), R0    // src.ptr
	MOVD    pred_base+24(FP), R3  // pred.ptr

	// Load src[0..7] and src[8..15]
	VLD1    (R0), [V0.H8]
	ADD     $16, R0, R6
	VLD1    (R6), [V1.H8]

	// Load pred[0..7] and pred[8..15]
	VLD1    (R3), [V2.H8]
	ADD     $16, R3, R7
	VLD1    (R7), [V3.H8]

	// Zero V4, V5 before UABD (avoids lane-6 garbage issue on some microarchs)
	VEOR    V4.B16, V4.B16, V4.B16
	VEOR    V5.B16, V5.B16, V5.B16

	// UABD V4.8H, V0.8H, V2.8H  (unsigned absolute difference, 8 int16 lanes)
	// Encoding: Q=1 U=1 size=01 Rm=2 op=011111 Rn=0 Rd=4
	WORD    $0x6E627C04
	// UABD V5.8H, V1.8H, V3.8H
	// Encoding: Q=1 U=1 size=01 Rm=3 op=011111 Rn=1 Rd=5
	WORD    $0x6E637C25

	// UADDLV V4(S-scalar), V4.8H  — sum all 8 uint16 abs-diffs → uint32 in S[0]
	// Encoding: Q=1 U=1 size=01 opcode=00011 Rn=4 Rd=4 (via verified formula)
	WORD    $0x6E703884
	// UADDLV V5(S-scalar), V5.8H
	WORD    $0x6E7038A5

	// Extract S[0] (32-bit, zero-extended) from each and sum → ret
	VMOV    V4.S[0], R0
	VMOV    V5.S[0], R1
	ADD     R1, R0, R0
	MOVD    R0, ret+48(FP)

	RET
