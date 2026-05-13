// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// NEON instructions used via WORD encoding (Go assembler does not accept
// these mnemonics directly on arm64):
//
//   SMULL  Vd.4S, Vn.4H, Vm.4H
//     Signed multiply long: Vd.S[i] = Vn.H[i] * Vm.H[i] for i=0..3
//     Encoding: 0_Q_U_01110_size_1_Rm_opcode_00_Rn_Rd
//       Q=0, U=0, size=01, opcode=1100
//       base = 0x0E60C000 + (Rm<<16) + (Rn<<5) + Rd
//
//   SMLAL  Vd.4S, Vn.4H, Vm.4H   (Q=0)
//   SMLAL2 Vd.4S, Vn.8H, Vm.8H   (Q=1, operates on high 4 lanes)
//     Signed multiply-accumulate long:
//       Vd.S[i] += Vn.H[lane] * Vm.H[lane]
//     opcode=1000
//       SMLAL  base = 0x0E608000 + (Rm<<16) + (Rn<<5) + Rd
//       SMLAL2 base = 0x4E608000 + (Rm<<16) + (Rn<<5) + Rd
//
//   ADDV  Sd, Vn.4S
//     Sum 4 int32 lanes of Vn → 32-bit Sd (zero-extended into Vd.S[0]).
//     Encoding (Advanced SIMD across lanes):
//       0_Q_U_01110_size_11000_opcode_10_Rn_Rd
//       Q=1, U=0, size=10, opcode=11011
//       base = 0x4EB1B800 + (Rn<<5) + Rd

// func ssd4x4(src, pred []int16) int64
//
// Computes sum of (src[i] - pred[i])² for i in 0..15 using NEON.
//
// Plan:
//   1. Load 16 int16 from src  → V0.8H, V1.8H
//   2. Load 16 int16 from pred → V2.8H, V3.8H
//   3. V4.8H = V0.8H - V2.8H, V5.8H = V1.8H - V3.8H
//   4. V6.4S  = (V4.lo)²       (SMULL  V6.4S, V4.4H, V4.4H)
//      V6.4S += (V4.hi)²       (SMLAL2 V6.4S, V4.8H, V4.8H)
//      V6.4S += (V5.lo)²       (SMLAL  V6.4S, V5.4H, V5.4H)
//      V6.4S += (V5.hi)²       (SMLAL2 V6.4S, V5.8H, V5.8H)
//   5. ADDV S6, V6.4S          horizontal sum to S[0]
//   6. Move S[0] → R0, zero-extend, store ret

TEXT ·ssd4x4(SB),NOSPLIT,$0-56
	MOVD    src_base+0(FP), R0
	MOVD    pred_base+24(FP), R3

	// Load src[0..7] and src[8..15]
	VLD1    (R0), [V0.H8]
	ADD     $16, R0, R6
	VLD1    (R6), [V1.H8]

	// Load pred[0..7] and pred[8..15]
	VLD1    (R3), [V2.H8]
	ADD     $16, R3, R7
	VLD1    (R7), [V3.H8]

	// Compute signed diffs (16 lanes total across V4, V5)
	VSUB    V2.H8, V0.H8, V4.H8   // V4 = src[0..7] - pred[0..7]
	VSUB    V3.H8, V1.H8, V5.H8   // V5 = src[8..15] - pred[8..15]

	// V6.4S = (V4.4H)² (signed; first write sets V6)
	// SMULL V6.4S, V4.4H, V4.4H : 0x0E60C000 + (4<<16) + (4<<5) + 6
	WORD    $0x0E64C086
	// V6.4S += (V4.8H high)²
	// SMLAL2 V6.4S, V4.8H, V4.8H : 0x4E608000 + (4<<16) + (4<<5) + 6
	WORD    $0x4E648086
	// V6.4S += (V5.4H)²
	// SMLAL V6.4S, V5.4H, V5.4H : 0x0E608000 + (5<<16) + (5<<5) + 6
	WORD    $0x0E6580A6
	// V6.4S += (V5.8H high)²
	// SMLAL2 V6.4S, V5.8H, V5.8H : 0x4E608000 + (5<<16) + (5<<5) + 6
	WORD    $0x4E6580A6

	// ADDV S6, V6.4S : horizontal sum of 4 int32 lanes
	// 0x4EB1B800 + (6<<5) + 6
	WORD    $0x4EB1B8C6

	// Extract S[0] (32-bit, zero-extended) and return
	VMOV    V6.S[0], R0
	MOVD    R0, ret+48(FP)
	RET

// func ssd16x16(src, pred []int16) int64
//
// Computes sum of (src[i] - pred[i])² for i in 0..255 using NEON.
//
// Strategy: loop 16 times, processing 16 int16 (32 bytes) per pass.
// The maximum possible result is 256 × 255² = 16,646,400, which fits in
// int32 (max ~2.1e9), so an int32 accumulator (V16.4S) suffices.

TEXT ·ssd16x16(SB),NOSPLIT,$0-56
	MOVD    src_base+0(FP), R0
	MOVD    pred_base+24(FP), R3

	// Zero the accumulator V16
	VEOR    V16.B16, V16.B16, V16.B16

	// Process 256 int16 = 16 vectors of 8H. We loop 16 times, each
	// pass loads 16 int16 from src (V0+V1 = 32 bytes) and 16 from pred.
	// Each pass accumulates 16 squared diffs into V16.4S.

	MOVD    $16, R4   // counter: 16 passes × 16 int16 = 256 lanes
loop16:
	// Load 16 int16 from src (V0.8H + V1.8H)
	VLD1    (R0), [V0.H8]
	ADD     $16, R0, R6
	VLD1    (R6), [V1.H8]
	ADD     $32, R0, R0

	// Load 16 int16 from pred (V2.8H + V3.8H)
	VLD1    (R3), [V2.H8]
	ADD     $16, R3, R7
	VLD1    (R7), [V3.H8]
	ADD     $32, R3, R3

	// Diffs
	VSUB    V2.H8, V0.H8, V4.H8
	VSUB    V3.H8, V1.H8, V5.H8

	// Accumulate squares into V16.4S
	// SMLAL  V16.4S, V4.4H, V4.4H : 0x0E608000 + (4<<16) + (4<<5) + 16
	WORD    $0x0E648090
	// SMLAL2 V16.4S, V4.8H, V4.8H : 0x4E608000 + (4<<16) + (4<<5) + 16
	WORD    $0x4E648090
	// SMLAL  V16.4S, V5.4H, V5.4H : 0x0E608000 + (5<<16) + (5<<5) + 16
	WORD    $0x0E6580B0
	// SMLAL2 V16.4S, V5.8H, V5.8H : 0x4E608000 + (5<<16) + (5<<5) + 16
	WORD    $0x4E6580B0

	SUBS    $1, R4, R4
	BNE     loop16

	// ADDV S16, V16.4S : 0x4EB1B800 + (16<<5) + 16
	WORD    $0x4EB1BA10

	// Extract S[0] (32-bit zero-extended) and return
	VMOV    V16.S[0], R0
	MOVD    R0, ret+48(FP)
	RET
