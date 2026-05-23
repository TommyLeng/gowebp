// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// pred4Context layout (from intra4.go):
//   left    [4]int    — bytes  0..31  (4 × int64)
//   topLeft int       — bytes 32..39
//   top     [8]int    — bytes 40..103 (8 × int64; only top[0..4] are read)
// Total: 104 bytes.
//
// For all four NEON helpers below the FP frame is:
//   ctx (104 bytes)            at offset 0
//   pred (slice: ptr/len/cap)  at offset 104
// argsize = 128.
//
// We load each ctx field individually through a base pointer kept in R10
// (set once at function entry) so that go vet does not flag offset-into-
// struct accesses. ABIInternal still receives args on the FP-based frame.
//
// All neighborhood pixel values are in [0,255], so they fit losslessly in
// int16. We narrow each int64 to int16 via VMOV …, V_.H[i] which keeps the
// low 16 bits — safe because the source values are small.
//
// NEON instructions encoded via WORD (Go assembler does not accept SMAX/SMIN):
//   SMAX Vd.4H, Vn.4H, Vm.4H : 0x0E606400 + (Rm<<16) + (Rn<<5) + Rd
//   SMIN Vd.4H, Vn.4H, Vm.4H : 0x0E606C00 + (Rm<<16) + (Rn<<5) + Rd

// ─────────────────────────────────────────────────────────────────────────────
// func intra4PredDC(ctx pred4Context, pred []int16)
//
// DC mode: dc = (top[0]+top[1]+top[2]+top[3]+left[0]+left[1]+left[2]+left[3]+4) >> 3
// Then broadcast int16(dc) to all 16 elements of pred.
//
// Each input value is in [0,255]. Sum + 4 is at most 8×255 + 4 = 2044, fits
// in 16 bits with room to spare. We sum in scalar GPRs (cheap; only 7 ADDs)
// and broadcast via VDUP.
// ─────────────────────────────────────────────────────────────────────────────
TEXT ·intra4PredDC(SB),NOSPLIT,$0-128
	// R10 = &ctx (FP-relative). Read fields via offsets into R10.
	MOVD	$ctx+0(FP), R10
	MOVD	0(R10), R0     // left[0]
	MOVD	8(R10), R1     // left[1]
	MOVD	16(R10), R2    // left[2]
	MOVD	24(R10), R3    // left[3]
	MOVD	40(R10), R4    // top[0]
	MOVD	48(R10), R5    // top[1]
	MOVD	56(R10), R6    // top[2]
	MOVD	64(R10), R7    // top[3]

	ADD	R1, R0, R0
	ADD	R3, R2, R2
	ADD	R5, R4, R4
	ADD	R7, R6, R6
	ADD	R2, R0, R0          // sum(left)
	ADD	R6, R4, R4          // sum(top)
	ADD	R4, R0, R0          // sum(all 8)
	ADD	$4, R0, R0          // +4
	LSR	$3, R0, R0          // >> 3, result in [0,255]

	// Broadcast low 16 bits of R0 across V0.8H (all 8 lanes).
	VDUP	R0, V0.H8

	// Store 32 bytes = 16 int16 to pred.
	MOVD	pred_base+104(FP), R8
	VST1	[V0.H8], (R8)
	ADD	$16, R8, R9
	VST1	[V0.H8], (R9)

	RET

// ─────────────────────────────────────────────────────────────────────────────
// func intra4PredTM(ctx pred4Context, pred []int16)
//
// TrueMotion: pred[y*4+x] = clip8(top[x] + left[y] - topLeft)
//
// Plan:
//   1. Load topLeft (scalar) and compute base[x] = top[x] - topLeft for x=0..3.
//      Range: [-255, 255], fits in int16.
//   2. Pack base[0..3] into V0.4H.
//   3. For each row y in 0..3:
//        - Load left[y] (scalar)
//        - VDUP left[y] across V1.4H
//        - V2 = V0 + V1   (range [-255, 510])
//        - SMAX V2, 0     (clamp lower)
//        - SMIN V2, 255   (clamp upper)
//        - Store V2.4H (8 bytes) to pred + y*8
// ─────────────────────────────────────────────────────────────────────────────
TEXT ·intra4PredTM(SB),NOSPLIT,$0-128
	MOVD	$ctx+0(FP), R10
	MOVD	32(R10), R0    // topLeft
	MOVD	40(R10), R1    // top[0]
	MOVD	48(R10), R2    // top[1]
	MOVD	56(R10), R3    // top[2]
	MOVD	64(R10), R4    // top[3]

	// base[x] = top[x] - topLeft, fits in int16 (range [-255, 255]).
	SUB	R0, R1, R1
	SUB	R0, R2, R2
	SUB	R0, R3, R3
	SUB	R0, R4, R4

	// Pack base[0..3] into V0.H[0..3] (low 64 bits of V0).
	VMOV	R1, V0.H[0]
	VMOV	R2, V0.H[1]
	VMOV	R3, V0.H[2]
	VMOV	R4, V0.H[3]

	// V4 = 0 (for SMAX clamp lower).
	VEOR	V4.B16, V4.B16, V4.B16
	// V5 = 255 (for SMIN clamp upper).
	MOVD	$255, R5
	VDUP	R5, V5.H4

	MOVD	pred_base+104(FP), R8

	// Row 0: left[0]
	MOVD	0(R10), R6
	VDUP	R6, V1.H4
	VADD	V1.H4, V0.H4, V2.H4
	// SMAX V2.4H, V2.4H, V4.4H : 0x0E606400 + (4<<16) + (2<<5) + 2 = 0x0E646442
	WORD	$0x0E646442
	// SMIN V2.4H, V2.4H, V5.4H : 0x0E606C00 + (5<<16) + (2<<5) + 2 = 0x0E656C42
	WORD	$0x0E656C42
	VST1	[V2.H4], (R8)

	// Row 1: left[1]
	MOVD	8(R10), R6
	VDUP	R6, V1.H4
	VADD	V1.H4, V0.H4, V2.H4
	WORD	$0x0E646442
	WORD	$0x0E656C42
	ADD	$8, R8, R9
	VST1	[V2.H4], (R9)

	// Row 2: left[2]
	MOVD	16(R10), R6
	VDUP	R6, V1.H4
	VADD	V1.H4, V0.H4, V2.H4
	WORD	$0x0E646442
	WORD	$0x0E656C42
	ADD	$16, R8, R9
	VST1	[V2.H4], (R9)

	// Row 3: left[3]
	MOVD	24(R10), R6
	VDUP	R6, V1.H4
	VADD	V1.H4, V0.H4, V2.H4
	WORD	$0x0E646442
	WORD	$0x0E656C42
	ADD	$24, R8, R9
	VST1	[V2.H4], (R9)

	RET

// ─────────────────────────────────────────────────────────────────────────────
// func intra4PredVE(ctx pred4Context, pred []int16)
//
// Vertical: each row of pred[y] = vals[0..3]
//   vals[x] = AVG3(topEx[x], topEx[x+1], topEx[x+2])
//   topEx   = [X, A, B, C, D, E]     where X=topLeft, A..E = top[0..4]
//   AVG3(a,b,c) = (a + 2*b + c + 2) >> 2
//
// All inputs are in [0,255]; AVG3 is also in [0,255]. Sum a+2b+c+2 ≤ 1022,
// fits in 16 bits, and is always non-negative, so we use unsigned shift.
//
// Plan:
//   1. Load X, A, B, C, D, E into V0.H[0..5] (6 of 8 lanes used).
//   2. V_a = V0.4H lanes [0..3] = [X, A, B, C]
//      V_b = VEXT 2-byte = [A, B, C, D]
//      V_c = VEXT 4-byte = [B, C, D, E]
//   3. V_sum = V_a + V_b + V_b + V_c + 2
//   4. vals = V_sum >> 2  (unsigned)
//   5. Store vals.4H to each of 4 rows.
// ─────────────────────────────────────────────────────────────────────────────
TEXT ·intra4PredVE(SB),NOSPLIT,$0-128
	MOVD	$ctx+0(FP), R10
	// Load 6 input values into V0.H[0..5].
	MOVD	32(R10), R0       // X (topLeft)
	VMOV	R0, V0.H[0]
	MOVD	40(R10), R0       // A = top[0]
	VMOV	R0, V0.H[1]
	MOVD	48(R10), R0       // B = top[1]
	VMOV	R0, V0.H[2]
	MOVD	56(R10), R0       // C = top[2]
	VMOV	R0, V0.H[3]
	MOVD	64(R10), R0       // D = top[3]
	VMOV	R0, V0.H[4]
	MOVD	72(R10), R0       // E = top[4]
	VMOV	R0, V0.H[5]

	// V_a = V0.4H lanes [0..3] = [X, A, B, C]
	// V_b = VEXT 2-byte → [A, B, C, D]  (low 64 bits of result)
	// V_c = VEXT 4-byte → [B, C, D, E]
	VEXT	$2, V0.B16, V0.B16, V1.B16
	VEXT	$4, V0.B16, V0.B16, V2.B16

	// V3 = V_a + V_b + V_b + V_c + 2 (low 4 lanes; all non-negative).
	VADD	V0.H4, V1.H4, V3.H4    // V3 = a + b
	VADD	V1.H4, V3.H4, V3.H4    // V3 += b
	VADD	V2.H4, V3.H4, V3.H4    // V3 += c
	MOVD	$2, R0
	VDUP	R0, V4.H4
	VADD	V4.H4, V3.H4, V3.H4    // V3 += 2

	// V3 >>= 2 (unsigned). All lanes are in [0, 1022] → [0, 255].
	VUSHR	$2, V3.H4, V3.H4

	// Store V3.4H to each of 4 rows (8 bytes per row).
	MOVD	pred_base+104(FP), R8
	VST1	[V3.H4], (R8)
	ADD	$8, R8, R9
	VST1	[V3.H4], (R9)
	ADD	$16, R8, R9
	VST1	[V3.H4], (R9)
	ADD	$24, R8, R9
	VST1	[V3.H4], (R9)

	RET

// ─────────────────────────────────────────────────────────────────────────────
// func intra4PredHE(ctx pred4Context, pred []int16)
//
// Horizontal: each row of pred[y] is filled with a single value vals[y].
//   vals[0] = AVG3(X, I, J)        // X=topLeft, I..L = left[0..3]
//   vals[1] = AVG3(I, J, K)
//   vals[2] = AVG3(J, K, L)
//   vals[3] = AVG3(K, L, L)        // note: trailing L duplicates last left
//
// All inputs in [0,255]; result in [0,255]; unsigned shift safe.
//
// Plan:
//   1. Load X, I, J, K, L, L into V0.H[0..5].
//   2. V_a = V0.4H lanes [0..3] = [X, I, J, K]
//      V_b = VEXT 2-byte → [I, J, K, L]
//      V_c = VEXT 4-byte → [J, K, L, L]
//   3. V_avg = (V_a + 2*V_b + V_c + 2) >> 2
//   4. For y in 0..3: VDUP V_avg.H[y] → V_row.4H; store 8 bytes.
// ─────────────────────────────────────────────────────────────────────────────
TEXT ·intra4PredHE(SB),NOSPLIT,$0-128
	MOVD	$ctx+0(FP), R10
	// Load X, I, J, K, L into V0.H[0..4], plus L into V0.H[5] for the
	// trailing AVG3(K, L, L) term.
	MOVD	32(R10), R0       // X (topLeft)
	VMOV	R0, V0.H[0]
	MOVD	0(R10), R0        // I = left[0]
	VMOV	R0, V0.H[1]
	MOVD	8(R10), R0        // J = left[1]
	VMOV	R0, V0.H[2]
	MOVD	16(R10), R0       // K = left[2]
	VMOV	R0, V0.H[3]
	MOVD	24(R10), R0       // L = left[3]
	VMOV	R0, V0.H[4]
	VMOV	R0, V0.H[5]       // L again (for the K,L,L tail)

	// V_a = V0.4H = [X, I, J, K]
	// V_b = VEXT 2 → [I, J, K, L]
	// V_c = VEXT 4 → [J, K, L, L]
	VEXT	$2, V0.B16, V0.B16, V1.B16
	VEXT	$4, V0.B16, V0.B16, V2.B16

	// V3 = (V_a + V_b + V_b + V_c + 2) >> 2
	VADD	V0.H4, V1.H4, V3.H4
	VADD	V1.H4, V3.H4, V3.H4
	VADD	V2.H4, V3.H4, V3.H4
	MOVD	$2, R0
	VDUP	R0, V4.H4
	VADD	V4.H4, V3.H4, V3.H4
	VUSHR	$2, V3.H4, V3.H4

	// Broadcast each lane of V3 to a 4-lane row and store.
	MOVD	pred_base+104(FP), R8

	VDUP	V3.H[0], V4.H4
	VST1	[V4.H4], (R8)

	ADD	$8, R8, R9
	VDUP	V3.H[1], V5.H4
	VST1	[V5.H4], (R9)

	ADD	$16, R8, R9
	VDUP	V3.H[2], V6.H4
	VST1	[V6.H4], (R9)

	ADD	$24, R8, R9
	VDUP	V3.H[3], V7.H4
	VST1	[V7.H4], (R9)

	RET
