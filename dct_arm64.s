// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// func fTransform(src []int16, ref []int16, out []int16)
//
// Computes the 4×4 forward DCT of (src - ref), storing 16 int16 coefficients into out.
// Horizontal pass vectorised with NEON; vertical pass in scalar ARM64.
//
// Stack convention (FP-based, NOSPLIT with $64 locals):
//   src_base+0(FP), ref_base+24(FP), out_base+48(FP)
//
// Note: R18 = macOS platform register — forbidden.
//       R28 = goroutine pointer — forbidden.
//       R30 = LR — managed by assembler.
// Available GP registers: R0-R17, R19-R27 (R26=LR on Go calls, but fine in leaf).
//
// NEON Plan9 VZIP note: "VZIP1 Vm, Vn, Vd" → ARM: ZIP1 Vd, Vn, Vm
//   → Vd = [Vn[0],Vm[0], Vn[1],Vm[1], ...]  (Vn comes first in output)
//
// SHL #3 on 4H (×8):
//   WORD $0x0F1355CE → SHL V14.4H, V14.4H, #3  (Rn=14, Rd=14)
//   WORD $0x0F1356B5 → SHL V21.4H, V21.4H, #3  (Rn=21, Rd=21)
//
// SSHR #9 on 4S (signed arithmetic right shift):
//   WORD $0x4F370610 → SSHR V16.4S, V16.4S, #9
//   WORD $0x4F370631 → SSHR V17.4S, V17.4S, #9
//
// SMULL/SMLAL/SMLSL using V15 (Rm=15, Rm[4]=0, even elements only):
//   H=0,L=0 → elem 0 = V15.H[0] = 2217
//   H=0,L=1 → elem 2 = V15.H[2] = 5352

TEXT ·fTransform(SB),NOSPLIT,$64-72
	// Uses FP-based arg convention (src_base+0, ref_base+24, out_base+48)
	// ── LOAD POINTERS ────────────────────────────────────────────────────────
	MOVD    src_base+0(FP), R0
	MOVD    ref_base+24(FP), R3
	MOVD    out_base+48(FP), R19  // R19 = out.ptr (callee-saved region)

	// ── LOAD 16 int16 from src and ref ───────────────────────────────────────
	VLD1    (R0), [V0.H8]
	ADD     $16, R0, R0
	VLD1    (R0), [V1.H8]
	VLD1    (R3), [V2.H8]
	ADD     $16, R3, R3
	VLD1    (R3), [V3.H8]

	// ── SUBTRACT ─────────────────────────────────────────────────────────────
	VSUB    V2.H8, V0.H8, V4.H8   // V4 = diffs rows 0,1
	VSUB    V3.H8, V1.H8, V5.H8   // V5 = diffs rows 2,3

	// ── TRANSPOSE: 2×VZIP ────────────────────────────────────────────────────
	// "VZIP1 Vm, Vn, Vd" → Vd=[Vn[0],Vm[0], Vn[1],Vm[1],...]
	// Round 1: Vn=V4 (rows01), Vm=V5 (rows23)
	VZIP1   V5.H8, V4.H8, V8.H8
	VZIP2   V5.H8, V4.H8, V9.H8
	// Round 2: Vn=V8, Vm=V9
	VZIP1   V9.H8, V8.H8, V6.H8   // V6 = col0&col1 for all rows
	VZIP2   V9.H8, V8.H8, V7.H8   // V7 = col2&col3 for all rows

	// Extract upper 4H of V6 and V7 (d1 and d3)
	VEXT    $8, V6.B16, V6.B16, V8.B16   // V8.4H = d1 all rows
	VEXT    $8, V7.B16, V7.B16, V9.B16   // V9.4H = d3 all rows

	// ── BUTTERFLY ────────────────────────────────────────────────────────────
	VADD    V9.H4, V6.H4, V10.H4  // a0 = d0+d3
	VADD    V7.H4, V8.H4, V11.H4  // a1 = d1+d2
	VSUB    V7.H4, V8.H4, V12.H4  // a2 = d1-d2
	VSUB    V9.H4, V6.H4, V13.H4  // a3 = d0-d3

	// ── DC TERMS (pos0 and pos2, int16 ×8) ───────────────────────────────────
	VADD    V11.H4, V10.H4, V14.H4   // V14 = a0+a1
	VSUB    V11.H4, V10.H4, V21.H4   // V21 = a0-a1
	// SHL V14.4H, V14.4H, #3  (×8)
	WORD    $0x0F1355CE
	// SHL V21.4H, V21.4H, #3
	WORD    $0x0F1356B5

	// ── ROTATION CONSTANTS ─────────────────────────────────────────────────
	MOVD    $2217, R9
	VMOV    R9, V15.H[0]
	MOVD    $5352, R10
	VMOV    R10, V15.H[2]
	MOVD    $1812, R9
	WORD    $0x4E040D33   // DUP V19.4S, R9
	MOVD    $937, R10
	WORD    $0x4E040D54   // DUP V20.4S, R10

	// ── ROTATION TERM 1: (a2*2217 + a3*5352 + 1812) >> 9 ────────────────────
	WORD    $0x0F4FA190   // SMULL V16.4S, V12.4H, V15.H[0]
	WORD    $0x0F6F21B0   // SMLAL V16.4S, V13.4H, V15.H[2]
	VADD    V19.S4, V16.S4, V16.S4
	WORD    $0x4F370610   // SSHR  V16.4S, V16.4S, #9

	// ── ROTATION TERM 3: (a3*2217 - a2*5352 + 937) >> 9 ─────────────────────
	WORD    $0x0F4FA1B1   // SMULL V17.4S, V13.4H, V15.H[0]
	WORD    $0x0F6F6191   // SMLSL V17.4S, V12.4H, V15.H[2]
	VADD    V20.S4, V17.S4, V17.S4
	WORD    $0x4F370631   // SSHR  V17.4S, V17.4S, #9

	// ── WIDEN DC TERMS TO INT32 ──────────────────────────────────────────────
	// SSHLL V22.4S, V14.4H, #0  (pos0; Rn=14, Rd=22)
	WORD    $0x0F10A5D6
	// SSHLL V23.4S, V21.4H, #0  (pos2; Rn=21, Rd=23)
	WORD    $0x0F10A6B7

	// ── VST4: store row-major int32 tmp[16] to stack ─────────────────────────
	// Pack into contiguous V24-V27
	VORR    V22.B16, V22.B16, V24.B16
	VORR    V16.B16, V16.B16, V25.B16
	VORR    V23.B16, V23.B16, V26.B16
	VORR    V17.B16, V17.B16, V27.B16
	MOVD    RSP, R11
	VST4    [V24.S4, V25.S4, V26.S4, V27.S4], (R11)

	// ── VERTICAL PASS ────────────────────────────────────────────────────────
	// R11=tmp base, R19=out.ptr
	// Registers: R0=i, R1=i*4, R2-R5=tmp values, R6=a0, R7=a1, R8=a2, R9=a3
	//            R10,R12=scratch, R13=$2217, R14=$5352, R15,R16=mul results
	MOVD    $0, R0
	MOVD    $2217, R13
	MOVD    $5352, R14
vloop:
	LSL     $2, R0, R1             // R1 = i*4 (byte offset)
	MOVW    (R11)(R1), R2           // tmp[0+i]
	// Compute base+offset for subsequent loads
	ADD     $16, R11, R20
	MOVW    (R20)(R1), R3           // tmp[4+i]  (R11+16+R1)
	ADD     $32, R11, R20
	MOVW    (R20)(R1), R4           // tmp[8+i]
	ADD     $48, R11, R20
	MOVW    (R20)(R1), R5           // tmp[12+i]

	ADD     R2, R5, R6    // a0 = tmp[0+i] + tmp[12+i]
	ADD     R3, R4, R7    // a1 = tmp[4+i] + tmp[8+i]
	SUB     R4, R3, R8    // a2 = tmp[4+i] - tmp[8+i]
	SUB     R5, R2, R9    // a3 = tmp[0+i] - tmp[12+i]

	// out[0+i] = (a0+a1+7) >> 4
	ADD     R7, R6, R10
	ADD     $7, R10, R10
	ASR     $4, R10, R10
	MOVH    R10, (R19)(R0<<1)          // out[0+i] at R19 + i*2

	// out[8+i] = (a0-a1+7) >> 4
	SUB     R7, R6, R10
	ADD     $7, R10, R10
	ASR     $4, R10, R10
	ADD     $16, R19, R20              // R20 = &out[8]
	MOVH    R10, (R20)(R0<<1)          // out[8+i] at (R19+16) + i*2

	// out[4+i] = ((a2*2217 + a3*5352 + 12000) >> 16) + (a3 != 0)
	MUL     R13, R8, R15   // a2 * 2217
	MUL     R14, R9, R16   // a3 * 5352
	ADD     R16, R15, R15
	ADD     $12000, R15, R15
	ASR     $16, R15, R15
	CMP     $0, R9
	CSET    NE, R16
	ADD     R16, R15, R15
	ADD     $8, R19, R20               // R20 = &out[4]
	MOVH    R15, (R20)(R0<<1)          // out[4+i] at (R19+8) + i*2

	// out[12+i] = (a3*2217 - a2*5352 + 51000) >> 16
	MUL     R13, R9, R15   // a3 * 2217
	MUL     R14, R8, R16   // a2 * 5352
	SUB     R16, R15, R15
	ADD     $51000, R15, R15
	ASR     $16, R15, R15
	ADD     $24, R19, R20              // R20 = &out[12]
	MOVH    R15, (R20)(R0<<1)          // out[12+i] at (R19+24) + i*2

	ADD     $1, R0, R0
	CMP     $4, R0
	BLT     vloop

	RET
