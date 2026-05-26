// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// func fTransform(src []int16, ref []int16, out []int16)
//
// Computes the 4×4 forward DCT of (src - ref), storing 16 int16 coefficients into out.
// Both passes fully vectorised with NEON (int32x4 lanes = one lane per column).
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
//
// Vertical pass WORD encodings (instruction base | Rn<<5 | Rd):
//   DUP  Vd.4S, Wn              base = 0x4E040C00 + (Rn<<5) + Rd
//   MUL  Vd.4S, Vn.4S, Vm.4S   base = 0x4EA09C00 + (Rm<<16) + (Rn<<5) + Rd
//   CMEQ Vd.4S, Vn.4S, #0      base = 0x4EA09800 + (Rn<<5) + Rd
//   NOT  Vd.16B, Vn.16B         base = 0x6E205800 + (Rn<<5) + Rd
//   USHR Vd.4S, Vn.4S, #sh     base = 0x6F000400 + ((64-sh)<<16) + (Rn<<5) + Rd
//   SSHR Vd.4S, Vn.4S, #sh     base = 0x4F000400 + ((64-sh)<<16) + (Rn<<5) + Rd
//   XTN  Vd.4H, Vn.4S          base = 0x0E612800 + (Rn<<5) + Rd

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

	// ── VERTICAL PASS (NEON, fully vectorised) ───────────────────────────────
	// VST4 layout at R11: each group of 4 consecutive int32s at offsets 0,16,32,48
	// forms one position (row) across all 4 columns.  Lane i = column i.
	//
	// Load 4 rows from stack into int32x4 vectors:
	//   V0[i] = tmp[0+i]  (pos 0, col i)
	//   V1[i] = tmp[4+i]  (pos 1, col i)
	//   V2[i] = tmp[8+i]  (pos 2, col i)
	//   V3[i] = tmp[12+i] (pos 3, col i)
	VLD1    (R11), [V0.S4]
	ADD     $16, R11, R0
	VLD1    (R0), [V1.S4]
	ADD     $32, R11, R0
	VLD1    (R0), [V2.S4]
	ADD     $48, R11, R0
	VLD1    (R0), [V3.S4]

	// Butterfly:  V4=a0, V5=a1, V6=a2, V7=a3
	VADD    V3.S4, V0.S4, V4.S4   // a0 = row0 + row3
	VADD    V2.S4, V1.S4, V5.S4   // a1 = row1 + row2
	VSUB    V2.S4, V1.S4, V6.S4   // a2 = row1 - row2
	VSUB    V3.S4, V0.S4, V7.S4   // a3 = row0 - row3

	// Load rotation constants as int32x4 broadcast vectors
	// V8=2217, V9=5352, V10=12000, V11=51000
	MOVD    $2217, R5
	WORD    $0x4E040CA8   // DUP V8.4S, R5
	MOVD    $5352, R6
	WORD    $0x4E040CC9   // DUP V9.4S, R6
	MOVD    $12000, R7
	WORD    $0x4E040CEA   // DUP V10.4S, R7
	MOVD    $51000, R8
	WORD    $0x4E040D0B   // DUP V11.4S, R8

	// out[0..3] = (a0+a1+7) >> 4  →  V20
	VADD    V5.S4, V4.S4, V20.S4
	MOVD    $7, R5
	WORD    $0x4E040CA0   // DUP V0.4S, R5  (broadcast 7)
	VADD    V0.S4, V20.S4, V20.S4
	WORD    $0x4F3C0694   // SSHR V20.4S, V20.4S, #4

	// out[8..11] = (a0-a1+7) >> 4  →  V22
	VSUB    V5.S4, V4.S4, V22.S4
	VADD    V0.S4, V22.S4, V22.S4
	WORD    $0x4F3C06D6   // SSHR V22.4S, V22.4S, #4

	// out[4..7] = ((a2*2217 + a3*5352 + 12000) >> 16) + (a3 != 0)  →  V21
	//   Step 1: a2*2217 in V12, a3*5352 in V13
	WORD    $0x4EA89CCC   // MUL V12.4S, V6.4S, V8.4S  (a2 * 2217)
	WORD    $0x4EA99CED   // MUL V13.4S, V7.4S, V9.4S  (a3 * 5352)
	VADD    V13.S4, V12.S4, V21.S4
	VADD    V10.S4, V21.S4, V21.S4
	WORD    $0x4F3006B5   // SSHR V21.4S, V21.4S, #16
	//   Step 2: branchless (a3 != 0) → +1
	//   CMEQ V16.4S, V7.4S, #0  → 0xFFFFFFFF where a3==0
	WORD    $0x4EA098F0   // CMEQ V16.4S, V7.4S, #0
	//   NOT V17.16B, V16.16B   → 0xFFFFFFFF where a3!=0
	WORD    $0x6E205A11   // NOT V17.16B, V16.16B
	//   USHR V17.4S, V17.4S, #31  → 1 where a3!=0, 0 otherwise
	WORD    $0x6F210631   // USHR V17.4S, V17.4S, #31
	VADD    V17.S4, V21.S4, V21.S4

	// out[12..15] = (a3*2217 - a2*5352 + 51000) >> 16  →  V23
	WORD    $0x4EA89CEE   // MUL V14.4S, V7.4S, V8.4S  (a3 * 2217)
	WORD    $0x4EA99CCF   // MUL V15.4S, V6.4S, V9.4S  (a2 * 5352)
	VSUB    V15.S4, V14.S4, V23.S4
	VADD    V11.S4, V23.S4, V23.S4
	WORD    $0x4F3006F7   // SSHR V23.4S, V23.4S, #16

	// Narrow int32x4 → int16x4 for all 4 output groups
	WORD    $0x0E612A94   // XTN V20.4H, V20.4S  (out[0..3])
	WORD    $0x0E612AB5   // XTN V21.4H, V21.4S  (out[4..7])
	WORD    $0x0E612AD6   // XTN V22.4H, V22.4S  (out[8..11])
	WORD    $0x0E612AF7   // XTN V23.4H, V23.4S  (out[12..15])

	// Store 4 × int16x4 = 32 bytes = out[0..15]
	// out[0..3] at R19+0, out[4..7] at R19+8, out[8..11] at R19+16, out[12..15] at R19+24
	VST1    [V20.H4, V21.H4, V22.H4, V23.H4], (R19)

	RET

// func iTransform4x4(coeffs []int16, pred []int16, out []int16)
//
// Computes the inverse 4x4 DCT, adds pred, clamps to [0,255], writes int16 out.
// Fully vectorised — both passes process all 4 columns / rows in parallel.
//
// Stack convention (FP-based, NOSPLIT $0):
//   coeffs_base+0(FP), pred_base+24(FP), out_base+48(FP)
//
// Algorithm (matches dct.go scalar):
//   const c1 = 85627, c2 = 35468
//   mul1(a) = (a * c1) >> 16    ≡  a + ((a * 20091) >> 16)  (int32 wrapping)
//   mul2(a) = (a * c2) >> 16
//
//   Vertical pass (per column i):
//     a = row0[i] + row2[i]
//     b = row0[i] - row2[i]
//     c = mul2(row1[i]) - mul1(row3[i])
//     d = mul1(row1[i]) + mul2(row3[i])
//     tmp[i][0]=a+d, tmp[i][1]=b+c, tmp[i][2]=b-c, tmp[i][3]=a-d
//
//   Horizontal pass (per row j):
//     dc = tmp[0][j] + 4
//     ha = dc + tmp[2][j],  hb = dc - tmp[2][j]
//     hc = mul2(tmp[1][j]) - mul1(tmp[3][j])
//     hd = mul1(tmp[1][j]) + mul2(tmp[3][j])
//     out[j*4+0..3] = clip8(pred[j*4+0..3] + ((ha+hd, hb+hc, hb-hc, ha-hd) >> 3))
//
// Register map:
//   V0..V3:   loaded rows (int32x4 after SXTL)
//   V4..V11:  vertical-pass scratch (a, b, c, d, mul1/mul2 of row1/row3)
//   V12..V15: vertical pass results t0,t1,t2,t3 (row-type across cols)
//   V16..V19: transpose scratch
//   V20..V23: tmp col0,col2,col1,col3 (each holds 4 row values)
//   V24..V27: horizontal pass results before transpose
//   V16..V19: re-used for second transpose scratch
//   V20..V23: re-used for output rows after second transpose
//   V0..V3:   re-used to hold sign-extended pred rows
//   V28: V_c2  (35468 in all 4 S lanes)
//   V29: V_c1  (20091 in all 4 S lanes)
//   V30: V_four (4 in all 4 S lanes)
//   V31: V_255 (255 in all 4 S lanes)
//
// NEON instructions used via WORD encoding (Go assembler does not accept these):
//   SXTL  Vd.4S, Vn.4H        = SSHLL Vd.4S, Vn.4H, #0
//     base = 0x0F10A400 + (Rn<<5) + Rd
//   SXTL2 Vd.4S, Vn.8H        = SSHLL2 Vd.4S, Vn.8H, #0
//     base = 0x4F10A400 + (Rn<<5) + Rd
//   SSHR  Vd.4S, Vn.4S, #16   base = 0x4F300400 + (Rn<<5) + Rd
//   SSHR  Vd.4S, Vn.4S, #3    base = 0x4F3D0400 + (Rn<<5) + Rd
//   MUL   Vd.4S, Vn.4S, Vm.4S base = 0x4EA09C00 + (Rm<<16) + (Rn<<5) + Rd
//   DUP   Vd.4S, Wn           base = 0x4E040C00 + (Rn<<5) + Rd
//   SMAX  Vd.4S, Vn.4S, Vm.4S base = 0x4EA06400 + (Rm<<16) + (Rn<<5) + Rd
//   SMIN  Vd.4S, Vn.4S, Vm.4S base = 0x4EA06C00 + (Rm<<16) + (Rn<<5) + Rd
//   XTN   Vd.4H, Vn.4S        base = 0x0E612800 + (Rn<<5) + Rd
//   TRN1  Vd.4S, Vn.4S, Vm.4S base = 0x4E802800 + (Rm<<16) + (Rn<<5) + Rd
//   TRN2  Vd.4S, Vn.4S, Vm.4S base = 0x4E806800 + (Rm<<16) + (Rn<<5) + Rd
//   ZIP1  Vd.2D, Vn.2D, Vm.2D base = 0x4EC03800 + (Rm<<16) + (Rn<<5) + Rd
//   ZIP2  Vd.2D, Vn.2D, Vm.2D base = 0x4EC07800 + (Rm<<16) + (Rn<<5) + Rd

TEXT ·iTransform4x4(SB),NOSPLIT,$0-72
	MOVD    coeffs_base+0(FP), R0
	MOVD    pred_base+24(FP), R1
	MOVD    out_base+48(FP), R2

	// ── LOAD CONSTANTS ──────────────────────────────────────────────────────
	MOVD    $35468, R3
	WORD    $0x4E040C7C   // DUP V28.4S, W3  (V_c2 = 35468)
	MOVD    $20091, R3
	WORD    $0x4E040C7D   // DUP V29.4S, W3  (V_c1 = 20091)
	MOVD    $4, R3
	WORD    $0x4E040C7E   // DUP V30.4S, W3  (V_four = 4)
	MOVD    $255, R3
	WORD    $0x4E040C7F   // DUP V31.4S, W3  (V_255 = 255)

	// ── LOAD 16 int16 coeffs as 4 int16x4 rows ──────────────────────────────
	VLD1    (R0), [V0.H4, V1.H4, V2.H4, V3.H4]

	// ── SIGN-EXTEND int16x4 → int32x4 ───────────────────────────────────────
	WORD    $0x0F10A400   // SXTL  V0.4S, V0.4H
	WORD    $0x0F10A421   // SXTL  V1.4S, V1.4H
	WORD    $0x0F10A442   // SXTL  V2.4S, V2.4H
	WORD    $0x0F10A463   // SXTL  V3.4S, V3.4H

	// ── VERTICAL PASS ───────────────────────────────────────────────────────
	// a = row0 + row2, b = row0 - row2
	VADD    V2.S4, V0.S4, V4.S4   // V4 = a = V0 + V2
	VSUB    V2.S4, V0.S4, V5.S4   // V5 = b = V0 - V2

	// mul2(row1) = (V1 * 35468) >> 16   →  V7
	WORD    $0x4EBC9C27   // MUL   V7.4S, V1.4S, V28.4S
	WORD    $0x4F3004E7   // SSHR  V7.4S, V7.4S, #16
	// mul1(row1) = V1 + ((V1 * 20091) >> 16)   →  V6
	WORD    $0x4EBD9C26   // MUL   V6.4S, V1.4S, V29.4S
	WORD    $0x4F3004C6   // SSHR  V6.4S, V6.4S, #16
	VADD    V1.S4, V6.S4, V6.S4   // V6 = mul1(row1)

	// mul2(row3) = (V3 * 35468) >> 16   →  V9
	WORD    $0x4EBC9C69   // MUL   V9.4S, V3.4S, V28.4S
	WORD    $0x4F300529   // SSHR  V9.4S, V9.4S, #16
	// mul1(row3) = V3 + ((V3 * 20091) >> 16)   →  V8
	WORD    $0x4EBD9C68   // MUL   V8.4S, V3.4S, V29.4S
	WORD    $0x4F300508   // SSHR  V8.4S, V8.4S, #16
	VADD    V3.S4, V8.S4, V8.S4   // V8 = mul1(row3)

	// c = mul2(row1) - mul1(row3)   = V7 - V8   →  V10
	VSUB    V8.S4, V7.S4, V10.S4
	// d = mul1(row1) + mul2(row3)   = V6 + V9   →  V11
	VADD    V9.S4, V6.S4, V11.S4

	// tmp row-type 0..3 (each is row-type-i across the 4 columns):
	VADD    V11.S4, V4.S4, V12.S4   // V12 = a + d  (t0)
	VADD    V10.S4, V5.S4, V13.S4   // V13 = b + c  (t1)
	VSUB    V10.S4, V5.S4, V14.S4   // V14 = b - c  (t2)
	VSUB    V11.S4, V4.S4, V15.S4   // V15 = a - d  (t3)

	// ── 4×4 INT32 TRANSPOSE: row-type-across-cols → col-across-rows ─────────
	// V16 = TRN1.4S V12, V13   = [t0[0],t1[0], t0[2],t1[2]]
	WORD    $0x4E8D2990   // TRN1 V16.4S, V12.4S, V13.4S
	// V17 = TRN2.4S V12, V13   = [t0[1],t1[1], t0[3],t1[3]]
	WORD    $0x4E8D6991   // TRN2 V17.4S, V12.4S, V13.4S
	// V18 = TRN1.4S V14, V15   = [t2[0],t3[0], t2[2],t3[2]]
	WORD    $0x4E8F29D2   // TRN1 V18.4S, V14.4S, V15.4S
	// V19 = TRN2.4S V14, V15   = [t2[1],t3[1], t2[3],t3[3]]
	WORD    $0x4E8F69D3   // TRN2 V19.4S, V14.4S, V15.4S

	// V20 = ZIP1.2D V16, V18   = tmp col0 across rows = [tmp[0][0..3]]
	WORD    $0x4ED23A14   // ZIP1 V20.2D, V16.2D, V18.2D
	// V21 = ZIP2.2D V16, V18   = tmp col2 across rows = [tmp[2][0..3]]
	WORD    $0x4ED27A15   // ZIP2 V21.2D, V16.2D, V18.2D
	// V22 = ZIP1.2D V17, V19   = tmp col1 across rows = [tmp[1][0..3]]
	WORD    $0x4ED33A36   // ZIP1 V22.2D, V17.2D, V19.2D
	// V23 = ZIP2.2D V17, V19   = tmp col3 across rows = [tmp[3][0..3]]
	WORD    $0x4ED37A37   // ZIP2 V23.2D, V17.2D, V19.2D

	// ── HORIZONTAL PASS (all 4 rows in parallel) ────────────────────────────
	// dc = tmp[0] + 4       →  V4 (reuse)
	VADD    V30.S4, V20.S4, V4.S4
	// ha = dc + tmp[2]      →  V5
	VADD    V21.S4, V4.S4, V5.S4
	// hb = dc - tmp[2]      →  V6
	VSUB    V21.S4, V4.S4, V6.S4

	// mul2(tmp[1])  = (V22 * 35468) >> 16   →  V8
	WORD    $0x4EBC9EC8   // MUL   V8.4S, V22.4S, V28.4S
	WORD    $0x4F300508   // SSHR  V8.4S, V8.4S, #16
	// mul1(tmp[1])  = V22 + ((V22 * 20091) >> 16)   →  V7
	WORD    $0x4EBD9EC7   // MUL   V7.4S, V22.4S, V29.4S
	WORD    $0x4F3004E7   // SSHR  V7.4S, V7.4S, #16
	VADD    V22.S4, V7.S4, V7.S4

	// mul2(tmp[3])  = (V23 * 35468) >> 16   →  V10
	WORD    $0x4EBC9EEA   // MUL   V10.4S, V23.4S, V28.4S
	WORD    $0x4F30054A   // SSHR  V10.4S, V10.4S, #16
	// mul1(tmp[3])  = V23 + ((V23 * 20091) >> 16)   →  V9
	WORD    $0x4EBD9EE9   // MUL   V9.4S, V23.4S, V29.4S
	WORD    $0x4F300529   // SSHR  V9.4S, V9.4S, #16
	VADD    V23.S4, V9.S4, V9.S4

	// hc = mul2(tmp[1]) - mul1(tmp[3])   = V8 - V9    →  V11
	VSUB    V9.S4, V8.S4, V11.S4
	// hd = mul1(tmp[1]) + mul2(tmp[3])   = V7 + V10   →  V12
	VADD    V10.S4, V7.S4, V12.S4

	// Output column results (each lane = output row r, given column index):
	VADD    V12.S4, V5.S4, V24.S4   // V24 = ha + hd  (out col 0 per row)
	VADD    V11.S4, V6.S4, V25.S4   // V25 = hb + hc  (out col 1)
	VSUB    V11.S4, V6.S4, V26.S4   // V26 = hb - hc  (out col 2)
	VSUB    V12.S4, V5.S4, V27.S4   // V27 = ha - hd  (out col 3)

	// >>3 arithmetic shift right
	WORD    $0x4F3D0718   // SSHR  V24.4S, V24.4S, #3
	WORD    $0x4F3D0739   // SSHR  V25.4S, V25.4S, #3
	WORD    $0x4F3D075A   // SSHR  V26.4S, V26.4S, #3
	WORD    $0x4F3D077B   // SSHR  V27.4S, V27.4S, #3

	// ── SECOND 4×4 TRANSPOSE: per-col-across-rows → per-row-across-cols ─────
	// V16 = TRN1.4S V24, V25   = [V24[0],V25[0], V24[2],V25[2]]
	WORD    $0x4E992B10   // TRN1 V16.4S, V24.4S, V25.4S
	// V17 = TRN2.4S V24, V25   = [V24[1],V25[1], V24[3],V25[3]]
	WORD    $0x4E996B11   // TRN2 V17.4S, V24.4S, V25.4S
	// V18 = TRN1.4S V26, V27   = [V26[0],V27[0], V26[2],V27[2]]
	WORD    $0x4E9B2B52   // TRN1 V18.4S, V26.4S, V27.4S
	// V19 = TRN2.4S V26, V27   = [V26[1],V27[1], V26[3],V27[3]]
	WORD    $0x4E9B6B53   // TRN2 V19.4S, V26.4S, V27.4S

	// V20 = ZIP1.2D V16, V18   = output row 0 = [out[0,0..3]]
	WORD    $0x4ED23A14   // ZIP1 V20.2D, V16.2D, V18.2D
	// V21 = ZIP1.2D V17, V19   = output row 1 = [out[1,0..3]]
	WORD    $0x4ED33A35   // ZIP1 V21.2D, V17.2D, V19.2D
	// V22 = ZIP2.2D V16, V18   = output row 2 = [out[2,0..3]]
	WORD    $0x4ED27A16   // ZIP2 V22.2D, V16.2D, V18.2D
	// V23 = ZIP2.2D V17, V19   = output row 3 = [out[3,0..3]]
	WORD    $0x4ED37A37   // ZIP2 V23.2D, V17.2D, V19.2D

	// ── LOAD PRED AS 4×int32x4 ──────────────────────────────────────────────
	// Load 16 int16 (4 rows of 4) as two int16x8 vectors
	VLD1    (R1), [V0.H8, V1.H8]
	// Sign-extend each half to int32x4
	WORD    $0x0F10A402   // SXTL  V2.4S, V0.4H   (pred row 0)
	WORD    $0x4F10A403   // SXTL2 V3.4S, V0.8H   (pred row 1)
	WORD    $0x0F10A424   // SXTL  V4.4S, V1.4H   (pred row 2)
	WORD    $0x4F10A425   // SXTL2 V5.4S, V1.8H   (pred row 3)

	// ── ADD PRED ────────────────────────────────────────────────────────────
	VADD    V2.S4, V20.S4, V20.S4   // row 0 + pred row 0
	VADD    V3.S4, V21.S4, V21.S4   // row 1 + pred row 1
	VADD    V4.S4, V22.S4, V22.S4   // row 2 + pred row 2
	VADD    V5.S4, V23.S4, V23.S4   // row 3 + pred row 3

	// ── CLAMP TO [0, 255] ───────────────────────────────────────────────────
	// SMIN with 255 first (saturates the high end), then SMAX with 0.
	// (Order doesn't matter since 0 < 255; pick SMAX then SMIN.)
	// SMAX V_, V_, V31?  No — we need SMAX with 0.
	// Use VEOR to make a zero vector in V6, then SMAX with V6.
	VEOR    V6.B16, V6.B16, V6.B16
	// SMAX V20.4S, V20.4S, V6.4S
	WORD    $0x4EA66694   // SMAX  V20.4S, V20.4S, V6.4S
	WORD    $0x4EA666B5   // SMAX  V21.4S, V21.4S, V6.4S
	WORD    $0x4EA666D6   // SMAX  V22.4S, V22.4S, V6.4S
	WORD    $0x4EA666F7   // SMAX  V23.4S, V23.4S, V6.4S
	// SMIN V20.4S, V20.4S, V31.4S
	WORD    $0x4EBF6E94   // SMIN  V20.4S, V20.4S, V31.4S
	WORD    $0x4EBF6EB5   // SMIN  V21.4S, V21.4S, V31.4S
	WORD    $0x4EBF6ED6   // SMIN  V22.4S, V22.4S, V31.4S
	WORD    $0x4EBF6EF7   // SMIN  V23.4S, V23.4S, V31.4S

	// ── NARROW int32 → int16 (values already in [0,255]) ───────────────────
	WORD    $0x0E612A94   // XTN   V20.4H, V20.4S
	WORD    $0x0E612AB5   // XTN   V21.4H, V21.4S
	WORD    $0x0E612AD6   // XTN   V22.4H, V22.4S
	WORD    $0x0E612AF7   // XTN   V23.4H, V23.4S

	// ── STORE 4 × int16x4 = 32 bytes to out ─────────────────────────────────
	VST1    [V20.H4, V21.H4, V22.H4, V23.H4], (R2)

	RET

// func fTransformWHT(in []int16, out []int16)
//
// Computes the 4×4 Walsh-Hadamard Transform on the 16 DC coefficients.
// Both passes are vectorised; pass 1 stays in int16, pass 2 widens to int32
// (because b0..b3 can reach 16-bit unsigned which doesn't fit int16) before
// the >>1 shift and narrow.
//
// Stack convention (FP-based, NOSPLIT $0):
//   in_base+0(FP), out_base+24(FP)
//
// Strategy:
//   1. VLD4 .4H deinterleaves the 16 int16 input into 4 col-vectors V0..V3
//      where V_k.lane[j] = in[j*4+k]. This makes pass 1 a simple butterfly
//      across the 4 vectors (lane-parallel for all 4 rows).
//   2. Pass 1 in int16 (input max 12b → tmp max 14b, fits int16):
//        a0 = V0 + V2,  a1 = V1 + V3,  a2 = V1 - V3,  a3 = V0 - V2
//        tmp_p0 = a0+a1,  tmp_p1 = a3+a2,  tmp_p2 = a3-a2,  tmp_p3 = a0-a1
//      tmp_p_k.lane[j] = tmp[k+j*4] = tmp col k.
//   3. 4×4 int16 transpose so we obtain ROW-vectors of tmp:
//        U_r = (tmp[r*4+0], tmp[r*4+1], tmp[r*4+2], tmp[r*4+3])
//      Done via TRN1/TRN2 on .H4 (round 1) then TRN1/TRN2 on .S2 (round 2).
//   4. SXTL widen each U_r to int32x4.
//   5. Pass 2 in int32 (a0..a3 are 15b which would fit int16, but a0+a1 is 16b
//      which doesn't — widening is the safe path):
//        a0 = U0 + U2,  a1 = U1 + U3,  a2 = U1 - U3,  a3 = U0 - U2
//        b0 = (a0+a1) >> 1,  b1 = (a3+a2) >> 1,  b2 = (a3-a2) >> 1,  b3 = (a0-a1) >> 1
//      b_r.lane[c] = out[r*4+c].
//   6. XTN narrow int32→int16, VST1 stores 4 × int16x4 contiguously to out.
//
// Register map:
//   V0..V3   : col-vectors of in (after VLD4)
//   V4..V7   : pass-1 a0..a3
//   V8..V11  : pass-1 tmp_p0..tmp_p3 (col-vectors of tmp)
//   V12..V19 : transpose scratch + transpose output
//   V20..V23 : pass-2 inputs U_r (row-vectors of tmp, after SXTL → int32x4)
//   V24..V27 : pass-2 a0..a3 (int32x4)
//   V28..V31 : pass-2 b0..b3 (int32x4, then narrowed to int16x4 in V20..V23)
//
// Encodings (Plan9 syntax does not accept these instructions):
//   TRN1 Vd.4H, Vn.4H, Vm.4H : 0x0E402800 | (Rm<<16) | (Rn<<5) | Rd
//   TRN2 Vd.4H, Vn.4H, Vm.4H : 0x0E406800 | (Rm<<16) | (Rn<<5) | Rd
//   TRN1 Vd.2S, Vn.2S, Vm.2S : 0x0E802800 | (Rm<<16) | (Rn<<5) | Rd
//   TRN2 Vd.2S, Vn.2S, Vm.2S : 0x0E806800 | (Rm<<16) | (Rn<<5) | Rd
//   SXTL Vd.4S, Vn.4H        : 0x0F10A400 | (Rn<<5) | Rd
//   SSHR Vd.4S, Vn.4S, #1    : 0x4F3F0400 | (Rn<<5) | Rd
//   XTN  Vd.4H, Vn.4S        : 0x0E612800 | (Rn<<5) | Rd

TEXT ·fTransformWHT(SB),NOSPLIT,$0-48
	MOVD    in_base+0(FP), R0
	MOVD    out_base+24(FP), R1

	// ── LOAD 16 int16 with de-interleave (4-way) ────────────────────────────
	// VLD4 .4H: V_k.lane[j] = in[j*4+k] for k,j in [0..3].
	// So V0=(in[0],in[4],in[8],in[12]), V1=(in[1],...,in[13]),
	//    V2=(in[2],in[6],in[10],in[14]), V3=(in[3],in[7],in[11],in[15]).
	VLD4    (R0), [V0.H4, V1.H4, V2.H4, V3.H4]

	// ── PASS 1 in int16 ─────────────────────────────────────────────────────
	// a0 = V0 + V2 → V4 ;  a1 = V1 + V3 → V5
	// a2 = V1 - V3 → V6 ;  a3 = V0 - V2 → V7
	VADD    V2.H4, V0.H4, V4.H4
	VADD    V3.H4, V1.H4, V5.H4
	VSUB    V3.H4, V1.H4, V6.H4
	VSUB    V2.H4, V0.H4, V7.H4

	// tmp_p0 = a0+a1 → V8
	// tmp_p1 = a3+a2 → V9
	// tmp_p2 = a3-a2 → V10
	// tmp_p3 = a0-a1 → V11
	VADD    V5.H4, V4.H4, V8.H4
	VADD    V6.H4, V7.H4, V9.H4
	VSUB    V6.H4, V7.H4, V10.H4
	VSUB    V5.H4, V4.H4, V11.H4

	// ── 4×4 INT16 TRANSPOSE (col-vectors → row-vectors of tmp) ──────────────
	// Round 1 (TRN.4H on adjacent pairs):
	//   V12 = TRN1.4H V8, V9   = (V8[0],V9[0], V8[2],V9[2])
	//   V13 = TRN2.4H V8, V9   = (V8[1],V9[1], V8[3],V9[3])
	//   V14 = TRN1.4H V10, V11 = (V10[0],V11[0], V10[2],V11[2])
	//   V15 = TRN2.4H V10, V11 = (V10[1],V11[1], V10[3],V11[3])
	WORD    $0x0E49290C   // TRN1 V12.4H, V8.4H, V9.4H    (Rm=9, Rn=8, Rd=12)
	WORD    $0x0E49690D   // TRN2 V13.4H, V8.4H, V9.4H    (Rm=9, Rn=8, Rd=13)
	WORD    $0x0E4B294E   // TRN1 V14.4H, V10.4H, V11.4H  (Rm=11, Rn=10, Rd=14)
	WORD    $0x0E4B694F   // TRN2 V15.4H, V10.4H, V11.4H  (Rm=11, Rn=10, Rd=15)

	// Round 2 (TRN.2S — treat 32-bit pairs in 64-bit registers):
	//   V16 = TRN1.2S V12, V14 = (V12[0..31], V14[0..31])
	//        = ((V8[0],V9[0]), (V10[0],V11[0]))
	//        = (tmp[0+0*4], tmp[1+0*4], tmp[2+0*4], tmp[3+0*4]) = tmp ROW 0
	//   V17 = TRN2.2S V12, V14 = ((V8[2],V9[2]), (V10[2],V11[2])) = tmp ROW 2
	//   V18 = TRN1.2S V13, V15 = ((V8[1],V9[1]), (V10[1],V11[1])) = tmp ROW 1
	//   V19 = TRN2.2S V13, V15 = ((V8[3],V9[3]), (V10[3],V11[3])) = tmp ROW 3
	WORD    $0x0E8E2990   // TRN1 V16.2S, V12.2S, V14.2S  (Rm=14, Rn=12, Rd=16)
	WORD    $0x0E8E6991   // TRN2 V17.2S, V12.2S, V14.2S  (Rm=14, Rn=12, Rd=17)
	WORD    $0x0E8F29B2   // TRN1 V18.2S, V13.2S, V15.2S  (Rm=15, Rn=13, Rd=18)
	WORD    $0x0E8F69B3   // TRN2 V19.2S, V13.2S, V15.2S  (Rm=15, Rn=13, Rd=19)

	// After round 2: V16=row0, V18=row1, V17=row2, V19=row3 (int16x4 each).

	// ── WIDEN TO INT32 ──────────────────────────────────────────────────────
	// SXTL Vd.4S, Vn.4H : V20 = row0_i32, V21 = row1_i32, V22 = row2_i32, V23 = row3_i32
	WORD    $0x0F10A614   // SXTL V20.4S, V16.4H  (Rn=16, Rd=20)
	WORD    $0x0F10A655   // SXTL V21.4S, V18.4H  (Rn=18, Rd=21)
	WORD    $0x0F10A636   // SXTL V22.4S, V17.4H  (Rn=17, Rd=22)
	WORD    $0x0F10A677   // SXTL V23.4S, V19.4H  (Rn=19, Rd=23)

	// ── PASS 2 in int32 ─────────────────────────────────────────────────────
	// Now V20 = tmp_row0, V21 = tmp_row1, V22 = tmp_row2, V23 = tmp_row3.
	// Scalar pass 2 reads: a0 = tmp[0+i] + tmp[8+i] = tmp_row0[i] + tmp_row2[i]
	//                       a1 = tmp[4+i] + tmp[12+i] = tmp_row1[i] + tmp_row3[i]
	//                       a2 = tmp_row1[i] - tmp_row3[i]
	//                       a3 = tmp_row0[i] - tmp_row2[i]
	VADD    V22.S4, V20.S4, V24.S4   // V24 = a0 = row0 + row2
	VADD    V23.S4, V21.S4, V25.S4   // V25 = a1 = row1 + row3
	VSUB    V23.S4, V21.S4, V26.S4   // V26 = a2 = row1 - row3
	VSUB    V22.S4, V20.S4, V27.S4   // V27 = a3 = row0 - row2

	// b0 = (a0+a1) >> 1 → V28 ;  b1 = (a3+a2) >> 1 → V29
	// b2 = (a3-a2) >> 1 → V30 ;  b3 = (a0-a1) >> 1 → V31
	VADD    V25.S4, V24.S4, V28.S4   // V28 = a0 + a1
	VADD    V26.S4, V27.S4, V29.S4   // V29 = a3 + a2
	VSUB    V26.S4, V27.S4, V30.S4   // V30 = a3 - a2
	VSUB    V25.S4, V24.S4, V31.S4   // V31 = a0 - a1

	// SSHR by #1 (signed arithmetic right shift, per int32 lane)
	WORD    $0x4F3F079C   // SSHR V28.4S, V28.4S, #1
	WORD    $0x4F3F07BD   // SSHR V29.4S, V29.4S, #1
	WORD    $0x4F3F07DE   // SSHR V30.4S, V30.4S, #1
	WORD    $0x4F3F07FF   // SSHR V31.4S, V31.4S, #1

	// ── NARROW int32 → int16 ────────────────────────────────────────────────
	// b_r is row r of output (lane c = out[r*4+c]).
	// XTN Vd.4H, Vn.4S : narrow int32x4 → int16x4 (low 16 bits, no saturation).
	WORD    $0x0E612B94   // XTN V20.4H, V28.4S  (Rn=28, Rd=20)
	WORD    $0x0E612BB5   // XTN V21.4H, V29.4S  (Rn=29, Rd=21)
	WORD    $0x0E612BD6   // XTN V22.4H, V30.4S  (Rn=30, Rd=22)
	WORD    $0x0E612BF7   // XTN V23.4H, V31.4S  (Rn=31, Rd=23)

	// ── STORE 4 × int16x4 = 32 bytes to out ─────────────────────────────────
	// V20=out_row0, V21=out_row1, V22=out_row2, V23=out_row3 → contiguous.
	VST1    [V20.H4, V21.H4, V22.H4, V23.H4], (R1)

	RET
