// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING
//
// SSE2 Y-luma kernel for *image.NRGBA / *image.RGBA inputs (yuv.go).
//
// Both image types store pixels as [R, G, B, A] bytes — identical layout.
// One XMM register holds 4 pixels (16 bytes).
//
// Algorithm per 4-pixel batch:
//
//   1. Load 16 bytes; zero-extend to int16: pixels 0&1 in X1, pixels 2&3 in X0.
//   2. Build X_RG = [R0,G0,R1,G1, R2,G2,R3,G3] via PSHUFD+PUNPCKLQDQ.
//   3. Build X_GB = [G0,B0,G1,B1, G2,B2,G3,B3] via PSRLQ+PSHUFD+PUNPCKLQDQ.
//   4. PMADDWL: pass1 = 16839·R + 32767·G (exact int32×4)
//               pass2 =   292·G +  6420·B (exact int32×4)
//              (sum gives 16839·R + 33059·G + 6420·B; 33059 = 32767 + 292)
//   5. PADDL bias (1081344), PSRAL $16 → Y ∈ [16,235] as int32×4.
//   6. PACKSSLW→int16, PMAXSW/PMINSW clamp, PACKUSWB→uint8; store 4 bytes.
//
// Plan 9 ↔ Intel mnemonic map used here:
//   PMADDWL = PMADDWD   PSRAL = PSRAD   PACKSSLW = PACKSSDW
//   PSHUFD requires 3 operands: PSHUFD $imm, Xsrc, Xdst
//   PUNPCKLBW / PUNPCKHBW / PUNPCKLQDQ / PACKUSWB / PMAXSW / PMINSW
//   PSRLQ / PSHUFD all use their natural Intel names.
//
// IMPORTANT: R14 = goroutine ptr, R15 = p — must not be used as scratch.
// Safe GPRs: AX, BX, CX, DX, SI, DI, R8–R13.

#include "textflag.h"

// ── Coefficient and bias constants ────────────────────────────────────────

// yCoeffRG: [16839, 32767] × 4 — PMADDWL coefficients for (R, G) pairs.
DATA	·yCoeffRG<>+0(SB)/2,  $16839
DATA	·yCoeffRG<>+2(SB)/2,  $32767
DATA	·yCoeffRG<>+4(SB)/2,  $16839
DATA	·yCoeffRG<>+6(SB)/2,  $32767
DATA	·yCoeffRG<>+8(SB)/2,  $16839
DATA	·yCoeffRG<>+10(SB)/2, $32767
DATA	·yCoeffRG<>+12(SB)/2, $16839
DATA	·yCoeffRG<>+14(SB)/2, $32767
GLOBL	·yCoeffRG<>(SB), (NOPTR+RODATA), $16

// yCoeffGB: [292, 6420] × 4 — PMADDWL coefficients for (G, B) pairs.
DATA	·yCoeffGB<>+0(SB)/2,  $292
DATA	·yCoeffGB<>+2(SB)/2,  $6420
DATA	·yCoeffGB<>+4(SB)/2,  $292
DATA	·yCoeffGB<>+6(SB)/2,  $6420
DATA	·yCoeffGB<>+8(SB)/2,  $292
DATA	·yCoeffGB<>+10(SB)/2, $6420
DATA	·yCoeffGB<>+12(SB)/2, $292
DATA	·yCoeffGB<>+14(SB)/2, $6420
GLOBL	·yCoeffGB<>(SB), (NOPTR+RODATA), $16

// yBias: [1081344] × 4 as int32 — yuvHalf + (16 << 16) = 32768 + 1048576.
DATA	·yBias<>+0(SB)/4,  $1081344
DATA	·yBias<>+4(SB)/4,  $1081344
DATA	·yBias<>+8(SB)/4,  $1081344
DATA	·yBias<>+12(SB)/4, $1081344
GLOBL	·yBias<>(SB), (NOPTR+RODATA), $16

// yClampLo: [16] × 8 as int16 — lower Y clamp.
DATA	·yClampLo<>+0(SB)/2,  $16
DATA	·yClampLo<>+2(SB)/2,  $16
DATA	·yClampLo<>+4(SB)/2,  $16
DATA	·yClampLo<>+6(SB)/2,  $16
DATA	·yClampLo<>+8(SB)/2,  $16
DATA	·yClampLo<>+10(SB)/2, $16
DATA	·yClampLo<>+12(SB)/2, $16
DATA	·yClampLo<>+14(SB)/2, $16
GLOBL	·yClampLo<>(SB), (NOPTR+RODATA), $16

// yClampHi: [235] × 8 as int16 — upper Y clamp.
DATA	·yClampHi<>+0(SB)/2,  $235
DATA	·yClampHi<>+2(SB)/2,  $235
DATA	·yClampHi<>+4(SB)/2,  $235
DATA	·yClampHi<>+6(SB)/2,  $235
DATA	·yClampHi<>+8(SB)/2,  $235
DATA	·yClampHi<>+10(SB)/2, $235
DATA	·yClampHi<>+12(SB)/2, $235
DATA	·yClampHi<>+14(SB)/2, $235
GLOBL	·yClampHi<>(SB), (NOPTR+RODATA), $16

// func yuvYRowNRGBASSE2(pix []uint8, srcOff, n4 int, dst []uint8, dstOff int)
//
// FP frame (ABI0):
//   pix_base+0(FP)   pix_len+8(FP)   pix_cap+16(FP)
//   srcOff+24(FP)    (byte offset into pix)
//   n4+32(FP)        (pixel count, multiple of 4)
//   dst_base+40(FP)  dst_len+48(FP)  dst_cap+56(FP)
//   dstOff+64(FP)    (byte offset into dst)
//
// Register map (live across iterations):
//   AX = src pointer   DI = dst pointer   CX = n4/4 (loop counter)
//   X4 = yCoeffRG      X5 = yCoeffGB      X6 = yBias
//   X7 = yClampLo      X8 = yClampHi      X15 = zero
//
// Per-iteration temporaries: X0–X3, X9–X13.
TEXT ·yuvYRowNRGBASSE2(SB),NOSPLIT,$0-72
	MOVQ	pix_base+0(FP), AX
	MOVQ	srcOff+24(FP), BX
	ADDQ	BX, AX			// AX = &pix[srcOff]
	MOVQ	n4+32(FP), CX
	MOVQ	dst_base+40(FP), DI
	MOVQ	dstOff+64(FP), DX
	ADDQ	DX, DI			// DI = &dst[dstOff]
	SHRQ	$2, CX			// CX = n4/4 iterations

	// Load constants once into XMM registers; they survive all iterations.
	PXOR	X15, X15
	MOVOU	·yCoeffRG<>(SB), X4
	MOVOU	·yCoeffGB<>(SB), X5
	MOVOU	·yBias<>(SB), X6
	MOVOU	·yClampLo<>(SB), X7
	MOVOU	·yClampHi<>(SB), X8

loopY:
	// ── Step 1: load 4 NRGBA pixels, zero-extend bytes → int16 ─────────
	MOVOU	(AX), X0		// X0 = [R0,G0,B0,A0, R1,G1,B1,A1, R2,G2,B2,A2, R3,G3,B3,A3]

	MOVO	X0, X1
	PUNPCKLBW X15, X1		// X1 = [R0,G0,B0,A0, R1,G1,B1,A1] as int16×8
	PUNPCKHBW X15, X0		// X0 = [R2,G2,B2,A2, R3,G3,B3,A3] as int16×8

	// ── Step 2: build X_RG = [R0,G0,R1,G1, R2,G2,R3,G3] ───────────────
	// PSHUFD $0x88 selects dwords [0,2,0,2] → low 64 bits = [R0,G0,R1,G1].
	PSHUFD	$0x88, X1, X3		// X3 = [R0,G0,R1,G1, R0,G0,R1,G1]
	PSHUFD	$0x88, X0, X9		// X9 = [R2,G2,R3,G3, R2,G2,R3,G3]
	PUNPCKLQDQ X9, X3		// X3 = [R0,G0,R1,G1, R2,G2,R3,G3]  ← X_RG

	// ── Step 3: build X_GB = [G0,B0,G1,B1, G2,B2,G3,B3] ───────────────
	// PSRLQ $16 shifts each 64-bit lane right by 16 bits (1 int16):
	//   [R,G,B,A] → [G,B,A,0]  ⟹  low 64 bits = [G0,B0,A0,0, G1,B1,A1,0]
	// PSHUFD $0x88 then picks [G,B] pairs into both halves.
	MOVO	X1, X10
	PSRLQ	$16, X10		// X10 = [G0,B0,A0,0, G1,B1,A1,0] as int16
	PSHUFD	$0x88, X10, X10		// X10 = [G0,B0,G1,B1, G0,B0,G1,B1]
	MOVO	X0, X11
	PSRLQ	$16, X11		// X11 = [G2,B2,A2,0, G3,B3,A3,0]
	PSHUFD	$0x88, X11, X11		// X11 = [G2,B2,G3,B3, G2,B2,G3,B3]
	PUNPCKLQDQ X11, X10		// X10 = [G0,B0,G1,B1, G2,B2,G3,B3]  ← X_GB

	// ── Step 4: PMADDWL — two passes for exact int32 result ─────────────
	// pass1: 16839·R + 32767·G  (→ 16839·R + 32767·G as int32×4)
	MOVO	X3, X12
	PMADDWL	X4, X12		// X12 = [16839·R0+32767·G0, ..., 16839·R3+32767·G3]
	// pass2: 292·G + 6420·B
	MOVO	X10, X13
	PMADDWL	X5, X13		// X13 = [292·G0+6420·B0, ..., 292·G3+6420·B3]

	// ── Step 5: accumulate, add bias, arithmetic shift right 16 ─────────
	PADDL	X13, X12		// X12 = 16839·R + 33059·G + 6420·B  per pixel
	PADDL	X6, X12		// X12 += 1081344
	PSRAL	$16, X12		// X12 = [Y0,Y1,Y2,Y3] as int32 ∈ [16,235]

	// ── Step 6: pack to uint8 and store 4 bytes ──────────────────────────
	PACKSSLW X12, X12		// int32×4 → int16×8: [Y0,Y1,Y2,Y3] doubled
	PMAXSW	X7, X12		// clamp ≥ 16
	PMINSW	X8, X12		// clamp ≤ 235
	PACKUSWB X12, X12		// int16×8 → uint8×16: [Y0,Y1,Y2,Y3,...] in low bytes
	MOVL	X12, R10		// extract low 32 bits = 4 bytes [Y0,Y1,Y2,Y3]
	MOVL	R10, (DI)		// store 4 Y values

	ADDQ	$16, AX
	ADDQ	$4, DI
	SUBQ	$1, CX
	JNZ	loopY
	RET
