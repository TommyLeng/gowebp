// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING
//
// SSE2 kernels for VP8L spatial-predictor residuals (lossless/transform.go).
// Each pixel is a color.NRGBA = 4 bytes [R,G,B,A]; one SSE2 register (XMM)
// holds 4 pixels (16 bytes).  Residuals use 8-bit wrapping subtraction (PSUBB),
// which matches uint8(a−b) exactly — output is byte-identical to the scalar.
//
// IMPORTANT: amd64 ABI register conventions:
//   R14 = goroutine pointer (g) — MUST NOT be used as scratch
//   R15 = m's processor (p)    — avoid as scratch
//   Safe GPRs: AX, BX, CX, DX, SI, DI, R8, R9, R10, R11, R12, R13

#include "textflag.h"

// func predResSubRowSSE2(pixels []color.NRGBA, dstOff, srcOff int, out []color.NRGBA, n int)
//
// out[k] = pixels[dstOff+k] − pixels[srcOff+k]  for k in 0..n-1, n mult of 4.
// PSUBB: packed 8-bit wrapping subtract, identical to Go's uint8(a−b).
// Covers predictors 1 (left), 2 (top), 3 (top-right), 4 (top-left) — all of
// which are "current pixel minus a fixed offset into the image".
//
// FP frame (amd64 ABI0):
//   pixels_base+0(FP)   = pixels.ptr
//   pixels_len+8(FP)    = pixels.len   (unused)
//   pixels_cap+16(FP)   = pixels.cap   (unused)
//   dstOff+24(FP)       = dstOff  (pixel index)
//   srcOff+32(FP)       = srcOff  (pixel index)
//   out_base+40(FP)     = out.ptr
//   out_len+48(FP)      = out.len      (unused)
//   out_cap+56(FP)      = out.cap      (unused)
//   n+64(FP)            = n            (multiple of 4; caller guarantees n >= 4)
TEXT ·predResSubRowSSE2(SB),NOSPLIT,$0-72
	MOVQ	pixels_base+0(FP), AX	// AX = pixels.ptr
	MOVQ	dstOff+24(FP), BX	// BX = dstOff  (pixel index)
	MOVQ	srcOff+32(FP), CX	// CX = srcOff  (pixel index)
	MOVQ	out_base+40(FP), DI	// DI = out.ptr
	MOVQ	n+64(FP), SI		// SI = n  (multiple of 4)

	// Convert pixel indices to byte offsets (4 bytes per color.NRGBA).
	LEAQ	(AX)(BX*4), R8		// R8 = &pixels[dstOff]
	LEAQ	(AX)(CX*4), R9		// R9 = &pixels[srcOff]
	SHRQ	$2, SI			// SI = n/4 = number of 16-byte (4-pixel) iterations

loopSub:
	MOVOU	(R8), X0		// load 4 current pixels   (16 bytes, unaligned ok)
	MOVOU	(R9), X1		// load 4 predictor pixels (16 bytes, unaligned ok)
	PSUBB	X1, X0			// X0 = current − predictor  (8-bit wrapping per byte)
	MOVOU	X0, (DI)		// store 4 result pixels
	ADDQ	$16, R8
	ADDQ	$16, R9
	ADDQ	$16, DI
	SUBQ	$1, SI
	JNZ	loopSub
	RET

// Black-pixel constant: {R=0,G=0,B=0,A=255} × 4 pixels = 16 bytes.
// color.NRGBA memory layout: [R,G,B,A] = bytes [0x00,0x00,0x00,0xFF] per pixel.
// As a uint32 little-endian: (A<<24)|(B<<16)|(G<<8)|R = 0xFF000000.
DATA	·blackMask<>+0(SB)/4, $0xFF000000
DATA	·blackMask<>+4(SB)/4, $0xFF000000
DATA	·blackMask<>+8(SB)/4, $0xFF000000
DATA	·blackMask<>+12(SB)/4, $0xFF000000
GLOBL	·blackMask<>(SB), (NOPTR+RODATA), $16

// func predResBlackRowSSE2(pixels []color.NRGBA, dstOff int, out []color.NRGBA, n int)
//
// out[k] = pixels[dstOff+k] − {0,0,0,255}  for k in 0..n-1, n mult of 4.
// Predictor 0 ("black"): constant {R=0,G=0,B=0,A=255} for all x>0, y>0.
//
// FP frame:
//   pixels_base+0(FP)   = pixels.ptr
//   pixels_len+8(FP)    = pixels.len   (unused)
//   pixels_cap+16(FP)   = pixels.cap   (unused)
//   dstOff+24(FP)       = dstOff  (pixel index)
//   out_base+32(FP)     = out.ptr
//   out_len+40(FP)      = out.len      (unused)
//   out_cap+48(FP)      = out.cap      (unused)
//   n+56(FP)            = n            (multiple of 4; caller guarantees n >= 4)
TEXT ·predResBlackRowSSE2(SB),NOSPLIT,$0-64
	MOVQ	pixels_base+0(FP), AX
	MOVQ	dstOff+24(FP), BX
	MOVQ	out_base+32(FP), DI
	MOVQ	n+56(FP), SI

	LEAQ	(AX)(BX*4), R8		// R8 = &pixels[dstOff]
	SHRQ	$2, SI			// SI = n/4
	MOVOU	·blackMask<>(SB), X1	// X1 = {0,0,0,255} × 4 pixels (constant)

loopBlack:
	MOVOU	(R8), X0		// load 4 current pixels
	PSUBB	X1, X0			// subtract {0,0,0,255} from each pixel (8-bit wrap)
	MOVOU	X0, (DI)		// store result
	ADDQ	$16, R8
	ADDQ	$16, DI
	SUBQ	$1, SI
	JNZ	loopBlack
	RET

// LSB correction mask for floor_avg via PAVGB fixup.
// PAVGB computes (a+b+1)>>1; subtracting PAND(PXOR(a,b), avgMask) converts
// it to (a+b)>>1 — matching uint8((a+b)/2) exactly.
// Proof: (a+b) is odd iff bit-0 of (a XOR b) is 1; that is exactly when
// PAVGB rounds up by 1. So correction = (a XOR b) & 0x01 per byte.
DATA	·avgMask<>+0(SB)/8, $0x0101010101010101
DATA	·avgMask<>+8(SB)/8, $0x0101010101010101
GLOBL	·avgMask<>(SB), (NOPTR+RODATA), $16

// func predResAvgRowSSE2(pixels []color.NRGBA, mode, curOff, upOff int, out []color.NRGBA, n int)
//
// Averaging predictors 5–10.  floor_avg(a,b) is computed as:
//   PAVGB(a,b) − PAND(PXOR(a,b), avgMask)
// which is (a+b)>>1 — byte-identical to uint8((a+b)/2) for all 256×256 pairs.
//
// Pointer layout (4 bytes per color.NRGBA pixel):
//   cur = &pixels[curOff]       l  = &pixels[curOff−1]
//   t   = &pixels[upOff]        tl = &pixels[upOff−1]
//   tr  = &pixels[upOff+1]
//
// FP frame (ABI0):
//   pixels_base+0(FP)  pixels_len+8(FP)  pixels_cap+16(FP)
//   mode+24(FP)  curOff+32(FP)  upOff+40(FP)
//   out_base+48(FP)  out_len+56(FP)  out_cap+64(FP)
//   n+72(FP)
//
// Register map (set before dispatch, live throughout):
//   AX = pixels.ptr  DI = out.ptr  SI = n/4 (loop counter)
//   R8 = cur         R9 = t        X4 = avgMask
//   R10/R11/R12 set per-mode for l/tl/tr as needed.
//   X0 = cur batch   X1 = first operand / result
//   X2 = second operand   X3 = third load (mode 10 tr)
//   X5 = floor_avg temp   X6 = second floor_avg temp (mode 10)
TEXT ·predResAvgRowSSE2(SB),NOSPLIT,$0-80
	MOVQ	pixels_base+0(FP), AX
	MOVQ	mode+24(FP), BX
	MOVQ	curOff+32(FP), CX
	MOVQ	upOff+40(FP), DX
	MOVQ	out_base+48(FP), DI
	MOVQ	n+72(FP), SI

	LEAQ	(AX)(CX*4), R8		// R8 = cur = &pixels[curOff]
	LEAQ	(AX)(DX*4), R9		// R9 = t   = &pixels[upOff]
	SHRQ	$2, SI			// SI = n/4
	MOVOU	·avgMask<>(SB), X4	// X4 = 0x01×16 correction mask (constant)

	CMPQ	BX, $5;  JEQ	avgMode5
	CMPQ	BX, $6;  JEQ	avgMode6
	CMPQ	BX, $7;  JEQ	avgMode7
	CMPQ	BX, $8;  JEQ	avgMode8
	CMPQ	BX, $9;  JEQ	avgMode9
	JMP	avgMode10

// ── mode 5: floor_avg(floor_avg(l, tr), t) ───────────────────────────────
avgMode5:
	LEAQ	-4(R8), R10		// R10 = l  = &pixels[curOff−1]
	LEAQ	4(R9), R12		// R12 = tr = &pixels[upOff+1]
avgLoop5:
	MOVOU	(R10), X1
	MOVOU	(R12), X2
	MOVO	X1, X5;  PXOR X2, X5;  PAND X4, X5;  PAVGB X2, X1;  PSUBB X5, X1	// X1 = floor_avg(l, tr)
	MOVOU	(R9), X2
	MOVO	X1, X5;  PXOR X2, X5;  PAND X4, X5;  PAVGB X2, X1;  PSUBB X5, X1	// X1 = floor_avg(a, t)
	MOVOU	(R8), X0;  PSUBB X1, X0;  MOVOU X0, (DI)
	ADDQ	$16, R8;  ADDQ	$16, R10;  ADDQ	$16, R9;  ADDQ	$16, R12;  ADDQ	$16, DI
	SUBQ	$1, SI;  JNZ	avgLoop5
	RET

// ── mode 6: floor_avg(l, tl) ─────────────────────────────────────────────
avgMode6:
	LEAQ	-4(R8), R10		// R10 = l
	LEAQ	-4(R9), R11		// R11 = tl = &pixels[upOff−1]
avgLoop6:
	MOVOU	(R10), X1
	MOVOU	(R11), X2
	MOVO	X1, X5;  PXOR X2, X5;  PAND X4, X5;  PAVGB X2, X1;  PSUBB X5, X1
	MOVOU	(R8), X0;  PSUBB X1, X0;  MOVOU X0, (DI)
	ADDQ	$16, R8;  ADDQ	$16, R10;  ADDQ	$16, R11;  ADDQ	$16, DI
	SUBQ	$1, SI;  JNZ	avgLoop6
	RET

// ── mode 7: floor_avg(l, t) ──────────────────────────────────────────────
avgMode7:
	LEAQ	-4(R8), R10		// R10 = l
avgLoop7:
	MOVOU	(R10), X1
	MOVOU	(R9), X2
	MOVO	X1, X5;  PXOR X2, X5;  PAND X4, X5;  PAVGB X2, X1;  PSUBB X5, X1
	MOVOU	(R8), X0;  PSUBB X1, X0;  MOVOU X0, (DI)
	ADDQ	$16, R8;  ADDQ	$16, R10;  ADDQ	$16, R9;  ADDQ	$16, DI
	SUBQ	$1, SI;  JNZ	avgLoop7
	RET

// ── mode 8: floor_avg(tl, t) ─────────────────────────────────────────────
avgMode8:
	LEAQ	-4(R9), R11		// R11 = tl
avgLoop8:
	MOVOU	(R11), X1
	MOVOU	(R9), X2
	MOVO	X1, X5;  PXOR X2, X5;  PAND X4, X5;  PAVGB X2, X1;  PSUBB X5, X1
	MOVOU	(R8), X0;  PSUBB X1, X0;  MOVOU X0, (DI)
	ADDQ	$16, R8;  ADDQ	$16, R11;  ADDQ	$16, R9;  ADDQ	$16, DI
	SUBQ	$1, SI;  JNZ	avgLoop8
	RET

// ── mode 9: floor_avg(t, tr) ─────────────────────────────────────────────
avgMode9:
	LEAQ	4(R9), R12		// R12 = tr
avgLoop9:
	MOVOU	(R9), X1
	MOVOU	(R12), X2
	MOVO	X1, X5;  PXOR X2, X5;  PAND X4, X5;  PAVGB X2, X1;  PSUBB X5, X1
	MOVOU	(R8), X0;  PSUBB X1, X0;  MOVOU X0, (DI)
	ADDQ	$16, R8;  ADDQ	$16, R9;  ADDQ	$16, R12;  ADDQ	$16, DI
	SUBQ	$1, SI;  JNZ	avgLoop9
	RET

// ── mode 10: floor_avg(floor_avg(l, tl), floor_avg(t, tr)) ───────────────
avgMode10:
	LEAQ	-4(R8), R10		// R10 = l
	LEAQ	-4(R9), R11		// R11 = tl
	LEAQ	4(R9), R12		// R12 = tr
avgLoop10:
	MOVOU	(R10), X1;  MOVOU	(R11), X2				// a = floor_avg(l, tl)
	MOVO	X1, X5;  PXOR X2, X5;  PAND X4, X5;  PAVGB X2, X1;  PSUBB X5, X1
	MOVOU	(R9), X2;  MOVOU	(R12), X3				// b = floor_avg(t, tr)
	MOVO	X2, X6;  PXOR X3, X6;  PAND X4, X6;  PAVGB X3, X2;  PSUBB X6, X2
	MOVO	X1, X5;  PXOR X2, X5;  PAND X4, X5;  PAVGB X2, X1;  PSUBB X5, X1	// pred = floor_avg(a, b)
	MOVOU	(R8), X0;  PSUBB X1, X0;  MOVOU X0, (DI)
	ADDQ	$16, R8;  ADDQ	$16, R10;  ADDQ	$16, R11;  ADDQ	$16, R9;  ADDQ	$16, R12;  ADDQ	$16, DI
	SUBQ	$1, SI;  JNZ	avgLoop10
	RET
