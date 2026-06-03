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
