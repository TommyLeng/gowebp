// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING
//
// NEON kernels for the VP8L spatial-predictor residual (lossless/transform.go).
// Each pixel is a color.NRGBA = 4 bytes [R,G,B,A]; a 16-byte NEON vector holds
// 4 pixels. Residuals are computed with 8-bit wraparound (matching uint8(a-b)),
// so the encoded bitstream is byte-identical to the scalar reference.
//
// Hand-written assembly uses ABI0: arguments are read from the FP-relative
// frame via name+offset(FP).

#include "textflag.h"

// func predResSubRow(pixels []color.NRGBA, dstOff, srcOff int, out []color.NRGBA, n int)
//
// out[k] = pixels[dstOff+k] - pixels[srcOff+k]  for k in 0..n-1, n a multiple
// of 4 (the caller handles the n&3 tail). Covers predictors 1/2/3/4 — they are
// all "current minus a shifted copy of the image", differing only in srcOff:
//   mode 1 (left)      srcOff = dstOff - 1
//   mode 2 (top)       srcOff = dstOff - width
//   mode 3 (top-right) srcOff = dstOff - width + 1   (wraps at the row edge)
//   mode 4 (top-left)  srcOff = dstOff - width - 1
//
// FP frame: pixels{ptr@0,len@8,cap@16} dstOff@24 srcOff@32 out{ptr@40} n@64.
TEXT ·predResSubRow(SB),NOSPLIT,$0-72
	MOVD	pixels_base+0(FP), R0   // pixels.ptr
	MOVD	dstOff+24(FP), R1
	MOVD	srcOff+32(FP), R2
	MOVD	out_base+40(FP), R3     // out.ptr
	MOVD	n+64(FP), R4            // n (multiple of 4)

	LSL	$2, R1, R1              // dstOff*4 (4 bytes / pixel)
	ADD	R1, R0, R5             // R5 = &pixels[dstOff]
	LSL	$2, R2, R2              // srcOff*4
	ADD	R2, R0, R6             // R6 = &pixels[srcOff]
	LSR	$2, R4, R7             // R7 = n/4 vector iterations

loopSub:
	VLD1	(R5), [V0.B16]         // 4 current pixels
	VLD1	(R6), [V1.B16]         // 4 predictor pixels
	VSUB	V1.B16, V0.B16, V2.B16 // V2 = cur - pred (8-bit wrap)
	VST1	[V2.B16], (R3)
	ADD	$16, R5, R5
	ADD	$16, R6, R6
	ADD	$16, R3, R3
	SUBS	$1, R7, R7
	BNE	loopSub
	RET

// func predResBlackRow(pixels []color.NRGBA, dstOff int, out []color.NRGBA, n int)
//
// out[k] = pixels[dstOff+k] - {0,0,0,255}  (predictor 0, "black"). Only the
// alpha lane changes (A-255 == A+1 mod 256); R/G/B pass through.
//
// FP frame: pixels{ptr@0} dstOff@24 out{ptr@32} n@56.
TEXT ·predResBlackRow(SB),NOSPLIT,$0-64
	MOVD	pixels_base+0(FP), R0
	MOVD	dstOff+24(FP), R1
	MOVD	out_base+32(FP), R3
	MOVD	n+56(FP), R4

	LSL	$2, R1, R1
	ADD	R1, R0, R5             // &pixels[dstOff]
	LSR	$2, R4, R7             // n/4

	// V1 = {0,0,0,255} × 4. Bytes [00 00 00 FF] per pixel → LE u64 word
	// 0xFF000000FF000000, broadcast to both 64-bit lanes.
	MOVD	$0xFF000000FF000000, R8
	VDUP	R8, V1.D2

loopBlack:
	VLD1	(R5), [V0.B16]
	VSUB	V1.B16, V0.B16, V2.B16
	VST1	[V2.B16], (R3)
	ADD	$16, R5, R5
	ADD	$16, R3, R3
	SUBS	$1, R7, R7
	BNE	loopBlack
	RET

// func predResAvgRow(pixels []color.NRGBA, mode int, curOff, upOff int, out []color.NRGBA, n int)
//
// out[k] = pixels[curOff+k] - predictor(mode), for the averaging predictors
// 5..10. avg2(a,b) = (a+b)/2 (truncating) == NEON UHADD. Context streams are
// just the image read at pixel offsets relative to cur (curOff+k):
//   l  = cur-1 (curOff-1)   t  = upOff   tl = upOff-1   tr = upOff+1
//   5: avg2(avg2(l,tr), t)              6: avg2(l,tl)     7: avg2(l,t)
//   8: avg2(tl,t)            9: avg2(t,tr)               10: avg2(avg2(l,tl), avg2(t,tr))
//
// UHADD Vd.16B,Vn.16B,Vm.16B = (Vn+Vm)>>1 ; Go assembler lacks it, so the
// encoding 0x6E200400 | Rm<<16 | Rn<<5 | Rd is emitted via WORD.
//
// FP frame: pixels{ptr@0} mode@24 curOff@32 upOff@40 out{ptr@48} n@72.
TEXT ·predResAvgRow(SB),NOSPLIT,$0-80
	MOVD	pixels_base+0(FP), R0
	MOVD	mode+24(FP), R12
	MOVD	curOff+32(FP), R1
	MOVD	upOff+40(FP), R2
	MOVD	out_base+48(FP), R3
	MOVD	n+72(FP), R4

	LSL	$2, R1, R1
	ADD	R1, R0, R5             // R5  = &cur  (curOff)
	SUB	$4, R5, R9             // R9  = &l    (curOff-1)
	LSL	$2, R2, R2
	ADD	R2, R0, R6             // R6  = &t    (upOff)
	SUB	$4, R6, R10            // R10 = &tl   (upOff-1)
	ADD	$4, R6, R11            // R11 = &tr   (upOff+1)
	LSR	$2, R4, R7             // R7  = n/4 iterations

	CMP	$5, R12
	BEQ	m5
	CMP	$6, R12
	BEQ	m6
	CMP	$7, R12
	BEQ	m7
	CMP	$8, R12
	BEQ	m8
	CMP	$9, R12
	BEQ	m9
	CMP	$10, R12
	BEQ	m10
	RET

// 5: pred = avg2(avg2(l,tr), t)
m5:
	VLD1	(R5), [V0.B16]         // cur
	VLD1	(R9), [V1.B16]         // l
	VLD1	(R6), [V2.B16]         // t
	VLD1	(R11), [V4.B16]        // tr
	WORD	$0x6E240425            // UHADD V5,V1,V4  : (l+tr)>>1
	WORD	$0x6E2204A7            // UHADD V7,V5,V2  : (that+t)>>1
	VSUB	V7.B16, V0.B16, V8.B16
	VST1	[V8.B16], (R3)
	ADD	$16, R5, R5
	ADD	$16, R9, R9
	ADD	$16, R6, R6
	ADD	$16, R11, R11
	ADD	$16, R3, R3
	SUBS	$1, R7, R7
	BNE	m5
	RET

// 6: pred = avg2(l,tl)
m6:
	VLD1	(R5), [V0.B16]         // cur
	VLD1	(R9), [V1.B16]         // l
	VLD1	(R10), [V3.B16]        // tl
	WORD	$0x6E230427            // UHADD V7,V1,V3  : (l+tl)>>1
	VSUB	V7.B16, V0.B16, V8.B16
	VST1	[V8.B16], (R3)
	ADD	$16, R5, R5
	ADD	$16, R9, R9
	ADD	$16, R10, R10
	ADD	$16, R3, R3
	SUBS	$1, R7, R7
	BNE	m6
	RET

// 7: pred = avg2(l,t)
m7:
	VLD1	(R5), [V0.B16]         // cur
	VLD1	(R9), [V1.B16]         // l
	VLD1	(R6), [V2.B16]         // t
	WORD	$0x6E220427            // UHADD V7,V1,V2  : (l+t)>>1
	VSUB	V7.B16, V0.B16, V8.B16
	VST1	[V8.B16], (R3)
	ADD	$16, R5, R5
	ADD	$16, R9, R9
	ADD	$16, R6, R6
	ADD	$16, R3, R3
	SUBS	$1, R7, R7
	BNE	m7
	RET

// 8: pred = avg2(tl,t)
m8:
	VLD1	(R5), [V0.B16]         // cur
	VLD1	(R10), [V3.B16]        // tl
	VLD1	(R6), [V2.B16]         // t
	WORD	$0x6E220467            // UHADD V7,V3,V2  : (tl+t)>>1
	VSUB	V7.B16, V0.B16, V8.B16
	VST1	[V8.B16], (R3)
	ADD	$16, R5, R5
	ADD	$16, R10, R10
	ADD	$16, R6, R6
	ADD	$16, R3, R3
	SUBS	$1, R7, R7
	BNE	m8
	RET

// 9: pred = avg2(t,tr)
m9:
	VLD1	(R5), [V0.B16]         // cur
	VLD1	(R6), [V2.B16]         // t
	VLD1	(R11), [V4.B16]        // tr
	WORD	$0x6E240447            // UHADD V7,V2,V4  : (t+tr)>>1
	VSUB	V7.B16, V0.B16, V8.B16
	VST1	[V8.B16], (R3)
	ADD	$16, R5, R5
	ADD	$16, R6, R6
	ADD	$16, R11, R11
	ADD	$16, R3, R3
	SUBS	$1, R7, R7
	BNE	m9
	RET

// 10: pred = avg2(avg2(l,tl), avg2(t,tr))
m10:
	VLD1	(R5), [V0.B16]         // cur
	VLD1	(R9), [V1.B16]         // l
	VLD1	(R10), [V3.B16]        // tl
	VLD1	(R6), [V2.B16]         // t
	VLD1	(R11), [V4.B16]        // tr
	WORD	$0x6E230425            // UHADD V5,V1,V3  : (l+tl)>>1
	WORD	$0x6E240446            // UHADD V6,V2,V4  : (t+tr)>>1
	WORD	$0x6E2604A7            // UHADD V7,V5,V6  : avg of the two
	VSUB	V7.B16, V0.B16, V8.B16
	VST1	[V8.B16], (R3)
	ADD	$16, R5, R5
	ADD	$16, R9, R9
	ADD	$16, R10, R10
	ADD	$16, R6, R6
	ADD	$16, R11, R11
	ADD	$16, R3, R3
	SUBS	$1, R7, R7
	BNE	m10
	RET
