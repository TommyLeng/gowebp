// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build amd64

package gowebp

func init() {
	computeYNRGBA = yuvYRowNRGBASSE2
}

// yuvYRowNRGBASSE2 computes Y luma values for n4 NRGBA/RGBA pixels using SSE2.
//
// Input:  pix[srcOff .. srcOff + n4*4 - 1]  (4 bytes per pixel: R,G,B,A)
// Output: dst[dstOff .. dstOff + n4 - 1]    (1 byte per Y value)
//
// n4 must be a multiple of 4 (the caller handles the n%4 tail with the scalar
// path).  Formula per pixel:
//
//	Y = (16839*R + 33059*G + 6420*B + 1081344) >> 16, clamped to [16, 235]
//
// 33059 exceeds int16 range, so it is split across two PMADDWD passes:
//
//	pass1 = PMADDWD([R,G], [16839, 32767]) → 16839*R + 32767*G  (exact int32)
//	pass2 = PMADDWD([G,B], [292,   6420])  →  292*G  +  6420*B  (exact int32)
//	Y     = (pass1 + pass2 + bias) >> 16   where bias = 1081344
//
// Output is byte-identical to the scalar loop in rgbaToYUV420 for every
// (R,G,B) ∈ [0,255]³.
//
//go:noescape
func yuvYRowNRGBASSE2(pix []uint8, srcOff, n4 int, dst []uint8, dstOff int)
