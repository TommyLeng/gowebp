// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build !amd64 && !arm64

package gowebp

// fTransform computes the 4x4 forward DCT of (src - ref), storing into out[16].
// Scalar fallback for non-SIMD platforms.
//
// Ported from FTransform_C() in libwebp/src/dsp/enc.c.
// The coefficients 2217 and 5352 approximate cos(pi/8)*sqrt(2) and sin(pi/8)*sqrt(2)
// scaled by 4096, matching the VP8 specification section 14.4.
func fTransform(src []int16, ref []int16, out []int16) {
	var tmp [16]int32
	// Horizontal pass
	for i := 0; i < 4; i++ {
		d0 := int32(src[i*4+0]) - int32(ref[i*4+0])
		d1 := int32(src[i*4+1]) - int32(ref[i*4+1])
		d2 := int32(src[i*4+2]) - int32(ref[i*4+2])
		d3 := int32(src[i*4+3]) - int32(ref[i*4+3])
		a0 := d0 + d3
		a1 := d1 + d2
		a2 := d1 - d2
		a3 := d0 - d3
		tmp[0+i*4] = (a0 + a1) * 8
		tmp[1+i*4] = (a2*2217 + a3*5352 + 1812) >> 9
		tmp[2+i*4] = (a0 - a1) * 8
		tmp[3+i*4] = (a3*2217 - a2*5352 + 937) >> 9
	}
	// Vertical pass
	for i := 0; i < 4; i++ {
		a0 := tmp[0+i] + tmp[12+i]
		a1 := tmp[4+i] + tmp[8+i]
		a2 := tmp[4+i] - tmp[8+i]
		a3 := tmp[0+i] - tmp[12+i]
		out[0+i] = int16((a0 + a1 + 7) >> 4)
		extra := int32(0)
		if a3 != 0 {
			extra = 1
		}
		out[4+i] = int16(((a2*2217 + a3*5352 + 12000) >> 16) + extra)
		out[8+i] = int16((a0 - a1 + 7) >> 4)
		out[12+i] = int16((a3*2217 - a2*5352 + 51000) >> 16)
	}
}
