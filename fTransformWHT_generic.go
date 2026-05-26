// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build !arm64 && !amd64

package gowebp

// fTransformWHT computes the 4x4 Walsh-Hadamard Transform on the 16 DC
// coefficients (one per 4x4 block in a 16x16 macroblock).
//
// Scalar fallback for non-SIMD platforms.
//
// Ported from FTransformWHT_C() in libwebp/src/dsp/enc.c.
func fTransformWHT(in []int16, out []int16) {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		a0 := int32(in[0+i*4]) + int32(in[2+i*4])
		a1 := int32(in[1+i*4]) + int32(in[3+i*4])
		a2 := int32(in[1+i*4]) - int32(in[3+i*4])
		a3 := int32(in[0+i*4]) - int32(in[2+i*4])
		tmp[0+i*4] = a0 + a1
		tmp[1+i*4] = a3 + a2
		tmp[2+i*4] = a3 - a2
		tmp[3+i*4] = a0 - a1
	}
	for i := 0; i < 4; i++ {
		a0 := tmp[0+i] + tmp[8+i]
		a1 := tmp[4+i] + tmp[12+i]
		a2 := tmp[4+i] - tmp[12+i]
		a3 := tmp[0+i] - tmp[8+i]
		b0 := a0 + a1
		b1 := a3 + a2
		b2 := a3 - a2
		b3 := a0 - a1
		out[0+i] = int16(b0 >> 1)
		out[4+i] = int16(b1 >> 1)
		out[8+i] = int16(b2 >> 1)
		out[12+i] = int16(b3 >> 1)
	}
}
