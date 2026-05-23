// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build !arm64

package gowebp

// iTransform4x4 computes the inverse 4x4 DCT and adds residuals to pred.
// coeffs[16] are dequantized DCT coefficients in raster order (row-major).
// pred[16] is the 4x4 prediction block.
// out[16] receives the reconstructed pixels (clamped to [0,255]).
//
// Ported from ITransformOne in libwebp/src/dsp/enc.c.
// c1 = 20091 + 65536 = 85627, c2 = 35468 (= sin(pi/8)*sqrt(2)*65536)
// MUL1(a) = ((a * 20091) >> 16) + a = (a * 85627) >> 16
// MUL2(a) = (a * 35468) >> 16
//
// Scalar fallback for non-arm64 platforms. The arm64 NEON implementation
// lives in dct_arm64.s.
func iTransform4x4(coeffs []int16, pred []int16, out []int16) {
	// c1 = 85627 = 65536 * cos(pi/8) * sqrt(2) (exact same value as decoder's idct.go).
	// c2 = 35468 = 65536 * sin(pi/8) * sqrt(2).
	// Using the decoder's exact formula: (a * c1) >> 16 to match integer rounding.
	// The two-step form ((a*20091)>>16)+a is algebraically equivalent but differs
	// for some negative integer values due to different rounding behavior.
	const c1 = 85627 // matches golang.org/x/image/vp8/idct.go
	const c2 = 35468

	mul1 := func(a int32) int32 { return (a * c1) >> 16 }
	mul2 := func(a int32) int32 { return (a * c2) >> 16 }

	// Vertical pass: for each column i, butterfly over the 4 rows.
	// coeffs is in raster order: row r, col c → coeffs[r*4+c].
	// Column i has elements: coeffs[0+i], coeffs[4+i], coeffs[8+i], coeffs[12+i].
	// tmp[row][col] layout: stored as tmp[col*4+row] to match the decoder's m[i][j].
	var tmp [4][4]int32
	for i := 0; i < 4; i++ { // column i
		a := int32(coeffs[0+i]) + int32(coeffs[8+i])
		b := int32(coeffs[0+i]) - int32(coeffs[8+i])
		c := mul2(int32(coeffs[4+i])) - mul1(int32(coeffs[12+i]))
		d := mul1(int32(coeffs[4+i])) + mul2(int32(coeffs[12+i]))
		tmp[i][0] = a + d
		tmp[i][1] = b + c
		tmp[i][2] = b - c
		tmp[i][3] = a - d
	}

	// Horizontal pass: for each row j, butterfly over the 4 columns.
	// Each output row j = [pixel(j,0), pixel(j,1), pixel(j,2), pixel(j,3)].
	for j := 0; j < 4; j++ { // row j
		dc := tmp[0][j] + 4
		a := dc + tmp[2][j]
		b := dc - tmp[2][j]
		c := mul2(tmp[1][j]) - mul1(tmp[3][j])
		d := mul1(tmp[1][j]) + mul2(tmp[3][j])
		out[j*4+0] = int16(clip8(int(pred[j*4+0]) + int((a+d)>>3)))
		out[j*4+1] = int16(clip8(int(pred[j*4+1]) + int((b+c)>>3)))
		out[j*4+2] = int16(clip8(int(pred[j*4+2]) + int((b-c)>>3)))
		out[j*4+3] = int16(clip8(int(pred[j*4+3]) + int((a-d)>>3)))
	}
}
