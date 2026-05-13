// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// fTransformWHT computes the 4x4 Walsh-Hadamard Transform on the 16 DC
// coefficients (one per 4x4 block in a 16x16 macroblock).
//
// in[0..15] are the DC values from each 4x4 block's DCT output,
// but they are passed as in[n*16] (every 16th element) for the 4 rows of 4.
// We take them as a flat in[0..15] here (caller lays them out).
//
// Ported from FTransformWHT_C() in libwebp/src/dsp/enc.c.
func fTransformWHT(in []int16, out []int16) {
	var tmp [16]int32
	// Input is in[0*16], in[1*16], in[2*16], in[3*16] for each of 4 rows
	// but in libwebp the inner loop advances 'in' by 64 each iteration.
	// We receive 16 values directly, in row-major order [0..15].
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

// iTransform4x4 computes the inverse 4x4 DCT and adds residuals to pred.
// coeffs[16] are dequantized DCT coefficients in raster order (row-major).
// pred[16] is the 4x4 prediction block.
// out[16] receives the reconstructed pixels (clamped to [0,255]).
//
// Ported from ITransformOne in libwebp/src/dsp/enc.c.
// c1 = 20091 + 65536 = 85627, c2 = 35468 (= sin(pi/8)*sqrt(2)*65536)
// MUL1(a) = ((a * 20091) >> 16) + a = (a * 85627) >> 16
// MUL2(a) = (a * 35468) >> 16
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

// iTransformWHT computes the inverse WHT and adds the DC values back
// into the first coefficient of each 4x4 block's DCT array.
// out[n] gets added to tmp[n][0].
//
// Ported from VP8TransformWHT / ITransformWHT in libwebp.
func iTransformWHT(in []int16, out []int16) {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		a0 := int32(in[0+i]) + int32(in[12+i])
		a1 := int32(in[4+i]) + int32(in[8+i])
		a2 := int32(in[4+i]) - int32(in[8+i])
		a3 := int32(in[0+i]) - int32(in[12+i])
		tmp[0+i*4] = a0 + a1
		tmp[1+i*4] = a3 + a2
		tmp[2+i*4] = a3 - a2
		tmp[3+i*4] = a0 - a1
	}
	for i := 0; i < 4; i++ {
		dc0 := tmp[0+i*4] + tmp[1+i*4] + tmp[2+i*4] + tmp[3+i*4]
		dc1 := tmp[0+i*4] + tmp[1+i*4] - tmp[2+i*4] - tmp[3+i*4]
		dc2 := tmp[0+i*4] - tmp[1+i*4] - tmp[2+i*4] + tmp[3+i*4]
		dc3 := tmp[0+i*4] - tmp[1+i*4] + tmp[2+i*4] - tmp[3+i*4]
		out[i*4+0] = int16((dc0 + 3) >> 3)
		out[i*4+1] = int16((dc1 + 3) >> 3)
		out[i*4+2] = int16((dc2 + 3) >> 3)
		out[i*4+3] = int16((dc3 + 3) >> 3)
	}
}

// inverseWHT16 is a direct port of the decoder's inverseWHT16 from
// golang.org/x/image/vp8/idct.go.
//
// in[0..15] = dequantized WHT coefficients in raster order (row-major 4x4).
// out[n] = DC coefficient for block n (n = by*4+bx), matching d.coeff[n*16].
func inverseWHT16(in []int16, out []int16) {
	var m [16]int32
	// First pass: for each column i (0..3), butterfly over the 4 rows.
	// Reads: in[0+i], in[4+i], in[8+i], in[12+i] = column i of the 4x4 raster grid.
	// Stores results in m with layout: m[row_type*4 + col_i].
	for i := 0; i < 4; i++ {
		a0 := int32(in[0+i]) + int32(in[12+i])
		a1 := int32(in[4+i]) + int32(in[8+i])
		a2 := int32(in[4+i]) - int32(in[8+i])
		a3 := int32(in[0+i]) - int32(in[12+i])
		m[0+i] = a0 + a1  // row_type 0, col i
		m[8+i] = a0 - a1  // row_type 1, col i  (stored at m[8+i])
		m[4+i] = a3 + a2  // row_type 2, col i  (stored at m[4+i])
		m[12+i] = a3 - a2 // row_type 3, col i  (stored at m[12+i])
	}
	// Second pass: for each "output row" i (0..3), butterfly over the 4 columns.
	// Each output row i maps to 4 blocks (one per column bx=0..3).
	// The decoder writes: coeff[outBase+0]=block(bx=0), coeff[outBase+16]=block(bx=1),
	// coeff[outBase+32]=block(bx=2), coeff[outBase+48]=block(bx=3), then outBase+=64.
	// outBase sequence: 0, 64, 128, 192 → row types map to block rows 0,1,2,3.
	// We map to out[by*4+bx] = block at (bx,by).
	for i := 0; i < 4; i++ {
		// Row type i: reads m[i*4+0..3] = row_type_i for columns 0..3.
		dc := m[i*4+0] + 3
		a0 := dc + m[i*4+3]
		a1 := m[i*4+1] + m[i*4+2]
		a2 := m[i*4+1] - m[i*4+2]
		a3 := dc - m[i*4+3]
		// Decoder: coeff[outBase+0]=v0, coeff[outBase+16]=v1, coeff[outBase+32]=v2, coeff[outBase+48]=v3
		// outBase = i*64. coeff[n*16] = block n. So:
		// coeff[i*64+0] = coeff[(i*4+0)*16] → block i*4+0 (bx=0, by=i)
		// coeff[i*64+16]= coeff[(i*4+1)*16] → block i*4+1 (bx=1, by=i)
		// coeff[i*64+32]= coeff[(i*4+2)*16] → block i*4+2 (bx=2, by=i)
		// coeff[i*64+48]= coeff[(i*4+3)*16] → block i*4+3 (bx=3, by=i)
		out[i*4+0] = int16((a0 + a1) >> 3) // block(bx=0, by=i)
		out[i*4+1] = int16((a3 + a2) >> 3) // block(bx=1, by=i)
		out[i*4+2] = int16((a0 - a1) >> 3) // block(bx=2, by=i)
		out[i*4+3] = int16((a3 - a2) >> 3) // block(bx=3, by=i)
	}
}
