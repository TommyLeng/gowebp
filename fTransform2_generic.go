// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build !arm64 && !amd64

package gowebp

// fTransform2Plane computes the 4×4 forward DCT for two horizontally-adjacent
// blocks, reading directly from the pixel plane (uint8) and int16 prediction.
//
// srcPlane: the full Y (or U/V) plane as uint8
// srcStride: row stride of srcPlane (in bytes)
// srcX, srcY: top-left pixel coordinates of the FIRST block (second is at srcX+4)
// pred: int16 prediction buffer, row-major, predStride int16 elements per row
//
//	pred[r*predStride + 0..3] = row r of first block's prediction
//	pred[r*predStride + 4..7] = row r of second block's prediction
//
// out: [32]int16, out[0..15]=DCT of first block, out[16..31]=DCT of second block
func fTransform2Plane(srcPlane []byte, srcStride, srcX, srcY int, pred []int16, predStride int, out []int16) {
	var tmp0, tmp1, p0, p1 [16]int16
	for r := 0; r < 4; r++ {
		row := srcPlane[(srcY+r)*srcStride:]
		pr := pred[r*predStride:]
		for c := 0; c < 4; c++ {
			tmp0[r*4+c] = int16(row[srcX+c])
			p0[r*4+c] = pr[c]
			tmp1[r*4+c] = int16(row[srcX+4+c])
			p1[r*4+c] = pr[4+c]
		}
	}
	fTransform(tmp0[:], p0[:], out[:16])
	fTransform(tmp1[:], p1[:], out[16:])
}
