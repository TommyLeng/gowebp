// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build !amd64 && !arm64

package gowebp

// sad4x4 computes the Sum of Absolute Differences between a 4×4 source
// and prediction block. Used for cheap mode pre-screening before DCT.
func sad4x4(src, pred []int16) int64 {
	var s int64
	for i := 0; i < 16; i++ {
		d := int64(src[i]) - int64(pred[i])
		if d < 0 {
			d = -d
		}
		s += d
	}
	return s
}
