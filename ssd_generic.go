// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// ssd4x4 computes Sum of Squared Differences between src[16] and pred[16].
// Used for distortion measurement of reconstructed vs. source pixels.
func ssd4x4(src, pred []int16) int64 {
	var sum int64
	for i := 0; i < 16; i++ {
		d := int64(src[i]) - int64(pred[i])
		sum += d * d
	}
	return sum
}

// ssd16x16 computes Sum of Squared Differences between src[256] and pred[256]
// for an entire 16x16 macroblock.
func ssd16x16(src, pred []int16) int64 {
	var sum int64
	for i := 0; i < 256; i++ {
		d := int64(src[i]) - int64(pred[i])
		sum += d * d
	}
	return sum
}
