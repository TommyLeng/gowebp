// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// predictUV fills pred[64] (flat 8×8 array) with the chroma prediction for the
// given UV mode, reading from the reconstructed chroma buffer (reconPlane).
// stride is yuv.uvStride.
// mbX, mbY are macroblock coordinates (each chroma block is 8×8 pixels).
// imgW, imgH are the original luma image dimensions; uvW = (imgW+1)/2, uvH = (imgH+1)/2.
//
// Border conventions (same as intra16PredictFromRecon):
//   no top row  → use 127
//   no left col → use 129
func predictUV(mode int, reconPlane []uint8, stride, mbX, mbY, imgW, imgH int, pred []int16) {
	switch mode {
	case 0:
		predictUVDC(reconPlane, stride, mbX, mbY, imgW, imgH, pred)
	case 1:
		predictUVVE(reconPlane, stride, mbX, mbY, imgW, imgH, pred)
	case 2:
		predictUVHE(reconPlane, stride, mbX, mbY, imgW, imgH, pred)
	case 3:
		predictUVTM(reconPlane, stride, mbX, mbY, imgW, imgH, pred)
	}
}

// getReconUV reads a reconstructed chroma pixel with border padding.
// px, py are the top-left corner of the 8×8 chroma block in chroma-plane coords.
func getReconUV(reconPlane []uint8, stride, x, y, uvW, uvH int) int {
	if x < 0 {
		return 129
	}
	if y < 0 {
		return 127
	}
	if x >= uvW {
		x = uvW - 1
	}
	if y >= uvH {
		y = uvH - 1
	}
	return int(reconPlane[y*stride+x])
}

// predictUVDC fills pred[64] with the DC (mode 0) chroma prediction.
// DC = average of top row (8 pixels) + left column (8 pixels) from reconPlane.
func predictUVDC(reconPlane []uint8, stride, mbX, mbY, imgW, imgH int, pred []int16) {
	hasTop := mbY > 0
	hasLeft := mbX > 0
	px := mbX * 8
	py := mbY * 8
	uvW := (imgW + 1) / 2
	uvH := (imgH + 1) / 2
	_ = uvW
	_ = uvH

	dc := 0
	n := 0
	if hasTop {
		for i := 0; i < 8; i++ {
			x := px + i
			if x >= uvW {
				x = uvW - 1
			}
			dc += int(reconPlane[(py-1)*stride+x])
		}
		n += 8
	}
	if hasLeft {
		for i := 0; i < 8; i++ {
			y := py + i
			if y >= uvH {
				y = uvH - 1
			}
			dc += int(reconPlane[y*stride+(px-1)])
		}
		n += 8
	}
	var dcVal int
	if n == 0 {
		dcVal = 128
	} else {
		dcVal = (dc + n/2) / n
	}
	for i := range pred {
		pred[i] = int16(dcVal)
	}
}

// predictUVVE fills pred[64] with the vertical (mode 1) chroma prediction.
// Copies the top row of reconPlane down 8 times.
// Requires mbY > 0 (caller must check before calling).
func predictUVVE(reconPlane []uint8, stride, mbX, mbY, imgW, imgH int, pred []int16) {
	px := mbX * 8
	py := mbY * 8
	uvW := (imgW + 1) / 2

	if mbY == 0 {
		// No top row: use 127
		for i := range pred {
			pred[i] = 127
		}
		return
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			cx := px + x
			if cx >= uvW {
				cx = uvW - 1
			}
			pred[y*8+x] = int16(reconPlane[(py-1)*stride+cx])
		}
	}
}

// predictUVHE fills pred[64] with the horizontal (mode 2) chroma prediction.
// Copies the left column of reconPlane across 8 times.
// Requires mbX > 0 (caller must check before calling).
func predictUVHE(reconPlane []uint8, stride, mbX, mbY, imgW, imgH int, pred []int16) {
	px := mbX * 8
	py := mbY * 8
	uvH := (imgH + 1) / 2

	if mbX == 0 {
		// No left col: use 129
		for i := range pred {
			pred[i] = 129
		}
		return
	}
	for y := 0; y < 8; y++ {
		cy := py + y
		if cy >= uvH {
			cy = uvH - 1
		}
		v := int16(reconPlane[cy*stride+(px-1)])
		for x := 0; x < 8; x++ {
			pred[y*8+x] = v
		}
	}
}

// predictUVTM fills pred[64] with the TrueMotion (mode 3) chroma prediction.
// pred[y*8+x] = clip(top[x] + left[y] - topLeft), clamped to [0,255].
// Requires mbX > 0 and mbY > 0 (caller must check before calling).
func predictUVTM(reconPlane []uint8, stride, mbX, mbY, imgW, imgH int, pred []int16) {
	px := mbX * 8
	py := mbY * 8
	uvW := (imgW + 1) / 2
	uvH := (imgH + 1) / 2

	// topLeft: pixel at (px-1, py-1)
	topLeft := 128
	if mbX > 0 && mbY > 0 {
		topLeft = int(reconPlane[(py-1)*stride+(px-1)])
	} else if mbY > 0 {
		topLeft = 127 // no left, so topLeft is 127 (top border)
	} else if mbX > 0 {
		topLeft = 129 // no top, so topLeft is 129 (left border)
	}

	for y := 0; y < 8; y++ {
		leftV := 129
		if mbX > 0 {
			cy := py + y
			if cy >= uvH {
				cy = uvH - 1
			}
			leftV = int(reconPlane[cy*stride+(px-1)])
		}
		for x := 0; x < 8; x++ {
			topV := 127
			if mbY > 0 {
				cx := px + x
				if cx >= uvW {
					cx = uvW - 1
				}
				topV = int(reconPlane[(py-1)*stride+cx])
			}
			v := topV + leftV - topLeft
			pred[y*8+x] = int16(clip8i(v))
		}
	}
}
