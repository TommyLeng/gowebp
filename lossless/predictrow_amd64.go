// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build amd64

package lossless

import "image/color"

// predResSubRowSSE2 computes out[k] = pixels[dstOff+k] - pixels[srcOff+k]
// for k in 0..n-1, n a multiple of 4. Each element is a color.NRGBA (4 bytes).
// Uses SSE2 PSUBB: packed 8-bit wraparound subtract — identical to uint8(a-b).
// Covers predictors 1 (left), 2 (top), 3 (top-right), 4 (top-left).
//
//go:noescape
func predResSubRowSSE2(pixels []color.NRGBA, dstOff, srcOff int, out []color.NRGBA, n int)

// predResBlackRowSSE2 computes out[k] = pixels[dstOff+k] - {0,0,0,255}
// for k in 0..n-1, n a multiple of 4. Predictor 0 ("black").
//
//go:noescape
func predResBlackRowSSE2(pixels []color.NRGBA, dstOff int, out []color.NRGBA, n int)

// predictResidualsRow (amd64): SSE2-vectorised for modes 0–4, scalar fallback
// for modes 5–13 and y==0. Output is byte-identical to predictResidualsRowScalar.
func predictResidualsRow(pixels []color.NRGBA, width, mode, xStart, xEnd, y int, out []color.NRGBA) {
	// Row y==0: boundary rules change per-pixel for every mode — fall to scalar.
	// Modes 5–13: not yet vectorised; scalar reference.
	if y == 0 || mode > 4 {
		predictResidualsRowScalar(pixels, width, mode, xStart, xEnd, y, out)
		return
	}

	base := y * width
	iStart := xStart
	if iStart == 0 {
		// x==0, y>0: predictor is always the pixel directly above, regardless
		// of mode (applyFilter special-cases x==0 before the mode switch).
		p := pixels[base]
		d := pixels[base-width]
		out[0] = color.NRGBA{p.R - d.R, p.G - d.G, p.B - d.B, p.A - d.A}
		iStart = 1
	}

	n := xEnd - iStart
	if n <= 0 {
		return
	}

	curOff := base + iStart
	upOff := curOff - width
	n4 := n &^ 3 // floor to multiple of 4 pixels (16 bytes per SSE2 batch)
	o := out[iStart-xStart:]

	if n4 == 0 {
		predictResidualsRowScalar(pixels, width, mode, iStart, xEnd, y, o)
		return
	}

	switch mode {
	case 0:
		predResBlackRowSSE2(pixels, curOff, o, n4)
	case 1:
		predResSubRowSSE2(pixels, curOff, curOff-1, o, n4) // left
	case 2:
		predResSubRowSSE2(pixels, curOff, upOff, o, n4) // top
	case 3:
		predResSubRowSSE2(pixels, curOff, upOff+1, o, n4) // top-right
	case 4:
		predResSubRowSSE2(pixels, curOff, upOff-1, o, n4) // top-left
	}

	// Scalar tail: n & 3 leftover pixels not covered by SSE2 batches.
	if iStart+n4 < xEnd {
		predictResidualsRowScalar(pixels, width, mode, iStart+n4, xEnd, y, out[iStart+n4-xStart:])
	}
}
