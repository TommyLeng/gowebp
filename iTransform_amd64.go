// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build amd64

package gowebp

// iTransform4x4 computes the inverse 4x4 DCT and adds residuals to pred.
// coeffs[16] are dequantized DCT coefficients in raster order (row-major).
// pred[16] is the 4x4 prediction block (int16 values in [0,255]).
// out[16] receives the reconstructed pixels (int16 clamped to [0,255]).
//
// Implemented in iTransform_amd64.s using SSE2. Both vertical and horizontal
// passes are vectorised across all 4 columns / rows in parallel, working
// entirely in int16 (no 32-bit intermediates needed).
//
// The SSE2 "trick" for the multiply constants (from libwebp dec_sse2.c):
//   K1 = 85627 = 65536 + 20091  →  k1 = 20091 (fits int16)
//   K2 = 35468 = 65536 - 30068  →  k2 = -30068 (as signed int16)
//   MUL(x, K) = PMULHW(x, k) + x   (since (x*(k+65536))>>16 = PMULHW(x,k)+x)
//
//go:noescape
func iTransform4x4(coeffs []int16, pred []int16, out []int16)
