// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// fTransform computes the 4x4 forward DCT of (src - ref), storing into out[16].
// Both passes are fully vectorised with NEON (int32x4, one lane per column).
// The data-dependent +1 when a3 != 0 is handled branchlessly via CMEQ/NOT/USHR.
//
//go:noescape
func fTransform(src []int16, ref []int16, out []int16)

// iTransform4x4 computes the inverse 4x4 DCT and adds residuals to pred.
// coeffs[16] are dequantized DCT coefficients in raster order (row-major).
// pred[16] is the 4x4 prediction block (int16 values in [0,255]).
// out[16] receives the reconstructed pixels (int16 clamped to [0,255]).
//
// Implemented in dct_arm64.s using NEON. Both vertical and horizontal passes
// are vectorised across all 4 columns / rows in parallel via int32x4 lanes.
//
//go:noescape
func iTransform4x4(coeffs []int16, pred []int16, out []int16)
