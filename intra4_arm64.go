// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build arm64

package gowebp

// intra4Predict fills pred[16] with the 4x4 prediction for the given mode.
// The four simple modes (DC/TM/VE/HE) are dispatched to NEON-vectorised
// helpers implemented in intra4_arm64.s. The six diagonal modes fall through
// to the scalar reference implementation in intra4.go.
func intra4Predict(mode int, ctx pred4Context, pred []int16) {
	switch mode {
	case B_DC_PRED:
		intra4PredDC(ctx, pred)
	case B_TM_PRED:
		intra4PredTM(ctx, pred)
	case B_VE_PRED:
		intra4PredVE(ctx, pred)
	case B_HE_PRED:
		intra4PredHE(ctx, pred)
	default:
		intra4PredScalar(mode, ctx, pred)
	}
}

// intra4PredDC computes B_DC_PRED: broadcasts a single DC value to all 16
// elements of pred. Implemented in intra4_arm64.s.
//
//go:noescape
func intra4PredDC(ctx pred4Context, pred []int16)

// intra4PredTM computes B_TM_PRED (TrueMotion): pred[y,x] = clip8(top[x] + left[y] - topLeft).
// Implemented in intra4_arm64.s.
//
//go:noescape
func intra4PredTM(ctx pred4Context, pred []int16)

// intra4PredVE computes B_VE_PRED (Vertical): each row equals the 4-pixel
// AVG3 filter over the top neighborhood. Implemented in intra4_arm64.s.
//
//go:noescape
func intra4PredVE(ctx pred4Context, pred []int16)

// intra4PredHE computes B_HE_PRED (Horizontal): each row r is the AVG3 filter
// applied to (leftEx[r], leftEx[r+1], leftEx[r+2]) broadcast across 4 columns.
// Implemented in intra4_arm64.s.
//
//go:noescape
func intra4PredHE(ctx pred4Context, pred []int16)
