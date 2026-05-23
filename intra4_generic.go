// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build !arm64

package gowebp

// intra4Predict fills pred[16] with the 4x4 prediction for the given mode.
// Non-arm64 platforms use the scalar implementation directly. The arm64 build
// dispatches the four simple modes (DC/TM/VE/HE) to NEON in intra4_arm64.go.
func intra4Predict(mode int, ctx pred4Context, pred []int16) {
	intra4PredScalar(mode, ctx, pred)
}
