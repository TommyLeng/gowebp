// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// sad4x4 computes the Sum of Absolute Differences between a 4×4 source
// and prediction block (16 int16 elements each).
// Implemented in sad_arm64.s using NEON UABD/UADDLV instructions.
//
//go:noescape
func sad4x4(src, pred []int16) int64
