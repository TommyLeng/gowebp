// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// ssd4x4 computes the Sum of Squared Differences between a 4x4 source and
// reconstructed block (16 int16 elements each).
// Implemented in ssd_arm64.s using NEON SMULL/SMLAL2/ADDV.
//
//go:noescape
func ssd4x4(src, pred []int16) int64

// ssd16x16 computes the Sum of Squared Differences between a 16x16 source
// and prediction macroblock (256 int16 elements each).
// Implemented in ssd_arm64.s using NEON SMULL/SMLAL2/ADDV.
//
//go:noescape
func ssd16x16(src, pred []int16) int64
