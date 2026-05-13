// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// fTransform computes the 4x4 forward DCT of (src - ref), storing into out[16].
// The horizontal pass is vectorised with NEON; the vertical pass is scalar
// because it contains a data-dependent conditional (+1 when a3 != 0).
//
//go:noescape
func fTransform(src []int16, ref []int16, out []int16)
