// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// fTransform computes the 4x4 forward DCT of (src - ref), storing into out[16].
// The horizontal pass is vectorised with SSE2; the vertical pass is scalar
// because it contains a data-dependent conditional (+1 when a3 != 0).
//
// Implemented in dct_amd64.s. All computation stays in XMM registers for the
// horizontal pass; the scalar vertical loop uses only the runtime-safe GPRs
// (AX, BX, CX, DX, SI, DI, R8..R13) — R14 (goroutine pointer) is never
// touched, so signal-based preemption cannot corrupt the g register.
//
//go:noescape
func fTransform(src []int16, ref []int16, out []int16)
