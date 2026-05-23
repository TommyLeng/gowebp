// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build arm64

package gowebp

import "testing"

// Direct microbenchmarks comparing the NEON helpers against the scalar
// reference. These do not exercise the encoder; they isolate the per-call
// cost of each mode.

func benchIntra4Mode(b *testing.B, mode int, neon func(pred4Context, []int16)) {
	ctx := pred4Context{
		topLeft: 128,
		left:    [4]int{100, 110, 120, 130},
		top:     [8]int{140, 150, 160, 170, 180, 190, 200, 210},
	}
	var out [16]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		neon(ctx, out[:])
	}
}

func benchIntra4Scalar(b *testing.B, mode int) {
	ctx := pred4Context{
		topLeft: 128,
		left:    [4]int{100, 110, 120, 130},
		top:     [8]int{140, 150, 160, 170, 180, 190, 200, 210},
	}
	var out [16]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		intra4PredScalar(mode, ctx, out[:])
	}
}

func BenchmarkIntra4_DC_NEON(b *testing.B)     { benchIntra4Mode(b, B_DC_PRED, intra4PredDC) }
func BenchmarkIntra4_DC_Scalar(b *testing.B)   { benchIntra4Scalar(b, B_DC_PRED) }
func BenchmarkIntra4_TM_NEON(b *testing.B)     { benchIntra4Mode(b, B_TM_PRED, intra4PredTM) }
func BenchmarkIntra4_TM_Scalar(b *testing.B)   { benchIntra4Scalar(b, B_TM_PRED) }
func BenchmarkIntra4_VE_NEON(b *testing.B)     { benchIntra4Mode(b, B_VE_PRED, intra4PredVE) }
func BenchmarkIntra4_VE_Scalar(b *testing.B)   { benchIntra4Scalar(b, B_VE_PRED) }
func BenchmarkIntra4_HE_NEON(b *testing.B)     { benchIntra4Mode(b, B_HE_PRED, intra4PredHE) }
func BenchmarkIntra4_HE_Scalar(b *testing.B)   { benchIntra4Scalar(b, B_HE_PRED) }
