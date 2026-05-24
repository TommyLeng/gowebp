// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build amd64

package gowebp

import (
	"math/rand"
	"testing"
)

// TestIntra4amd64Equivalence verifies the amd64 helpers (DC/TM/VE/HE) produce
// bit-identical output to the scalar reference across a wide range of
// neighborhood pixel values, including boundary cases (0, 127, 128, 129, 255).
func TestIntra4amd64Equivalence(t *testing.T) {
	rng := rand.New(rand.NewSource(0xCAFEBABE))

	fixtures := buildIntra4Fixturesamd64(rng, 2048)

	type modeCase struct {
		name string
		mode int
		impl func(ctx pred4Context, pred []int16)
	}
	cases := []modeCase{
		{"DC", B_DC_PRED, intra4PredDCamd64},
		{"TM", B_TM_PRED, intra4PredTMamd64},
		{"VE", B_VE_PRED, intra4PredVEamd64},
		{"HE", B_HE_PRED, intra4PredHEamd64},
	}

	var got, want [16]int16
	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			for i, ctx := range fixtures {
				for j := range got {
					got[j] = 0xCC
					want[j] = 0xCC
				}
				c.impl(ctx, got[:])
				intra4PredScalar(c.mode, ctx, want[:])
				if got != want {
					t.Fatalf("mode %s, fixture %d: mismatch\n  ctx  = %+v\n  got  = %v\n  want = %v",
						c.name, i, ctx, got, want)
				}
			}
		})
	}
}

func buildIntra4Fixturesamd64(rng *rand.Rand, nRandom int) []pred4Context {
	corners := []int{0, 1, 7, 31, 63, 127, 128, 129, 191, 254, 255}

	var out []pred4Context

	// Constant fixtures: every neighbor equal to one of the corner values.
	for _, v := range corners {
		var ctx pred4Context
		ctx.topLeft = v
		for i := range ctx.left {
			ctx.left[i] = v
		}
		for i := range ctx.top {
			ctx.top[i] = v
		}
		out = append(out, ctx)
	}

	// Mixed extreme fixtures.
	for _, tl := range corners {
		for _, lv := range corners {
			for _, tv := range corners {
				var ctx pred4Context
				ctx.topLeft = tl
				for i := range ctx.left {
					ctx.left[i] = lv
				}
				for i := range ctx.top {
					ctx.top[i] = tv
				}
				out = append(out, ctx)
			}
		}
	}

	// Random fixtures.
	for i := 0; i < nRandom; i++ {
		var ctx pred4Context
		ctx.topLeft = rng.Intn(256)
		for j := range ctx.left {
			ctx.left[j] = rng.Intn(256)
		}
		for j := range ctx.top {
			ctx.top[j] = rng.Intn(256)
		}
		out = append(out, ctx)
	}

	return out
}

// BenchmarkIntra4_*_amd64 benchmarks the amd64 helpers vs scalar.

func benchIntra4Modeamd64(b *testing.B, fn func(pred4Context, []int16)) {
	ctx := pred4Context{
		topLeft: 128,
		left:    [4]int{100, 110, 120, 130},
		top:     [8]int{140, 150, 160, 170, 180, 190, 200, 210},
	}
	var out [16]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(ctx, out[:])
	}
}

func benchIntra4Scalaramd64(b *testing.B, mode int) {
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

func BenchmarkIntra4_DC_amd64(b *testing.B)     { benchIntra4Modeamd64(b, intra4PredDCamd64) }
func BenchmarkIntra4_DC_Scalar_amd64(b *testing.B) { benchIntra4Scalaramd64(b, B_DC_PRED) }
func BenchmarkIntra4_TM_amd64(b *testing.B)     { benchIntra4Modeamd64(b, intra4PredTMamd64) }
func BenchmarkIntra4_TM_Scalar_amd64(b *testing.B) { benchIntra4Scalaramd64(b, B_TM_PRED) }
func BenchmarkIntra4_VE_amd64(b *testing.B)     { benchIntra4Modeamd64(b, intra4PredVEamd64) }
func BenchmarkIntra4_VE_Scalar_amd64(b *testing.B) { benchIntra4Scalaramd64(b, B_VE_PRED) }
func BenchmarkIntra4_HE_amd64(b *testing.B)     { benchIntra4Modeamd64(b, intra4PredHEamd64) }
func BenchmarkIntra4_HE_Scalar_amd64(b *testing.B) { benchIntra4Scalaramd64(b, B_HE_PRED) }
