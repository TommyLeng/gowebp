// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build arm64

package gowebp

import (
	"math/rand"
	"testing"
)

// TestIntra4NEONEquivalence verifies the NEON helpers (DC/TM/VE/HE) produce
// bit-identical output to the scalar reference across a wide range of
// neighborhood pixel values, including boundary cases (0, 127, 128, 129, 255).
func TestIntra4NEONEquivalence(t *testing.T) {
	rng := rand.New(rand.NewSource(0xCAFEBABE))

	// Deterministic + random fixtures.
	fixtures := buildIntra4Fixtures(rng, 2048)

	type modeCase struct {
		name string
		mode int
		neon func(ctx pred4Context, pred []int16)
	}
	cases := []modeCase{
		{"DC", B_DC_PRED, intra4PredDC},
		{"TM", B_TM_PRED, intra4PredTM},
		{"VE", B_VE_PRED, intra4PredVE},
		{"HE", B_HE_PRED, intra4PredHE},
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
				c.neon(ctx, got[:])
				intra4PredScalar(c.mode, ctx, want[:])
				if got != want {
					t.Fatalf("mode %s, fixture %d: mismatch\n  ctx  = %+v\n  got  = %v\n  want = %v",
						c.name, i, ctx, got, want)
				}
			}
		})
	}
}

func buildIntra4Fixtures(rng *rand.Rand, nRandom int) []pred4Context {
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

	// Mixed extreme fixtures: top all 0 and left all 255, etc.
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
