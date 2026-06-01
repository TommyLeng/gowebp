package gowebp

import (
	"math/rand"
	"testing"
)

// scalarFTransform is the reference scalar implementation of fTransform,
// inlined here so the test runs on both arm64 (NEON) and amd64 (SSE2) builds
// and always compares against a known-good scalar path.
func scalarFTransform(src []int16, ref []int16, out []int16) {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		d0 := int32(src[i*4+0]) - int32(ref[i*4+0])
		d1 := int32(src[i*4+1]) - int32(ref[i*4+1])
		d2 := int32(src[i*4+2]) - int32(ref[i*4+2])
		d3 := int32(src[i*4+3]) - int32(ref[i*4+3])
		a0 := d0 + d3
		a1 := d1 + d2
		a2 := d1 - d2
		a3 := d0 - d3
		tmp[0+i*4] = (a0 + a1) * 8
		tmp[1+i*4] = (a2*2217 + a3*5352 + 1812) >> 9
		tmp[2+i*4] = (a0 - a1) * 8
		tmp[3+i*4] = (a3*2217 - a2*5352 + 937) >> 9
	}
	for i := 0; i < 4; i++ {
		a0 := tmp[0+i] + tmp[12+i]
		a1 := tmp[4+i] + tmp[8+i]
		a2 := tmp[4+i] - tmp[8+i]
		a3 := tmp[0+i] - tmp[12+i]
		out[0+i] = int16((a0 + a1 + 7) >> 4)
		extra := int32(0)
		if a3 != 0 {
			extra = 1
		}
		out[4+i] = int16(((a2*2217 + a3*5352 + 12000) >> 16) + extra)
		out[8+i] = int16((a0 - a1 + 7) >> 4)
		out[12+i] = int16((a3*2217 - a2*5352 + 51000) >> 16)
	}
}

func TestFTransformMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for trial := 0; trial < 500; trial++ {
		var src, ref [16]int16
		for i := range src {
			src[i] = int16(rng.Intn(512) - 256)
			ref[i] = int16(rng.Intn(256))
		}
		var simd, scalar [16]int16
		fTransform(src[:], ref[:], simd[:])
		scalarFTransform(src[:], ref[:], scalar[:])
		for i := range simd {
			if simd[i] != scalar[i] {
				t.Errorf("trial %d idx %d: simd=%d scalar=%d (src=%v ref=%v)",
					trial, i, simd[i], scalar[i], src, ref)
				break
			}
		}
	}
}

// TestFTransformFlatBlock checks the DC coefficient for a constant-residual block,
// which is the case that was broken by the vertical-pass transpose bug.
func TestFTransformFlatBlock(t *testing.T) {
	cases := []struct {
		residual int16
		wantDC   int16
	}{
		{0, 0},
		{-26, -208},
		{64, 512},
		{-64, -512},
		{127, 1016},
		{-128, -1024},
	}
	for _, tc := range cases {
		var src, ref [16]int16
		for i := range src {
			src[i] = tc.residual
			ref[i] = 0
		}
		var out [16]int16
		fTransform(src[:], ref[:], out[:])
		if out[0] != tc.wantDC {
			t.Errorf("residual=%d: DC got %d, want %d (full out: %v)",
				tc.residual, out[0], tc.wantDC, out)
		}
	}
}
