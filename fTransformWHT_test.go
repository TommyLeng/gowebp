package gowebp

import (
	"math/rand"
	"testing"
)

// scalarFTransformWHT is the reference scalar implementation, inlined here for
// the test (since the SIMD build tags hide the generic one on arm64/amd64).
func scalarFTransformWHT(in []int16, out []int16) {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		a0 := int32(in[0+i*4]) + int32(in[2+i*4])
		a1 := int32(in[1+i*4]) + int32(in[3+i*4])
		a2 := int32(in[1+i*4]) - int32(in[3+i*4])
		a3 := int32(in[0+i*4]) - int32(in[2+i*4])
		tmp[0+i*4] = a0 + a1
		tmp[1+i*4] = a3 + a2
		tmp[2+i*4] = a3 - a2
		tmp[3+i*4] = a0 - a1
	}
	for i := 0; i < 4; i++ {
		a0 := tmp[0+i] + tmp[8+i]
		a1 := tmp[4+i] + tmp[12+i]
		a2 := tmp[4+i] - tmp[12+i]
		a3 := tmp[0+i] - tmp[8+i]
		b0 := a0 + a1
		b1 := a3 + a2
		b2 := a3 - a2
		b3 := a0 - a1
		out[0+i] = int16(b0 >> 1)
		out[4+i] = int16(b1 >> 1)
		out[8+i] = int16(b2 >> 1)
		out[12+i] = int16(b3 >> 1)
	}
}

func TestFTransformWHTMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	// Fixed seed exercise; many random 12b-signed inputs.
	for trial := 0; trial < 200; trial++ {
		var in [16]int16
		for i := 0; i < 16; i++ {
			in[i] = int16(rng.Intn(4095) - 2047) // 12-bit signed range
		}

		var simd, ref [16]int16
		fTransformWHT(in[:], simd[:])
		scalarFTransformWHT(in[:], ref[:])

		for i := 0; i < 16; i++ {
			if simd[i] != ref[i] {
				t.Errorf("trial %d, idx %d: simd=%d ref=%d (in=%v)", trial, i, simd[i], ref[i], in)
				break
			}
		}
	}
}

func TestFTransformWHTEdgeCases(t *testing.T) {
	cases := [][16]int16{
		{},
		{2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047, 2047},
		{-2047, -2047, -2047, -2047, -2047, -2047, -2047, -2047, -2047, -2047, -2047, -2047, -2047, -2047, -2047, -2047},
		{2047, -2047, 2047, -2047, -2047, 2047, -2047, 2047, 2047, -2047, 2047, -2047, -2047, 2047, -2047, 2047},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		// Test pass 2 b0 = a0+a1 reaching ±32767 boundary (16b overflow boundary)
		{2047, 2047, -2047, -2047, 2047, 2047, -2047, -2047, 2047, 2047, -2047, -2047, 2047, 2047, -2047, -2047},
	}
	for i, in := range cases {
		var simd, ref [16]int16
		fTransformWHT(in[:], simd[:])
		scalarFTransformWHT(in[:], ref[:])
		for k := 0; k < 16; k++ {
			if simd[k] != ref[k] {
				t.Errorf("case %d, idx %d: simd=%d ref=%d (in=%v simd=%v ref=%v)", i, k, simd[k], ref[k], in, simd, ref)
				break
			}
		}
	}
}
