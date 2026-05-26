package gowebp

import (
	"math/rand"
	"testing"
)

// Verify the dual-block wrappers match calling the single-block transforms
// twice (block A in [0:16], block B in [16:32]).

func TestFTransform2MatchesTwoFTransform(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 200; trial++ {
		var src, ref [32]int16
		for i := 0; i < 32; i++ {
			src[i] = int16(rng.Intn(256))
			ref[i] = int16(rng.Intn(256))
		}
		var dual, ref1, ref2 [32]int16
		fTransform2(src[:], ref[:], dual[:])
		fTransform(src[0:16], ref[0:16], ref1[0:16])
		fTransform(src[16:32], ref[16:32], ref2[16:32])

		for i := 0; i < 32; i++ {
			var expected int16
			if i < 16 {
				expected = ref1[i]
			} else {
				expected = ref2[i]
			}
			if dual[i] != expected {
				t.Errorf("trial %d, idx %d: dual=%d expected=%d", trial, i, dual[i], expected)
				break
			}
		}
	}
}

func TestITransform4x4x2MatchesTwoITransform(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 200; trial++ {
		var coeffs, pred [32]int16
		for i := 0; i < 32; i++ {
			coeffs[i] = int16(rng.Intn(1024) - 512)
			pred[i] = int16(rng.Intn(256))
		}
		var dual, ref1, ref2 [32]int16
		iTransform4x4x2(coeffs[:], pred[:], dual[:])
		iTransform4x4(coeffs[0:16], pred[0:16], ref1[0:16])
		iTransform4x4(coeffs[16:32], pred[16:32], ref2[16:32])

		for i := 0; i < 32; i++ {
			var expected int16
			if i < 16 {
				expected = ref1[i]
			} else {
				expected = ref2[i]
			}
			if dual[i] != expected {
				t.Errorf("trial %d, idx %d: dual=%d expected=%d", trial, i, dual[i], expected)
				break
			}
		}
	}
}
