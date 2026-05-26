package gowebp

import (
	"math/rand"
	"testing"
)

// Microbenchmarks comparing the dual-block wrappers (fTransform2 /
// iTransform4x4x2) against calling the single-block kernels twice.
// Used to document the wrapper's call-overhead cost — see transform_dual.go
// for why the encoder loops do NOT use these wrappers.

func BenchmarkFTransformPair_Single(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	var src, ref [32]int16
	for i := 0; i < 32; i++ {
		src[i] = int16(rng.Intn(256))
		ref[i] = int16(rng.Intn(256))
	}
	var out [32]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fTransform(src[0:16], ref[0:16], out[0:16])
		fTransform(src[16:32], ref[16:32], out[16:32])
	}
}

func BenchmarkFTransformPair_Dual(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	var src, ref [32]int16
	for i := 0; i < 32; i++ {
		src[i] = int16(rng.Intn(256))
		ref[i] = int16(rng.Intn(256))
	}
	var out [32]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fTransform2(src[:], ref[:], out[:])
	}
}

func BenchmarkITransform4x4Pair_Single(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	var coeffs, pred [32]int16
	for i := 0; i < 32; i++ {
		coeffs[i] = int16(rng.Intn(1024) - 512)
		pred[i] = int16(rng.Intn(256))
	}
	var out [32]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		iTransform4x4(coeffs[0:16], pred[0:16], out[0:16])
		iTransform4x4(coeffs[16:32], pred[16:32], out[16:32])
	}
}

func BenchmarkITransform4x4Pair_Dual(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	var coeffs, pred [32]int16
	for i := 0; i < 32; i++ {
		coeffs[i] = int16(rng.Intn(1024) - 512)
		pred[i] = int16(rng.Intn(256))
	}
	var out [32]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		iTransform4x4x2(coeffs[:], pred[:], out[:])
	}
}
