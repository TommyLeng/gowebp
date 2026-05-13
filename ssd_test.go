// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"math/rand"
	"testing"
)

// ssd4x4Scalar is the reference scalar implementation used to validate
// the SIMD ssd4x4 across architectures.
func ssd4x4Scalar(src, pred []int16) int64 {
	var sum int64
	for i := 0; i < 16; i++ {
		d := int64(src[i]) - int64(pred[i])
		sum += d * d
	}
	return sum
}

// ssd16x16Scalar is the reference scalar implementation for ssd16x16.
func ssd16x16Scalar(src, pred []int16) int64 {
	var sum int64
	for i := 0; i < 256; i++ {
		d := int64(src[i]) - int64(pred[i])
		sum += d * d
	}
	return sum
}

func TestSSD4x4_Equivalence(t *testing.T) {
	src := make([]int16, 16)
	pred := make([]int16, 16)
	r := rand.New(rand.NewSource(0x12345))

	for trial := 0; trial < 1000; trial++ {
		for i := range src {
			src[i] = int16(r.Intn(256))
			pred[i] = int16(r.Intn(256))
		}
		got := ssd4x4(src, pred)
		want := ssd4x4Scalar(src, pred)
		if got != want {
			t.Fatalf("ssd4x4 mismatch (trial %d): got %d, want %d\nsrc=%v\npred=%v",
				trial, got, want, src, pred)
		}
	}
}

func TestSSD4x4_EdgeCases(t *testing.T) {
	cases := []struct {
		name string
		src  []int16
		pred []int16
		want int64
	}{
		{
			name: "all zero",
			src:  make([]int16, 16),
			pred: make([]int16, 16),
			want: 0,
		},
		{
			name: "max positive diff",
			src:  []int16{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			pred: make([]int16, 16),
			want: 16 * 255 * 255,
		},
		{
			name: "max negative diff",
			src:  make([]int16, 16),
			pred: []int16{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			want: 16 * 255 * 255,
		},
		{
			// Pattern repeats over 16 lanes: each |diff| value appears 4 times.
			name: "alternating signs",
			src:  []int16{100, 0, 200, 0, 50, 0, 75, 0, 100, 0, 200, 0, 50, 0, 75, 0},
			pred: []int16{0, 100, 0, 200, 0, 50, 0, 75, 0, 100, 0, 200, 0, 50, 0, 75},
			want: 4 * (100*100 + 200*200 + 50*50 + 75*75),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ssd4x4(tc.src, tc.pred)
			if got != tc.want {
				t.Fatalf("ssd4x4 = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestSSD16x16_Equivalence(t *testing.T) {
	src := make([]int16, 256)
	pred := make([]int16, 256)
	r := rand.New(rand.NewSource(0xABCDE))

	for trial := 0; trial < 200; trial++ {
		for i := range src {
			src[i] = int16(r.Intn(256))
			pred[i] = int16(r.Intn(256))
		}
		got := ssd16x16(src, pred)
		want := ssd16x16Scalar(src, pred)
		if got != want {
			t.Fatalf("ssd16x16 mismatch (trial %d): got %d, want %d", trial, got, want)
		}
	}
}

func TestSSD16x16_EdgeCases(t *testing.T) {
	t.Run("all zero", func(t *testing.T) {
		src := make([]int16, 256)
		pred := make([]int16, 256)
		if got := ssd16x16(src, pred); got != 0 {
			t.Fatalf("ssd16x16 = %d, want 0", got)
		}
	})
	t.Run("max diff", func(t *testing.T) {
		src := make([]int16, 256)
		pred := make([]int16, 256)
		for i := range src {
			src[i] = 255
		}
		want := int64(256) * 255 * 255
		if got := ssd16x16(src, pred); got != want {
			t.Fatalf("ssd16x16 = %d, want %d", got, want)
		}
	})
	t.Run("negative diff", func(t *testing.T) {
		src := make([]int16, 256)
		pred := make([]int16, 256)
		for i := range src {
			pred[i] = 200
		}
		want := int64(256) * 200 * 200
		if got := ssd16x16(src, pred); got != want {
			t.Fatalf("ssd16x16 = %d, want %d", got, want)
		}
	})
}

func BenchmarkSSD4x4(b *testing.B) {
	src := make([]int16, 16)
	pred := make([]int16, 16)
	r := rand.New(rand.NewSource(1))
	for i := range src {
		src[i] = int16(r.Intn(256))
		pred[i] = int16(r.Intn(256))
	}
	b.ResetTimer()
	var sum int64
	for i := 0; i < b.N; i++ {
		sum += ssd4x4(src, pred)
	}
	_ = sum
}

func BenchmarkSSD4x4Scalar(b *testing.B) {
	src := make([]int16, 16)
	pred := make([]int16, 16)
	r := rand.New(rand.NewSource(1))
	for i := range src {
		src[i] = int16(r.Intn(256))
		pred[i] = int16(r.Intn(256))
	}
	b.ResetTimer()
	var sum int64
	for i := 0; i < b.N; i++ {
		sum += ssd4x4Scalar(src, pred)
	}
	_ = sum
}

func BenchmarkSSD16x16(b *testing.B) {
	src := make([]int16, 256)
	pred := make([]int16, 256)
	r := rand.New(rand.NewSource(1))
	for i := range src {
		src[i] = int16(r.Intn(256))
		pred[i] = int16(r.Intn(256))
	}
	b.ResetTimer()
	var sum int64
	for i := 0; i < b.N; i++ {
		sum += ssd16x16(src, pred)
	}
	_ = sum
}

func BenchmarkSSD16x16Scalar(b *testing.B) {
	src := make([]int16, 256)
	pred := make([]int16, 256)
	r := rand.New(rand.NewSource(1))
	for i := range src {
		src[i] = int16(r.Intn(256))
		pred[i] = int16(r.Intn(256))
	}
	b.ResetTimer()
	var sum int64
	for i := 0; i < b.N; i++ {
		sum += ssd16x16Scalar(src, pred)
	}
	_ = sum
}
