package gowebp

import (
	"bytes"
	"image"
	"image/color"
	"math/rand"
	"testing"
)

// syntheticImage creates a reproducible pseudo-random NRGBA image.
func syntheticImage(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	r := rand.New(rand.NewSource(42))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(r.Intn(256)),
				G: uint8(r.Intn(256)),
				B: uint8(r.Intn(256)),
				A: 255,
			})
		}
	}
	return img
}

// BenchmarkEncodeSynthetic300x300 encodes a 300×300 synthetic image at quality 90.
// Uses a reproducible PRNG image so no test files are needed.
func BenchmarkEncodeSynthetic300x300(b *testing.B) {
	img := syntheticImage(300, 300)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		if err := Encode(&buf, img, &Options{Quality: 90}); err != nil {
			b.Fatal(err)
		}
		b.SetBytes(int64(img.Bounds().Dx() * img.Bounds().Dy()))
	}
}

// BenchmarkEncodeSynthetic768x512 encodes a 768×512 synthetic image at quality 90.
func BenchmarkEncodeSynthetic768x512(b *testing.B) {
	img := syntheticImage(768, 512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		if err := Encode(&buf, img, &Options{Quality: 90}); err != nil {
			b.Fatal(err)
		}
		b.SetBytes(int64(img.Bounds().Dx() * img.Bounds().Dy()))
	}
}
