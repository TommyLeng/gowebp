// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"bytes"
	"image"
	"image/draw"
	_ "image/jpeg"
	"os"
	"testing"

	xdraw "golang.org/x/image/draw"
)

// BenchmarkEncode300x300 encodes a 300x300 JPEG resized image at quality 90.
// Uses the wave-front parallel encoding path (parallelThreshold=0, all sizes parallelised).
func BenchmarkEncode300x300(b *testing.B) {
	f, err := os.Open("test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	if err != nil {
		b.Skip("test image not found: test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	}
	defer f.Close()

	src, _, err := image.Decode(f)
	if err != nil {
		b.Fatal(err)
	}

	// Resize to 300x300
	dst := image.NewNRGBA(image.Rect(0, 0, 300, 300))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Src, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		if err := Encode(&buf, dst, &Options{Quality: 90}); err != nil {
			b.Fatal(err)
		}
		b.SetBytes(int64(dst.Bounds().Dx() * dst.Bounds().Dy()))
	}
}

// BenchmarkEncodeKodak768x512 encodes kodim05.png (768×512) at quality 90.
func BenchmarkEncodeKodak768x512(b *testing.B) {
	f, err := os.Open("test_data/original/kodak/kodim05.png")
	if err != nil {
		b.Skip("kodak test image not found: test_data/original/kodak/kodim05.png")
	}
	defer f.Close()

	src, _, err := image.Decode(f)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		if err := Encode(&buf, src, &Options{Quality: 90}); err != nil {
			b.Fatal(err)
		}
		b.SetBytes(int64(src.Bounds().Dx() * src.Bounds().Dy()))
	}
}

// BenchmarkEncodeLarge encodes a 1536×2048 JPEG at quality 90.
// Uses the wave-front parallel encoding path (parallelThreshold=0, all sizes parallelised).
func BenchmarkEncodeLarge(b *testing.B) {
	f, err := os.Open("test_data/original/jablehk_snexxxxxxx_0055.jpg")
	if err != nil {
		b.Skip("large test image not found: test_data/original/jablehk_snexxxxxxx_0055.jpg")
	}
	defer f.Close()

	src, _, err := image.Decode(f)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		if err := Encode(&buf, src, &Options{Quality: 90}); err != nil {
			b.Fatal(err)
		}
		b.SetBytes(int64(src.Bounds().Dx() * src.Bounds().Dy()))
	}
}
