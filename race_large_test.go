// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"bytes"
	"image"
	_ "image/jpeg"
	"os"
	"testing"

	"golang.org/x/image/webp"
)

// TestEncodeLargeParallelCorrectness encodes jablehk_snexxxxxxx_0055.jpg
// (1536×2048) using the wave-front parallel
// path and verifies the output decodes correctly.
func TestEncodeLargeParallelCorrectness(t *testing.T) {
	f, err := os.Open("test_data/original/jablehk_snexxxxxxx_0055.jpg")
	if err != nil {
		t.Skip("large test image not found: test_data/original/jablehk_snexxxxxxx_0055.jpg")
	}
	defer f.Close()

	src, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := Encode(&buf, src, &Options{Quality: 90}); err != nil {
		t.Fatalf("parallel encode failed: %v", err)
	}

	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("parallel-encoded image failed to decode: %v", err)
	}

	if dec.Bounds().Dx() != src.Bounds().Dx() || dec.Bounds().Dy() != src.Bounds().Dy() {
		t.Fatalf("dimension mismatch: src=%v dec=%v", src.Bounds(), dec.Bounds())
	}

	t.Logf("parallel encoded: %d bytes (%.1f kb)", buf.Len(), float64(buf.Len())/1024)
	t.Logf("decoded dimensions: %dx%d OK", dec.Bounds().Dx(), dec.Bounds().Dy())
}
