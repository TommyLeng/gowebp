package gowebp

import (
	"bytes"
	"image"
	"image/color"
	_ "image/png"
	"os"
	"testing"

	"golang.org/x/image/webp"
)

// TestEncodeAlpha verifies that a PNG with transparency is encoded as WebP Extended format
// and that the alpha channel is preserved after round-trip decode.
func TestEncodeAlpha(t *testing.T) {
	// Build a 4×4 NRGBA test image with a known alpha pattern.
	img := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	alphaVals := []uint8{0, 64, 128, 255}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			a := alphaVals[(y*4+x)%4]
			img.SetNRGBA(x, y, color.NRGBA{R: 200, G: 100, B: 50, A: a})
		}
	}

	// Encode to WebP.
	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 90}); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	encoded := buf.Bytes()

	// Verify the output contains VP8X and ALPH markers.
	if !bytes.Contains(encoded, []byte("VP8X")) {
		t.Error("output missing VP8X chunk")
	}
	if !bytes.Contains(encoded, []byte("ALPH")) {
		t.Error("output missing ALPH chunk")
	}

	// Decode the WebP and verify alpha is preserved.
	decoded, err := webp.Decode(bytes.NewReader(encoded))
	if err != nil {
		t.Fatalf("webp.Decode: %v", err)
	}

	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			_, _, _, gotA := decoded.At(x, y).RGBA()
			gotA8 := uint8(gotA >> 8)
			wantA := alphaVals[(y*4+x)%4]
			if gotA8 != wantA {
				t.Errorf("pixel (%d,%d): alpha = %d, want %d", x, y, gotA8, wantA)
			}
		}
	}
}

// TestEncodeAlphaOpaque verifies that a fully opaque image does NOT use Extended format.
func TestEncodeAlphaOpaque(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: 200, G: 100, B: 50, A: 255})
		}
	}

	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 90}); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	encoded := buf.Bytes()

	if bytes.Contains(encoded, []byte("VP8X")) {
		t.Error("fully opaque image should NOT use Extended format (VP8X)")
	}
}

// TestEncodeAlphaRealImages encodes the three real-world alpha PNGs, decodes the result,
// and checks that fully-transparent pixels remain transparent (alpha == 0).
func TestEncodeAlphaRealImages(t *testing.T) {
	files := []string{
		"test_data/original/i1-a.png",
		"test_data/original/i11-a.png",
		"test_data/original/i18-a.png",
	}
	for _, path := range files {
		t.Run(path, func(t *testing.T) {
			f, err := os.Open(path)
			if err != nil {
				t.Skipf("test image not found: %v", err)
			}
			defer f.Close()

			src, _, err := image.Decode(f)
			if err != nil {
				t.Fatalf("decode PNG: %v", err)
			}

			// Encode to WebP with alpha.
			var buf bytes.Buffer
			if err := Encode(&buf, src, &Options{Quality: 90}); err != nil {
				t.Fatalf("Encode: %v", err)
			}
			encoded := buf.Bytes()

			if !bytes.Contains(encoded, []byte("VP8X")) {
				t.Error("output missing VP8X chunk — alpha not detected")
			}
			if !bytes.Contains(encoded, []byte("ALPH")) {
				t.Error("output missing ALPH chunk")
			}

			// Decode and verify transparent pixels stayed transparent.
			decoded, err := webp.Decode(bytes.NewReader(encoded))
			if err != nil {
				t.Fatalf("webp.Decode: %v", err)
			}
			b := src.Bounds()
			transparent, checked := 0, 0
			for y := b.Min.Y; y < b.Max.Y; y++ {
				for x := b.Min.X; x < b.Max.X; x++ {
					_, _, _, srcA := src.At(x, y).RGBA()
					if srcA == 0 {
						_, _, _, gotA := decoded.At(x, y).RGBA()
						if gotA != 0 {
							t.Errorf("pixel (%d,%d): expected transparent, got alpha=%d", x, y, gotA>>8)
						}
						transparent++
					}
					checked++
				}
			}
			t.Logf("%s: %dx%d, %d/%d transparent pixels verified, WebP size=%d bytes",
				path, b.Dx(), b.Dy(), transparent, checked, len(encoded))
		})
	}
}
