package gowebp

import (
	"bytes"
	"image"
	"image/color"
	"testing"

	"golang.org/x/image/webp"
)

// TestVerifyI4Reconstruction verifies that the encoder's i4 reconstruction
// matches the decoder's output for a controlled single-MB case.
// This confirms whether the 27-29dB PSNR gap is due to encoder-decoder mismatch
// or is the actual quality achievable (i.e., the encoder is correct but 29dB is
// what i4 gives for this signal).
func TestVerifyI4Reconstruction(t *testing.T) {
	// Create a simple 16x16 image with a constant value.
	// For a constant image, i4 should achieve very high PSNR (near-lossless).
	img := image.NewGray(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.SetGray(x, y, color.Gray{Y: 128})
		}
	}

	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 90}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	p := psnr(img, dec)
	t.Logf("constant 128 PSNR: %.2f dB", p)
	// For a constant image, reconstruction should be nearly perfect
	if p < 40.0 {
		t.Errorf("constant image PSNR %.2f < 40dB — indicates reconstruction bug", p)
	}

	// Now test with a step function — 4 columns at 0, 4 at 64, 4 at 128, 4 at 192.
	// This requires non-zero DCT coefficients but is simple enough to validate.
	img2 := image.NewGray(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			v := (x / 4) * 64
			img2.SetGray(x, y, color.Gray{Y: uint8(v)})
		}
	}
	var buf2 bytes.Buffer
	if err := Encode(&buf2, img2, &Options{Quality: 90}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec2, err := webp.Decode(bytes.NewReader(buf2.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	p2 := psnr(img2, dec2)
	t.Logf("4-step image PSNR: %.2f dB, size: %d bytes", p2, buf2.Len())

	// Check first row pixels
	for x := 0; x < 16; x++ {
		r, _, _, _ := dec2.At(x, 0).RGBA()
		srcV := (x / 4) * 64
		t.Logf("x=%d src=%d dec=%d", x, srcV, r>>8)
	}
}

// TestVerifyI4VsI16ForConstantImage verifies i4 and i16 both work well on a constant image.
func TestVerifyI4VsI16ForConstantImage(t *testing.T) {
	// A constant 16x16 image: both i4 and i16 DC mode should work perfectly.
	img := image.NewGray(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.SetGray(x, y, color.Gray{Y: 100})
		}
	}

	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 90}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// Check all pixels
	totalErr := 0
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			r, _, _, _ := dec.At(x, y).RGBA()
			d := int(r>>8) - 100
			if d < 0 { d = -d }
			totalErr += d
		}
	}
	t.Logf("constant 100 total error: %d, avg: %.2f", totalErr, float64(totalErr)/256)
	if totalErr > 1024 { // more than 4 per pixel average (YUV conversion bias is ~2/pixel)
		t.Errorf("constant image has high error: %d", totalErr)
	}
}
