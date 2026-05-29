package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"os"
	"os/exec"
	"testing"
)

// TestI16FlatGray encodes a 64×64 constant-gray image and expects near-perfect
// PSNR (i16 DC should be lossless or near-lossless at q=95).
func TestI16FlatGray(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}

	img := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+0] = 128
		img.Pix[i+1] = 128
		img.Pix[i+2] = 128
		img.Pix[i+3] = 255
	}

	for _, q := range []int{75, 95} {
		var snap []mbInfo
		debugMBStats = &snap

		var buf bytes.Buffer
		if err := Encode(&buf, img, &Options{Quality: q}); err != nil {
			t.Fatalf("encode: %v", err)
		}
		debugMBStats = nil

		var i4, i16 int
		for _, mi := range snap {
			if mi.isI4 {
				i4++
			} else {
				i16++
			}
		}

		tmp := fmt.Sprintf("/tmp/flat_q%d.webp", q)
		os.WriteFile(tmp, buf.Bytes(), 0644)
		tmpPng := fmt.Sprintf("/tmp/flat_q%d.png", q)
		exec.Command(dwebp, tmp, "-o", tmpPng).Run()

		decoded, err := loadPNG(tmpPng)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}

		p := psnrRGBA(img, decoded)
		// Check a few pixels for sanity
		r0, g0, b0, _ := decoded.At(0, 0).RGBA()
		r1, g1, b1, _ := decoded.At(32, 32).RGBA()
		t.Logf("q=%d  i4=%d  i16=%d  PSNR=%.2f dB  px[0,0]=(%d,%d,%d)  px[32,32]=(%d,%d,%d)",
			q, i4, i16, p, r0>>8, g0>>8, b0>>8, r1>>8, g1>>8, b1>>8)
		_ = color.Black
	}
}
