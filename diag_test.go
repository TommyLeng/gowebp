package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"math"
	"testing"

	"golang.org/x/image/webp"
)

func TestDiagnosticUV(t *testing.T) {
	// Create a solid colored image: pure red
	// RGB (255,0,0) -> Y≈81, U≈90, V≈240
	img := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.Set(x, y, color.NRGBA{R: 255, G: 0, B: 0, A: 255})
		}
	}

	// Check UV values
	yuv := rgbaToYUV420(img)
	fmt.Printf("Y[0]=%d U[0]=%d V[0]=%d\n", yuv.y[0], yuv.u[0], yuv.v[0])

	// Encode
	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 90}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	t.Logf("size: %d bytes", buf.Len())

	// Decode
	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// Check decoded pixels
	for x := 0; x < 4; x++ {
		r, g, b, _ := dec.At(x*4, 0).RGBA()
		t.Logf("pixel(%d,0): R=%d G=%d B=%d", x*4, r>>8, g>>8, b>>8)
	}

	// PSNR
	var mse float64
	n := 0
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			r1, g1, b1, _ := img.At(x, y).RGBA()
			r2, g2, b2, _ := dec.At(x, y).RGBA()
			dr := float64(int(r1>>8) - int(r2>>8))
			dg := float64(int(g1>>8) - int(g2>>8))
			db := float64(int(b1>>8) - int(b2>>8))
			mse += dr*dr + dg*dg + db*db
			n += 3
		}
	}
	mse /= float64(n)
	if mse > 0 {
		t.Logf("PSNR: %.2f dB", 10*math.Log10(255*255/mse))
	} else {
		t.Logf("PSNR: inf")
	}
}
