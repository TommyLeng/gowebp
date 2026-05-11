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

func TestDiagnosticQ100(t *testing.T) {
	// Create a solid colored image: pure red at quality 100
	img := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.Set(x, y, color.NRGBA{R: 255, G: 0, B: 0, A: 255})
		}
	}

	// Check UV values
	yuv := rgbaToYUV420(img)
	fmt.Printf("Y=%d U=%d V=%d\n", yuv.y[0], yuv.u[0], yuv.v[0])

	// Build matrices at quality 100
	qm := buildQuantMatrices(100)
	fmt.Printf("UV DC step=%d AC step=%d\n", qm.uv.q[0], qm.uv.q[1])
	fmt.Printf("Y1 DC step=%d AC step=%d\n", qm.y1.q[0], qm.y1.q[1])

	// Compute UV source DC for first block
	uSrc := int(yuv.u[0]) // =119
	dcU := computeDCUV(yuv.u, yuv.uvStride, 0, 0, yuv.width, yuv.height)
	fmt.Printf("uSrc=%d dcU=%d residual=%d\n", uSrc, dcU, uSrc-dcU)
	
	// DCT of the residual
	var src4 [16]int16
	var pred4 [16]int16
	extractBlock4x4UV(yuv.u, yuv.uvStride, 0, 0, yuv.width, yuv.height, src4[:])
	fillPred4x4(pred4[:], uint8(dcU))
	fmt.Printf("src4[0]=%d pred4[0]=%d diff=%d\n", src4[0], pred4[0], src4[0]-pred4[0])
	var dctOut [16]int16
	fTransform(src4[:], pred4[:], dctOut[:])
	fmt.Printf("DCT[0]=%d (DC coeff)\n", dctOut[0])
	var quant [16]int16
	quantizeBlock(dctOut[:], quant[:], &qm.uv, 0)
	fmt.Printf("Quant[0]=%d\n", quant[0])

	// Encode at quality 100
	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 100}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	t.Logf("size: %d bytes", buf.Len())

	// Decode
	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// Check decoded pixels
	r, g, b, _ := dec.At(0, 0).RGBA()
	t.Logf("decoded(0,0): R=%d G=%d B=%d (expected ~255,0,0)", r>>8, g>>8, b>>8)

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
