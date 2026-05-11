package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"testing"

	"golang.org/x/image/webp"
)

func TestDiagnosticCoeffs(t *testing.T) {
	// 16x16 solid red
	img := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.Set(x, y, color.NRGBA{R: 255, G: 0, B: 0, A: 255})
		}
	}

	yuv := rgbaToYUV420(img)
	qm := buildQuantMatrices(100)

	// Simulate what encoder does for UV
	dcU := computeDCUV(yuv.u, yuv.uvStride, 0, 0, yuv.width, yuv.height)
	dcV := computeDCUV(yuv.v, yuv.uvStride, 0, 0, yuv.width, yuv.height)
	fmt.Printf("dcU=%d dcV=%d (both should be 128 for topleft MB)\n", dcU, dcV)
	fmt.Printf("U[0]=%d V[0]=%d\n", yuv.u[0], yuv.v[0])
	fmt.Printf("U residual=%d V residual=%d\n", int(yuv.u[0])-dcU, int(yuv.v[0])-dcV)

	// First U block (by=0, bx=0): coords (0, 0)
	var uSrc4 [16]int16
	var uPred4 [16]int16
	extractBlock4x4UV(yuv.u, yuv.uvStride, 0, 0, yuv.width, yuv.height, uSrc4[:])
	fillPred4x4(uPred4[:], uint8(dcU))
	var uDct [16]int16
	fTransform(uSrc4[:], uPred4[:], uDct[:])
	var uQuant [16]int16
	quantizeBlock(uDct[:], uQuant[:], &qm.uv, 0)
	fmt.Printf("U block(0,0) src=%v\n", uSrc4)
	fmt.Printf("U block(0,0) pred=%v\n", uPred4)
	fmt.Printf("U block(0,0) dct=%v\n", uDct)
	fmt.Printf("U block(0,0) quant=%v\n", uQuant)
	
	// First V block (by=0, bx=0): coords (0, 0)
	var vSrc4 [16]int16
	var vPred4 [16]int16
	extractBlock4x4UV(yuv.v, yuv.uvStride, 0, 0, yuv.width, yuv.height, vSrc4[:])
	fillPred4x4(vPred4[:], uint8(dcV))
	var vDct [16]int16
	fTransform(vSrc4[:], vPred4[:], vDct[:])
	var vQuant [16]int16
	quantizeBlock(vDct[:], vQuant[:], &qm.uv, 0)
	fmt.Printf("V block(0,0) src=%v\n", vSrc4)
	fmt.Printf("V block(0,0) quant=%v\n", vQuant)

	// Encode and decode
	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 100}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	r, g, b, _ := dec.At(0, 0).RGBA()
	t.Logf("decoded: R=%d G=%d B=%d", r>>8, g>>8, b>>8)
}
