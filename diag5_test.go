package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"testing"

	"golang.org/x/image/webp"
)

// Test with a known 16x16 image where we know what the output should be
func TestDCOnlyBlock(t *testing.T) {
	// Solid gray 128 -> Y=126 (close to 128)
	// For the first MB, DC prediction = 128
	// Residual should be ~0, no coefficients
	img := image.NewGray(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.SetGray(x, y, color.Gray{Y: 128})
		}
	}

	yuv := rgbaToYUV420(img)
	fmt.Printf("Y[0]=%d (for input gray=128)\n", yuv.y[0])
	
	// Build quantization matrices
	qm := buildQuantMatrices(90)
	
	// Simulate encoding of first MB luma
	// DC prediction = 128 (no top/left neighbors)
	dc := computeDCY(yuv, 0, 0)
	fmt.Printf("DC prediction = %d\n", dc)
	
	// Build i16 prediction
	var pred16 [256]int16
	intra16Predict(I16_DC_PRED, yuv, 0, 0, pred16[:])
	fmt.Printf("pred16[0]=%d\n", pred16[0])

	// Extract source  
	var src16 [256]int16
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			src16[y*16+x] = int16(yuv.y[y*yuv.yStride+x])
		}
	}
	fmt.Printf("src16[0]=%d\n", src16[0])

	// DCT of 4x4 blocks
	var yDcLevels [16]int16
	var yAcLevels [16][16]int16
	for by := 0; by < 4; by++ {
		for bx := 0; bx < 4; bx++ {
			n := by*4 + bx
			var src4, pred4 [16]int16
			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					src4[y*4+x] = src16[(by*4+y)*16+(bx*4+x)]
					pred4[y*4+x] = pred16[(by*4+y)*16+(bx*4+x)]
				}
			}
			var dctOut [16]int16
			fTransform(src4[:], pred4[:], dctOut[:])
			yDcLevels[n] = dctOut[0]
			dctOut[0] = 0
			var acQ [16]int16
			quantizeBlock(dctOut[:], acQ[:], &qm.y1, 1)
			yAcLevels[n] = acQ
			fmt.Printf("Block(%d,%d): DC_raw=%d, AC[0]=%d\n", bx, by, yDcLevels[n], acQ[1])
		}
	}

	// WHT
	var whtOut [16]int16
	var dcQuantLevels [16]int16
	fTransformWHT(yDcLevels[:], whtOut[:])
	quantizeBlockWHT(whtOut[:], dcQuantLevels[:], &qm.y2)
	fmt.Printf("WHT output[0]=%d, quant[0]=%d\n", whtOut[0], dcQuantLevels[0])

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
	r, g, b, _ := dec.At(0, 0).RGBA()
	t.Logf("decoded(0,0): R=%d G=%d B=%d", r>>8, g>>8, b>>8)
}
