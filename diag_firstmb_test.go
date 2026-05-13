package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"testing"

	"golang.org/x/image/webp"
)

func TestFirstMBRecon(t *testing.T) {
	// Simple gradient: rows have increasing Y values
	// So DC pred for MB(1,0) depends on right column of MB(0,0)
	img := image.NewGray(image.Rect(0, 0, 32, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.SetGray(x, y, color.Gray{Y: 100}) // MB(0,0) = 100
		}
		for x := 16; x < 32; x++ {
			img.SetGray(x, y, color.Gray{Y: 50}) // MB(1,0) = 50
		}
	}

	yuv := rgbaToYUV420(img, &frameArena{})
	fmt.Printf("Y[0,0]=%d (MB0), Y[16,0]=%d (MB1)\n", yuv.y[0], yuv.y[16])

	qm := buildQuantMatrices(90)

	// Manually simulate MB(0,0) encoding and check recon
	reconStride := 2 * 16 // 2 MBs wide
	recon := make([]uint8, reconStride*16)

	// Predict from recon for MB(0,0)
	var pred0 [256]int16
	intra16PredictFromRecon(I16_DC_PRED, recon, reconStride, 0, 0, 32, 16, pred0[:])
	fmt.Printf("MB(0,0) pred[0]=%d (expect 128)\n", pred0[0])

	// DCT, WHT, quantize
	var src16 [256]int16
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			src16[y*16+x] = int16(yuv.y[y*yuv.yStride+x])
		}
	}
	var yDcRaw [16]int16
	var yAcQ [16][16]int16
	for by := 0; by < 4; by++ {
		for bx := 0; bx < 4; bx++ {
			n := by*4 + bx
			var s4, p4 [16]int16
			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					s4[y*4+x] = src16[(by*4+y)*16+bx*4+x]
					p4[y*4+x] = pred0[(by*4+y)*16+bx*4+x]
				}
			}
			var dct [16]int16
			fTransform(s4[:], p4[:], dct[:])
			yDcRaw[n] = dct[0]
			dct[0] = 0
			quantizeBlock(dct[:], yAcQ[n][:], &qm.y1, 1)
		}
	}

	var whtOut [16]int16
	var dcQ [16]int16
	fTransformWHT(yDcRaw[:], whtOut[:])
	quantizeBlockWHT(whtOut[:], dcQ[:], &qm.y2)
	fmt.Printf("WHT quant[0]=%d\n", dcQ[0])

	// Reconstruct
	var whtDeq [16]int16
	for n := 0; n < 16; n++ {
		j := int(kZigzag[n])
		whtDeq[n] = int16(int32(dcQ[n]) * int32(qm.y2.q[j]))
	}
	var dcBlock [16]int16
	inverseWHT16(whtDeq[:], dcBlock[:])
	fmt.Printf("dcBlock[0]=%d\n", dcBlock[0])

	// Reconstruct block (0,0) of MB(0,0)
	var ras [16]int16
	dequantizeBlock(yAcQ[0][:], ras[:], &qm.y1, dcBlock[0])
	var p4 [16]int16
	for i := range p4 { p4[i] = pred0[i] }
	var rec [16]int16
	iTransform4x4(ras[:], p4[:], rec[:])
	fmt.Printf("Block(0,0) recon[0]=%d (src=%d, pred=%d)\n", rec[0], src16[0], pred0[0])

	// Store in recon
	for by := 0; by < 4; by++ {
		for bx := 0; bx < 4; bx++ {
			n := by*4 + bx
			var p4b [16]int16
			var rasb [16]int16
			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					p4b[y*4+x] = pred0[(by*4+y)*16+bx*4+x]
				}
			}
			dequantizeBlock(yAcQ[n][:], rasb[:], &qm.y1, dcBlock[n])
			var recb [16]int16
			iTransform4x4(rasb[:], p4b[:], recb[:])
			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					recon[(by*4+y)*reconStride+bx*4+x] = uint8(recb[y*4+x])
				}
			}
		}
	}
	fmt.Printf("recon[0..3]: %d %d %d %d (expect ~%d)\n", recon[0], recon[1], recon[2], recon[3], yuv.y[0])
	fmt.Printf("recon[15] (right col MB0): %d\n", recon[15])

	// Now predict for MB(1,0) from recon
	var pred1 [256]int16
	intra16PredictFromRecon(I16_DC_PRED, recon, reconStride, 1, 0, 32, 16, pred1[:])
	fmt.Printf("MB(1,0) pred[0]=%d (left-only, expect ~%d)\n", pred1[0], recon[15])

	// Encode actual image and decode
	var buf bytes.Buffer
	Encode(&buf, img, &Options{Quality: 90})
	dec, _ := webp.Decode(bytes.NewReader(buf.Bytes()))
	decYCbCr, _ := dec.(*image.YCbCr)
	if decYCbCr != nil {
		fmt.Printf("\nDecoded: Y[0,0]=%d (src=%d), Y[16,0]=%d (src=%d)\n",
			decYCbCr.Y[0], yuv.y[0], decYCbCr.Y[16], yuv.y[16])
	}
}
