package gowebp

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"testing"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

// Diagnose chroma encoding for the actual test image
func TestDiagChromaReal(t *testing.T) {
	f, _ := os.Open("test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	defer f.Close()
	src, _ := jpeg.Decode(f)
	
	dst := image.NewNRGBA(image.Rect(0, 0, 300, 300))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), xdraw.Src, nil)
	
	// Check source pixel (0,0) 
	r, g, b, _ := dst.At(0, 0).RGBA()
	t.Logf("src(0,0): R=%d G=%d B=%d", r>>8, g>>8, b>>8)
	
	// Convert to YUV
	yuv := rgbaToYUV420(dst)
	t.Logf("yuv(0,0): Y=%d U=%d V=%d", yuv.y[0], yuv.u[0], yuv.v[0])
	
	// Check what DC prediction would be for top-left MB's UV
	dcU := computeDCUV(yuv.u, yuv.uvStride, 0, 0, yuv.width, yuv.height)
	dcV := computeDCUV(yuv.v, yuv.uvStride, 0, 0, yuv.width, yuv.height)
	t.Logf("MB(0,0) DC pred: U=%d V=%d", dcU, dcV)
	
	// Compute UV residual DC coeff for first block
	var uSrc [16]int16
	var vSrc [16]int16
	var uPred [16]int16
	var vPred [16]int16
	extractBlock4x4UV(yuv.u, yuv.uvStride, 0, 0, yuv.width, yuv.height, uSrc[:])
	extractBlock4x4UV(yuv.v, yuv.uvStride, 0, 0, yuv.width, yuv.height, vSrc[:])
	fillPred4x4(uPred[:], uint8(dcU))
	fillPred4x4(vPred[:], uint8(dcV))
	
	qm := buildQuantMatrices(90)
	
	var uDct, vDct [16]int16
	fTransform(uSrc[:], uPred[:], uDct[:])
	fTransform(vSrc[:], vPred[:], vDct[:])
	
	var uQ, vQ [16]int16
	quantizeBlock(uDct[:], uQ[:], &qm.uv, 0)
	quantizeBlock(vDct[:], vQ[:], &qm.uv, 0)
	
	t.Logf("U DC quant=%d (raw=%d) V DC quant=%d (raw=%d)", uQ[0], uDct[0], vQ[0], vDct[0])
	t.Logf("uSrc[0..3]=%v uPred[0..3]=%v", uSrc[:4], uPred[:4])
	t.Logf("vSrc[0..3]=%v vPred[0..3]=%v", vSrc[:4], vPred[:4])
	
	// Encode and decode
	var buf bytes.Buffer
	Encode(&buf, dst, &Options{Quality: 90})
	dec, _ := webp.Decode(bytes.NewReader(buf.Bytes()))
	
	// Decoded pixel (0,0)
	r2, g2, b2, _ := dec.At(0, 0).RGBA()
	t.Logf("decoded(0,0): R=%d G=%d B=%d", r2>>8, g2>>8, b2>>8)
	
	// Convert decoded back to YUV (approximate)
	decYCbCr := image.NewYCbCr(image.Rect(0,0,300,300), image.YCbCrSubsampleRatio420)
	for py := 0; py < 300; py++ {
		for px := 0; px < 300; px++ {
			rr, gg, bb, _ := dec.At(px, py).RGBA()
			y, cb, cr := color.RGBToYCbCr(uint8(rr>>8), uint8(gg>>8), uint8(bb>>8))
			decYCbCr.Y[py*decYCbCr.YStride+px] = y
			if px%2 == 0 && py%2 == 0 {
				decYCbCr.Cb[(py/2)*decYCbCr.CStride+(px/2)] = cb
				decYCbCr.Cr[(py/2)*decYCbCr.CStride+(px/2)] = cr
			}
		}
	}
	t.Logf("dec Y[0]=%d Cb[0]=%d Cr[0]=%d (src U=%d V=%d)", 
		decYCbCr.Y[0], decYCbCr.Cb[0], decYCbCr.Cr[0],
		yuv.u[0], yuv.v[0])
}
