package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"os"
	"testing"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

func TestLumaQuality(t *testing.T) {
	f, _ := os.Open("test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	defer f.Close()
	src, _ := jpeg.Decode(f)
	
	dst := image.NewNRGBA(image.Rect(0, 0, 300, 300))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), xdraw.Src, nil)
	
	yuv := rgbaToYUV420(dst)
	
	// Encode
	var buf bytes.Buffer
	Encode(&buf, dst, &Options{Quality: 90})
	dec, _ := webp.Decode(bytes.NewReader(buf.Bytes()))
	decYCbCr, ok := dec.(*image.YCbCr)
	if !ok {
		t.Fatal("decoded image is not YCbCr")
	}
	
	// Compare Y channel
	var mseY, sumYsrc, sumYdec float64
	for py := 0; py < 300; py++ {
		for px := 0; px < 300; px++ {
			ysrc := float64(yuv.y[py*yuv.yStride+px])
			ydec := float64(decYCbCr.Y[py*decYCbCr.YStride+px])
			diff := ysrc - ydec
			mseY += diff*diff
			sumYsrc += ysrc
			sumYdec += ydec
		}
	}
	n := float64(300*300)
	mseY /= n
	t.Logf("Luma: src_mean=%.1f dec_mean=%.1f PSNR=%.2f dB", sumYsrc/n, sumYdec/n, 10*math.Log10(255*255/mseY))
	
	// Check some pixels
	for _, coord := range [][2]int{{0,0},{100,100},{150,150},{200,200}} {
		px, py := coord[0], coord[1]
		ysrc := yuv.y[py*yuv.yStride+px]
		ydec := decYCbCr.Y[py*decYCbCr.YStride+px]
		usrc := yuv.u[(py/2)*(yuv.uvStride)+(px/2)]
		udec := decYCbCr.Cb[(py/2)*decYCbCr.CStride+(px/2)]
		vsrc := yuv.v[(py/2)*(yuv.uvStride)+(px/2)]
		vdec := decYCbCr.Cr[(py/2)*decYCbCr.CStride+(px/2)]
		fmt.Printf("(%d,%d): Y_src=%d Y_dec=%d U_src=%d U_dec=%d V_src=%d V_dec=%d\n",
			px, py, ysrc, ydec, usrc, udec, vsrc, vdec)
	}
}
