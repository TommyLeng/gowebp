package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"testing"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

func TestMB66Trace(t *testing.T) {
	f, _ := os.Open("test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	defer f.Close()
	src, _ := jpeg.Decode(f)
	dst := image.NewNRGBA(image.Rect(0, 0, 300, 300))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), xdraw.Src, nil)

	yuv := rgbaToYUV420(dst)

	// Check pixel (100,100) → MB(6,6) → local (4,4) → block (by=1,bx=0) within MB
	// Source Y at (100,100)
	px100, py100 := 100, 100
	mbX, mbY := px100/16, py100/16
	fmt.Printf("Pixel (%d,%d) → MB(%d,%d)\n", px100, py100, mbX, mbY)
	fmt.Printf("Source Y at (100,100) = %d\n", yuv.y[py100*yuv.yStride+px100])

	// Check a few Y values around MB(6,6)
	fmt.Printf("Y values in MB(6,6) top-left area:\n")
	for by := 0; by < 2; by++ {
		for bx := 0; bx < 4; bx++ {
			vy := yuv.y[(mbY*16+by)*yuv.yStride+mbX*16+bx]
			fmt.Printf("  Y(%d,%d)=%d", mbX*16+bx, mbY*16+by, vy)
		}
		fmt.Println()
	}

	// Source border pixels for MB(6,6) DC prediction
	// Top row = Y at (mbX*16..mbX*16+15, mbY*16-1)
	// Left col = Y at (mbX*16-1, mbY*16..mbY*16+15)
	fmt.Printf("\nSource border pixels for MB(6,6):\n")
	fmt.Printf("Top row: ")
	sumTop := 0
	for x := 0; x < 16; x++ {
		v := int(yuv.y[(mbY*16-1)*yuv.yStride+(mbX*16+x)])
		sumTop += v
		if x < 4 { fmt.Printf("%d ", v) }
	}
	fmt.Printf("... mean=%d\n", sumTop/16)

	fmt.Printf("Left col: ")
	sumLeft := 0
	for y := 0; y < 16; y++ {
		v := int(yuv.y[(mbY*16+y)*yuv.yStride+(mbX*16-1)])
		sumLeft += v
		if y < 4 { fmt.Printf("%d ", v) }
	}
	fmt.Printf("... mean=%d\n", sumLeft/16)

	dcFromSource := (sumTop + sumLeft + 16) >> 5
	fmt.Printf("DC pred from source = %d\n", dcFromSource)

	// Encode and check decoded pixel
	var buf bytes.Buffer
	Encode(&buf, dst, &Options{Quality: 90})
	dec, _ := webp.Decode(bytes.NewReader(buf.Bytes()))
	decYCbCr, _ := dec.(*image.YCbCr)
	if decYCbCr != nil {
		ydec := decYCbCr.Y[py100*decYCbCr.YStride+px100]
		fmt.Printf("\nDecoded Y at (100,100) = %d (src=%d)\n", ydec, yuv.y[py100*yuv.yStride+px100])
	}
}
