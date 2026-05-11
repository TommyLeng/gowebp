package gowebp

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"os"
	"testing"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

// TestCascadeTrace traces the reconstruction of a single 16x16 MB
// with a known gradient, comparing encoder mbReconI4 vs decoder Y output.
// This pinpoints where the divergence happens.
func TestCascadeTrace(t *testing.T) {
	// 16x16 horizontal gradient: src[x] = x*16
	img := image.NewGray(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.Set(x, y, color.Gray{Y: uint8(x * 16)})
		}
	}

	yuv := rgbaToYUV420(img)
	qm := buildQuantMatrices(86)
	baseQ := qualityToLevel(86)

	// Manually run i4 encoding on the single MB, capturing mbReconI4 state
	// after each 4x4 block.
	mbX, mbY := 0, 0
	px, py := mbX*16, mbY*16
	reconStride := yuv.mbW
	recon := make([]uint8, reconStride*yuv.mbH)
	var mbReconI4 [16 * 16]uint8

	var y1qSum int
	for i := 0; i < 16; i++ {
		y1qSum += int(qm.y1.q[i])
	}
	qI4 := (y1qSum + 8) >> 4
	lambdaI4 := (3 * qI4 * qI4) >> 7
	if lambdaI4 < 1 { lambdaI4 = 1 }

	var localI4Modes [16]int
	var i4AcLevels [16][16]int16
	topBlkMode := [4]int{}
	leftBlkMode := [4]int{}

	for by := 0; by < 4; by++ {
		for bx := 0; bx < 4; bx++ {
			blkIdx := by*4 + bx
			bpx := px + bx*4
			bpy := py + by*4

			ctx := buildPred4ContextWithMBRecon(yuv, recon, reconStride, mbReconI4[:], px, py, bpx, bpy)

			var src4 [16]int16
			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					sx := bpx + x
					sy := bpy + y
					if sx >= yuv.width { sx = yuv.width - 1 }
					if sy >= yuv.height { sy = yuv.height - 1 }
					src4[y*4+x] = int16(yuv.y[sy*yuv.yStride+sx])
				}
			}

			topPred := topBlkMode[bx]
			leftPred := 0
			if bx > 0 {
				leftPred = localI4Modes[blkIdx-1]
			} else {
				leftPred = leftBlkMode[by]
			}

			bestBlkMode := B_DC_PRED
			bestBlkScore := int64(1<<62 - 1)
			var bestBlkRecon [16]int16
			var bestBlkPred [16]int16
			var bestBlkAcLevels [16]int16

			for mode := 0; mode < numI4Modes; mode++ {
				var pred4 [16]int16
				intra4Predict(mode, ctx, pred4[:])

				var dctOut [16]int16
				fTransform(src4[:], pred4[:], dctOut[:])

				var acQ [16]int16
				quantizeBlock(dctOut[:], acQ[:], &qm.y1, 0)

				var raster [16]int16
				for n := 0; n < 16; n++ {
					j := int(kZigzag[n])
					raster[j] = int16(int32(acQ[n]) * int32(qm.y1.q[j]))
				}
				var recBlock [16]int16
				iTransform4x4(raster[:], pred4[:], recBlock[:])

				var distortion int64
				for i := 0; i < 16; i++ {
					d := int64(src4[i]) - int64(recBlock[i])
					distortion += d * d
				}

				modeBits := i4ModeBitCost(mode, topPred, leftPred)
				score := distortion + int64(lambdaI4)*modeBits
				if score < bestBlkScore {
					bestBlkScore = score
					bestBlkMode = mode
					copy(bestBlkAcLevels[:], acQ[:])
					copy(bestBlkRecon[:], recBlock[:])
					copy(bestBlkPred[:], pred4[:])
				}
			}

			localI4Modes[blkIdx] = bestBlkMode
			i4AcLevels[blkIdx] = bestBlkAcLevels

			if by == 0 && bx <= 1 {
				t.Logf("Block (%d,%d) mode=%d src[0]=%d pred[0]=%d recon[0]=%d", bx, by, bestBlkMode, src4[0], bestBlkPred[0], bestBlkRecon[0])
				t.Logf("  ac[0]=%d (DC level), recon row0: %d %d %d %d", bestBlkAcLevels[0], bestBlkRecon[0], bestBlkRecon[1], bestBlkRecon[2], bestBlkRecon[3])
				t.Logf("  ctx left=%v top=%v", ctx.left, ctx.top[:4])
			}

			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					mbReconI4[(by*4+y)*16+(bx*4+x)] = uint8(bestBlkRecon[y*4+x])
				}
			}

			topBlkMode[bx] = bestBlkMode
		}
		leftBlkMode[by] = localI4Modes[by*4+3]
	}

	t.Logf("\nEncoder mbReconI4 row 0: %v", mbReconI4[:16])
	t.Logf("Encoder mbReconI4 row 1: %v", mbReconI4[16:32])

	// Now encode with force-i4 and decode, compare Y channel
	vp8Data := encodeFrameForceI4(yuv, qm, baseQ)
	var buf bytes.Buffer
	writeWebPHeader(&buf, vp8Data)

	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	ycbcr, ok := dec.(*image.YCbCr)
	if !ok {
		t.Fatalf("not YCbCr")
	}

	t.Logf("\nDecoder Y row 0:")
	for x := 0; x < 16; x++ {
		yi := ycbcr.YOffset(x, 0)
		t.Logf("  x=%d enc=%d dec=%d diff=%d", x, int(mbReconI4[x]), int(ycbcr.Y[yi]), int(mbReconI4[x])-int(ycbcr.Y[yi]))
	}

	// Check if ac levels match what the decoder would decode
	t.Logf("\nAC levels for block (0,0): %v", i4AcLevels[0])
	t.Logf("AC levels for block (1,0): %v", i4AcLevels[1])
}

// TestCascadeMultiMB tests cascade behavior for a 2-MB horizontal strip (32x16).
func TestCascadeMultiMB(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 32, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 32; x++ {
			v := x * 8
			if v > 255 { v = 255 }
			img.Set(x, y, color.Gray{Y: uint8(v)})
		}
	}

	yuv := rgbaToYUV420(img)
	qm := buildQuantMatrices(86)
	baseQ := qualityToLevel(86)

	vp8Data := encodeFrameForceI4(yuv, qm, baseQ)
	var buf bytes.Buffer
	writeWebPHeader(&buf, vp8Data)

	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	ycbcr, ok := dec.(*image.YCbCr)
	if !ok {
		t.Fatalf("not YCbCr")
	}

	p := psnrYLuma(img, dec)
	t.Logf("32x16 force-i4 luma PSNR: %.2f dB", p)

	// Compare source vs decoder for each column
	for x := 0; x < 32; x++ {
		v := x * 8
		if v > 255 { v = 255 }
		yi := ycbcr.YOffset(x, 0)
		decY := int(ycbcr.Y[yi])
		if x < 4 || (x >= 16 && x < 20) || x == 31 {
			t.Logf("x=%d src=%d decY=%d diff=%d", x, v, decY, v-decY)
		}
	}
}

// TestCascadeMultiRow tests cascade for multiple MB rows (32x32).
func TestCascadeMultiRow(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 32, 32))
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			v := (x + y) * 4
			if v > 255 { v = 255 }
			img.Set(x, y, color.Gray{Y: uint8(v)})
		}
	}

	yuv := rgbaToYUV420(img)
	qm := buildQuantMatrices(86)
	baseQ := qualityToLevel(86)

	vp8Data := encodeFrameForceI4(yuv, qm, baseQ)
	var buf bytes.Buffer
	writeWebPHeader(&buf, vp8Data)

	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	p := psnrYLuma(img, dec)
	t.Logf("32x32 force-i4 luma PSNR: %.2f dB", p)
}


// TestCompareI4vsI16Small compares force-i4 vs i16 for 48x48 photo.
func TestCompareI4vsI16Small(t *testing.T) {
	f, err := os.Open("test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	if err != nil { t.Fatalf("open: %v", err) }
	defer f.Close()
	src, _, err := image.Decode(f)
	if err != nil { t.Fatalf("decode: %v", err) }
	dst := image.NewNRGBA(image.Rect(0, 0, 48, 48))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Src, nil)

	yuv := rgbaToYUV420(dst)
	qm := buildQuantMatrices(86)
	baseQ := qualityToLevel(86)

	// Force-i4
	vp8DataI4 := encodeFrameForceI4(yuv, qm, baseQ)
	var bufI4 bytes.Buffer
	writeWebPHeader(&bufI4, vp8DataI4)
	decI4, _ := webp.Decode(bytes.NewReader(bufI4.Bytes()))
	pI4luma := psnrYLuma(dst, decI4)
	pI4rgb := psnr(dst, decI4)

	// i16-only
	vp8DataI16 := encodeFrame(yuv, qm, baseQ)
	var bufI16 bytes.Buffer
	writeWebPHeader(&bufI16, vp8DataI16)
	decI16, _ := webp.Decode(bytes.NewReader(bufI16.Bytes()))
	pI16luma := psnrYLuma(dst, decI16)
	pI16rgb := psnr(dst, decI16)

	t.Logf("Force-i4: luma PSNR=%.2f dB, RGB PSNR=%.2f dB, size=%d", pI4luma, pI4rgb, bufI4.Len())
	t.Logf("i16-only: luma PSNR=%.2f dB, RGB PSNR=%.2f dB, size=%d", pI16luma, pI16rgb, bufI16.Len())
}
