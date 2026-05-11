package gowebp

import (
	"bytes"
	"image"
	"image/color"
	"testing"

	"golang.org/x/image/webp"
)

// TestI4ReconVsDecoder checks whether the encoder's internal reconstruction
// buffer (what we store in mbReconI4) matches what the decoder produces.
// If there's a mismatch, we have the cascade bug.
//
// Strategy: run the encoder's i4 loop on a simple 16x16 image, capture
// the mbReconI4 buffer, encode the image normally, decode it, and compare
// the decoded pixels to mbReconI4.
func TestI4ReconVsDecoder(t *testing.T) {
	// Create a simple gradient image (16x16)
	img := image.NewGray(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			v := x * 16
			img.SetGray(x, y, color.Gray{Y: uint8(v)})
		}
	}

	// Run the encoder's i4 reconstruction manually to capture encoder's internal state.
	yuv := rgbaToYUV420(img)
	qm := buildQuantMatrices(90)

	mbX, mbY := 0, 0
	px, py := mbX*16, mbY*16
	reconStride := yuv.mbW // padded width
	recon := make([]uint8, reconStride*yuv.mbH)
	var mbReconI4 [16 * 16]uint8

	var src16 [256]int16
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			sx := px + x
			sy := py + y
			if sx >= yuv.width { sx = yuv.width - 1 }
			if sy >= yuv.height { sy = yuv.height - 1 }
			src16[y*16+x] = int16(yuv.y[sy*yuv.yStride+sx])
		}
	}

	topBlkMode := make([]int, 4)
	leftBlkMode := [4]int{}

	for by := 0; by < 4; by++ {
		for bx := 0; bx < 4; bx++ {
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
			leftPred := leftBlkMode[by]
			if bx > 0 {
				// leftPred is from localI4Modes[blkIdx-1], but we just use DC
				leftPred = 0
			}

			// Try all modes and pick best (simplified: just use DC for this test)
			bestMode := B_DC_PRED
			bestScore := int64(1<<62 - 1)
			var bestRecon [16]uint8
			var bestAcLevels [16]int16

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
				lambdaI4 := 3
				score := distortion + int64(lambdaI4)*modeBits

				if score < bestScore {
					bestScore = score
					bestMode = mode
					copy(bestAcLevels[:], acQ[:])
					for i := 0; i < 16; i++ {
						bestRecon[i] = uint8(recBlock[i])
					}
				}
			}

			_ = bestAcLevels
			_ = bestMode

			// Update mbReconI4
			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					mbReconI4[(by*4+y)*16+(bx*4+x)] = bestRecon[y*4+x]
				}
			}

			topBlkMode[bx] = bestMode
			if bx == 3 {
				_ = leftBlkMode
			}
		}
	}

	// Now encode the image normally and decode
	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 90}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	ycbcr, ok := dec.(*image.YCbCr)
	if !ok {
		t.Fatalf("decoded image is not YCbCr")
	}

	// Compare encoder's mbReconI4 to decoder's output (Y channel)
	t.Logf("Comparing encoder's mbReconI4 vs decoder Y channel:")
	totalMismatch := 0
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			encRecon := int(mbReconI4[y*16+x])
			yi := ycbcr.YOffset(x, y)
			decY := int(ycbcr.Y[yi])
			if encRecon != decY {
				totalMismatch++
				if totalMismatch <= 8 {
					t.Logf("  (%d,%d): enc=%d dec=%d diff=%d", x, y, encRecon, decY, encRecon-decY)
				}
			}
		}
	}
	t.Logf("Total mismatches: %d / 256", totalMismatch)
	if totalMismatch == 0 {
		t.Logf("PERFECT MATCH: encoder reconstruction = decoder output")
	} else if totalMismatch < 10 {
		t.Logf("NEAR MATCH: minimal mismatch (may be border effects)")
	} else {
		t.Logf("MISMATCH DETECTED: encoder-decoder reconstruction diverges")
	}
}
