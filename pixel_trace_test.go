package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	"os"
	"testing"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

// TestPixelTrace traces the reconstruction of a specific pixel that mismatches.
// First mismatch found: (29, 0) enc=133 dec=128.
// This is in MB(1,0), block (bx=3, by=0) (bx=29-16=13 → 13/4=3, by=0).
func TestPixelTrace(t *testing.T) {
	f, err := os.Open("test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	if err != nil { t.Fatalf("open: %v", err) }
	defer f.Close()
	srcImg, _, err := image.Decode(f)
	if err != nil { t.Fatalf("decode jpeg: %v", err) }
	dst := image.NewNRGBA(image.Rect(0, 0, 48, 48))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), srcImg, srcImg.Bounds(), draw.Src, nil)

	yuv := rgbaToYUV420(dst)
	qm := buildQuantMatrices(86)
	baseQ := qualityToLevel(86)

	mbW := yuv.mbW / 16
	mbH := yuv.mbH / 16

	var y1qSum int
	for i := 0; i < 16; i++ {
		y1qSum += int(qm.y1.q[i])
	}
	qI4 := (y1qSum + 8) >> 4
	lambdaI4 := (3 * qI4 * qI4) >> 7
	if lambdaI4 < 1 { lambdaI4 = 1 }

	reconStride := mbW * 16
	recon := make([]uint8, reconStride*mbH*16)
	topI4Modes := make([]int, mbW*4)

	// Process MB(0,0) first — this fills recon with MB0's reconstruction
	// so we can then trace MB(1,0).
	for mbY := 0; mbY < mbH; mbY++ {
		leftI4Mode := [4]int{}

		for mbX := 0; mbX < mbW; mbX++ {
			px := mbX * 16
			py := mbY * 16

			var mbReconI4 [16 * 16]uint8
			var localI4Modes [16]int

			topBlkMode := make([]int, 4)
			for bx := 0; bx < 4; bx++ {
				topBlkMode[bx] = topI4Modes[mbX*4+bx]
			}
			leftBlkMode := [4]int{leftI4Mode[0], leftI4Mode[1], leftI4Mode[2], leftI4Mode[3]}

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
					var bestBlkRecon [16]uint8
					var bestBlkAcLevels [16]int16

					// Trace MB(1,0) block (bx=3, by=0)
					isTracedBlock := mbX == 1 && mbY == 0 && bx == 3 && by == 0

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
							for i := 0; i < 16; i++ {
								bestBlkRecon[i] = uint8(recBlock[i])
							}

							if isTracedBlock {
								t.Logf("MB(1,0) block(3,0) mode=%d score=%d distortion=%d", mode, score, distortion)
								t.Logf("  pred[0]=%d src[0]=%d recon[0]=%d", pred4[0], src4[0], recBlock[0])
								t.Logf("  pred row0: %v", pred4[:4])
								t.Logf("  recon row0: %d %d %d %d", recBlock[0], recBlock[1], recBlock[2], recBlock[3])
								t.Logf("  ac[0]=%d ac[1]=%d", acQ[0], acQ[1])
							}
						}
					}

					if isTracedBlock {
						t.Logf("MB(1,0) block(3,0) BEST mode=%d", bestBlkMode)
						t.Logf("  src: %d %d %d %d", src4[0], src4[1], src4[2], src4[3])
						t.Logf("  recon: %d %d %d %d", bestBlkRecon[0], bestBlkRecon[1], bestBlkRecon[2], bestBlkRecon[3])
						t.Logf("  ctx left=%v topLeft=%d top=%v", ctx.left, ctx.topLeft, ctx.top[:4])
						t.Logf("  ctx top[4..7]=%v", ctx.top[4:])
						// Check what left column of mbReconI4 looks like (for context)
						fmt.Printf("  mbReconI4 column 11 (x=27 within MB1): ")
						for y := 0; y < 4; y++ {
							fmt.Printf("%d ", mbReconI4[y*16+11])
						}
						fmt.Println()
						// Actual column used: bpx-1 = 28-1 = 27, which is at mbReconI4 col 27-16=11
					}

					localI4Modes[blkIdx] = bestBlkMode

					for y := 0; y < 4; y++ {
						for x := 0; x < 4; x++ {
							mbReconI4[(by*4+y)*16+(bx*4+x)] = bestBlkRecon[y*4+x]
						}
					}

					topBlkMode[bx] = bestBlkMode
				}
				leftBlkMode[by] = localI4Modes[by*4+3]
			}

			for y := 0; y < 16; y++ {
				for x := 0; x < 16; x++ {
					recon[(py+y)*reconStride+(px+x)] = mbReconI4[y*16+x]
				}
			}

			for bx := 0; bx < 4; bx++ {
				topI4Modes[mbX*4+bx] = localI4Modes[3*4+bx]
			}
			for by := 0; by < 4; by++ {
				leftI4Mode[by] = localI4Modes[by*4+3]
			}
		}
	}

	// Now encode with force-i4 and decode
	vp8Data, _, _ := encodeFrameForceI4WithRecon(yuv, qm, baseQ)
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

	// Print decoder's pixels for blocks (bx=2,3) of MB1
	t.Logf("Decoder Y for pixels (24..31, 0..3):")
	for y := 0; y < 4; y++ {
		for x := 24; x < 32; x++ {
			yi := ycbcr.YOffset(x, y)
			decY := int(ycbcr.Y[yi])
			fmt.Printf("(%d,%d)=%d ", x, y, decY)
		}
		fmt.Println()
	}

	// Also print encoder recon for x=24..31 from encodeFrameForceI4WithRecon
	vp8DataFull, encReconFull, reconStrideFull := encodeFrameForceI4WithRecon(yuv, qm, baseQ)
	_ = vp8DataFull
	t.Logf("Encoder recon for pixels (24..31, 0..3):")
	for y := 0; y < 4; y++ {
		for x := 24; x < 32; x++ {
			encY := int(encReconFull[y*reconStrideFull+x])
			fmt.Printf("(%d,%d)=%d ", x, y, encY)
		}
		fmt.Println()
	}
}
