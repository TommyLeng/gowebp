package gowebp

import (
	"bytes"
	"image"
	"image/draw"
	_ "image/jpeg"
	"os"
	"testing"
	"time"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

// encodeI4Enabled is a copy of encodeFrame with i4 enabled via RD comparison.
// Used to test i4 encoding without modifying the main encoder.
func encodeFrameI4Enabled(yuv *yuvImage, qm quantMatrices, baseQ int) []byte {
	w := yuv.width
	h := yuv.height
	mbW := yuv.mbW / 16
	mbH := yuv.mbH / 16

	var y1qSum, y2qSum int
	for i := 0; i < 16; i++ {
		y1qSum += int(qm.y1.q[i])
		y2qSum += int(qm.y2.q[i])
	}
	qI4 := (y1qSum + 8) >> 4
	qI16 := (y2qSum + 8) >> 4
	lambdaI4 := (3 * qI4 * qI4) >> 7
	if lambdaI4 < 1 { lambdaI4 = 1 }
	lambdaI16 := 3 * qI16 * qI16
	if lambdaI16 < 1 { lambdaI16 = 1 }
	lambdaMode := (1 * qI4 * qI4) >> 7
	if lambdaMode < 1 { lambdaMode = 1 }

	reconStride := mbW * 16
	recon := make([]uint8, reconStride*mbH*16)
	mbInfos := make([]mbInfo, mbW*mbH)
	tokenBW := newBoolEncoder()

	topNzY := make([]int, mbW*4+1)
	topNzU := make([]int, mbW*2+1)
	topNzV := make([]int, mbW*2+1)
	topNzDC := make([]int, mbW+1)
	topI4Modes := make([]int, mbW*4)

	i4Count := 0
	i16Count := 0

	for mbY := 0; mbY < mbH; mbY++ {
		leftNzY := [5]int{}
		leftNzU := [3]int{}
		leftNzV := [3]int{}
		leftI4Mode := [4]int{}

		for mbX := 0; mbX < mbW; mbX++ {
			mbIdx := mbY*mbW + mbX
			px := mbX * 16
			py := mbY * 16

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

			// Try i16
			bestI16Mode := I16_DC_PRED
			bestI16Score := int64(1<<62 - 1)
			var pred16Best [256]int16

			for mode := 0; mode < numI16Modes; mode++ {
				var pred16 [256]int16
				intra16Predict(mode, yuv, mbX, mbY, pred16[:])
				distortion := ssd16x16(src16[:], pred16[:])
				modeBits := i16ModeBitCost(mode)
				score := distortion + int64(lambdaI16)*modeBits
				if score < bestI16Score {
					bestI16Score = score
					bestI16Mode = mode
					copy(pred16Best[:], pred16[:])
				}
			}
			_ = pred16Best

			// Try i4
			var bestI4Score int64
			var i4AcLevels [16][16]int16
			var i4DcLevels [16]int16
			var localI4Modes [16]int
			var mbReconI4 [16 * 16]uint8

			{
				topBlkMode := make([]int, 4)
				for bx := 0; bx < 4; bx++ {
					topBlkMode[bx] = topI4Modes[mbX*4+bx]
				}
				leftBlkMode := [4]int{leftI4Mode[0], leftI4Mode[1], leftI4Mode[2], leftI4Mode[3]}

				var i4TotalScore int64
				var localI4AcLevels [16][16]int16
				var localI4DcLevels [16]int16

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
							}
						}

						localI4Modes[blkIdx] = bestBlkMode
						localI4AcLevels[blkIdx] = bestBlkAcLevels
						localI4DcLevels[blkIdx] = 0
						i4TotalScore += bestBlkScore

						for y := 0; y < 4; y++ {
							for x := 0; x < 4; x++ {
								mbReconI4[(by*4+y)*16+(bx*4+x)] = bestBlkRecon[y*4+x]
							}
						}

						topBlkMode[bx] = bestBlkMode
					}
					leftBlkMode[by] = localI4Modes[by*4+3]
				}

				bestI4Score = i4TotalScore
				i4AcLevels = localI4AcLevels
				i4DcLevels = localI4DcLevels
			}

			// Compute i16 post-quant distortion
			var mbI16AcLevels [16][16]int16
			var mbI16DcQuantLevels [16]int16
			var mbI16Pred [256]int16

			intra16PredictFromRecon(bestI16Mode, recon, reconStride, mbX, mbY, yuv.mbW, yuv.mbH, mbI16Pred[:])
			var yDcRaw16 [16]int16
			for by := 0; by < 4; by++ {
				for bx := 0; bx < 4; bx++ {
					n := by*4 + bx
					var src4, pred4 [16]int16
					for y := 0; y < 4; y++ {
						for x := 0; x < 4; x++ {
							src4[y*4+x] = src16[(by*4+y)*16+(bx*4+x)]
							pred4[y*4+x] = mbI16Pred[(by*4+y)*16+(bx*4+x)]
						}
					}
					var dctOut [16]int16
					fTransform(src4[:], pred4[:], dctOut[:])
					yDcRaw16[n] = dctOut[0]
					dctOut[0] = 0
					quantizeBlock(dctOut[:], mbI16AcLevels[n][:], &qm.y1, 1)
				}
			}
			var whtOut16 [16]int16
			fTransformWHT(yDcRaw16[:], whtOut16[:])
			quantizeBlockWHT(whtOut16[:], mbI16DcQuantLevels[:], &qm.y2)

			var whtRaster16 [16]int16
			for n := 0; n < 16; n++ {
				j := int(kZigzag[n])
				whtRaster16[j] = int16(int32(mbI16DcQuantLevels[n]) * int32(qm.y2.q[j]))
			}
			var dcBlockCoeffs16 [16]int16
			inverseWHT16(whtRaster16[:], dcBlockCoeffs16[:])

			var i16PostQuantDistortion int64
			for by := 0; by < 4; by++ {
				for bx := 0; bx < 4; bx++ {
					n := by*4 + bx
					var pred4 [16]int16
					for y := 0; y < 4; y++ {
						for x := 0; x < 4; x++ {
							pred4[y*4+x] = mbI16Pred[(by*4+y)*16+(bx*4+x)]
						}
					}
					var rasterCoeffs [16]int16
					dequantizeBlock(mbI16AcLevels[n][:], rasterCoeffs[:], &qm.y1, dcBlockCoeffs16[n])
					var recBlock [16]int16
					iTransform4x4(rasterCoeffs[:], pred4[:], recBlock[:])
					for y := 0; y < 4; y++ {
						for x := 0; x < 4; x++ {
							d := int64(src16[(by*4+y)*16+(bx*4+x)]) - int64(recBlock[y*4+x])
							i16PostQuantDistortion += d * d
						}
					}
				}
			}

			// RD decision: compare i4 vs i16 using same lambda scale
			i16Score := i16PostQuantDistortion + int64(lambdaI4)*i16ModeBitCost(bestI16Mode)
			i4Score := bestI4Score + int64(lambdaMode)*211 // add i4 header cost

			info := &mbInfos[mbIdx]
			if i4Score < i16Score {
				info.isI4 = true
				copy(info.i4Modes[:], localI4Modes[:])
				i4Count++
			} else {
				info.isI4 = false
				info.i16Mode = bestI16Mode
				i16Count++
			}
			info.uvMode = 0

			// Update recon buffer
			if info.isI4 {
				for y := 0; y < 16; y++ {
					for x := 0; x < 16; x++ {
						recon[(py+y)*reconStride+(px+x)] = mbReconI4[y*16+x]
					}
				}
			} else {
				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						var pred4 [16]int16
						for y := 0; y < 4; y++ {
							for x := 0; x < 4; x++ {
								pred4[y*4+x] = mbI16Pred[(by*4+y)*16+(bx*4+x)]
							}
						}
						var rasterCoeffs [16]int16
						dequantizeBlock(mbI16AcLevels[n][:], rasterCoeffs[:], &qm.y1, dcBlockCoeffs16[n])
						var recBlock [16]int16
						iTransform4x4(rasterCoeffs[:], pred4[:], recBlock[:])
						for y := 0; y < 4; y++ {
							for x := 0; x < 4; x++ {
								recon[(py+by*4+y)*reconStride+(px+bx*4+x)] = uint8(recBlock[y*4+x])
							}
						}
					}
				}
			}

			// Update i4 mode context
			if info.isI4 {
				for bx := 0; bx < 4; bx++ {
					topI4Modes[mbX*4+bx] = info.i4Modes[3*4+bx]
				}
				for by := 0; by < 4; by++ {
					leftI4Mode[by] = info.i4Modes[by*4+3]
				}
			} else {
				for bx := 0; bx < 4; bx++ {
					topI4Modes[mbX*4+bx] = info.i16Mode
				}
				for by := 0; by < 4; by++ {
					leftI4Mode[by] = info.i16Mode
				}
			}

			// Encode coefficients
			if info.isI4 {
				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						ctx := topNzY[mbX*4+bx] + leftNzY[by]
						last := findLast(i4AcLevels[n][:], 0)
						putCoeffs(tokenBW, ctx, i4AcLevels[n][:], 3, 0, last)
						nz := 0
						if last >= 0 { nz = 1 }
						topNzY[mbX*4+bx] = nz
						leftNzY[by] = nz
					}
				}
				topNzDC[mbX] = 0
				leftNzY[4] = 0
				_ = i4DcLevels
			} else {
				dcCtx := topNzDC[mbX] + leftNzY[4]
				lastDC := findLast(mbI16DcQuantLevels[:], 0)
				putCoeffs(tokenBW, dcCtx, mbI16DcQuantLevels[:], 1, 0, lastDC)
				dcNZ := 0
				if lastDC >= 0 { dcNZ = 1 }
				topNzDC[mbX] = dcNZ
				leftNzY[4] = dcNZ

				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						ctx := topNzY[mbX*4+bx] + leftNzY[by]
						last := findLast(mbI16AcLevels[n][:], 1)
						putCoeffs(tokenBW, ctx, mbI16AcLevels[n][:], 0, 1, last)
						nz := 0
						if last >= 1 { nz = 1 }
						topNzY[mbX*4+bx] = nz
						leftNzY[by] = nz
					}
				}
			}

			// UV encoding
			dcU := computeDCUV(yuv.u, yuv.uvStride, mbX, mbY, yuv.width, yuv.height)
			dcV := computeDCUV(yuv.v, yuv.uvStride, mbX, mbY, yuv.width, yuv.height)

			var uvLevels [8][16]int16
			for ch := 0; ch < 2; ch++ {
				plane := yuv.u
				if ch == 1 { plane = yuv.v }
				dcUV := dcU
				if ch == 1 { dcUV = dcV }
				for by := 0; by < 2; by++ {
					for bx := 0; bx < 2; bx++ {
						bn := ch*4 + by*2 + bx
						var src4 [16]int16
						var pred4 [16]int16
						extractBlock4x4UV(plane, yuv.uvStride, mbX*8+bx*4, mbY*8+by*4, yuv.width, yuv.height, src4[:])
						fillPred4x4(pred4[:], uint8(dcUV))
						var dctOut [16]int16
						fTransform(src4[:], pred4[:], dctOut[:])
						var quant [16]int16
						quantizeBlock(dctOut[:], quant[:], &qm.uv, 0)
						uvLevels[bn] = quant
					}
				}
			}

			for by := 0; by < 2; by++ {
				for bx := 0; bx < 2; bx++ {
					n := by*2 + bx
					ctx := topNzU[mbX*2+bx] + leftNzU[by]
					last := findLast(uvLevels[n][:], 0)
					putCoeffs(tokenBW, ctx, uvLevels[n][:], 2, 0, last)
					nz := 0
					if last >= 0 { nz = 1 }
					topNzU[mbX*2+bx] = nz
					leftNzU[by] = nz
				}
			}

			for by := 0; by < 2; by++ {
				for bx := 0; bx < 2; bx++ {
					n := by*2 + bx
					ctx := topNzV[mbX*2+bx] + leftNzV[by]
					last := findLast(uvLevels[4+n][:], 0)
					putCoeffs(tokenBW, ctx, uvLevels[4+n][:], 2, 0, last)
					nz := 0
					if last >= 0 { nz = 1 }
					topNzV[mbX*2+bx] = nz
					leftNzV[by] = nz
				}
			}
		}
	}

	tokenData := tokenBW.finish()
	part0BW := newBoolEncoder()
	encodePartition0Phase2(part0BW, mbW, mbH, baseQ, mbInfos)
	part0Data := part0BW.finish()

	_ = i4Count
	_ = i16Count

	frameHdr := buildVP8FrameHeader(w, h, len(part0Data))
	result := make([]byte, 0, len(frameHdr)+len(part0Data)+len(tokenData))
	result = append(result, frameHdr...)
	result = append(result, part0Data...)
	result = append(result, tokenData...)
	return result
}

func TestI4EnabledSmall(t *testing.T) {
	f, err := os.Open("test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	if err != nil { t.Fatalf("open: %v", err) }
	defer f.Close()
	src, _, err := image.Decode(f)
	if err != nil { t.Fatalf("decode jpeg: %v", err) }
	dst := image.NewNRGBA(image.Rect(0, 0, 48, 48))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Src, nil)

	yuv := rgbaToYUV420(dst)
	qm := buildQuantMatrices(86) // same as writer.go for quality=90
	baseQ := qualityToLevel(86)

	vp8Data := encodeFrameI4Enabled(yuv, qm, baseQ)
	var buf bytes.Buffer
	writeWebPHeader(&buf, vp8Data)

	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	p := psnrYLuma(dst, dec)
	t.Logf("48x48 i4-enabled: luma PSNR=%.2f dB, size=%d bytes", p, buf.Len())
	if p < 35.0 {
		t.Errorf("PSNR %.2f < 35dB — cascade bug still present", p)
	}
}

func TestI4Enabled300x300(t *testing.T) {
	f, err := os.Open("test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	if err != nil { t.Fatalf("open: %v", err) }
	defer f.Close()
	src, _, err := image.Decode(f)
	if err != nil { t.Fatalf("decode jpeg: %v", err) }
	dst := image.NewNRGBA(image.Rect(0, 0, 300, 300))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Src, nil)

	yuv := rgbaToYUV420(dst)
	qm := buildQuantMatrices(86)
	baseQ := qualityToLevel(86)

	start := time.Now()
	vp8Data := encodeFrameI4Enabled(yuv, qm, baseQ)
	elapsed := time.Since(start)

	var buf bytes.Buffer
	writeWebPHeader(&buf, vp8Data)

	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	sizeKB := float64(buf.Len()) / 1024
	p := psnrYLuma(dst, dec)
	t.Logf("300x300 i4-enabled: luma PSNR=%.2f dB, size=%.1f kb, time=%v", p, sizeKB, elapsed)

	os.WriteFile("test_data/CD15_lossy_i4_enabled_test.webp", buf.Bytes(), 0644)
}
