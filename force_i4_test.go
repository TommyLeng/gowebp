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

// encodeFrameForceI4 forces i4 for all MBs to test reconstruction quality.
// This isolates the reconstruction chain from the RD decision bug.
func encodeFrameForceI4(yuv *yuvImage, qm quantMatrices, baseQ int) []byte {
	w := yuv.width
	h := yuv.height
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
	mbInfos := make([]mbInfo, mbW*mbH)
	tokenBW := newBoolEncoder()

	topNzY := make([]int, mbW*4+1)
	topNzU := make([]int, mbW*2+1)
	topNzV := make([]int, mbW*2+1)
	topNzDC := make([]int, mbW+1)
	topI4Modes := make([]int, mbW*4)

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

			var i4AcLevels [16][16]int16
			var localI4Modes [16]int
			var mbReconI4 [16 * 16]uint8

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
					i4AcLevels[blkIdx] = bestBlkAcLevels

					for y := 0; y < 4; y++ {
						for x := 0; x < 4; x++ {
							mbReconI4[(by*4+y)*16+(bx*4+x)] = bestBlkRecon[y*4+x]
						}
					}

					topBlkMode[bx] = bestBlkMode
				}
				leftBlkMode[by] = localI4Modes[by*4+3]
			}

			info := &mbInfos[mbIdx]
			info.isI4 = true
			copy(info.i4Modes[:], localI4Modes[:])
			info.uvMode = 0

			// Update recon buffer
			for y := 0; y < 16; y++ {
				for x := 0; x < 16; x++ {
					recon[(py+y)*reconStride+(px+x)] = mbReconI4[y*16+x]
				}
			}

			// Update i4 mode context
			for bx := 0; bx < 4; bx++ {
				topI4Modes[mbX*4+bx] = info.i4Modes[3*4+bx]
			}
			for by := 0; by < 4; by++ {
				leftI4Mode[by] = info.i4Modes[by*4+3]
			}

			// Encode coefficients
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

	frameHdr := buildVP8FrameHeader(w, h, len(part0Data))
	result := make([]byte, 0, len(frameHdr)+len(part0Data)+len(tokenData))
	result = append(result, frameHdr...)
	result = append(result, part0Data...)
	result = append(result, tokenData...)
	return result
}

func TestForceI4Small(t *testing.T) {
	f, err := os.Open("test_data/original/hidden/CD15 - Gallarde,Nica_fix.jpg")
	if err != nil { t.Fatalf("open: %v", err) }
	defer f.Close()
	src, _, err := image.Decode(f)
	if err != nil { t.Fatalf("decode jpeg: %v", err) }
	dst := image.NewNRGBA(image.Rect(0, 0, 48, 48))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Src, nil)

	yuv := rgbaToYUV420(dst)
	qm := buildQuantMatrices(86)
	baseQ := qualityToLevel(86)

	vp8Data := encodeFrameForceI4(yuv, qm, baseQ)
	var buf bytes.Buffer
	writeWebPHeader(&buf, vp8Data)

	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	p := psnrYLuma(dst, dec)
	t.Logf("48x48 force-i4: luma PSNR=%.2f dB, size=%d bytes", p, buf.Len())
	if p < 35.0 {
		t.Errorf("PSNR %.2f < 35dB — cascade bug present (expected ≥35dB for correct reconstruction)", p)
	}
}

func TestForceI4_300x300(t *testing.T) {
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
	vp8Data := encodeFrameForceI4(yuv, qm, baseQ)
	elapsed := time.Since(start)

	var buf bytes.Buffer
	writeWebPHeader(&buf, vp8Data)

	dec, err := webp.Decode(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	sizeKB := float64(buf.Len()) / 1024
	p := psnrYLuma(dst, dec)
	t.Logf("300x300 force-i4: luma PSNR=%.2f dB, size=%.1f kb, time=%v", p, sizeKB, elapsed)
	os.WriteFile("test_data/CD15_lossy_force_i4.webp", buf.Bytes(), 0644)
}
