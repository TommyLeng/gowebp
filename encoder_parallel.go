// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import "sync"

// parallelThreshold is the minimum total MB count (mbW*mbH) above which
// encodeFrameParallel is used instead of the serial encodeFrame.
// Below ~360×360px the serial path is faster due to goroutine overhead.
const parallelThreshold = 500

// rowBottomNz stores the NZ context state that the last MB in a row leaves
// behind for the next row to consume. It is written by row ry after it
// finishes MB (mbX, ry) and read by row ry+1 before it starts MB (mbX, ry+1).
type rowBottomNz struct {
	// topNzY[bx] = NZ flag for luma 4x4 column bx (0..3) of this MB's bottom row
	topNzY [4]int
	// topNzU[bx], topNzV[bx] = NZ flag for chroma column bx (0..1)
	topNzU [2]int
	topNzV [2]int
	// topNzDC: NZ flag for i16 WHT-DC
	topNzDC int
	// topI4Modes[bx] = i4 mode for the bottom row of 4x4 blocks (by=3)
	topI4Modes [4]int
}

// encodeFrameParallel is a wave-front parallel implementation of encodeFrame.
// It processes each row as a separate goroutine, but each MB in row ry waits
// for the MB directly above it (row ry-1) to finish writing its recon buffer
// before proceeding. This ensures correct intra-prediction across row boundaries.
//
// The second pass (coefficient probability adaptation + entropy coding) remains
// sequential, as the output must be a single ordered bitstream.
func encodeFrameParallel(yuv *yuvImage, baseQ int) []byte {
	w := yuv.width
	h := yuv.height
	mbW := yuv.mbW / 16
	mbH := yuv.mbH / 16

	// --- SNS pre-analysis (serial, already independent) ---
	mbAlpha := make([]int, mbW*mbH)
	for mbY := 0; mbY < mbH; mbY++ {
		for mbX := 0; mbX < mbW; mbX++ {
			mbAlpha[mbY*mbW+mbX] = computeMBAlpha(yuv, mbX, mbY)
		}
	}

	seg0Quality, seg1Quality := computeSNSSegmentQualities(baseQ, mbW*mbH)
	seg0 := makeSegmentParams(seg0Quality)
	seg1 := makeSegmentParams(seg1Quality)
	segs := [2]segmentParams{seg0, seg1}

	alphaThreshold := computeAlphaThreshold(mbAlpha)
	mbSegment := make([]int, mbW*mbH)
	for i, a := range mbAlpha {
		if a <= alphaThreshold {
			mbSegment[i] = 0
		} else {
			mbSegment[i] = 1
		}
	}

	// --- Shared reconstruction buffers ---
	// These are written by row ry and read by row ry+1 (after channel sync).
	reconStride := mbW * 16
	recon := make([]uint8, reconStride*mbH*16)

	uvPlaneH := yuv.mbH / 2
	reconU := make([]uint8, yuv.uvStride*uvPlaneH)
	reconV := make([]uint8, yuv.uvStride*uvPlaneH)
	for i := range reconU {
		reconU[i] = 128
	}
	for i := range reconV {
		reconV[i] = 128
	}

	mbInfos := make([]mbInfo, mbW*mbH)
	mbCoeffs := make([]mbCoeffData, mbW*mbH)

	// --- Wave-front synchronisation ---
	// done[ry][mbX] is closed when MB (mbX, ry) has finished writing its recon buffer.
	// This allows row ry+1 to safely read the top-neighbor recon data.
	//
	// We allocate mbW+1 channels per row: done[ry][mbX] for mbX in [0, mbW].
	// The sentinel done[ry][mbW] is closed when the whole row is done (not needed
	// here but makes the index math simpler).
	done := make([][]chan struct{}, mbH)
	for y := range done {
		done[y] = make([]chan struct{}, mbW)
		for x := range done[y] {
			done[y][x] = make(chan struct{})
		}
	}

	// Row -1 sentinels: pre-closed so row 0 MBs can start immediately.
	topRowDone := make([]chan struct{}, mbW)
	for x := range topRowDone {
		topRowDone[x] = make(chan struct{})
		close(topRowDone[x])
	}

	// --- Per-column NZ context shared between rows ---
	// Row ry writes these after finishing MB (mbX, ry); row ry+1 reads them
	// before starting MB (mbX, ry+1). Safe because of the done-channel sync.
	//
	// Indexed as [mbX] for DC/i4modes, [mbX*4+bx] for Y, [mbX*2+bx] for UV.
	topNzYShared := make([]int, mbW*4)
	topNzUShared := make([]int, mbW*2)
	topNzVShared := make([]int, mbW*2)
	topNzDCShared := make([]int, mbW)
	topI4ModesShared := make([]int, mbW*4)

	var wg sync.WaitGroup

	for rowY := 0; rowY < mbH; rowY++ {
		wg.Add(1)

		// Capture loop variables.
		ry := rowY
		prevDone := topRowDone
		if ry > 0 {
			prevDone = done[ry-1]
		}

		go func(ry int, prevDone []chan struct{}) {
			defer wg.Done()

			// Per-row left-neighbor NZ state (reset at each row start).
			var leftNzY [5]int
			var leftNzU [3]int
			var leftNzV [3]int

			// Per-row left i4 mode context.
			var leftI4Mode [4]int

			for mbX := 0; mbX < mbW; mbX++ {
				// Wait for MB (mbX, ry-1) above to finish before we read
				// its recon buffer and NZ state.
				<-prevDone[mbX]

				// Also wait for MB (mbX+1, ry-1) if it exists.
				// buildPred4ContextWithMBRecon reads recon[topY*stride + x] for
				// x >= px+16 (the 4 right-of-top pixels used by diagonal i4 modes
				// in the rightmost 4x4 block column). Those pixels belong to MB
				// (mbX+1, ry-1), so we must wait for that MB to complete too.
				if mbX+1 < mbW {
					<-prevDone[mbX+1]
				}

				// Now safe to read top-neighbor recon and NZ state for column mbX.
				// Copy topNzY/U/V/DC/i4Modes for this column from the shared arrays.
				// (Written by row ry-1 after closing prevDone[mbX].)
				var colTopNzY [4]int
				var colTopNzU [2]int
				var colTopNzV [2]int
				colTopNzDC := topNzDCShared[mbX]
				var colTopI4Modes [4]int
				for bx := 0; bx < 4; bx++ {
					colTopNzY[bx] = topNzYShared[mbX*4+bx]
					colTopI4Modes[bx] = topI4ModesShared[mbX*4+bx]
				}
				for bx := 0; bx < 2; bx++ {
					colTopNzU[bx] = topNzUShared[mbX*2+bx]
					colTopNzV[bx] = topNzVShared[mbX*2+bx]
				}

				mbIdx := ry*mbW + mbX
				px := mbX * 16
				py := ry * 16

				seg := &segs[mbSegment[mbIdx]]
				qm := seg.qm
				mbLambdaI4 := seg.lambdaI4
				mbLambdaI16 := seg.lambdaI16
				mbLambdaMode := seg.lambdaMode
				mbLambdaTrellisI4 := seg.lambdaTrellisI4
				mbLambdaTrellisI16 := seg.lambdaTrellisI16
				mbLambdaTrellisUV := seg.lambdaTrellisUV
				trellisI4Costs := &seg.trellisI4Costs
				trellisI16Costs := &seg.trellisI16Costs
				trellisUVCosts := &seg.trellisUVCosts

				// Extract full 16x16 source block
				var src16 [256]int16
				for y := 0; y < 16; y++ {
					for x := 0; x < 16; x++ {
						sx := px + x
						sy := py + y
						if sx >= yuv.width {
							sx = yuv.width - 1
						}
						if sy >= yuv.height {
							sy = yuv.height - 1
						}
						src16[y*16+x] = int16(yuv.y[sy*yuv.yStride+sx])
					}
				}

				// -------------------------------------------------------
				// Try all 4 intra16 modes, pick best
				// -------------------------------------------------------
				bestI16Mode := I16_DC_PRED
				bestI16Score := int64(1<<62 - 1)
				var pred16Best [256]int16

				for mode := 0; mode < numI16Modes; mode++ {
					var pred16 [256]int16
					intra16Predict(mode, yuv, mbX, ry, pred16[:])
					distortion := ssd16x16(src16[:], pred16[:])
					modeBits := i16ModeBitCost(mode)
					score := distortion + int64(mbLambdaI16)*modeBits
					if score < bestI16Score {
						bestI16Score = score
						bestI16Mode = mode
						copy(pred16Best[:], pred16[:])
					}
				}
				_ = pred16Best

				// -------------------------------------------------------
				// Try intra4
				// -------------------------------------------------------
				var bestI4Score int64
				var i4AcLevels [16][16]int16
				var i4DcLevels [16]int16
				var localI4Modes [16]int
				var mbReconI4 [16 * 16]uint8

				{
					topBlkMode := make([]int, 4)
					for bx := 0; bx < 4; bx++ {
						topBlkMode[bx] = colTopI4Modes[bx]
					}
					leftBlkMode := [4]int{leftI4Mode[0], leftI4Mode[1], leftI4Mode[2], leftI4Mode[3]}

					var topNzI4 [4]int
					var leftNzI4 [4]int
					for bx := 0; bx < 4; bx++ {
						topNzI4[bx] = colTopNzY[bx]
					}
					for by := 0; by < 4; by++ {
						leftNzI4[by] = leftNzY[by]
					}

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
									if sx >= yuv.width {
										sx = yuv.width - 1
									}
									if sy >= yuv.height {
										sy = yuv.height - 1
									}
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

							const sadTopN = 4
							var sadScores [numI4Modes]int64
							for i := 0; i < numI4Modes; i++ {
								var p [16]int16
								intra4Predict(i, ctx, p[:])
								sadScores[i] = sad4x4(src4[:], p[:])
							}
							var sadTmp [numI4Modes]int64
							copy(sadTmp[:], sadScores[:])
							for k := 0; k < sadTopN; k++ {
								minIdx := k
								for j := k + 1; j < numI4Modes; j++ {
									if sadTmp[j] < sadTmp[minIdx] {
										minIdx = j
									}
								}
								sadTmp[k], sadTmp[minIdx] = sadTmp[minIdx], sadTmp[k]
							}
							sadCutoff := sadTmp[sadTopN-1]

							bestBlkMode := B_DC_PRED
							bestBlkScore := int64(1<<62 - 1)
							var bestBlkRecon [16]uint8
							var bestBlkAcLevels [16]int16

							for mode := 0; mode < numI4Modes; mode++ {
								if sadScores[mode] > sadCutoff {
									continue
								}
								var pred4 [16]int16
								intra4Predict(mode, ctx, pred4[:])

								var dctOut [16]int16
								fTransform(src4[:], pred4[:], dctOut[:])

								trellisCtx0 := topNzI4[bx] + leftNzI4[by]
								if trellisCtx0 > 2 {
									trellisCtx0 = 2
								}
								var acQ [16]int16
								trellisQuantize(dctOut[:], acQ[:], &qm.y1, 0, mbLambdaTrellisI4, trellisI4Costs,
									(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[3]), trellisCtx0)

								var recBlock [16]int16
								iTransform4x4(dctOut[:], pred4[:], recBlock[:])

								var distortion int64
								for i := 0; i < 16; i++ {
									d := int64(src4[i]) - int64(recBlock[i])
									distortion += d * d
								}

								modeBits := i4ModeBitCost(mode, topPred, leftPred)
								score := distortion + int64(mbLambdaI4)*modeBits
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

							bestNZ := 0
							if findLast(bestBlkAcLevels[:], 0) >= 0 {
								bestNZ = 1
							}
							topNzI4[bx] = bestNZ
							leftNzI4[by] = bestNZ

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

				// -------------------------------------------------------
				// i16 post-quantization RD
				// -------------------------------------------------------
				var mbI16AcLevels [16][16]int16
				var mbI16DcQuantLevels [16]int16
				var mbI16Pred [256]int16

				intra16PredictFromRecon(bestI16Mode, recon, reconStride, mbX, ry, yuv.mbW, yuv.mbH, mbI16Pred[:])
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
						trellisQuantize(dctOut[:], mbI16AcLevels[n][:], &qm.y1, 1, mbLambdaTrellisI16, trellisI16Costs,
							(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[0]), 0)
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

				i16Score := i16PostQuantDistortion + int64(mbLambdaI16)*i16ModeBitCost(bestI16Mode)
				i4HeaderCost := int64(mbLambdaMode) * 211
				i4Score := bestI4Score + i4HeaderCost

				info := &mbInfos[mbIdx]
				if i4Score < i16Score {
					info.isI4 = true
					copy(info.i4Modes[:], localI4Modes[:])
				} else {
					info.isI4 = false
					info.i16Mode = bestI16Mode
				}

				// UV: RD-optimal prediction mode selection.
				bestUVMode := 0
				bestUVScore := int64(1<<62 - 1)
				for uvMode := 0; uvMode < 4; uvMode++ {
					if uvMode == 1 && ry == 0 {
						continue // VE needs top row
					}
					if uvMode == 2 && mbX == 0 {
						continue // HE needs left column
					}
					if uvMode == 3 && (mbX == 0 || ry == 0) {
						continue // TM needs both top and left
					}
					var scoreU, scoreV int64
					for ch := 0; ch < 2; ch++ {
						rPlane := reconU
						sPlane := yuv.u
						if ch == 1 {
							rPlane = reconV
							sPlane = yuv.v
						}
						var pred8 [64]int16
						predictUV(uvMode, rPlane, yuv.uvStride, mbX, ry, yuv.width, yuv.height, pred8[:])
						uvW := (yuv.width + 1) / 2
						uvH := (yuv.height + 1) / 2
						for j := 0; j < 8; j++ {
							for i := 0; i < 8; i++ {
								bpx := mbX*8 + i
								bpy := ry*8 + j
								if bpx >= uvW {
									bpx = uvW - 1
								}
								if bpy >= uvH {
									bpy = uvH - 1
								}
								src := int64(sPlane[bpy*yuv.uvStride+bpx])
								d := src - int64(pred8[j*8+i])
								if ch == 0 {
									scoreU += d * d
								} else {
									scoreV += d * d
								}
							}
						}
					}
					score := scoreU + scoreV
					if score < bestUVScore {
						bestUVScore = score
						bestUVMode = uvMode
					}
				}
				info.uvMode = bestUVMode
				info.segment = mbSegment[mbIdx]

				// -------------------------------------------------------
				// Update global recon buffer
				// -------------------------------------------------------
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

				// -------------------------------------------------------
				// Update top/left i4 mode contexts
				// -------------------------------------------------------
				if info.isI4 {
					for bx := 0; bx < 4; bx++ {
						colTopI4Modes[bx] = info.i4Modes[3*4+bx]
					}
					for by := 0; by < 4; by++ {
						leftI4Mode[by] = info.i4Modes[by*4+3]
					}
				} else {
					for bx := 0; bx < 4; bx++ {
						colTopI4Modes[bx] = info.i16Mode
					}
					for by := 0; by < 4; by++ {
						leftI4Mode[by] = info.i16Mode
					}
				}

				// -------------------------------------------------------
				// UV quantization using RD-selected prediction mode.
				// -------------------------------------------------------
				var predU8 [64]int16
				var predV8 [64]int16
				predictUV(info.uvMode, reconU, yuv.uvStride, mbX, ry, yuv.width, yuv.height, predU8[:])
				predictUV(info.uvMode, reconV, yuv.uvStride, mbX, ry, yuv.width, yuv.height, predV8[:])

				var uvLevels [8][16]int16
				for ch := 0; ch < 2; ch++ {
					plane := yuv.u
					pred8 := predU8
					if ch == 1 {
						plane = yuv.v
						pred8 = predV8
					}
					for by := 0; by < 2; by++ {
						for bx := 0; bx < 2; bx++ {
							bn := ch*4 + by*2 + bx
							var src4 [16]int16
							var pred4 [16]int16
							extractBlock4x4UV(plane, yuv.uvStride, mbX*8+bx*4, ry*8+by*4, yuv.width, yuv.height, src4[:])
							// Extract 4×4 sub-block from the 8×8 prediction.
							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									pred4[y*4+x] = pred8[(by*4+y)*8+(bx*4+x)]
								}
							}
							var dctOut [16]int16
							fTransform(src4[:], pred4[:], dctOut[:])
							var quant [16]int16
							trellisQuantize(dctOut[:], quant[:], &qm.uv, 0, mbLambdaTrellisUV, trellisUVCosts,
								(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[2]), 0)
							uvLevels[bn] = quant
						}
					}
				}

				// Reconstruct chroma and update reconU/reconV.
				for ch := 0; ch < 2; ch++ {
					reconPlane := reconU
					pred8 := predU8
					if ch == 1 {
						reconPlane = reconV
						pred8 = predV8
					}
					for by := 0; by < 2; by++ {
						for bx := 0; bx < 2; bx++ {
							bn := ch*4 + by*2 + bx
							var pred4 [16]int16
							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									pred4[y*4+x] = pred8[(by*4+y)*8+(bx*4+x)]
								}
							}
							var raster [16]int16
							for n := 0; n < 16; n++ {
								j := int(kZigzag[n])
								raster[j] = int16(int32(uvLevels[bn][n]) * int32(qm.uv.q[j]))
							}
							var recBlock [16]int16
							iTransform4x4(raster[:], pred4[:], recBlock[:])
							bPX := mbX*8 + bx*4
							bPY := ry*8 + by*4
							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									reconPlane[(bPY+y)*yuv.uvStride+(bPX+x)] = uint8(recBlock[y*4+x])
								}
							}
						}
					}
				}

				// -------------------------------------------------------
				// Store coefficient levels and update NZ context
				// -------------------------------------------------------
				cd := &mbCoeffs[mbIdx]
				cd.isI4 = info.isI4

				if info.isI4 {
					for by := 0; by < 4; by++ {
						for bx := 0; bx < 4; bx++ {
							n := by*4 + bx
							cd.i4AC[n] = i4AcLevels[n]
							last := findLast(i4AcLevels[n][:], 0)
							nz := 0
							if last >= 0 {
								nz = 1
							}
							colTopNzY[bx] = nz
							leftNzY[by] = nz
						}
					}
					colTopNzDC = 0
					leftNzY[4] = 0
					_ = i4DcLevels
				} else {
					cd.i16DC = mbI16DcQuantLevels
					lastDC := findLast(mbI16DcQuantLevels[:], 0)
					dcNZ := 0
					if lastDC >= 0 {
						dcNZ = 1
					}
					colTopNzDC = dcNZ
					leftNzY[4] = dcNZ

					for by := 0; by < 4; by++ {
						for bx := 0; bx < 4; bx++ {
							n := by*4 + bx
							cd.i16AC[n] = mbI16AcLevels[n]
							last := findLast(mbI16AcLevels[n][:], 1)
							nz := 0
							if last >= 1 {
								nz = 1
							}
							colTopNzY[bx] = nz
							leftNzY[by] = nz
						}
					}
				}

				// UV NZ context
				for by := 0; by < 2; by++ {
					for bx := 0; bx < 2; bx++ {
						n := by*2 + bx
						cd.uv[n] = uvLevels[n]
						last := findLast(uvLevels[n][:], 0)
						nz := 0
						if last >= 0 {
							nz = 1
						}
						colTopNzU[bx] = nz
						leftNzU[by] = nz
					}
				}
				for by := 0; by < 2; by++ {
					for bx := 0; bx < 2; bx++ {
						n := by*2 + bx
						cd.uv[4+n] = uvLevels[4+n]
						last := findLast(uvLevels[4+n][:], 0)
						nz := 0
						if last >= 0 {
							nz = 1
						}
						colTopNzV[bx] = nz
						leftNzV[by] = nz
					}
				}

				// -------------------------------------------------------
				// Write back column NZ/i4-mode state for the next row.
				// This MUST happen before close(done[ry][mbX]) so that the
				// next row reads the updated values, not stale ones.
				// -------------------------------------------------------
				for bx := 0; bx < 4; bx++ {
					topNzYShared[mbX*4+bx] = colTopNzY[bx]
					topI4ModesShared[mbX*4+bx] = colTopI4Modes[bx]
				}
				for bx := 0; bx < 2; bx++ {
					topNzUShared[mbX*2+bx] = colTopNzU[bx]
					topNzVShared[mbX*2+bx] = colTopNzV[bx]
				}
				topNzDCShared[mbX] = colTopNzDC

				// Signal: MB (mbX, ry) is complete. Row ry+1 may now read
				// this column's recon data and NZ state.
				close(done[ry][mbX])
			}
		}(ry, prevDone)
	}

	wg.Wait()

	// --- Two-pass coefficient probability adaptation (sequential) ---
	var stats coeffStats
	collectCoeffStats(mbCoeffs, mbW, mbH, &stats)

	adaptedProbs, updatedFlags := finalizeTokenProbas(&stats)

	tokenBW := newBoolEncoder()
	encodeTokenPartition(tokenBW, mbCoeffs, mbW, mbH, &adaptedProbs)
	tokenData := tokenBW.finish()

	part0BW := newBoolEncoder()
	encodePartition0WithProbs(part0BW, mbW, mbH, seg0.baseQ, seg1.baseQ, mbInfos, &adaptedProbs, &updatedFlags)
	part0Data := part0BW.finish()

	frameHdr := buildVP8FrameHeader(w, h, len(part0Data))
	result := make([]byte, 0, len(frameHdr)+len(part0Data)+len(tokenData))
	result = append(result, frameHdr...)
	result = append(result, part0Data...)
	result = append(result, tokenData...)
	return result
}
