// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// mbWorkspace holds all temporary buffers for one MB's encoding work.
// Pre-allocated once per goroutine (parallel path) or once per frame (serial path)
// and reused across all MBs, avoiding repeated stack frame setup and improving
// CPU cache locality.
type mbWorkspace struct {
	// 16×16 luma source
	src16 [256]int16
	// i16 prediction
	pred16Best [256]int16
	pred16     [256]int16 // scratch for each i16 mode trial
	// i16 coefficients and reconstruction
	mbI16Pred          [256]int16
	mbI16AcLevels      [16][16]int16
	mbI16DcQuantLevels [16]int16
	yDcRaw16           [16]int16
	whtOut16           [16]int16
	whtRaster16        [16]int16
	dcBlockCoeffs16    [16]int16
	// i4 per-MB accumulators
	i4AcLevels  [16][16]int16
	i4DcLevels  [16]int16
	localI4Modes [16]int
	mbReconI4   [256]uint8
	// i4 per-block temporaries
	src4           [16]int16
	pred4          [16]int16
	dctOut         [16]int16
	acQ            [16]int16
	recBlock       [16]int16
	bestBlkAcLevels [16]int16
	bestBlkRecon   [16]uint8
	// SAD pre-screening temporaries
	sadPred   [16]int16 // scratch for one SAD prediction
	sadScores [numI4Modes]int64
	sadTmp    [numI4Modes]int64
	// UV
	predU8      [64]int16
	predV8      [64]int16
	uvLevels    [8][16]int16
	uvSrc4      [16]int16
	uvPred4     [16]int16
	uvDctOut    [16]int16
	uvQuant     [16]int16
	uvRaster    [16]int16
	uvRecBlock  [16]int16
	// i16 inner loop
	i16Src4     [16]int16
	i16Pred4    [16]int16
	i16DctOut   [16]int16
	i16RasterCoeffs [16]int16
	i16RecBlock [16]int16
	i16Pred4b   [16]int16
	// local copies of i4 accumulators (used within the i4 block, then assigned to ws fields)
	localI4AcLevels [16][16]int16
	localI4DcLevels [16]int16
}

// parallelThreshold is the minimum total MB count (mbW*mbH) above which
// encodeFrameParallel is used instead of the serial encodeFrame.
// Set to 0 so ALL images use the wave-front parallel encoder regardless of size.
const parallelThreshold = 0

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
//
// arena supplies reusable backing slices to avoid per-call allocation churn.
func encodeFrameParallel(yuv *yuvImage, baseQ int, arena *frameArena) []byte {
	w := yuv.width
	h := yuv.height
	mbW := yuv.mbW / 16
	mbH := yuv.mbH / 16

	// --- SNS pre-analysis (serial, already independent) ---
	sns := computeSNS(yuv, mbW, mbH, baseQ)
	mbSegment := sns.mbSegment
	numSegs := sns.numSegs
	segs := make([]segmentParams, numSegs)
	for i := 0; i < numSegs; i++ {
		segs[i] = makeSegmentParamsFromQ(sns.segQs[i])
	}

	// --- Shared reconstruction buffers ---
	// These are written by row ry and read by row ry+1 (after channel sync).
	reconStride := mbW * 16
	arena.recon = growSliceU8(arena.recon, reconStride*mbH*16)
	recon := arena.recon
	clear(recon)

	uvPlaneH := yuv.mbH / 2
	arena.reconU = growSliceU8(arena.reconU, yuv.uvStride*uvPlaneH)
	arena.reconV = growSliceU8(arena.reconV, yuv.uvStride*uvPlaneH)
	reconU := arena.reconU
	reconV := arena.reconV
	for i := range reconU {
		reconU[i] = 128
	}
	for i := range reconV {
		reconV[i] = 128
	}

	arena.mbInfos = growSliceMBInfo(arena.mbInfos, mbW*mbH)
	mbInfos := arena.mbInfos
	clear(mbInfos)
	arena.mbCoeffs = growSliceMBCoeff(arena.mbCoeffs, mbW*mbH)
	mbCoeffs := arena.mbCoeffs
	clear(mbCoeffs)

	// --- Wave-front synchronisation ---
	// rowProgress[ry] stores the mbX index of the last completed MB in row ry.
	// -1 means the row has not started. The spin-wait loop below reads this
	// atomically so no goroutine scheduler involvement is needed for typical
	// cases, eliminating the pthread_cond overhead seen in the channel version.
	rowProgress := make([]atomic.Int32, mbH)
	for i := range rowProgress {
		rowProgress[i].Store(-1)
	}

	// sentinelRow represents row -1: all MBs are already "done" so row 0
	// can start immediately without waiting.
	var sentinelRow atomic.Int32
	sentinelRow.Store(int32(mbW - 1))

	// --- Per-column NZ context shared between rows ---
	// Row ry writes these after finishing MB (mbX, ry); row ry+1 reads them
	// before starting MB (mbX, ry+1). Safe because of the done-channel sync.
	//
	// Indexed as [mbX] for DC/i4modes, [mbX*4+bx] for Y, [mbX*2+bx] for UV.
	arena.topNzYShared = growSliceInt(arena.topNzYShared, mbW*4)
	topNzYShared := arena.topNzYShared
	clear(topNzYShared)
	arena.topNzUShared = growSliceInt(arena.topNzUShared, mbW*2)
	topNzUShared := arena.topNzUShared
	clear(topNzUShared)
	arena.topNzVShared = growSliceInt(arena.topNzVShared, mbW*2)
	topNzVShared := arena.topNzVShared
	clear(topNzVShared)
	arena.topNzDCShared = growSliceInt(arena.topNzDCShared, mbW)
	topNzDCShared := arena.topNzDCShared
	clear(topNzDCShared)
	arena.topI4ModesShared = growSliceInt(arena.topI4ModesShared, mbW*4)
	topI4ModesShared := arena.topI4ModesShared
	clear(topI4ModesShared)

	var wg sync.WaitGroup

	for rowY := 0; rowY < mbH; rowY++ {
		wg.Add(1)

		// Capture loop variables.
		ry := rowY
		var prev *atomic.Int32
		if ry == 0 {
			prev = &sentinelRow
		} else {
			prev = &rowProgress[ry-1]
		}

		go func(ry int, prev *atomic.Int32) {
			defer wg.Done()

			// Allocate workspace once per goroutine; reused for every MB in this row.
			ws := new(mbWorkspace)

			// Per-row left-neighbor NZ state (reset at each row start).
			var leftNzY [5]int
			var leftNzU [3]int
			var leftNzV [3]int

			// Per-row left i4 mode context.
			var leftI4Mode [4]int

			for mbX := 0; mbX < mbW; mbX++ {
				// Wait for the previous row to complete MB (mbX+1, ry-1) if it
				// exists, otherwise just MB (mbX, ry-1).
				// buildPred4ContextWithMBRecon reads recon pixels that belong to
				// MB (mbX+1, ry-1) for diagonal i4 modes, so we need that too.
				needed := int32(mbX + 1)
				if mbX == mbW-1 {
					needed = int32(mbX)
				}
				for i := 0; prev.Load() < needed; i++ {
					if i < 16 {
						runtime.Gosched()
					} else {
						time.Sleep(1 * time.Microsecond)
					}
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

				// Extract full 16x16 source block into workspace.
				src16 := &ws.src16
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

				for mode := 0; mode < numI16Modes; mode++ {
					intra16Predict(mode, yuv, mbX, ry, ws.pred16[:])
					distortion := ssd16x16(src16[:], ws.pred16[:])
					modeBits := i16ModeBitCost(mode)
					// Match libwebp's SetRDScore: score = lambda*(R+H) + 256*(D+SD).
					// See encoder.go for full rationale.
					score := int64(rdDistoMult)*distortion + int64(mbLambdaI16)*modeBits
					if score < bestI16Score {
						bestI16Score = score
						bestI16Mode = mode
						copy(ws.pred16Best[:], ws.pred16[:])
					}
				}
				_ = ws.pred16Best

				// -------------------------------------------------------
				// Try intra4
				// -------------------------------------------------------
				var bestI4Score int64

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

					for by := 0; by < 4; by++ {
						for bx := 0; bx < 4; bx++ {
							blkIdx := by*4 + bx
							bpx := px + bx*4
							bpy := py + by*4

							ctx := buildPred4ContextWithMBRecon(yuv, recon, reconStride, ws.mbReconI4[:], px, py, bpx, bpy)

							src4 := &ws.src4
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
								leftPred = ws.localI4Modes[blkIdx-1]
							} else {
								leftPred = leftBlkMode[by]
							}

							// Flat-block early exit: if block variance is very low, only try
							// B_DC_PRED (mode 0). DC is provably optimal for constant blocks
							// and near-optimal for very flat ones, so the other 9 modes can
							// be skipped safely with minimal quality impact.
							const flatThreshold16 = 16 * 16 * 16 // variance per pixel < 16²
							var varSum, varSumSq int
							for _, v := range src4 {
								iv := int(v)
								varSum += iv
								varSumSq += iv * iv
							}
							variance16 := varSumSq*16 - varSum*varSum

							bestBlkMode := B_DC_PRED
							bestBlkScore := int64(1<<62 - 1)
							// Old-scale score for the MB-level i4-vs-i16 comparison.
							// See encoder.go for rationale.
							bestBlkOldScore := int64(1<<62 - 1)

							// runRD evaluates mode mode's full RD cost using the prediction
							// already in ws.pred4. It updates best{Blk*} if this mode wins.
							runRD := func(mode int) {
								fTransform(src4[:], ws.pred4[:], ws.dctOut[:])

								trellisCtx0 := topNzI4[bx] + leftNzI4[by]
								if trellisCtx0 > 2 {
									trellisCtx0 = 2
								}
								trellisQuantize(ws.dctOut[:], ws.acQ[:], &qm.y1, 0, mbLambdaTrellisI4, trellisI4Costs,
									(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[3]), trellisCtx0)

								iTransform4x4(ws.dctOut[:], ws.pred4[:], ws.recBlock[:])

								distortion := ssd4x4(src4[:], ws.recBlock[:])

								modeBits := i4ModeBitCost(mode, topPred, leftPred)

								// Coefficient bit cost R: see encoder.go for rationale.
								rCost := coeffBitCost(trellisCtx0, ws.acQ[:], 0, trellisI4Costs,
									(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[3]))

								// Flatness penalty added to R (libwebp PickBestIntra4
								// quant_enc.c:1097-1103). See encoder.go for rationale.
								flatBitsR := int64(0)
								if mode > 0 && isFlatI4Levels(ws.acQ[:]) {
									flatBitsR = flatnessPenalty
								}

								// Match libwebp's SetRDScore: score = lambda*(R+H) + 256*(D+SD).
								// See encoder.go for full rationale.
								score := int64(rdDistoMult)*distortion + int64(mbLambdaI4)*(modeBits+int64(rCost)+flatBitsR)
								if score < bestBlkScore {
									bestBlkScore = score
									// Keep distortion at 1× for the MB-level i4-vs-i16
									// comparison so it still matches the unscaled
									// i16PostQuantDistortion term in i16Score.
									bestBlkOldScore = distortion + int64(mbLambdaI4)*modeBits
									bestBlkMode = mode
									copy(ws.bestBlkAcLevels[:], ws.acQ[:])
									for i := 0; i < 16; i++ {
										ws.bestBlkRecon[i] = uint8(ws.recBlock[i])
									}
								}
							}

							if variance16 < flatThreshold16 {
								// Very flat block: only DC mode is worth trying.
								intra4Predict(B_DC_PRED, ctx, ws.pred4[:])
								runRD(B_DC_PRED)
							} else {
								const sadTopN = 4
								// Cache all 10 SAD-phase predictions to avoid repeating them in the RD phase.
								var sadPreds [numI4Modes][16]int16
								for i := 0; i < numI4Modes; i++ {
									intra4Predict(i, ctx, sadPreds[i][:])
									ws.sadScores[i] = sad4x4(src4[:], sadPreds[i][:])
								}
								copy(ws.sadTmp[:], ws.sadScores[:])
								for k := 0; k < sadTopN; k++ {
									minIdx := k
									for j := k + 1; j < numI4Modes; j++ {
										if ws.sadTmp[j] < ws.sadTmp[minIdx] {
											minIdx = j
										}
									}
									ws.sadTmp[k], ws.sadTmp[minIdx] = ws.sadTmp[minIdx], ws.sadTmp[k]
								}
								sadCutoff := ws.sadTmp[sadTopN-1]

								for mode := 0; mode < numI4Modes; mode++ {
									if ws.sadScores[mode] > sadCutoff {
										continue
									}
									copy(ws.pred4[:], sadPreds[mode][:])
									runRD(mode)
								}
							}

							ws.localI4Modes[blkIdx] = bestBlkMode
							ws.localI4AcLevels[blkIdx] = ws.bestBlkAcLevels
							ws.localI4DcLevels[blkIdx] = 0
							i4TotalScore += bestBlkOldScore

							bestNZ := 0
							if findLast(ws.bestBlkAcLevels[:], 0) >= 0 {
								bestNZ = 1
							}
							topNzI4[bx] = bestNZ
							leftNzI4[by] = bestNZ

							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									ws.mbReconI4[(by*4+y)*16+(bx*4+x)] = ws.bestBlkRecon[y*4+x]
								}
							}

							topBlkMode[bx] = bestBlkMode
						}
						leftBlkMode[by] = ws.localI4Modes[by*4+3]
					}

					bestI4Score = i4TotalScore
				}

				// -------------------------------------------------------
				// i16 post-quantization RD
				// -------------------------------------------------------
				intra16PredictFromRecon(bestI16Mode, recon, reconStride, mbX, ry, yuv.mbW, yuv.mbH, ws.mbI16Pred[:])
				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						for y := 0; y < 4; y++ {
							for x := 0; x < 4; x++ {
								ws.i16Src4[y*4+x] = src16[(by*4+y)*16+(bx*4+x)]
								ws.i16Pred4[y*4+x] = ws.mbI16Pred[(by*4+y)*16+(bx*4+x)]
							}
						}
						fTransform(ws.i16Src4[:], ws.i16Pred4[:], ws.i16DctOut[:])
						ws.yDcRaw16[n] = ws.i16DctOut[0]
						ws.i16DctOut[0] = 0
						trellisQuantize(ws.i16DctOut[:], ws.mbI16AcLevels[n][:], &qm.y1, 1, mbLambdaTrellisI16, trellisI16Costs,
							(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[0]), 0)
					}
				}
				fTransformWHT(ws.yDcRaw16[:], ws.whtOut16[:])
				quantizeBlockWHT(ws.whtOut16[:], ws.mbI16DcQuantLevels[:], &qm.y2)

				for n := 0; n < 16; n++ {
					j := int(kZigzag[n])
					ws.whtRaster16[j] = int16(int32(ws.mbI16DcQuantLevels[n]) * int32(qm.y2.q[j]))
				}
				inverseWHT16(ws.whtRaster16[:], ws.dcBlockCoeffs16[:])

				var i16PostQuantDistortion int64
				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						for y := 0; y < 4; y++ {
							for x := 0; x < 4; x++ {
								ws.i16Pred4b[y*4+x] = ws.mbI16Pred[(by*4+y)*16+(bx*4+x)]
							}
						}
						dequantizeBlock(ws.mbI16AcLevels[n][:], ws.i16RasterCoeffs[:], &qm.y1, ws.dcBlockCoeffs16[n])
						iTransform4x4(ws.i16RasterCoeffs[:], ws.i16Pred4b[:], ws.i16RecBlock[:])
						for y := 0; y < 4; y++ {
							for x := 0; x < 4; x++ {
								d := int64(src16[(by*4+y)*16+(bx*4+x)]) - int64(ws.i16RecBlock[y*4+x])
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
					copy(info.i4Modes[:], ws.localI4Modes[:])
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
						predictUV(uvMode, rPlane, yuv.uvStride, mbX, ry, yuv.width, yuv.height, ws.predU8[:])
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
								d := src - int64(ws.predU8[j*8+i])
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
							recon[(py+y)*reconStride+(px+x)] = ws.mbReconI4[y*16+x]
						}
					}
				} else {
					for by := 0; by < 4; by++ {
						for bx := 0; bx < 4; bx++ {
							n := by*4 + bx
							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									ws.i16Pred4b[y*4+x] = ws.mbI16Pred[(by*4+y)*16+(bx*4+x)]
								}
							}
							dequantizeBlock(ws.mbI16AcLevels[n][:], ws.i16RasterCoeffs[:], &qm.y1, ws.dcBlockCoeffs16[n])
							iTransform4x4(ws.i16RasterCoeffs[:], ws.i16Pred4b[:], ws.i16RecBlock[:])
							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									recon[(py+by*4+y)*reconStride+(px+bx*4+x)] = uint8(ws.i16RecBlock[y*4+x])
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
				predictUV(info.uvMode, reconU, yuv.uvStride, mbX, ry, yuv.width, yuv.height, ws.predU8[:])
				predictUV(info.uvMode, reconV, yuv.uvStride, mbX, ry, yuv.width, yuv.height, ws.predV8[:])

				for ch := 0; ch < 2; ch++ {
					plane := yuv.u
					predSlice := ws.predU8
					if ch == 1 {
						plane = yuv.v
						predSlice = ws.predV8
					}
					for by := 0; by < 2; by++ {
						for bx := 0; bx < 2; bx++ {
							bn := ch*4 + by*2 + bx
							extractBlock4x4UV(plane, yuv.uvStride, mbX*8+bx*4, ry*8+by*4, yuv.width, yuv.height, ws.uvSrc4[:])
							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									ws.uvPred4[y*4+x] = predSlice[(by*4+y)*8+(bx*4+x)]
								}
							}
							fTransform(ws.uvSrc4[:], ws.uvPred4[:], ws.uvDctOut[:])
							trellisQuantize(ws.uvDctOut[:], ws.uvQuant[:], &qm.uv, 0, mbLambdaTrellisUV, trellisUVCosts,
								(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[2]), 0)
							ws.uvLevels[bn] = ws.uvQuant
						}
					}
				}

				// Reconstruct chroma and update reconU/reconV.
				for ch := 0; ch < 2; ch++ {
					reconPlane := reconU
					predSlice := ws.predU8
					if ch == 1 {
						reconPlane = reconV
						predSlice = ws.predV8
					}
					for by := 0; by < 2; by++ {
						for bx := 0; bx < 2; bx++ {
							bn := ch*4 + by*2 + bx
							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									ws.uvPred4[y*4+x] = predSlice[(by*4+y)*8+(bx*4+x)]
								}
							}
							for n := 0; n < 16; n++ {
								j := int(kZigzag[n])
								ws.uvRaster[j] = int16(int32(ws.uvLevels[bn][n]) * int32(qm.uv.q[j]))
							}
							iTransform4x4(ws.uvRaster[:], ws.uvPred4[:], ws.uvRecBlock[:])
							bPX := mbX*8 + bx*4
							bPY := ry*8 + by*4
							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									reconPlane[(bPY+y)*yuv.uvStride+(bPX+x)] = uint8(ws.uvRecBlock[y*4+x])
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
							cd.i4AC[n] = ws.localI4AcLevels[n]
							last := findLast(ws.localI4AcLevels[n][:], 0)
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
				} else {
					cd.i16DC = ws.mbI16DcQuantLevels
					lastDC := findLast(ws.mbI16DcQuantLevels[:], 0)
					dcNZ := 0
					if lastDC >= 0 {
						dcNZ = 1
					}
					colTopNzDC = dcNZ
					leftNzY[4] = dcNZ

					for by := 0; by < 4; by++ {
						for bx := 0; bx < 4; bx++ {
							n := by*4 + bx
							cd.i16AC[n] = ws.mbI16AcLevels[n]
							last := findLast(ws.mbI16AcLevels[n][:], 1)
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
						cd.uv[n] = ws.uvLevels[n]
						last := findLast(ws.uvLevels[n][:], 0)
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
						cd.uv[4+n] = ws.uvLevels[4+n]
						last := findLast(ws.uvLevels[4+n][:], 0)
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
				rowProgress[ry].Store(int32(mbX))
			}
		}(ry, prev)
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
	encodePartition0WithProbs(part0BW, mbW, mbH, sns.segQs, numSegs, mbInfos, &adaptedProbs, &updatedFlags)
	part0Data := part0BW.finish()

	frameHdr := buildVP8FrameHeader(w, h, len(part0Data))
	result := make([]byte, 0, len(frameHdr)+len(part0Data)+len(tokenData))
	result = append(result, frameHdr...)
	result = append(result, part0Data...)
	result = append(result, tokenData...)
	return result
}
