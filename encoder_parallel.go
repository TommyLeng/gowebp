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
	// sadPreds caches the 10 SAD-phase predictions so they can be reused
	// in the RD phase without recomputing intra4Predict.
	// Moved from a per-block stack array (320 B) to ws to cut stack growth.
	sadPreds [numI4Modes][16]int16
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
	// uvPredCand{U,V}: per-mode candidate predictions for UV RD selection.
	// Filled once per (mbX, mbY) for all 4 modes so the winner's prediction
	// can be reused without recomputation after the RD loop.
	uvPredCandU [4][64]int16
	uvPredCandV [4][64]int16
	// uvReconBestU/V: the chosen mode's 8×8 reconstruction (in pixel space),
	// written by quantizeUVFinal after mode selection.
	uvReconBestU [64]uint8
	uvReconBestV [64]uint8
	// uvLevelsTmp/uvReconTmpU/uvReconTmpV: scratch buffers used inside
	// uvCandidateScore (per-candidate greedy quant + recon for D and R).
	uvLevelsTmp [8][16]int16
	uvReconTmpU [64]uint8
	uvReconTmpV [64]uint8
	// uvSrcU/uvSrcV: 8×8 source chroma copy extracted once per MB and used
	// by every RD candidate (avoids per-candidate plane reads inside the
	// inner SSE loop).
	uvSrcU [64]uint8
	uvSrcV [64]uint8
	// i16 inner loop
	i16Src4     [16]int16
	i16Pred4    [16]int16
	i16DctOut   [16]int16
	// fTransform2Plane scratch: holds DCT output for two adjacent 4x4 blocks
	dctPair     [32]int16
	i16RasterCoeffs [16]int16
	i16RecBlock [16]int16
	i16Pred4b   [16]int16
	// local copies of i4 accumulators (used within the i4 block, then assigned to ws fields)
	localI4AcLevels [16][16]int16
	localI4DcLevels [16]int16

	// i4Patch is a precomputed 21×21 neighbourhood extracted once per MB,
	// before the 16-block inner loop. It eliminates the per-pixel bounds-check
	// closure inside buildPred4ContextWithMBRecon.
	//
	// Layout (37 bytes total):
	//   i4Patch[0..20]  = top border row: y=mbPY-1, x=mbPX-1..mbPX+19 (21 pixels)
	//   i4Patch[21..36] = left border col: y=mbPY..mbPY+15, x=mbPX-1   (16 pixels)
	//
	// The MB interior (ws.mbReconI4) is NOT copied here; it is read directly and
	// updated block-by-block as before, which is required for correct intra context.
	i4Patch [37]uint8
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

			// Precomputed probability-table pointers (frame-invariant). Hoisted
			// out of the per-block path so the pointer casts don't run per block.
			// UV no longer uses trellis quantization (libwebp DO_TRELLIS_UV=0).
			i4ProbsPtr := (*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[3])
			i16ProbsPtr := (*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[0])

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
				mbLambdaUV := seg.lambdaUV
				mbLambdaMode := seg.lambdaMode
				mbLambdaTrellisI4 := seg.lambdaTrellisI4
				mbLambdaTrellisI16 := seg.lambdaTrellisI16
				// UV uses greedy quant (libwebp DO_TRELLIS_UV=0); trellis lambda
				// for UV is no longer used in the inner loop.
				trellisI4Costs := &seg.trellisI4Costs
				trellisI16Costs := &seg.trellisI16Costs
				trellisUVCosts := &seg.trellisUVCosts

				// Extract full 16x16 source block into workspace.
				src16 := &ws.src16
				for y := 0; y < 16; y++ {
					for x := 0; x < 16; x++ {
						src16[y*16+x] = int16(yuv.y[(py+y)*yuv.yStride+(px+x)])
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
				// i16 post-quantization RD (moved before i4 loop so that
				// i16Score is available for per-sub-block early-out).
				// Safe: intra16PredictFromRecon reads only the global recon
				// buffer (neighbor MBs), unaffected by the i4 loop below.
				// -------------------------------------------------------
				intra16PredictFromRecon(bestI16Mode, recon, reconStride, mbX, ry, yuv.mbW, yuv.mbH, ws.mbI16Pred[:])
				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx += 2 {
						n0, n1 := by*4+bx, by*4+bx+1
						fTransform2Plane(yuv.y, yuv.yStride, px+bx*4, py+by*4,
							ws.mbI16Pred[by*4*16:], 16, ws.dctPair[:])
						ws.yDcRaw16[n0] = ws.dctPair[0]
						ws.dctPair[0] = 0
						trellisQuantize(ws.dctPair[0:16], ws.mbI16AcLevels[n0][:], &qm.y1, 1, mbLambdaTrellisI16, trellisI16Costs,
							i16ProbsPtr, 0)
						ws.yDcRaw16[n1] = ws.dctPair[16]
						ws.dctPair[16] = 0
						trellisQuantize(ws.dctPair[16:32], ws.mbI16AcLevels[n1][:], &qm.y1, 1, mbLambdaTrellisI16, trellisI16Costs,
							i16ProbsPtr, 0)
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
				// i4HeaderCost: fixed overhead for signalling i4 mode in partition 0.
				// Defined here so it is available both in the per-block early-out and
				// in the final i4-vs-i16 comparison below.
				i4HeaderCost := int64(mbLambdaMode) * 211

				// -------------------------------------------------------
				// Try intra4
				// -------------------------------------------------------
				var bestI4Score int64 // set inside block below; value valid after i4EarlyOut label

				{
					var topBlkMode [4]int
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

					// Precompute top-row and left-column patch for this MB once,
					// eliminating 16 × 13 per-pixel bounds-check calls in the inner loop.
					fillI4Patch(&ws.i4Patch, recon, reconStride, px, py, yuv.mbW, yuv.mbH)
					mbHasTop := ry > 0

					var i4TotalScore int64

					for by := 0; by < 4; by++ {
						for bx := 0; bx < 4; bx++ {
							blkIdx := by*4 + bx
							bpx := px + bx*4
							bpy := py + by*4

							ctx := buildPred4ContextFromPatch(&ws.i4Patch, ws.mbReconI4[:], px, py, bpx, bpy, mbHasTop, yuv.mbW)

							src4 := &ws.src4
							for y := 0; y < 4; y++ {
								for x := 0; x < 4; x++ {
									src4[y*4+x] = int16(yuv.y[(bpy+y)*yuv.yStride+(bpx+x)])
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

							// trellisCtx0 is a loop-invariant for this block.
							trellisCtx0 := topNzI4[bx] + leftNzI4[by]
							if trellisCtx0 > 2 {
								trellisCtx0 = 2
							}

							if variance16 < flatThreshold16 {
								// Very flat block: only DC mode is worth trying.
								intra4Predict(B_DC_PRED, ctx, ws.pred4[:])
								// --- runRD(B_DC_PRED) inlined ---
								mode := B_DC_PRED
								fTransform(src4[:], ws.pred4[:], ws.dctOut[:])
								trellisQuantize(ws.dctOut[:], ws.acQ[:], &qm.y1, 0, mbLambdaTrellisI4, trellisI4Costs,
									i4ProbsPtr, trellisCtx0)
								iTransform4x4(ws.dctOut[:], ws.pred4[:], ws.recBlock[:])
								distortion := ssd4x4(src4[:], ws.recBlock[:])
								modeBits := i4ModeBitCost(mode, topPred, leftPred)
								rCost := coeffBitCost(trellisCtx0, ws.acQ[:], 0, trellisI4Costs)
								// mode==0 (DC), so no flatness penalty.
								score := int64(rdDistoMult)*distortion + int64(mbLambdaI4)*(modeBits+int64(rCost))
								if score < bestBlkScore {
									bestBlkScore = score
									bestBlkOldScore = distortion + int64(mbLambdaI4)*modeBits
									bestBlkMode = mode
									copy(ws.bestBlkAcLevels[:], ws.acQ[:])
									for i := 0; i < 16; i++ {
										ws.bestBlkRecon[i] = uint8(ws.recBlock[i])
									}
								}
							} else {
								const sadTopN = 4
								// Cache all 10 SAD-phase predictions in workspace (sadPreds
								// is part of ws to avoid 320 B per-iteration stack growth).
								var localSAD [numI4Modes]int64
								for i := 0; i < numI4Modes; i++ {
									intra4Predict(i, ctx, ws.sadPreds[i][:])
									s := sad4x4(src4[:], ws.sadPreds[i][:])
									ws.sadScores[i] = s
									localSAD[i] = s
								}
								// Find sadTopN-th lowest SAD via partial selection sort on a
								// stack-local copy (avoids bounds-check on ws.sadTmp slice).
								for k := 0; k < sadTopN; k++ {
									minIdx := k
									minVal := localSAD[k]
									for j := k + 1; j < numI4Modes; j++ {
										if localSAD[j] < minVal {
											minIdx = j
											minVal = localSAD[j]
										}
									}
									if minIdx != k {
										localSAD[minIdx] = localSAD[k]
										localSAD[k] = minVal
									}
								}
								sadCutoff := localSAD[sadTopN-1]

								for mode := 0; mode < numI4Modes; mode++ {
									if ws.sadScores[mode] > sadCutoff {
										continue
									}
									// --- runRD(mode) inlined ---
									// Use the cached SAD prediction directly (avoids a 16-int16 copy).
									predSlice := ws.sadPreds[mode][:]
									fTransform(src4[:], predSlice, ws.dctOut[:])
									trellisQuantize(ws.dctOut[:], ws.acQ[:], &qm.y1, 0, mbLambdaTrellisI4, trellisI4Costs,
										i4ProbsPtr, trellisCtx0)
									iTransform4x4(ws.dctOut[:], predSlice, ws.recBlock[:])
									distortion := ssd4x4(src4[:], ws.recBlock[:])
									modeBits := i4ModeBitCost(mode, topPred, leftPred)

									// Phase-1 early-out, stage A: D-only lower bound (no flat
									// penalty, no coefficient rate). Cheapest possible test, run
									// before the flatness scan so the common path pays nothing
									// extra. score >= 256*D + λ*modeBits always, so a loss here is
									// a guaranteed loss.
									dScore := int64(rdDistoMult)*distortion + int64(mbLambdaI4)*modeBits
									if bestBlkScore < int64(1<<62-1) && dScore >= bestBlkScore {
										continue
									}

									// Flatness penalty (also needed for the final score below).
									flatBitsR := int64(0)
									if mode > 0 && isFlatI4Levels(ws.acQ[:]) {
										flatBitsR = flatnessPenalty
									}

									// Phase-1 early-out, stage B: tighten the bound with the flat
									// penalty (rCost >= 0, so this is still a valid lower bound).
									// Matches libwebp's early-out, which folds the flat penalty into
									// R before VP8GetCostLuma4. Only fires in the narrow window where
									// the flat penalty alone flips the decision, so coeffBitCost is
									// skipped in those cases too.
									if flatBitsR != 0 && bestBlkScore < int64(1<<62-1) {
										if dScore+int64(mbLambdaI4)*flatBitsR >= bestBlkScore {
											continue
										}
									}

									rCost := coeffBitCost(trellisCtx0, ws.acQ[:], 0, trellisI4Costs)
									score := dScore + int64(mbLambdaI4)*(int64(rCost)+flatBitsR)
									if score < bestBlkScore {
										bestBlkScore = score
										bestBlkOldScore = distortion + int64(mbLambdaI4)*modeBits
										bestBlkMode = mode
										copy(ws.bestBlkAcLevels[:], ws.acQ[:])
										for i := 0; i < 16; i++ {
											ws.bestBlkRecon[i] = uint8(ws.recBlock[i])
										}
									}
								}
							}

							ws.localI4Modes[blkIdx] = bestBlkMode
							ws.localI4AcLevels[blkIdx] = ws.bestBlkAcLevels
							ws.localI4DcLevels[blkIdx] = 0
							i4TotalScore += bestBlkOldScore

							// Per-block early-out: if accumulated i4 cost already exceeds i16,
							// remaining sub-blocks cannot recover — bail out and use i16.
							// Port of libwebp's PickBestIntra4 check (quant_enc.c:1121):
							//   if (rd_best.score >= rd_i16->score) return 0;
							if i4TotalScore+i4HeaderCost >= i16Score {
								goto i4EarlyOut
							}

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

				i4EarlyOut:
					bestI4Score = i4TotalScore
				}

				i4Score := bestI4Score + i4HeaderCost

				info := &mbInfos[mbIdx]
				if i4Score < i16Score {
					info.isI4 = true
					copy(info.i4Modes[:], ws.localI4Modes[:])
				} else {
					info.isI4 = false
					info.i16Mode = bestI16Mode
				}

				// UV: RD-style prediction mode selection (PickBestUV port).
				// See encoder.go for full rationale; same SSD pre-screen + top-N RD.
				const uvRDTopN = 1
				bestUVMode := 0
				ctxTopUBase := [2]int{colTopNzU[0], colTopNzU[1]}
				ctxLeftUBase := [2]int{leftNzU[0], leftNzU[1]}
				ctxTopVBase := [2]int{colTopNzV[0], colTopNzV[1]}
				ctxLeftVBase := [2]int{leftNzV[0], leftNzV[1]}

				// Precompute source 8×8 chroma for cheap reuse across candidates.
				fillUVSrc8x8(ws, yuv, mbX, ry)

				// Phase 1: raw prediction SSD for every valid mode.
				const uvInfSAD = int64(1<<62 - 1)
				var uvCandSAD [4]int64
				var uvCandValid [4]bool
				for uvMode := 0; uvMode < 4; uvMode++ {
					switch uvMode {
					case 1:
						if ry == 0 {
							uvCandSAD[uvMode] = uvInfSAD
							continue
						}
					case 2:
						if mbX == 0 {
							uvCandSAD[uvMode] = uvInfSAD
							continue
						}
					case 3:
						if mbX == 0 || ry == 0 {
							uvCandSAD[uvMode] = uvInfSAD
							continue
						}
					}
					uvCandValid[uvMode] = true
					predictUV(uvMode, reconU, yuv.uvStride, mbX, ry, yuv.width, yuv.height, ws.uvPredCandU[uvMode][:])
					predictUV(uvMode, reconV, yuv.uvStride, mbX, ry, yuv.width, yuv.height, ws.uvPredCandV[uvMode][:])
					var sad int64
					for ch := 0; ch < 2; ch++ {
						var src []uint8
						var pred []int16
						if ch == 0 {
							src = ws.uvSrcU[:]
							pred = ws.uvPredCandU[uvMode][:]
						} else {
							src = ws.uvSrcV[:]
							pred = ws.uvPredCandV[uvMode][:]
						}
						for k := 0; k < 64; k++ {
							d := int64(src[k]) - int64(pred[k])
							sad += d * d
						}
					}
					uvCandSAD[uvMode] = sad
				}
				// Rank modes by SSD (ascending) — partial selection of top uvRDTopN.
				var uvRank [4]int
				for i := 0; i < 4; i++ {
					uvRank[i] = i
				}
				for k := 0; k < uvRDTopN && k < 4; k++ {
					minIdx := k
					for j := k + 1; j < 4; j++ {
						if uvCandSAD[uvRank[j]] < uvCandSAD[uvRank[minIdx]] {
							minIdx = j
						}
					}
					uvRank[k], uvRank[minIdx] = uvRank[minIdx], uvRank[k]
				}

				// Phase 2: pick by RD when uvRDTopN > 1; SSD wins when N=1.
				if uvRDTopN == 1 {
					if uvCandValid[uvRank[0]] {
						bestUVMode = uvRank[0]
					} else {
						bestUVMode = 0
					}
				} else {
					bestUVScore := int64(1<<62 - 1)
					haveBest := false
					for k := 0; k < uvRDTopN; k++ {
						uvMode := uvRank[k]
						if !uvCandValid[uvMode] {
							continue
						}
						dCand, rCand := uvCandidateScore(ws, yuv, mbX, ry,
							ws.uvPredCandU[uvMode][:], ws.uvPredCandV[uvMode][:],
							&qm.uv, trellisUVCosts,
							ctxTopUBase, ctxLeftUBase, ctxTopVBase, ctxLeftVBase)

						hCost := uvModeBitCost(uvMode)
						flatR := int64(0)
						if uvMode > 0 && isFlatUVLevels(&ws.uvLevelsTmp) {
							flatR = int64(flatnessPenalty) * 8
						}
						score := int64(rdDistoMult)*dCand + int64(mbLambdaUV)*(hCost+int64(rCand)+flatR)
						if !haveBest || score < bestUVScore {
							bestUVScore = score
							bestUVMode = uvMode
							haveBest = true
						}
					}
					if !haveBest {
						bestUVMode = 0
					}
				}
				if !uvCandValid[bestUVMode] {
					predictUV(0, reconU, yuv.uvStride, mbX, ry, yuv.width, yuv.height, ws.uvPredCandU[0][:])
					predictUV(0, reconV, yuv.uvStride, mbX, ry, yuv.width, yuv.height, ws.uvPredCandV[0][:])
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
				// UV: trellis-quantize the chosen mode's prediction for the
				// final bitstream. Mode selection used greedy quant
				// (uvCandidateScore) for speed; the final pass uses trellis
				// for the compression gain. Writes ws.uvLevels and ws.uvReconBestU/V.
				// -------------------------------------------------------
				ws.predU8 = ws.uvPredCandU[info.uvMode]
				ws.predV8 = ws.uvPredCandV[info.uvMode]
				quantizeUVFinal(ws, yuv, mbX, ry,
					ws.predU8[:], ws.predV8[:],
					&qm.uv, seg.lambdaTrellisUV, trellisUVCosts,
					(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[2]))

				// Update reconU/reconV from the final 8×8 reconstruction.
				for by := 0; by < 8; by++ {
					rowDst := (ry*8 + by) * yuv.uvStride
					rowSrc := by * 8
					for bx := 0; bx < 8; bx++ {
						reconU[rowDst+mbX*8+bx] = ws.uvReconBestU[rowSrc+bx]
						reconV[rowDst+mbX*8+bx] = ws.uvReconBestV[rowSrc+bx]
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
							cd.i4ACLast[n] = int8(last)
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
					cd.i16DCLast = int8(lastDC)
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
							cd.i16ACLast[n] = int8(last)
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
						cd.uvLast[n] = int8(last)
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
						cd.uvLast[4+n] = int8(last)
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
