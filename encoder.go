// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// mbInfo holds the encoding decisions for one macroblock.
type mbInfo struct {
	isI4    bool      // true = intra4x4, false = intra16x16
	i16Mode int       // if !isI4: intra16 mode (0..3)
	i4Modes [16]int   // if isI4: per 4x4 block mode (0..9), scan order
	uvMode  int       // UV prediction mode (0=DC, 1=VE, 2=HE, 3=TM — libwebp order)
	segment int       // segment ID (0=coarse/smooth, 1=fine/textured) for SNS
}

// segmentParams holds per-segment quantizer matrices and associated lambda values.
type segmentParams struct {
	qm              quantMatrices
	baseQ           int
	lambdaI4        int
	lambdaI16       int
	lambdaMode      int
	lambdaTrellisI4  int
	lambdaTrellisI16 int
	lambdaTrellisUV  int
	trellisI4Costs  trellisCostTables
	trellisI16Costs trellisCostTables
	trellisUVCosts  trellisCostTables
}

// makeSegmentParams builds a segmentParams for a given quality level.
func makeSegmentParams(quality int) segmentParams {
	qm := buildQuantMatrices(quality)
	q := qualityToLevel(quality)

	var y1qSum, y2qSum, uvqSum int
	for i := 0; i < 16; i++ {
		y1qSum += int(qm.y1.q[i])
		y2qSum += int(qm.y2.q[i])
		uvqSum += int(qm.uv.q[i])
	}
	qI4 := (y1qSum + 8) >> 4
	qI16 := (y2qSum + 8) >> 4
	qUV := (uvqSum + 8) >> 4

	lambdaI4 := (3 * qI4 * qI4) >> 7
	if lambdaI4 < 1 {
		lambdaI4 = 1
	}
	lambdaI16 := 3 * qI16 * qI16
	if lambdaI16 < 1 {
		lambdaI16 = 1
	}
	lambdaMode := (1 * qI4 * qI4) >> 7
	if lambdaMode < 1 {
		lambdaMode = 1
	}
	lambdaTrellisI4 := (7 * qI4 * qI4) >> 3
	if lambdaTrellisI4 < 1 {
		lambdaTrellisI4 = 1
	}
	lambdaTrellisI16 := (qI16 * qI16) >> 2
	if lambdaTrellisI16 < 1 {
		lambdaTrellisI16 = 1
	}
	lambdaTrellisUV := (qUV * qUV) << 1
	if lambdaTrellisUV < 1 {
		lambdaTrellisUV = 1
	}

	return segmentParams{
		qm:               qm,
		baseQ:            q,
		lambdaI4:         lambdaI4,
		lambdaI16:        lambdaI16,
		lambdaMode:       lambdaMode,
		lambdaTrellisI4:  lambdaTrellisI4,
		lambdaTrellisI16: lambdaTrellisI16,
		lambdaTrellisUV:  lambdaTrellisUV,
		trellisI4Costs:   buildTrellisCostTables((*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[3])),
		trellisI16Costs:  buildTrellisCostTables((*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[0])),
		trellisUVCosts:   buildTrellisCostTables((*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[2])),
	}
}

// encodeFrame encodes a YUV image into a VP8 bitstream.
// Returns the raw VP8 bytes (frame header + partition0 + token partition).
// Phase 2: per-MB RD selection between intra16 (4 modes) and intra4 (10 modes).
//
// SNS: uses 2 segments — segment 0 (smooth MBs, coarser quant) and
// segment 1 (textured MBs, finer quant = baseQ). Segment quantizers are
// signaled in partition 0 using the VP8 segment header.
func encodeFrame(yuv *yuvImage, baseQ int) []byte {
	w := yuv.width
	h := yuv.height
	// mbW/mbH come from the padded yuvImage so they match the YUV plane dimensions.
	mbW := yuv.mbW / 16
	mbH := yuv.mbH / 16

	// SNS (Spatial Noise Shaping): pre-compute per-MB luma activity scores.
	// Mirrors VP8EncAnalyze() / MBAnalyze() in libwebp/src/enc/analysis_enc.c.
	mbAlpha := make([]int, mbW*mbH)
	for mbY := 0; mbY < mbH; mbY++ {
		for mbX := 0; mbX < mbW; mbX++ {
			mbAlpha[mbY*mbW+mbX] = computeMBAlpha(yuv, mbX, mbY)
		}
	}

	// Compute per-segment quantizers using the libwebp SNS formula.
	// VP8SetSegmentParams: for each segment alpha, expn = 1 - amp*alpha,
	// c = c_base^expn, q = 127*(1-c). With snsStrength=50, SNS_TO_DQ=0.9:
	//   amp = 0.9 * 50 / 100 / 128 ≈ 0.003516
	// We use 2 segments: segment 0 = smooth (higher q), segment 1 = textured (baseQ).
	// The alpha threshold separates the two groups.
	seg0Quality, seg1Quality := computeSNSSegmentQualities(baseQ, mbAlpha)

	// Build per-segment quantizer matrices and lambdas.
	seg0 := makeSegmentParams(seg0Quality)
	seg1 := makeSegmentParams(seg1Quality)
	segs := [2]segmentParams{seg0, seg1}

	// Assign each MB to a segment: below threshold → segment 0 (coarse/smooth).
	// Use median alpha as threshold (50th percentile).
	alphaThreshold := computeAlphaThreshold(mbAlpha)
	mbSegment := make([]int, mbW*mbH)
	for i, a := range mbAlpha {
		if a <= alphaThreshold {
			mbSegment[i] = 0 // smooth → coarser quant
		} else {
			mbSegment[i] = 1 // textured → finer quant
		}
	}

	// Reconstructed luma: used to build intra4 contexts across 4x4 blocks.
	// We maintain a reconstructed frame buffer so intra4 predictions use
	// previously decoded pixels within and across MBs.
	reconStride := mbW * 16
	recon := make([]uint8, reconStride*mbH*16)

	// Reconstructed chroma: UV DC prediction must use the same values the
	// decoder will reconstruct, not the original YUV samples. Without this,
	// the encoder's DC prediction diverges from the decoder's, causing a
	// systematic Cr offset (red bias) that accumulates across MBs.
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

	// mbCoeffs stores quantized coefficient levels for every MB.
	// Used for the two-pass coefficient probability adaptation:
	// pass 1 collects statistics, pass 2 emits entropy-coded bits.
	mbCoeffs := make([]mbCoeffData, mbW*mbH)

	// per-MB top/left i4 mode arrays for entropy context
	// topI4Modes[mbX*4 + bx]: top neighbor mode for 4x4 block column bx in mbX
	topI4Modes := make([]int, mbW*4)

	// NZ context arrays — still tracked during the MB loop so that we know
	// which ctx value to pass to recordCoeffs/putCoeffsWithProbs in the
	// second pass.  We re-derive them from mbCoeffs in encodeTokenPartition.
	// They are kept here only to compute the correct nz flags that we store
	// alongside the coefficient levels (ctx is implicit; we store it too).
	topNzY := make([]int, mbW*4+1)
	topNzU := make([]int, mbW*2+1)
	topNzV := make([]int, mbW*2+1)
	topNzDC := make([]int, mbW+1)

	for mbY := 0; mbY < mbH; mbY++ {
		leftNzY := [5]int{}
		leftNzU := [3]int{}
		leftNzV := [3]int{}

		// left I4 mode context: leftI4Mode[by] = mode of block (bx-1, by)
		leftI4Mode := [4]int{}

		for mbX := 0; mbX < mbW; mbX++ {
			mbIdx := mbY*mbW + mbX
			px := mbX * 16
			py := mbY * 16

			// SNS: select per-MB segment params based on activity score.
			// segment 0 = smooth/coarse quant, segment 1 = textured/fine quant.
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
				intra16Predict(mode, yuv, mbX, mbY, pred16[:])
				distortion := ssd16x16(src16[:], pred16[:])
				// bits cost for i16 mode: approximate with fixed costs
				// VP8FixedCostsI16: DC=0,TM=1,VE=2,HE=3 -> bit costs in millibits
				// Use rough estimate: all modes cost ~2-4 bits
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
			// Try intra4: for each of 16 4x4 blocks pick best mode
			// -------------------------------------------------------
			// We process blocks in raster order and update mbRecon with the
			// actual reconstructed pixels (pred + dequantized residual)
			// so subsequent blocks get correct intra4 prediction contexts.
			var bestI4Score int64
			var i4AcLevels [16][16]int16
			var i4DcLevels [16]int16
			var localI4Modes [16]int
			// mbReconI4 holds reconstructed pixels built during i4 mode selection.
			// It is also the mbRecon that is copied to the global recon buffer if i4 wins.
			var mbReconI4 [16 * 16]uint8

			{
				// Track per-block top/left mode context
				topBlkMode := make([]int, 4) // top block modes for bx=0..3
				for bx := 0; bx < 4; bx++ {
					topBlkMode[bx] = topI4Modes[mbX*4+bx]
				}
				leftBlkMode := [4]int{leftI4Mode[0], leftI4Mode[1], leftI4Mode[2], leftI4Mode[3]}

				// NZ context tracking for trellis: mirrors it->top_nz / it->left_nz in libwebp.
				// topNzI4[bx] = 1 if the last block in column bx had a non-zero coeff.
				// leftNzI4[by] = 1 if the last block in row by had a non-zero coeff.
				// Seeded from the global topNzY which tracks across MBs.
				var topNzI4 [4]int
				var leftNzI4 [4]int
				for bx := 0; bx < 4; bx++ {
					topNzI4[bx] = topNzY[mbX*4+bx]
				}
				// leftNzY[by] from previous MB (already tracks left edge)
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

						// Build prediction context (uses global recon for already-encoded MBs,
						// and mbReconI4 for within this MB — containing true reconstructed pixels).
						ctx := buildPred4ContextWithMBRecon(yuv, recon, reconStride, mbReconI4[:], px, py, bpx, bpy)

						// Extract 4x4 source
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

						// Mode context for bit cost
						topPred := topBlkMode[bx]
						leftPred := 0
						if bx > 0 {
							leftPred = localI4Modes[blkIdx-1]
						} else {
							leftPred = leftBlkMode[by]
						}

						// SAD pre-screening: compute cheap Sum of Absolute Differences for
						// all 10 modes, then only run the full DCT+quantize path on the
						// top sadTopN candidates. Gives ~2.5× speedup for the i4 path.
						const sadTopN = 4
						var sadScores [numI4Modes]int64
						for i := 0; i < numI4Modes; i++ {
							var p [16]int16
							intra4Predict(i, ctx, p[:])
							sadScores[i] = sad4x4(src4[:], p[:])
						}
						// Find sadTopN-th lowest SAD via partial selection sort on a copy.
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

						// Try all 10 I4 modes; track the best.
						bestBlkMode := B_DC_PRED
						bestBlkScore := int64(1<<62 - 1)
						var bestBlkRecon [16]uint8 // actual reconstructed pixels for best mode
						var bestBlkAcLevels [16]int16

						for mode := 0; mode < numI4Modes; mode++ {
							if sadScores[mode] > sadCutoff {
								continue
							}
							var pred4 [16]int16
							intra4Predict(mode, ctx, pred4[:])

							// Forward DCT of (src - pred)
							var dctOut [16]int16
							fTransform(src4[:], pred4[:], dctOut[:])

							// Trellis-quantize all 16 coefficients (i4: no WHT, first=0).
							// ctx0 = topNzI4[bx] + leftNzI4[by] mirrors libwebp's NZ context.
							// trellisQuantize writes dequantized values back into dctOut.
							trellisCtx0 := topNzI4[bx] + leftNzI4[by]
							if trellisCtx0 > 2 {
								trellisCtx0 = 2
							}
							var acQ [16]int16
							trellisQuantize(dctOut[:], acQ[:], &qm.y1, 0, mbLambdaTrellisI4, trellisI4Costs,
								(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[3]), trellisCtx0)

							// Inverse DCT → reconstructed block.
							// dctOut now holds dequantized raster-order coefficients.
							var recBlock [16]int16
							iTransform4x4(dctOut[:], pred4[:], recBlock[:])

							// Distortion = SSD of source vs actual reconstructed pixels
							var distortion int64
							for i := 0; i < 16; i++ {
								d := int64(src4[i]) - int64(recBlock[i])
								distortion += d * d
							}

							// Mode bit cost
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
						localI4DcLevels[blkIdx] = 0 // i4 has no WHT DC
						i4TotalScore += bestBlkScore

						// Update NZ context for subsequent blocks' trellis decisions.
						bestNZ := 0
						if findLast(bestBlkAcLevels[:], 0) >= 0 {
							bestNZ = 1
						}
						topNzI4[bx] = bestNZ
						leftNzI4[by] = bestNZ

						// Update mbReconI4 with actual reconstructed pixels.
						for y := 0; y < 4; y++ {
							for x := 0; x < 4; x++ {
								mbReconI4[(by*4+y)*16+(bx*4+x)] = bestBlkRecon[y*4+x]
							}
						}

						// Update top block mode context
						topBlkMode[bx] = bestBlkMode
					}
					// After this row, update left mode context for next MB's bx=0 at this row
					leftBlkMode[by] = localI4Modes[by*4+3]
				}

				bestI4Score = i4TotalScore
				i4AcLevels = localI4AcLevels
				i4DcLevels = localI4DcLevels
			}

			// -------------------------------------------------------
			// RD decision: i16 vs i4
			//
			// We need a fair comparison. Both scores must use:
			//   - post-quantization distortion (SSD of source vs actual reconstructed)
			//   - same lambda scale for bit costs
			//
			// For i16: compute the full reconstruction (DCT → WHT → quantize → dequant
			//   → iWHT → iDCT) and measure the post-quantization distortion.
			//   Then score = distortion + lambdaMode * headerBits.
			//
			// For i4: bestI4Score = sum over 16 blocks of (post-quant SSD + lambdaI4 * modeBits).
			//   The mode bit costs use lambdaI4 scale. The header cost (MB type bit) is separate.
			//
			// To compare at the same scale, we need to account for the fact that
			// i4 bit costs used lambdaI4 while i16 uses lambdaMode.
			// lambdaI4 = (3*qI4^2)>>7 and lambdaMode = (1*qI4^2)>>7, so lambdaI4 ≈ 3*lambdaMode.
			//
			// Strategy: compute i16 post-quant distortion, add lambdaMode * i16HdrBits.
			// For i4: use post-quant distortion from bestI4Score decomposition.
			//   Since bestI4Score = distortion_sum + lambdaI4 * mode_bits_sum, and
			//   we want distortion_sum + lambdaMode * mode_bits_sum:
			//   i4Score_normalized ≈ bestI4Score - lambdaI4*bits + lambdaMode*bits
			//                      = bestI4Score - 2/3 * lambdaI4 * bits
			// Approximation: use bestI4Score / 3 + i4HeaderPenalty (since lambdaI4≈3*lambdaMode).
			// This avoids needing to separately track distortion vs bit costs.
			//
			// Simpler equivalent: compare i16 and i4 at the same lambda=lambdaMode:
			//   i16Score = i16PostQuantDistortion + lambdaMode * i16ModeBits
			//   i4Score  = i4PostQuantDistortion  + lambdaMode * i4ModeBits + i4HeaderCost
			// where i4PostQuantDistortion = bestI4Score - lambdaI4 * i4ModeBits_sum.
			// But we don't have i4ModeBits separately from the score.
			//
			// Practical approach: since distortion dominates at quality=90 and
			// both scores share the same distortion unit, just scale the bit terms equally.
			// Use bestI4Score as-is (overestimates bit cost penalty for i4 by ~3x) but
			// also compute the i16 score with lambdaI4 scale to match.
			//
			// This biases toward i16 (penalizes i4 bit costs more), which is actually
			// fine — it means we choose i4 only when it meaningfully beats i16 on distortion.
			// -------------------------------------------------------

			// Compute i16 post-quantization distortion.
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
					trellisQuantize(dctOut[:], mbI16AcLevels[n][:], &qm.y1, 1, mbLambdaTrellisI16, trellisI16Costs,
						(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[0]), 0)
				}
			}
			var whtOut16 [16]int16
			fTransformWHT(yDcRaw16[:], whtOut16[:])
			quantizeBlockWHT(whtOut16[:], mbI16DcQuantLevels[:], &qm.y2)

			// Dequantize WHT → iWHT → per-block DC coeffs
			var whtRaster16 [16]int16
			for n := 0; n < 16; n++ {
				j := int(kZigzag[n])
				whtRaster16[j] = int16(int32(mbI16DcQuantLevels[n]) * int32(qm.y2.q[j]))
			}
			var dcBlockCoeffs16 [16]int16
			inverseWHT16(whtRaster16[:], dcBlockCoeffs16[:])

			// Reconstruct i16 and measure post-quantization distortion
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

			// MB-level RD decision: compare i4 vs i16.
			// i16Score uses lambdaI16 (large) which inflates the bit-cost term,
			// biasing the comparison toward i4. This matches libwebp's intent:
			// at high quality, i4 wins whenever its distortion is comparable to
			// i16, resulting in better compression for natural images.
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

			// UV: always use DC (simplest, good for chroma)
			info.uvMode = 0
			// SNS: record which segment this MB belongs to.
			info.segment = mbSegment[mbIdx]

			// -------------------------------------------------------
			// Update global recon buffer with chosen mode's reconstruction
			// -------------------------------------------------------
			if info.isI4 {
				// mbReconI4 holds actual reconstructed pixels (pred + quantized residual).
				// Copy to global recon buffer for adjacent MBs' prediction context.
				for y := 0; y < 16; y++ {
					for x := 0; x < 16; x++ {
						recon[(py+y)*reconStride+(px+x)] = mbReconI4[y*16+x]
					}
				}
			} else {
				// i16: all the quantization and reconstruction was already computed
				// in the RD decision section above (mbI16AcLevels, mbI16DcQuantLevels,
				// mbI16Pred, dcBlockCoeffs16). Just write the reconstructed pixels to recon.
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
			// Update top/left i4 mode contexts for next MB
			// -------------------------------------------------------
			if info.isI4 {
				for bx := 0; bx < 4; bx++ {
					topI4Modes[mbX*4+bx] = info.i4Modes[3*4+bx]
				}
				for by := 0; by < 4; by++ {
					leftI4Mode[by] = info.i4Modes[by*4+3]
				}
			} else {
				// i16 MB: set i4 context to the i16 mode.
				// The VP8 spec says: for i16 MBs, the top/left prediction mode
				// contexts for adjacent i4 blocks are set to the i16 mode.
				// The decoder's parsePredModeY16 sets d.upMB[mbx].pred[i] = i16Mode
				// and d.leftMB.pred[j] = i16Mode for all i,j.
				// Our i4 mode constants match the i16 constants:
				//   B_DC_PRED=0=I16_DC_PRED, B_TM_PRED=1=I16_TM_PRED,
				//   B_VE_PRED=2=I16_VE_PRED, B_HE_PRED=3=I16_HE_PRED.
				for bx := 0; bx < 4; bx++ {
					topI4Modes[mbX*4+bx] = info.i16Mode
				}
				for by := 0; by < 4; by++ {
					leftI4Mode[by] = info.i16Mode
				}
			}

			// -------------------------------------------------------
			// Encode coefficients
			// -------------------------------------------------------
			// -------------------------------------------------------
			// Store coefficient levels in mbCoeffs for the two-pass
			// probability adaptation (stats → adapted probs → entropy coding).
			// Also update NZ context arrays so subsequent MBs use the correct ctx.
			// -------------------------------------------------------
			cd := &mbCoeffs[mbIdx]
			cd.isI4 = info.isI4

			if info.isI4 {
				// i4: store all 16 block × 16 coeff levels and update NZ ctx.
				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						cd.i4AC[n] = i4AcLevels[n]
						last := findLast(i4AcLevels[n][:], 0)
						nz := 0
						if last >= 0 {
							nz = 1
						}
						topNzY[mbX*4+bx] = nz
						leftNzY[by] = nz
					}
				}
				topNzDC[mbX] = 0
				leftNzY[4] = 0
				_ = i4DcLevels

			} else {
				// i16: store DC and AC levels, update NZ ctx.
				cd.i16DC = mbI16DcQuantLevels
				lastDC := findLast(mbI16DcQuantLevels[:], 0)
				dcNZ := 0
				if lastDC >= 0 {
					dcNZ = 1
				}
				topNzDC[mbX] = dcNZ
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
						topNzY[mbX*4+bx] = nz
						leftNzY[by] = nz
					}
				}
			}

			// UV quantization — always DC prediction.
			dcU := computeDCUV(reconU, yuv.uvStride, mbX, mbY, yuv.width, yuv.height)
			dcV := computeDCUV(reconV, yuv.uvStride, mbX, mbY, yuv.width, yuv.height)

			var uvLevels [8][16]int16
			for ch := 0; ch < 2; ch++ {
				plane := yuv.u
				if ch == 1 {
					plane = yuv.v
				}
				dcUV := dcU
				if ch == 1 {
					dcUV = dcV
				}
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
						trellisQuantize(dctOut[:], quant[:], &qm.uv, 0, mbLambdaTrellisUV, trellisUVCosts,
							(*[numBands][numCtx][numProbas]uint8)(&defaultCoeffProbs[2]), 0)
						uvLevels[bn] = quant
					}
				}
			}

			// Reconstruct chroma and update reconU/reconV for subsequent MBs.
			// Re-dequantize from levels (since we need a fresh raster for iDCT).
			for ch := 0; ch < 2; ch++ {
				reconPlane := reconU
				if ch == 1 {
					reconPlane = reconV
				}
				dcUV := dcU
				if ch == 1 {
					dcUV = dcV
				}
				for by := 0; by < 2; by++ {
					for bx := 0; bx < 2; bx++ {
						bn := ch*4 + by*2 + bx
						var pred4 [16]int16
						fillPred4x4(pred4[:], uint8(dcUV))
						var raster [16]int16
						for n := 0; n < 16; n++ {
							j := int(kZigzag[n])
							raster[j] = int16(int32(uvLevels[bn][n]) * int32(qm.uv.q[j]))
						}
						var recBlock [16]int16
						iTransform4x4(raster[:], pred4[:], recBlock[:])
						bPX := mbX*8 + bx*4
						bPY := mbY*8 + by*4
						for y := 0; y < 4; y++ {
							for x := 0; x < 4; x++ {
								reconPlane[(bPY+y)*yuv.uvStride+(bPX+x)] = uint8(recBlock[y*4+x])
							}
						}
					}
				}
			}

			// Store UV levels and update NZ ctx.
			for by := 0; by < 2; by++ {
				for bx := 0; bx < 2; bx++ {
					n := by*2 + bx
					cd.uv[n] = uvLevels[n]
					last := findLast(uvLevels[n][:], 0)
					nz := 0
					if last >= 0 {
						nz = 1
					}
					topNzU[mbX*2+bx] = nz
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
					topNzV[mbX*2+bx] = nz
					leftNzV[by] = nz
				}
			}
		}
	}

	// --- Two-pass coefficient probability adaptation ---
	// Pass 1: collect statistics over all stored coefficient levels.
	var stats coeffStats
	collectCoeffStats(mbCoeffs, mbW, mbH, &stats)

	// Compute adapted probabilities and which entries to update.
	adaptedProbs, updatedFlags := finalizeTokenProbas(&stats)

	// Pass 2: entropy-encode all coefficients using the adapted probabilities.
	tokenBW := newBoolEncoder()
	encodeTokenPartition(tokenBW, mbCoeffs, mbW, mbH, &adaptedProbs)
	tokenData := tokenBW.finish()

	// --- Partition 0: frame-level headers + intra modes + updated probs ---
	// Pass SNS segment quantizer indices so the segment header can be emitted.
	// seg0.baseQ = quantizer for smooth MBs (segment 0)
	// seg1.baseQ = quantizer for textured MBs (segment 1)
	part0BW := newBoolEncoder()
	encodePartition0WithProbs(part0BW, mbW, mbH, seg0.baseQ, seg1.baseQ, mbInfos, &adaptedProbs, &updatedFlags)
	part0Data := part0BW.finish()

	// --- Assemble VP8 bitstream ---
	frameHdr := buildVP8FrameHeader(w, h, len(part0Data))
	result := make([]byte, 0, len(frameHdr)+len(part0Data)+len(tokenData))
	result = append(result, frameHdr...)
	result = append(result, part0Data...)
	result = append(result, tokenData...)
	return result
}

// buildPred4ContextWithMBRecon builds the intra4 context using both the
// global reconstructed frame buffer (for cross-MB neighbors) and the
// current MB's in-progress reconstruction (mbRecon, 16x16 flat).
func buildPred4ContextWithMBRecon(yuv *yuvImage, recon []uint8, reconStride int,
	mbRecon []uint8, mbPX, mbPY, bpx, bpy int) pred4Context {

	var ctx pred4Context

	// get returns the reconstructed pixel at global image coordinates (x, y).
	// Mirrors VP8's YBR buffer border conventions:
	//   x < 0 (left of image): returns 129 (VP8 left-column border = 0x81)
	//   y < 0 (above image):   returns 127 (VP8 top-row border = 0x7f)
	//   otherwise: reads from mbRecon (current MB) or recon (previous MBs)
	get := func(x, y int) int {
		if x < 0 {
			return 129
		}
		if y < 0 {
			return 127
		}
		if x >= yuv.mbW || y >= yuv.mbH {
			return 128
		}
		// Is the pixel within the current MB?
		rx := x - mbPX
		ry := y - mbPY
		if rx >= 0 && rx < 16 && ry >= 0 && ry < 16 {
			return int(mbRecon[ry*16+rx])
		}
		// Use the global recon buffer for previously encoded MBs.
		return int(recon[y*reconStride+x])
	}

	hasLeft := bpx > 0
	hasTop := bpy > 0

	for i := 0; i < 4; i++ {
		if hasLeft {
			ctx.left[i] = get(bpx-1, bpy+i)
		} else {
			ctx.left[i] = 129
		}
	}

	// Top-left pixel: use get() which correctly returns 129 for x<0 and 127 for y<0.
	// This mirrors VP8's YBR border conventions exactly.
	if hasTop || hasLeft {
		ctx.topLeft = get(bpx-1, bpy-1)
	} else {
		// No top, no left: top-left corner of the image.
		// VP8 prepareYBR sets ybr[0][7] = 0x7f for mby=0, 0x81 for mby>0.
		// In practice, this case only occurs for block (bx=0, by=0) of MB(0,0).
		// For mby=0: 127. For mby>0: 129.
		if mbPY == 0 {
			ctx.topLeft = 127
		} else {
			ctx.topLeft = 129
		}
	}

	for i := 0; i < 4; i++ {
		if hasTop {
			ctx.top[i] = get(bpx+i, bpy-1)
		} else {
			ctx.top[i] = 127
		}
	}

	// top[4..7]: right-of-top context (used by LD, VL diagonal predictors).
	// Key: VP8IteratorRotateI4 in libwebp replicates the MB's original right-of-top
	// context for all sub-rows within the MB (not using the reconstructed pixels
	// to the right of the current sub-row). Specifically, for blocks in the
	// rightmost column (bx=3) and sub-rows below the first (by>0), the right-of-top
	// comes from the TOP BORDER ROW of the MB (row mbPY-1), not from row bpy-1.
	// The decoder's prepareYBR copies d.ybr[0][24..27] to d.ybr[4,8,12][24..27].
	// We replicate this: for x >= mbPX+16, always use y = mbPY-1 (MB top border).
	mbHasTop := mbPY > 0
	for i := 0; i < 4; i++ {
		x := bpx + 4 + i
		if !hasTop {
			ctx.top[4+i] = 127
		} else if x >= mbPX+16 {
			// This pixel is to the right of the MB boundary.
			// Use the MB's top border row (y = mbPY-1), replicate the last valid column.
			if mbHasTop {
				topY := mbPY - 1
				if x < yuv.mbW {
					ctx.top[4+i] = int(recon[topY*reconStride+x])
				} else {
					// Beyond padded image: replicate last column
					ctx.top[4+i] = int(recon[topY*reconStride+(yuv.mbW-1)])
				}
			} else {
				// First MB row: replicate last top pixel (127) or last actual top pixel
				// The decoder uses 0x7f=127 for y=-1 (above the top row).
				ctx.top[4+i] = 127
			}
		} else {
			ctx.top[4+i] = get(x, bpy-1)
		}
	}

	return ctx
}

// i16ModeBitCost returns a rough bit cost for an intra16 mode.
// Based on the entropy coding tree in partition 0.
// DC=2 bits, VE=2, HE=2, TM=2 (approximate; actual depends on prob).
func i16ModeBitCost(mode int) int64 {
	// kVP8FixedCostsI16 from libwebp (in bits * 256):
	// These are rough bit costs for each mode.
	// DC_PRED (0): ~256, TM_PRED (1): ~512, V_PRED (2): ~384, H_PRED (3): ~512
	switch mode {
	case I16_DC_PRED:
		return 1
	case I16_VE_PRED:
		return 2
	case I16_HE_PRED:
		return 2
	case I16_TM_PRED:
		return 2
	}
	return 2
}

// i4ModeBitCost returns an approximate bit cost for an intra4 mode
// given the neighboring top and left modes.
// Based on kBModesProba probabilities.
func i4ModeBitCost(mode, topPred, leftPred int) int64 {
	// Approximate: count bits in the probability tree.
	// This is a simplified entropy estimate.
	prob := &kBModesProba[topPred][leftPred]

	bits := int64(0)
	// Bit 0: mode != B_DC_PRED
	if mode != B_DC_PRED {
		bits += probToBits(255 - int(prob[0]))
		// Bit 1: mode != B_TM_PRED
		if mode != B_TM_PRED {
			bits += probToBits(255 - int(prob[1]))
			// Bit 2: mode != B_VE_PRED
			if mode != B_VE_PRED {
				bits += probToBits(255 - int(prob[2]))
				if mode >= B_LD_PRED {
					bits += probToBits(int(prob[3]))
					if mode != B_LD_PRED {
						bits += probToBits(255 - int(prob[6]))
						if mode != B_VL_PRED {
							bits += probToBits(255 - int(prob[7]))
							// HD or HU
							bits += 1
						}
					}
				} else {
					bits += probToBits(255 - int(prob[3]))
					if mode != B_HE_PRED {
						bits += probToBits(255 - int(prob[4]))
						// RD or VR
						bits += 1
					}
				}
			}
		}
	} else {
		bits += probToBits(int(prob[0]))
	}
	return bits
}

// probToBits converts a VP8 probability (0..255) to a rough bit cost (0 or 1).
// Full entropy is -log2(p/256) but we use a table approximation.
func probToBits(prob int) int64 {
	// Rough: if prob > 128, costs 0 bits effectively; else costs 1 bit.
	// Better: use actual entropy ~-log2(prob/256).
	if prob >= 200 {
		return 0
	}
	if prob >= 128 {
		return 1
	}
	return 2
}

// computeDCY computes the DC prediction value for a 16x16 luma macroblock.
func computeDCY(yuv *yuvImage, mbX, mbY int) int {
	hasTop := mbY > 0
	hasLeft := mbX > 0
	px := mbX * 16
	py := mbY * 16

	if !hasTop && !hasLeft {
		return 128
	}

	dc := 0
	if hasTop {
		for i := 0; i < 16; i++ {
			x := px + i
			if x >= yuv.width {
				x = yuv.width - 1
			}
			dc += int(yuv.y[(py-1)*yuv.yStride+x])
		}
		if hasLeft {
			for i := 0; i < 16; i++ {
				y := py + i
				if y >= yuv.height {
					y = yuv.height - 1
				}
				dc += int(yuv.y[y*yuv.yStride+(px-1)])
			}
			return (dc + 16) >> 5
		}
		return (dc*2 + 16) >> 5
	}
	for i := 0; i < 16; i++ {
		y := py + i
		if y >= yuv.height {
			y = yuv.height - 1
		}
		dc += int(yuv.y[y*yuv.yStride+(px-1)])
	}
	return (dc*2 + 16) >> 5
}

// computeDCUV computes the DC prediction for an 8x8 chroma macroblock.
func computeDCUV(plane []uint8, stride, mbX, mbY, imgW, imgH int) int {
	hasTop := mbY > 0
	hasLeft := mbX > 0
	px := mbX * 8
	py := mbY * 8
	uvW := (imgW + 1) / 2
	uvH := (imgH + 1) / 2

	if !hasTop && !hasLeft {
		return 128
	}

	dc := 0
	if hasTop {
		for i := 0; i < 8; i++ {
			x := px + i
			if x >= uvW {
				x = uvW - 1
			}
			dc += int(plane[(py-1)*stride+x])
		}
		if hasLeft {
			for i := 0; i < 8; i++ {
				y := py + i
				if y >= uvH {
					y = uvH - 1
				}
				dc += int(plane[y*stride+(px-1)])
			}
			return (dc + 8) >> 4
		}
		return (dc*2 + 8) >> 4
	}
	for i := 0; i < 8; i++ {
		y := py + i
		if y >= uvH {
			y = uvH - 1
		}
		dc += int(plane[y*stride+(px-1)])
	}
	return (dc*2 + 8) >> 4
}

// extractBlock4x4Y extracts a 4x4 luma block starting at (px, py) into flat[16].
func extractBlock4x4Y(yuv *yuvImage, px, py int, flat []int16) {
	for dy := 0; dy < 4; dy++ {
		for dx := 0; dx < 4; dx++ {
			x := px + dx
			y := py + dy
			if x >= yuv.width {
				x = yuv.width - 1
			}
			if y >= yuv.height {
				y = yuv.height - 1
			}
			flat[dy*4+dx] = int16(yuv.y[y*yuv.yStride+x])
		}
	}
}

// extractBlock4x4UV extracts a 4x4 chroma block.
func extractBlock4x4UV(plane []uint8, stride, px, py, imgW, imgH int, flat []int16) {
	uvW := (imgW + 1) / 2
	uvH := (imgH + 1) / 2
	for dy := 0; dy < 4; dy++ {
		for dx := 0; dx < 4; dx++ {
			x := px + dx
			y := py + dy
			if x >= uvW {
				x = uvW - 1
			}
			if y >= uvH {
				y = uvH - 1
			}
			flat[dy*4+dx] = int16(plane[y*stride+x])
		}
	}
}

// fillPred4x4 fills a 4x4 prediction block with a constant value.
func fillPred4x4(pred []int16, val uint8) {
	for i := range pred {
		pred[i] = int16(val)
	}
}

// sad4x4 computes the Sum of Absolute Differences between a 4×4 source
// and prediction block. Used for cheap mode pre-screening before DCT.
func sad4x4(src, pred []int16) int64 {
	var s int64
	for i := 0; i < 16; i++ {
		d := int64(src[i]) - int64(pred[i])
		if d < 0 {
			d = -d
		}
		s += d
	}
	return s
}
