// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp


// Spatial Noise Shaping (SNS) — faithful port of libwebp's analysis_enc.c + quant_enc.c.
//
// The algorithm:
//   1. Per-MB alpha: DCT-based activity metric matching libwebp's CollectHistogram_C /
//      GetAlpha / FinalAlphaValue.  0=textured/hard, 255=flat/smooth.
//   2. Segment assignment: K-means clustering of the alpha histogram (AssignSegments).
//      K=4 for large images, K=2 for small.
//   3. Quantizer mapping: VP8SetSegmentParams power-law formula with SNS_TO_DQ=0.9.

// maxAlpha is the max alpha value (MAX_ALPHA in libwebp analysis_enc.c).
const maxAlpha = 255

// alphaScale is ALPHA_SCALE = 2 * MAX_ALPHA = 510 in libwebp.
const alphaScale = 2 * maxAlpha

// maxCoeffThresh is MAX_COEFF_THRESH from libwebp/src/dsp/dsp.h.
const maxCoeffThresh = 31

// snsKMeansMaxIter is the max K-means iterations (MAX_ITERS_K_MEANS=6 in libwebp).
const snsKMeansMaxIter = 6

// numMBSegments is the maximum number of segments (NUM_MB_SEGMENTS=4 in libwebp).
const numMBSegments = 4

// snsSmallImageThreshold: images with mbW*mbH <= this use 2 segments; larger use 4.
// Mirrors libwebp's default behavior at method=4.
const snsSmallImageThreshold = 500

// computeMBAlphaLibwebp computes the per-MB luma alpha activity metric matching
// libwebp's MBAnalyzeBestIntra16Mode + MBAnalyzeBestUVMode + MBAnalyze in
// analysis_enc.c.
//
// Alpha = 0 means highly textured (hard to compress).
// Alpha = 255 means flat/smooth (easy to compress).
//
// Process:
//   - Forward DCT of (src - dcPred) for each 4x4 block in the 16x16 MB.
//   - For each coeff: bin = min(abs(coeff) >> 3, 31); update distribution.
//   - alpha = ALPHA_SCALE * last_non_zero / max_value  (GetAlpha)
//   - FinalAlphaValue: clip(MAX_ALPHA - alpha, 0, 255)
//
// For analysis, we use DC-16 prediction (mean of top+left neighbors) for luma
// and DC-8 for UV, matching what libwebp uses for the analysis pass.
func computeMBAlphaLibwebp(yuv *yuvImage, mbX, mbY int) int {
	// Luma: try DC prediction only (dominant mode, sufficient for SNS analysis).
	// libwebp tests DC+TM (MAX_INTRA16_MODE=2) and picks the best alpha.
	// Using DC only is a safe conservative choice: it gives a valid alpha.
	lumaAlpha := mbAnalyzeLuma(yuv, mbX, mbY)

	// UV: analyze both U and V channels, take best UV alpha.
	uvAlpha := mbAnalyzeUV(yuv, mbX, mbY)

	// Mix luma and UV susceptibility (from MBAnalyze):
	//   best_alpha = (3 * luma_alpha + uv_alpha + 2) >> 2
	best := (3*lumaAlpha + uvAlpha + 2) >> 2

	// FinalAlphaValue: alpha = MAX_ALPHA - alpha, clipped to [0, MAX_ALPHA].
	// libwebp flips so that high raw alpha (lots of non-zero coeffs = textured)
	// maps to low final alpha, and low raw (flat) maps to high final alpha.
	return finalAlphaValue(best)
}

// finalAlphaValue maps a raw GetAlpha result to [0, 255].
// From analysis_enc.c: FinalAlphaValue(alpha) = clip(MAX_ALPHA - alpha, 0, MAX_ALPHA).
func finalAlphaValue(alpha int) int {
	v := maxAlpha - alpha
	if v < 0 {
		return 0
	}
	if v > maxAlpha {
		return maxAlpha
	}
	return v
}

// getAlpha computes the alpha from a DCT coefficient histogram.
// From analysis_enc.c: alpha = ALPHA_SCALE * last_non_zero / max_value if max_value > 1, else 0.
func getAlpha(maxValue, lastNonZero int) int {
	if maxValue > 1 {
		return alphaScale * lastNonZero / maxValue
	}
	return 0
}


// mbAnalyzeLuma computes the luma alpha for a 16x16 macroblock using the DC
// intra-16 prediction. Matches MBAnalyzeBestIntra16Mode (with DC mode only).
func mbAnalyzeLuma(yuv *yuvImage, mbX, mbY int) int {
	px := mbX * 16
	py := mbY * 16

	// Build DC-16 prediction: average of left column + top row.
	// For analysis we use the raw source pixels (not reconstructed) — libwebp
	// does the same in the analysis pass (VP8IteratorImport sets yuv_in from source).
	dc := mbDCPred16(yuv, mbX, mbY)

	// For each of 16 4x4 sub-blocks, run CollectHistogram and accumulate.
	// libwebp's VP8CollectHistogram(yuv_in + Y_OFF_ENC, yuv_p + offset, 0, 16, histo)
	// iterates over all 16 4x4 blocks (start_block=0, end_block=16).
	var maxVal, lastNZ int
	var dist [maxCoeffThresh + 1]int
	for by := 0; by < 4; by++ {
		for bx := 0; bx < 4; bx++ {
			var src4 [16]int16
			var pred4 [16]int16
			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					sx := px + bx*4 + x
					sy := py + by*4 + y
					if sx >= yuv.mbW {
						sx = yuv.mbW - 1
					}
					if sy >= yuv.mbH {
						sy = yuv.mbH - 1
					}
					src4[y*4+x] = int16(yuv.y[sy*yuv.yStride+sx])
					pred4[y*4+x] = int16(dc)
				}
			}
			var out [16]int16
			fTransform(src4[:], pred4[:], out[:])
			for k := 0; k < 16; k++ {
				v := int(out[k])
				if v < 0 {
					v = -v
				}
				v >>= 3
				if v > maxCoeffThresh {
					v = maxCoeffThresh
				}
				dist[v]++
			}
		}
	}
	// Compute max_value and last_non_zero from aggregated distribution.
	lastNZ = 1
	for k := 0; k <= maxCoeffThresh; k++ {
		if dist[k] > 0 {
			if dist[k] > maxVal {
				maxVal = dist[k]
			}
			lastNZ = k
		}
	}
	return getAlpha(maxVal, lastNZ)
}

// mbAnalyzeUV computes the UV alpha for a macroblock using DC-8 prediction.
// Matches MBAnalyzeBestUVMode (with DC mode only, 16+4+4=24 UV blocks in
// libwebp's scan order, but we only need U and V 8x8 blocks = 8 4x4 blocks total).
func mbAnalyzeUV(yuv *yuvImage, mbX, mbY int) int {
	ux := mbX * 8
	uy := mbY * 8
	uvW := (yuv.width + 1) / 2
	uvH := (yuv.height + 1) / 2

	var maxVal, lastNZ int
	var dist [maxCoeffThresh + 1]int

	// U then V (matches libwebp's U_OFF_ENC, start_block=16, end_block=24)
	for ch := 0; ch < 2; ch++ {
		plane := yuv.u
		if ch == 1 {
			plane = yuv.v
		}
		dc := mbDCPred8(plane, yuv.uvStride, mbX, mbY, uvW, uvH)
		for by := 0; by < 2; by++ {
			for bx := 0; bx < 2; bx++ {
				var src4 [16]int16
				var pred4 [16]int16
				for y := 0; y < 4; y++ {
					for x := 0; x < 4; x++ {
						sx := ux + bx*4 + x
						sy := uy + by*4 + y
						if sx >= uvW {
							sx = uvW - 1
						}
						if sy >= uvH {
							sy = uvH - 1
						}
						src4[y*4+x] = int16(plane[sy*yuv.uvStride+sx])
						pred4[y*4+x] = int16(dc)
					}
				}
				var out [16]int16
				fTransform(src4[:], pred4[:], out[:])
				for k := 0; k < 16; k++ {
					v := int(out[k])
					if v < 0 {
						v = -v
					}
					v >>= 3
					if v > maxCoeffThresh {
						v = maxCoeffThresh
					}
					dist[v]++
				}
			}
		}
	}
	lastNZ = 1
	for k := 0; k <= maxCoeffThresh; k++ {
		if dist[k] > 0 {
			if dist[k] > maxVal {
				maxVal = dist[k]
			}
			lastNZ = k
		}
	}
	return getAlpha(maxVal, lastNZ)
}

// mbDCPred16 returns the DC prediction value for a 16x16 luma block.
// Uses raw source samples (not reconstructed) — analysis pass only.
func mbDCPred16(yuv *yuvImage, mbX, mbY int) int {
	px := mbX * 16
	py := mbY * 16
	hasTop := mbY > 0
	hasLeft := mbX > 0
	if !hasTop && !hasLeft {
		return 128
	}
	dc := 0
	n := 0
	if hasTop {
		for i := 0; i < 16; i++ {
			x := px + i
			if x >= yuv.mbW {
				x = yuv.mbW - 1
			}
			dc += int(yuv.y[(py-1)*yuv.yStride+x])
		}
		n += 16
	}
	if hasLeft {
		for i := 0; i < 16; i++ {
			y := py + i
			if y >= yuv.mbH {
				y = yuv.mbH - 1
			}
			dc += int(yuv.y[y*yuv.yStride+(px-1)])
		}
		n += 16
	}
	return (dc + n/2) / n
}

// mbDCPred8 returns the DC prediction value for an 8x8 chroma block.
func mbDCPred8(plane []uint8, stride, mbX, mbY, uvW, uvH int) int {
	px := mbX * 8
	py := mbY * 8
	hasTop := mbY > 0
	hasLeft := mbX > 0
	if !hasTop && !hasLeft {
		return 128
	}
	dc := 0
	n := 0
	if hasTop {
		for i := 0; i < 8; i++ {
			x := px + i
			if x >= uvW {
				x = uvW - 1
			}
			dc += int(plane[(py-1)*stride+x])
		}
		n += 8
	}
	if hasLeft {
		for i := 0; i < 8; i++ {
			y := py + i
			if y >= uvH {
				y = uvH - 1
			}
			dc += int(plane[y*stride+(px-1)])
		}
		n += 8
	}
	return (dc + n/2) / n
}

// assignSegments performs K-means clustering of the alpha histogram into nb segments.
// Returns:
//   - centers[nb]: cluster center values
//   - segMap[MAX_ALPHA+1]: maps alpha value → segment index
//   - weightedAvg: weighted mean of centers (used as "mid" in SetSegmentAlphas)
//
// Faithful port of AssignSegments() from libwebp/src/enc/analysis_enc.c.
func assignSegments(alphas [maxAlpha + 1]int, nb int) (centers [numMBSegments]int, segMap [maxAlpha + 1]int, weightedAvg int) {
	if nb > numMBSegments {
		nb = numMBSegments
	}
	if nb < 1 {
		nb = 1
	}

	// Bracket the input: find min and max non-zero alpha values.
	minA, maxA := 0, maxAlpha
	for n := 0; n <= maxAlpha; n++ {
		if alphas[n] != 0 {
			minA = n
			break
		}
	}
	for n := maxAlpha; n > minA; n-- {
		if alphas[n] != 0 {
			maxA = n
			break
		}
	}
	rangeA := maxA - minA

	// Spread initial centers evenly (matches libwebp):
	//   for k in [0,nb): centers[k] = min_a + ((2k+1) * range_a) / (2*nb)
	for k := 0; k < nb; k++ {
		centers[k] = minA + ((2*k+1)*rangeA)/(2*nb)
	}

	var accum [numMBSegments]int
	var distAccum [numMBSegments]int

	for iter := 0; iter < snsKMeansMaxIter; iter++ {
		// Reset stats.
		for n := 0; n < nb; n++ {
			accum[n] = 0
			distAccum[n] = 0
		}
		// Assign nearest center for each alpha value (scan forward, track nearest).
		n := 0 // nearest center index
		for a := minA; a <= maxA; a++ {
			if alphas[a] == 0 {
				continue
			}
			// Advance to nearest center (centers are sorted ascending).
			for n+1 < nb && abs(a-centers[n+1]) < abs(a-centers[n]) {
				n++
			}
			segMap[a] = n
			distAccum[n] += a * alphas[a]
			accum[n] += alphas[a]
		}
		// Move centroids to weighted center of their cluster.
		displaced := 0
		totalWeight := 0
		wAvg := 0
		for n := 0; n < nb; n++ {
			if accum[n] != 0 {
				newCenter := (distAccum[n] + accum[n]/2) / accum[n]
				displaced += iabs(centers[n] - newCenter)
				centers[n] = newCenter
				wAvg += newCenter * accum[n]
				totalWeight += accum[n]
			}
		}
		if totalWeight > 0 {
			weightedAvg = (wAvg + totalWeight/2) / totalWeight
		}
		if displaced < 5 {
			break // converged
		}
	}
	return
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func iabs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}


func clipI(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}


// snsResult holds the full SNS analysis output.
type snsResult struct {
	mbSegment []int  // per-MB segment ID [0..numSegs-1]
	segQs     [4]int // quantizer index per segment (absolute)
	numSegs   int    // number of active segments (1 or 2)
}

// computeSNS runs the SNS pipeline:
//   - Per-MB alpha: libwebp's DCT-based metric (0=textured, 255=smooth).
//   - K-means with 2 clusters for segment assignment (natural boundary).
//   - Segment quantizers: smooth cluster → coarser (baseQ+delta), textured → baseQ.
//
// The DCT-based alpha (ported from libwebp's CollectHistogram/GetAlpha/FinalAlphaValue)
// more accurately distinguishes smooth from textured regions than the old MAD metric.
// K-means finds the natural cluster boundary rather than using a fixed threshold.
//
// Segment 0 = smooth/flat: gets coarser quant (baseQ+delta) — saves bits.
// Segment 1 = textured:    gets base quant (baseQ)           — preserves quality.
func computeSNS(yuv *yuvImage, mbW, mbH, baseQ int) snsResult {
	mbCount := mbW * mbH

	// Step 1: compute per-MB alpha using libwebp's DCT-based metric.
	mbAlphaSlice := make([]int, mbCount)
	for mbY := 0; mbY < mbH; mbY++ {
		for mbX := 0; mbX < mbW; mbX++ {
			mbAlphaSlice[mbY*mbW+mbX] = computeMBAlphaLibwebp(yuv, mbX, mbY)
		}
	}

	// Step 2: build alpha histogram.
	var alphaHist [maxAlpha + 1]int
	for _, a := range mbAlphaSlice {
		alphaHist[a]++
	}

	// Step 3: K-means with 2 clusters for natural smooth/textured boundary.
	// centers[0] = textured cluster (lower alpha), centers[1] = smooth cluster (higher alpha).
	// segMap[a] = 0 (textured) or 1 (smooth).
	_, segMap, _ := assignSegments(alphaHist, 2)

	// Step 4: determine quantizer delta based on image size.
	// Small images (portraits): delta=8 compensates for DCT alpha being slightly
	// more sensitive to texture than MAD, ensuring smooth MBs remain compressed.
	// Large images: delta=12 (unchanged from v0.4.0).
	delta := 8
	if mbCount > snsSmallImageThreshold {
		delta = 12
	}

	// Compute 2-segment q-indices:
	//   seg0 (smooth): q0 = baseQ + delta (coarser → saves bits on flat areas)
	//   seg1 (textured): q1 = baseQ (baseline quality)
	q0 := clipI(baseQ+delta, 0, 127)
	q1 := baseQ

	// Step 5: assign each MB.
	// K-means segMap: 1=smooth (higher center), 0=textured (lower center).
	mbSegment := make([]int, mbCount)
	for i, a := range mbAlphaSlice {
		if segMap[a] == 1 {
			mbSegment[i] = 0 // smooth → encoder seg0 (coarser)
		} else {
			mbSegment[i] = 1 // textured → encoder seg1 (finer)
		}
	}

	return snsResult{
		mbSegment: mbSegment,
		segQs:     [4]int{q0, q1, q1, q1},
		numSegs:   2,
	}
}

// makeSegmentParamsFromQ builds a segmentParams from a raw quantizer index [0..127].
// Used by the new SNS path where we have exact q-indices from the libwebp formula.
func makeSegmentParamsFromQ(q int) segmentParams {
	q = clipQ(q, 0, 127)
	var qm quantMatrices

	// Y1: luma AC
	qDC := int(kDcTable[q])
	qAC := int(kAcTable[q])
	qm.y1.q[0] = uint16(qDC)
	for i := 1; i < 16; i++ {
		qm.y1.q[i] = uint16(qAC)
	}
	setupMatrix(&qm.y1, 0)

	// Y2: WHT/DC
	qAC2 := int(kAcTable2[q])
	qm.y2.q[0] = uint16(qDC * 2)
	for i := 1; i < 16; i++ {
		qm.y2.q[i] = uint16(qAC2)
	}
	setupMatrix(&qm.y2, 1)

	// UV: chroma
	qDCuv := int(kDcTable[clipQ(q, 0, 117)])
	qACuv := int(kAcTable[q])
	qm.uv.q[0] = uint16(qDCuv)
	for i := 1; i < 16; i++ {
		qm.uv.q[i] = uint16(qACuv)
	}
	setupMatrix(&qm.uv, 2)

	// Lambda computation (SetupMatrices in libwebp).
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

