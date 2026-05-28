// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// fillUVSrc8x8 extracts the 8×8 source U and V planes for the current MB into
// ws.uvSrcU / ws.uvSrcV. Called once per MB before the UV RD loop so the inner
// candidate loops can read the source from a tight, cache-friendly 64-byte
// buffer instead of striding into the much larger plane.
func fillUVSrc8x8(ws *mbWorkspace, yuv *yuvImage, mbX, mbY int) {
	uvStride := yuv.uvStride
	for j := 0; j < 8; j++ {
		row := (mbY*8 + j) * uvStride
		off := j * 8
		for i := 0; i < 8; i++ {
			ws.uvSrcU[off+i] = yuv.u[row+mbX*8+i]
			ws.uvSrcV[off+i] = yuv.v[row+mbX*8+i]
		}
	}
}

// uvCandidateScore quantizes a candidate UV mode's prediction with greedy
// quantization (no trellis), reconstructs the chroma blocks, and returns the
// rate-distortion components needed for RD-style UV mode selection.
//
// Mirrors the body of libwebp's PickBestUV inner loop (quant_enc.c lines 1162-1186).
// libwebp uses greedy quant for UV (DO_TRELLIS_UV=0 in quant_enc.c:28). We follow
// suit here so the candidate loop is cheap. The FINAL UV quantization is done
// once after mode selection using trellis (matches libwebp's call from
// SimpleQuantize → ReconstructUV, where the "real" pass is greedy by default
// but our gowebp keeps trellis on the final output for compression gains).
//
// `pred8U` and `pred8V` are 8×8 chroma predictions (int16, row-major stride 8)
// for the candidate's mode. These are *not* mutated.
//
// Outputs: ws.uvLevelsTmp (8 blocks of [16]int16, greedy-quantized levels),
// ws.uvReconTmpU/V (8×8 uint8, reconstructed pixels).
// The caller is responsible for committing them to *Best* on a winning candidate.
//
// `ctxTop{U,V}` / `ctxLeft{U,V}` are the per-block NZ context entries used to
// compute the entropy cost. They are NOT mutated by this function (passed by
// value as [2]int).
func uvCandidateScore(
	ws *mbWorkspace,
	yuv *yuvImage,
	mbX, mbY int,
	pred8U, pred8V []int16,
	qmUV *quantMatrix,
	trellisUVCosts *trellisCostTables,
	ctxTopU, ctxLeftU [2]int,
	ctxTopV, ctxLeftV [2]int,
) (distortion int64, rate int) {
	uvStride := yuv.uvStride
	mbPX := mbX * 8
	mbPY := mbY * 8

	for ch := 0; ch < 2; ch++ {
		var (
			plane    []uint8
			pred     []int16
			src      []uint8
			reconOut []uint8
		)
		if ch == 0 {
			plane = yuv.u
			pred = pred8U
			src = ws.uvSrcU[:]
			reconOut = ws.uvReconTmpU[:]
		} else {
			plane = yuv.v
			pred = pred8V
			src = ws.uvSrcV[:]
			reconOut = ws.uvReconTmpV[:]
		}
		for by := 0; by < 2; by++ {
			bn0 := ch*4 + by*2
			bn1 := ch*4 + by*2 + 1

			// DCT pair: both bx=0 and bx=1 4×4 sub-blocks at once.
			fTransform2Plane(plane, uvStride, mbPX, mbPY+by*4,
				pred[by*4*8:], 8, ws.dctPair[:])

			// Greedy quantize each half of the DCT pair separately (libwebp
			// DO_TRELLIS_UV=0). Trellis would be more accurate but is too slow
			// for a per-candidate inner loop.
			quantizeBlock(ws.dctPair[0:16], ws.uvLevelsTmp[bn0][:], qmUV, 0)
			quantizeBlock(ws.dctPair[16:32], ws.uvLevelsTmp[bn1][:], qmUV, 0)

			// Reconstruct each 4×4 block: dequantize → iDCT(+pred) → write into
			// the local 8×8 recon plane, accumulating SSE.
			for bx := 0; bx < 2; bx++ {
				bn := ch*4 + by*2 + bx
				for y := 0; y < 4; y++ {
					predRow := (by*4 + y) * 8
					ws.uvPred4[y*4+0] = pred[predRow+bx*4+0]
					ws.uvPred4[y*4+1] = pred[predRow+bx*4+1]
					ws.uvPred4[y*4+2] = pred[predRow+bx*4+2]
					ws.uvPred4[y*4+3] = pred[predRow+bx*4+3]
				}
				for n := 0; n < 16; n++ {
					j := int(kZigzag[n])
					ws.uvRaster[j] = int16(int32(ws.uvLevelsTmp[bn][n]) * int32(qmUV.q[j]))
				}
				iTransform4x4(ws.uvRaster[:], ws.uvPred4[:], ws.uvRecBlock[:])
				for y := 0; y < 4; y++ {
					reconRow := (by*4 + y) * 8
					recRow := y * 4
					for x := 0; x < 4; x++ {
						r := uint8(ws.uvRecBlock[recRow+x])
						reconOut[reconRow+bx*4+x] = r
						s := int64(src[reconRow+bx*4+x])
						d := s - int64(r)
						distortion += d * d
					}
				}
			}
		}
	}

	// Rate: sum coeffBitCost over all 8 blocks, with NZ context tracking that
	// mirrors libwebp's VP8GetCostUV (cost_enc.c:263). U and V scans use
	// independent NZ context (top_nz[4+ch+x], left_nz[4+ch+y]).
	rate = 0
	{
		topNz := ctxTopU
		leftNz := ctxLeftU
		for by := 0; by < 2; by++ {
			for bx := 0; bx < 2; bx++ {
				n := by*2 + bx
				ctx := topNz[bx] + leftNz[by]
				if ctx > 2 {
					ctx = 2
				}
				rate += coeffBitCost(ctx, ws.uvLevelsTmp[n][:], 0, trellisUVCosts)
				nz := 0
				if findLast(ws.uvLevelsTmp[n][:], 0) >= 0 {
					nz = 1
				}
				topNz[bx] = nz
				leftNz[by] = nz
			}
		}
	}
	{
		topNz := ctxTopV
		leftNz := ctxLeftV
		for by := 0; by < 2; by++ {
			for bx := 0; bx < 2; bx++ {
				n := 4 + by*2 + bx
				ctx := topNz[bx] + leftNz[by]
				if ctx > 2 {
					ctx = 2
				}
				rate += coeffBitCost(ctx, ws.uvLevelsTmp[n][:], 0, trellisUVCosts)
				nz := 0
				if findLast(ws.uvLevelsTmp[n][:], 0) >= 0 {
					nz = 1
				}
				topNz[bx] = nz
				leftNz[by] = nz
			}
		}
	}

	return distortion, rate
}

// quantizeUVFinal performs the trellis-quantized UV pass for the chosen mode,
// writes the levels to ws.uvLevels, and reconstructs into ws.uvReconBestU/V.
//
// This is the gowebp post-selection step: even though libwebp uses greedy
// quant for UV (DO_TRELLIS_UV=0), gowebp keeps trellis on the final output for
// the compression gain. The mode SELECTION uses greedy quant (uvCandidateScore)
// for speed, then this function emits the trellis-quant bitstream.
func quantizeUVFinal(
	ws *mbWorkspace,
	yuv *yuvImage,
	mbX, mbY int,
	pred8U, pred8V []int16,
	qmUV *quantMatrix,
	lambdaTrellisUV int,
	trellisUVCosts *trellisCostTables,
	uvProbsPtr *[numBands][numCtx][numProbas]uint8,
) {
	uvStride := yuv.uvStride
	mbPX := mbX * 8
	mbPY := mbY * 8

	for ch := 0; ch < 2; ch++ {
		var (
			plane    []uint8
			pred     []int16
			reconOut []uint8
		)
		if ch == 0 {
			plane = yuv.u
			pred = pred8U
			reconOut = ws.uvReconBestU[:]
		} else {
			plane = yuv.v
			pred = pred8V
			reconOut = ws.uvReconBestV[:]
		}

		for by := 0; by < 2; by++ {
			bn0 := ch*4 + by*2
			bn1 := ch*4 + by*2 + 1

			fTransform2Plane(plane, uvStride, mbPX, mbPY+by*4,
				pred[by*4*8:], 8, ws.dctPair[:])

			trellisQuantize(ws.dctPair[0:16], ws.uvLevels[bn0][:], qmUV, 0,
				lambdaTrellisUV, trellisUVCosts, uvProbsPtr, 0)
			trellisQuantize(ws.dctPair[16:32], ws.uvLevels[bn1][:], qmUV, 0,
				lambdaTrellisUV, trellisUVCosts, uvProbsPtr, 0)

			for bx := 0; bx < 2; bx++ {
				bn := ch*4 + by*2 + bx
				for y := 0; y < 4; y++ {
					predRow := (by*4 + y) * 8
					ws.uvPred4[y*4+0] = pred[predRow+bx*4+0]
					ws.uvPred4[y*4+1] = pred[predRow+bx*4+1]
					ws.uvPred4[y*4+2] = pred[predRow+bx*4+2]
					ws.uvPred4[y*4+3] = pred[predRow+bx*4+3]
				}
				for n := 0; n < 16; n++ {
					j := int(kZigzag[n])
					ws.uvRaster[j] = int16(int32(ws.uvLevels[bn][n]) * int32(qmUV.q[j]))
				}
				iTransform4x4(ws.uvRaster[:], ws.uvPred4[:], ws.uvRecBlock[:])
				for y := 0; y < 4; y++ {
					reconRow := (by*4 + y) * 8
					recRow := y * 4
					for x := 0; x < 4; x++ {
						reconOut[reconRow+bx*4+x] = uint8(ws.uvRecBlock[recRow+x])
					}
				}
			}
		}
	}
}
