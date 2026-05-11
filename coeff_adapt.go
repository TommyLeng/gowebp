// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// mbCoeffData stores all quantized coefficient levels for one macroblock,
// after the RD selection pass. These are used for the two-pass coefficient
// probability adaptation: first pass collects stats, second pass emits bits.
type mbCoeffData struct {
	isI4 bool

	// i4 path: 16 blocks × 16 coefficients each
	i4AC [16][16]int16

	// i16 path:
	i16DC [16]int16     // WHT-quantized DC levels (zigzag)
	i16AC [16][16]int16 // AC levels per 4x4 block (zigzag, DC slot=0)

	// UV: 4 U-blocks + 4 V-blocks, each 16 coefficients
	uv [8][16]int16
}

// coeffStats accumulates binary decision counts for coefficient probability
// adaptation. Indexed [type][band][ctx][prob_index][0=count0, 1=count1].
type coeffStats [numTypes][numBands][numCtx][numProbas][2]int32

// recordCoeffs simulates putCoeffs but accumulates binary decision counts
// into stats instead of writing bits. Mirrors VP8RecordCoeffs in libwebp.
// Returns true if any non-zero coefficient was found.
func recordCoeffs(stats *coeffStats, ctx int, coeffs []int16, coeffType int, first int, last int) bool {
	n := first
	t := coeffType
	b := int(vp8EncBands[n])
	c := ctx

	// record a bit decision and return the bit value
	rec := func(bit int, pp int) {
		stats[t][b][c][pp][bit]++
	}

	// Is there any non-zero coeff?
	if last < 0 {
		rec(0, 0)
		return false
	}
	rec(1, 0)

	for n < 16 {
		cv := int(coeffs[n])
		n++
		v := cv
		if cv < 0 {
			v = -cv
		}

		if v == 0 {
			rec(0, 1)
			b = int(vp8EncBands[n])
			c = 0
			continue
		}
		rec(1, 1)

		if v == 1 {
			rec(0, 2)
			b = int(vp8EncBands[n])
			c = 1
		} else {
			rec(1, 2)

			if v <= 4 {
				rec(0, 3)
				if v == 2 {
					rec(0, 4)
				} else {
					rec(1, 4)
					if v == 4 {
						rec(1, 5)
					} else {
						rec(0, 5)
					}
				}
			} else if v <= 10 {
				rec(1, 3)
				rec(0, 6)
				if v <= 6 {
					rec(0, 7)
					// constant probs (159 for eq6): not tracked
				} else {
					rec(1, 7)
					// constant probs (165, 145): not tracked
				}
			} else {
				rec(1, 3)
				rec(1, 6)
				residue := v - 3
				if residue < (8 << 1) {
					rec(0, 8)
					rec(0, 9)
				} else if residue < (8 << 2) {
					rec(0, 8)
					rec(1, 9)
				} else if residue < (8 << 3) {
					rec(1, 8)
					rec(0, 10)
				} else {
					rec(1, 8)
					rec(1, 10)
				}
			}

			b = int(vp8EncBands[n])
			c = 2
		}

		// sign is a constant-prob bit (128), not tracked

		if n == 16 {
			return true
		}
		if n <= last {
			rec(1, 0)
		} else {
			rec(0, 0)
			return true
		}
	}
	return true
}

// vp8EntropyCost[p] ≈ -log2(p/256) * 256 for p in [1..255], 0 for p<=0 or p>=256.
// Mirrors kVP8EntropyCost in libwebp/src/enc/cost_enc.c.
// Generated from: round(-log2(p/256.0) * 256) for p in 1..255.
var vp8EntropyCost = [256]int{
	0,    // p=0: unused (clamped)
	2048, // p=1
	1792, // p=2
	1651, // p=3
	1536, // p=4
	1449, // p=5
	1373, // p=6
	1312, // p=7
	1280, // p=8 — note: libwebp has exact values; these are approximations
	1220, // p=9
	1171, // p=10
	1128, // p=11
	1096, // p=12
	1068, // p=13
	1014, // p=14
	980,  // p=15
	949,  // p=16
	910,  // p=17
	870,  // p=18
	840,  // p=19
	801,  // p=20
	774,  // p=21
	746,  // p=22
	720,  // p=23
	694,  // p=24
	668,  // p=25
	642,  // p=26
	616,  // p=27
	590,  // p=28
	564,  // p=29
	538,  // p=30
	512,  // p=31
	487,  // p=32
	461,  // p=33
	449,  // p=34
	436,  // p=35
	413,  // p=36
	390,  // p=37
	368,  // p=38
	345,  // p=39
	323,  // p=40
	300,  // p=41
	287,  // p=42
	266,  // p=43
	254,  // p=44
	231,  // p=45
	209,  // p=46
	186,  // p=47
	164,  // p=48
	141,  // p=49
	128,  // p=50
	116,  // p=51
	103,  // p=52
	90,   // p=53
	78,   // p=54
	65,   // p=55
	52,   // p=56
	40,   // p=57
	27,   // p=58
	14,   // p=59
	// p=60..255: cost rounds to 0 (outcome is very likely)
	0, 0, 0, 0, // 60..63
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 64..79
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 80..95
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 96..111
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 112..127
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 128..143
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 144..159
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 160..175
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 176..191
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 192..207
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 208..223
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 224..239
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 240..255
}

// vp8BitCost returns the cost in bits×256 of coding 'bit' with probability 'prob'.
// Mirrors VP8BitCost() in libwebp.
func vp8BitCost(bit int, prob int) int {
	if prob <= 0 {
		prob = 1
	}
	if prob >= 255 {
		prob = 254
	}
	var p int
	if bit == 0 {
		p = prob
	} else {
		p = 255 - prob
	}
	return vp8EntropyCost[p]
}

// calcTokenProba computes the new probability from observation counts.
// Mirrors CalcTokenProba() in libwebp/src/enc/frame_enc.c:
//   255 - nb*255/total, where nb = count of 1-decisions.
func calcTokenProba(nb, total int) int {
	if total == 0 || nb == 0 {
		return 255
	}
	p := 255 - nb*255/total
	if p < 1 {
		p = 1
	}
	return p
}

// branchCost returns the total bit-cost (in bits×256) for nb ones and
// (total-nb) zeros coded with probability prob.
// Mirrors BranchCost() in libwebp/src/enc/frame_enc.c.
func branchCost(nb, total, prob int) int {
	return nb*vp8BitCost(1, prob) + (total-nb)*vp8BitCost(0, prob)
}

// adaptedCoeffProbs is the coefficient probability table after adaptation.
type adaptedCoeffProbs [numTypes][numBands][numCtx][numProbas]uint8

// finalizeTokenProbas computes adapted coefficient probabilities from accumulated
// statistics. Mirrors FinalizeTokenProbas() in libwebp/src/enc/frame_enc.c.
//
// Returns (probs, updated) where:
//   probs[t][b][c][p] = chosen probability (new_p if cost-beneficial, else old_p)
//   updated[t][b][c][p] = true if we use new_p (will signal update in bitstream)
func finalizeTokenProbas(stats *coeffStats) (adaptedCoeffProbs, [numTypes][numBands][numCtx][numProbas]bool) {
	var probs adaptedCoeffProbs
	var updated [numTypes][numBands][numCtx][numProbas]bool

	for t := 0; t < numTypes; t++ {
		for b := 0; b < numBands; b++ {
			for c := 0; c < numCtx; c++ {
				for p := 0; p < numProbas; p++ {
					count0 := int(stats[t][b][c][p][0])
					count1 := int(stats[t][b][c][p][1])
					total := count0 + count1
					nb := count1 // count of "1" decisions

					updateProba := int(coeffsUpdateProba[t][b][c][p])
					oldP := int(defaultCoeffProbs[t][b][c][p])
					newP := calcTokenProba(nb, total)

					// Cost of NOT updating: keep old_p, plus cost of signaling "no update"
					oldCost := branchCost(nb, total, oldP) + vp8BitCost(0, updateProba)
					// Cost of updating: use new_p, plus cost of signaling "update" + 8 bits for new value
					newCost := branchCost(nb, total, newP) + vp8BitCost(1, updateProba) + 8*256

					if newCost < oldCost {
						probs[t][b][c][p] = uint8(newP)
						updated[t][b][c][p] = true
					} else {
						probs[t][b][c][p] = uint8(oldP)
					}
				}
			}
		}
	}
	return probs, updated
}

// putCoeffsWithProbs is identical to putCoeffs but uses an adapted probability
// table instead of the global defaultCoeffProbs. This is the second-pass entropy
// coding using updated probabilities.
func putCoeffsWithProbs(bw *boolEncoder, ctx int, coeffs []int16, coeffType int, first int, last int, probs *adaptedCoeffProbs) bool {
	n := first
	p := &probs[coeffType][n][ctx]

	// Emit: is there any non-zero coeff? (eob_prob)
	if last < 0 {
		bw.putBit(0, int(p[0]))
		return false
	}
	bw.putBit(1, int(p[0]))

	for n < 16 {
		c := int(coeffs[n])
		n++
		sign := 0
		v := c
		if c < 0 {
			sign = 1
			v = -c
		}

		// Emit: is this coeff non-zero?
		if v == 0 {
			bw.putBit(0, int(p[1]))
			band := int(vp8EncBands[n])
			p = &probs[coeffType][band][0]
			continue
		}
		bw.putBit(1, int(p[1]))

		// Emit: is v > 1?
		if v == 1 {
			bw.putBit(0, int(p[2]))
			band := int(vp8EncBands[n])
			p = &probs[coeffType][band][1]
		} else {
			bw.putBit(1, int(p[2]))

			// Emit: is v > 4?
			if v <= 4 {
				bw.putBit(0, int(p[3]))
				// Emit: is v != 2? (i.e., 3 or 4)
				if v == 2 {
					bw.putBit(0, int(p[4]))
				} else {
					bw.putBit(1, int(p[4]))
					// Emit: is v == 4?
					eq4 := 0
					if v == 4 {
						eq4 = 1
					}
					bw.putBit(eq4, int(p[5]))
				}
			} else if v <= 10 {
				bw.putBit(1, int(p[3]))
				bw.putBit(0, int(p[6]))
				// Emit: is v > 6?
				if v <= 6 {
					bw.putBit(0, int(p[7]))
					eq6 := 0
					if v == 6 {
						eq6 = 1
					}
					bw.putBit(eq6, 159)
				} else {
					bw.putBit(1, int(p[7]))
					ge9 := 0
					if v >= 9 {
						ge9 = 1
					}
					bw.putBit(ge9, 165)
					even := 0
					if v&1 == 0 {
						even = 1
					}
					bw.putBit(even, 145)
				}
			} else {
				bw.putBit(1, int(p[3]))
				bw.putBit(1, int(p[6]))
				// Large value: encode using cat tables
				residue := v - 3
				if residue < (8 << 1) { // Cat3
					bw.putBit(0, int(p[8]))
					bw.putBit(0, int(p[9]))
					residue -= 8
					for i := 2; i >= 0; i-- {
						bit := (residue >> uint(i)) & 1
						bw.putBit(bit, int(vp8Cat3[2-i]))
					}
				} else if residue < (8 << 2) { // Cat4
					bw.putBit(0, int(p[8]))
					bw.putBit(1, int(p[9]))
					residue -= 8 << 1
					for i := 3; i >= 0; i-- {
						bit := (residue >> uint(i)) & 1
						bw.putBit(bit, int(vp8Cat4[3-i]))
					}
				} else if residue < (8 << 3) { // Cat5
					bw.putBit(1, int(p[8]))
					bw.putBit(0, int(p[10]))
					residue -= 8 << 2
					for i := 4; i >= 0; i-- {
						bit := (residue >> uint(i)) & 1
						bw.putBit(bit, int(vp8Cat5[4-i]))
					}
				} else { // Cat6
					bw.putBit(1, int(p[8]))
					bw.putBit(1, int(p[10]))
					residue -= 8 << 3
					for i := 10; i >= 0; i-- {
						bit := (residue >> uint(i)) & 1
						bw.putBit(bit, int(vp8Cat6[10-i]))
					}
				}

				band := int(vp8EncBands[n])
				p = &probs[coeffType][band][2]
			}

			band := int(vp8EncBands[n])
			p = &probs[coeffType][band][2]
		}

		// Sign bit (uniform)
		bw.putBitUniform(sign)

		// Is there another non-zero coeff? (or EOB)
		if n == 16 {
			return true
		}
		more := 0
		if n <= last {
			more = 1
		}
		bw.putBit(more, int(p[0]))
		if more == 0 {
			return true
		}
	}
	return true
}

// collectCoeffStats iterates over all stored MB coefficient levels in raster
// order and calls recordCoeffs for each block, accumulating statistics.
// The NZ context tracking mirrors the original encoding order exactly.
func collectCoeffStats(mbCoeffs []mbCoeffData, mbW, mbH int, stats *coeffStats) {
	topNzY := make([]int, mbW*4+1)
	topNzU := make([]int, mbW*2+1)
	topNzV := make([]int, mbW*2+1)
	topNzDC := make([]int, mbW+1)

	for mbY := 0; mbY < mbH; mbY++ {
		leftNzY := [5]int{}
		leftNzU := [3]int{}
		leftNzV := [3]int{}

		for mbX := 0; mbX < mbW; mbX++ {
			cd := &mbCoeffs[mbY*mbW+mbX]

			if cd.isI4 {
				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						ctx := topNzY[mbX*4+bx] + leftNzY[by]
						last := findLast(cd.i4AC[n][:], 0)
						recordCoeffs(stats, ctx, cd.i4AC[n][:], 3, 0, last)
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
			} else {
				dcCtx := topNzDC[mbX] + leftNzY[4]
				lastDC := findLast(cd.i16DC[:], 0)
				recordCoeffs(stats, dcCtx, cd.i16DC[:], 1, 0, lastDC)
				dcNZ := 0
				if lastDC >= 0 {
					dcNZ = 1
				}
				topNzDC[mbX] = dcNZ
				leftNzY[4] = dcNZ

				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						ctx := topNzY[mbX*4+bx] + leftNzY[by]
						last := findLast(cd.i16AC[n][:], 1)
						recordCoeffs(stats, ctx, cd.i16AC[n][:], 0, 1, last)
						nz := 0
						if last >= 1 {
							nz = 1
						}
						topNzY[mbX*4+bx] = nz
						leftNzY[by] = nz
					}
				}
			}

			// U blocks
			for by := 0; by < 2; by++ {
				for bx := 0; bx < 2; bx++ {
					n := by*2 + bx
					ctx := topNzU[mbX*2+bx] + leftNzU[by]
					last := findLast(cd.uv[n][:], 0)
					recordCoeffs(stats, ctx, cd.uv[n][:], 2, 0, last)
					nz := 0
					if last >= 0 {
						nz = 1
					}
					topNzU[mbX*2+bx] = nz
					leftNzU[by] = nz
				}
			}
			// V blocks
			for by := 0; by < 2; by++ {
				for bx := 0; bx < 2; bx++ {
					n := by*2 + bx
					ctx := topNzV[mbX*2+bx] + leftNzV[by]
					last := findLast(cd.uv[4+n][:], 0)
					recordCoeffs(stats, ctx, cd.uv[4+n][:], 2, 0, last)
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
}

// encodeTokenPartition entropy-encodes all MB coefficient levels using the
// supplied adapted probability table. Mirrors the encoding order in encodeFrame.
func encodeTokenPartition(bw *boolEncoder, mbCoeffs []mbCoeffData, mbW, mbH int, probs *adaptedCoeffProbs) {
	topNzY := make([]int, mbW*4+1)
	topNzU := make([]int, mbW*2+1)
	topNzV := make([]int, mbW*2+1)
	topNzDC := make([]int, mbW+1)

	for mbY := 0; mbY < mbH; mbY++ {
		leftNzY := [5]int{}
		leftNzU := [3]int{}
		leftNzV := [3]int{}

		for mbX := 0; mbX < mbW; mbX++ {
			cd := &mbCoeffs[mbY*mbW+mbX]

			if cd.isI4 {
				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						ctx := topNzY[mbX*4+bx] + leftNzY[by]
						last := findLast(cd.i4AC[n][:], 0)
						putCoeffsWithProbs(bw, ctx, cd.i4AC[n][:], 3, 0, last, probs)
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
			} else {
				dcCtx := topNzDC[mbX] + leftNzY[4]
				lastDC := findLast(cd.i16DC[:], 0)
				putCoeffsWithProbs(bw, dcCtx, cd.i16DC[:], 1, 0, lastDC, probs)
				dcNZ := 0
				if lastDC >= 0 {
					dcNZ = 1
				}
				topNzDC[mbX] = dcNZ
				leftNzY[4] = dcNZ

				for by := 0; by < 4; by++ {
					for bx := 0; bx < 4; bx++ {
						n := by*4 + bx
						ctx := topNzY[mbX*4+bx] + leftNzY[by]
						last := findLast(cd.i16AC[n][:], 1)
						putCoeffsWithProbs(bw, ctx, cd.i16AC[n][:], 0, 1, last, probs)
						nz := 0
						if last >= 1 {
							nz = 1
						}
						topNzY[mbX*4+bx] = nz
						leftNzY[by] = nz
					}
				}
			}

			// U blocks
			for by := 0; by < 2; by++ {
				for bx := 0; bx < 2; bx++ {
					n := by*2 + bx
					ctx := topNzU[mbX*2+bx] + leftNzU[by]
					last := findLast(cd.uv[n][:], 0)
					putCoeffsWithProbs(bw, ctx, cd.uv[n][:], 2, 0, last, probs)
					nz := 0
					if last >= 0 {
						nz = 1
					}
					topNzU[mbX*2+bx] = nz
					leftNzU[by] = nz
				}
			}
			// V blocks
			for by := 0; by < 2; by++ {
				for bx := 0; bx < 2; bx++ {
					n := by*2 + bx
					ctx := topNzV[mbX*2+bx] + leftNzV[by]
					last := findLast(cd.uv[4+n][:], 0)
					putCoeffsWithProbs(bw, ctx, cd.uv[4+n][:], 2, 0, last, probs)
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
}

// putSegmentID encodes a 2-bit segment ID using the 3-node probability tree.
// Mirrors PutSegment() in libwebp/src/enc/tree_enc.c:
//   VP8PutBit(bw, s >= 2, p[0])
//   VP8PutBit(bw, s &  1, p[1])  if s < 2
//   VP8PutBit(bw, s &  1, p[2])  if s >= 2
//
// The default segment probabilities are all 255. The probability value matters
// for the VP8 range coder — encoder and decoder must use the same value, so we
// use putBit with the stored default (255), not putBitUniform (which uses 128).
func putSegmentID(bw *boolEncoder, s int, prob [3]uint8) {
	highBit := boolInt(s >= 2)
	bw.putBit(highBit, int(prob[0]))
	if highBit == 0 {
		bw.putBit(s&1, int(prob[1]))
	} else {
		bw.putBit(s&1, int(prob[2]))
	}
}

// encodePartition0WithProbs encodes partition 0 (frame headers + intra modes +
// coefficient probability updates) using the adapted probability table.
// Mirrors VP8WriteProbas() from libwebp/src/enc/tree_enc.c for the prob section.
//
// segQ0, segQ1: quantizer indices for SNS segments 0 (smooth) and 1 (textured).
// When segQ0 == segQ1, single-segment mode is used (no segment header overhead).
func encodePartition0WithProbs(bw *boolEncoder, mbW, mbH, segQ0, segQ1 int, infos []mbInfo,
	probs *adaptedCoeffProbs, updated *[numTypes][numBands][numCtx][numProbas]bool) {

	useSegments := segQ0 != segQ1

	// colorspace = 0, clamp_type = 0
	bw.putBitUniform(0)
	bw.putBitUniform(0)

	// Segment header (VP8 spec §9.3 / libwebp PutSegmentHeader in syntax_enc.c).
	// Bit order must match exactly: update_map, then update_data, then data, then probs.
	if useSegments {
		bw.putBitUniform(1) // update_mb_segmentation = 1

		// update_mb_segmentation_map (1 bit): whether per-MB segment IDs are signaled.
		bw.putBitUniform(1) // update_map = 1

		// update_segment_feature_data (1 bit): whether quantizer/filter tables follow.
		// If 1: segment_feature_mode + 4 quantizer values + 4 filter strength values.
		bw.putBitUniform(1) // update_data = 1

		// segment_feature_mode (1 bit): 1 = absolute values (not relative deltas).
		bw.putBitUniform(1)

		// Quantizer absolute values for 4 segments (7-bit signed each).
		// Unused segments (2, 3) replicate segment 1's value.
		bw.putSignedBits(segQ0, 7)
		bw.putSignedBits(segQ1, 7)
		bw.putSignedBits(segQ1, 7) // segment 2 — unused, same as 1
		bw.putSignedBits(segQ1, 7) // segment 3 — unused, same as 1

		// Filter strength absolute values for 4 segments (6-bit signed).
		// All 0: loop filtering is disabled.
		for s := 0; s < 4; s++ {
			bw.putSignedBits(0, 6)
		}

		// Segment map probability updates (only when update_map=1).
		// Each of probs[0..2]: 1-bit flag "update this prob?"; if 1, emit 8-bit value.
		// We keep defaults (255) so emit 0 for all three flags.
		for s := 0; s < 3; s++ {
			bw.putBitUniform(0) // no update (keep default prob=255)
		}
	} else {
		bw.putBitUniform(0) // update_mb_segmentation = 0
	}

	// Filter header
	bw.putBitUniform(0) // filter_type = 0 (simple)
	bw.putBits(0, 6)    // filter_level = 0
	bw.putBits(0, 3)    // filter_sharpness = 0
	bw.putBitUniform(0) // loop_filter_adj_enable = 0

	// Number of DCT partitions: log2(1) = 0
	bw.putBits(0, 2)

	// Quantizer indices.
	// base_q is the nominal quantizer (used for MBs without explicit segment override).
	// With update_map=1, every MB gets an explicit segment ID, so base_q is not
	// directly used for reconstruction. We set it to segQ1 (textured segment)
	// to match libwebp's enc->base_quant convention.
	baseQ := segQ1
	bw.putBits(uint32(baseQ), 7)
	bw.putSignedBits(0, 4) // y1_dc_delta = 0
	bw.putSignedBits(0, 4) // y2_dc_delta = 0
	bw.putSignedBits(0, 4) // y2_ac_delta = 0
	bw.putSignedBits(0, 4) // uv_dc_delta = 0
	bw.putSignedBits(0, 4) // uv_ac_delta = 0

	// refreshLastFrameBuffer = 0 (key frame)
	bw.putBitUniform(0)

	// Token probability updates.
	// For each [t][b][c][p]: if updated[t][b][c][p], signal "1" and write the
	// new 8-bit probability; otherwise signal "0" (keep default).
	// Mirrors VP8WriteProbas() in libwebp/src/enc/tree_enc.c.
	for t := 0; t < numTypes; t++ {
		for b := 0; b < numBands; b++ {
			for c := 0; c < numCtx; c++ {
				for p := 0; p < numProbas; p++ {
					updateProba := int(coeffsUpdateProba[t][b][c][p])
					if updated[t][b][c][p] {
						bw.putBit(1, updateProba)
						bw.putBits(uint32(probs[t][b][c][p]), 8)
					} else {
						bw.putBit(0, updateProba)
					}
				}
			}
		}
	}

	// use_skip_proba = 0
	bw.putBitUniform(0)

	// Per-MB data: [optional segment ID] + [optional skip] + mb_type + modes.
	topI4 := make([]int, mbW*4)

	for mbY := 0; mbY < mbH; mbY++ {
		leftI4 := [4]int{}

		for mbX := 0; mbX < mbW; mbX++ {
			info := &infos[mbY*mbW+mbX]

			// Emit segment ID if update_map is active.
			// Use default segment probs (255, 255, 255) — no prob updates signaled.
			if useSegments {
				putSegmentID(bw, info.segment, [3]uint8{255, 255, 255})
			}

			if info.isI4 {
				bw.putBit(0, 145)

				topPred := make([]int, 4)
				copy(topPred, topI4[mbX*4:mbX*4+4])

				for by := 0; by < 4; by++ {
					leftPred := leftI4[by]
					if by == 0 {
						leftPred = leftI4[0]
					}
					for bx := 0; bx < 4; bx++ {
						blkIdx := by*4 + bx
						mode := info.i4Modes[blkIdx]
						top := topPred[bx]
						left := leftPred
						putI4Mode(bw, mode, top, left)
						leftPred = mode
						topPred[bx] = mode
					}
					leftI4[by] = info.i4Modes[by*4+3]
				}
				for bx := 0; bx < 4; bx++ {
					topI4[mbX*4+bx] = info.i4Modes[3*4+bx]
				}
			} else {
				bw.putBit(1, 145)
				mode := info.i16Mode
				isTMorHE := mode == I16_TM_PRED || mode == I16_HE_PRED
				bw.putBit(boolInt(isTMorHE), 156)
				if isTMorHE {
					bw.putBit(boolInt(mode == I16_TM_PRED), 128)
				} else {
					bw.putBit(boolInt(mode == I16_VE_PRED), 163)
				}
				for bx := 0; bx < 4; bx++ {
					topI4[mbX*4+bx] = info.i16Mode
				}
				for by := 0; by < 4; by++ {
					leftI4[by] = info.i16Mode
				}
			}

			// UV mode encoding using the RD-selected mode.
			putUVMode(bw, info.uvMode)
		}
	}
}
