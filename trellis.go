// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// Trellis quantization: Viterbi DP to find the coefficient levels that
// minimise distortion + lambda * entropy_bits.
// Ported from TrellisQuantizeBlock() in libwebp/src/enc/quant_enc.c.

// maxVariableLevel is the last level with a variable coding cost (libwebp MAX_VARIABLE_LEVEL=67).
const maxVariableLevel = 67

// maxLevel is the maximum codable coefficient level (libwebp MAX_LEVEL=2047).
const maxLevel = 2047

// rdDistoMult is the distortion scaling factor (libwebp RD_DISTO_MULT=256).
const rdDistoMult = 256

// kWeightTrellis[j] is the distortion weight for raster position j.
// USE_TDISTO=1 values from libwebp/src/enc/quant_enc.c.
var kWeightTrellis = [16]int{
	30, 27, 19, 11, 27, 24, 17, 10, 19,
	17, 12, 8, 11, 10, 8, 6,
}

// vp8LevelFixedCosts[v] is the fixed part of the bit-cost of coding level v.
// From libwebp/src/dsp/cost.c VP8LevelFixedCosts[].
var vp8LevelFixedCosts [maxLevel + 2]int16

func init() {
	// Fill from libwebp/src/dsp/cost.c VP8LevelFixedCosts[].
	// Only the first 128 entries are needed in practice at quality≥10.
	src := []int16{
		0, 256, 256, 256, 256, 432, 618, 630, 731, 640, 640, 828,
		901, 948, 1021, 1101, 1174, 1221, 1294, 1042, 1085, 1115, 1158, 1202,
		1245, 1275, 1318, 1337, 1380, 1410, 1453, 1497, 1540, 1570, 1613, 1280,
		1295, 1317, 1332, 1358, 1373, 1395, 1410, 1454, 1469, 1491, 1506, 1532,
		1547, 1569, 1584, 1601, 1616, 1638, 1653, 1679, 1694, 1716, 1731, 1775,
		1790, 1812, 1827, 1853, 1868, 1890, 1905, 1727, 1733, 1742, 1748, 1759,
		1765, 1774, 1780, 1800, 1806, 1815, 1821, 1832, 1838, 1847, 1853, 1878,
		1884, 1893, 1899, 1910, 1916, 1925, 1931, 1951, 1957, 1966, 1972, 1983,
		1989, 1998, 2004, 2027, 2033, 2042, 2048, 2059, 2065, 2074, 2080, 2100,
		2106, 2115, 2121, 2132, 2138, 2147, 2153, 2178, 2184, 2193, 2199, 2210,
		2216, 2225, 2231, 2251, 2257, 2266, 2272, 2283, 2289, 2298, 2304,
	}
	copy(vp8LevelFixedCosts[:], src)
	// For levels beyond what we have, replicate the last entry.
	last := src[len(src)-1]
	for i := len(src); i < len(vp8LevelFixedCosts); i++ {
		vp8LevelFixedCosts[i] = last
	}
}

// vp8LevelCodes encodes each level's coding pattern.
// VP8LevelCodes[level-1] = {pattern, bits} from libwebp/src/enc/cost_enc.c.
var vp8LevelCodes = [maxVariableLevel][2]uint16{
	{0x001, 0x000}, {0x007, 0x001}, {0x00f, 0x005}, {0x00f, 0x00d},
	{0x033, 0x003}, {0x033, 0x003}, {0x033, 0x023}, {0x033, 0x023},
	{0x033, 0x023}, {0x033, 0x023}, {0x0d3, 0x013}, {0x0d3, 0x013},
	{0x0d3, 0x013}, {0x0d3, 0x013}, {0x0d3, 0x013}, {0x0d3, 0x013},
	{0x0d3, 0x013}, {0x0d3, 0x013}, {0x0d3, 0x093}, {0x0d3, 0x093},
	{0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093},
	{0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093},
	{0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093}, {0x0d3, 0x093},
	{0x0d3, 0x093}, {0x0d3, 0x093}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053}, {0x153, 0x053},
	{0x153, 0x053}, {0x153, 0x053}, {0x153, 0x153},
}

// trellisCostTables holds precomputed level cost tables for one coefficient type.
// Indexed [band][ctx][level] where level ranges 0..maxVariableLevel.
// table.level[band][ctx][v] = variable-cost of emitting level v in context (band, ctx).
// Total level cost = vp8LevelFixedCosts[level] + table.level[band][ctx][min(level,67)].
// table.eob[band][ctx]  = VP8BitCost(0, probs[band][ctx][0]) — EOB bit cost (block zero).
// table.eob1[band][ctx] = VP8BitCost(1, probs[band][ctx][0]) — non-zero block flag cost.
// Both precomputed to avoid cold 4D probs-array dereferences in hot paths.
type trellisCostTables struct {
	level [numBands][numCtx][maxVariableLevel + 1]int16
	eob   [numBands][numCtx]int32
	eob1  [numBands][numCtx]int32
}

// buildTrellisCostTables precomputes all level cost tables from a coefficient
// probability table. Mirrors VP8CalculateLevelCosts() in libwebp.
//
// tables.level[band][ctx][v] stores the TOTAL bit cost of emitting level v in
// context (band, ctx): vp8LevelFixedCosts[v] + variablePart(v, probs).
// This lets levelCostFromTable execute a single table lookup for v ≤ 67
// instead of two separate accesses to vp8LevelFixedCosts and the variable table.
func buildTrellisCostTables(probs *[numBands][numCtx][numProbas]uint8) trellisCostTables {
	var tables trellisCostTables
	for band := 0; band < numBands; band++ {
		for ctx := 0; ctx < numCtx; ctx++ {
			p := probs[band][ctx][:]
			cost0 := 0
			if ctx > 0 {
				cost0 = vp8BitCost(1, int(p[0]))
			}
			// EOB costs: probability of "no / at-least-one more non-zero" at (band, ctx).
			tables.eob[band][ctx] = int32(vp8BitCost(0, int(p[0])))
			tables.eob1[band][ctx] = int32(vp8BitCost(1, int(p[0])))
			// Level 0: emit the "non-zero = 0" bit. Fixed cost for level 0 is 0.
			tables.level[band][ctx][0] = int16(vp8BitCost(0, int(p[1])) + cost0)
			// Levels 1..maxVariableLevel: total = fixedCost[v] + variablePart.
			costBase := vp8BitCost(1, int(p[1])) + cost0
			for v := 1; v <= maxVariableLevel; v++ {
				tables.level[band][ctx][v] = int16(int(vp8LevelFixedCosts[v]) + costBase + variableLevelCost(v, p))
			}
		}
	}
	return tables
}

// variableLevelCost computes the probability-dependent part of coding level v.
// Mirrors VariableLevelCost() in libwebp/src/enc/cost_enc.c.
func variableLevelCost(level int, p []uint8) int {
	pattern := int(vp8LevelCodes[level-1][0])
	bits := int(vp8LevelCodes[level-1][1])
	cost := 0
	for i := 2; pattern != 0; i++ {
		if pattern&1 != 0 {
			cost += vp8BitCost(bits&1, int(p[i]))
		}
		bits >>= 1
		pattern >>= 1
	}
	return cost
}

// levelCostFromTable returns the total bit-cost of coding level v.
// table[v] already stores vp8LevelFixedCosts[v] + variablePart(v) for v ≤ 67,
// so the common case (level ≤ maxVariableLevel) is a single table lookup.
// For level > 67 (rare at quality ≥ 10): table[67] holds the variable part at
// the saturation point; we add vp8LevelFixedCosts[level] minus the fixed cost
// already embedded in table[67].  For simplicity we store the full total in the
// table for levels 0..67, and for level > 67 fall back to the two-table sum.
func levelCostFromTable(table *[maxVariableLevel + 1]int16, level int) int {
	if level <= maxVariableLevel {
		return int(table[level])
	}
	// level > 67: rare. vp8LevelFixedCosts saturates at maxLevel (2047).
	if level > maxLevel {
		level = maxLevel
	}
	// table[67] = fixedCosts[67] + variablePart(67); for level > 67 the variable
	// part is the same as level 67, so total = fixedCosts[level] + table[67] - fixedCosts[67].
	return int(vp8LevelFixedCosts[level]) + int(table[maxVariableLevel]) - int(vp8LevelFixedCosts[maxVariableLevel])
}

// trellisNode stores the Viterbi DP state for one trellis candidate.
// level is the fully signed quantized coefficient (negative if original was negative).
// Storing signed level avoids a branch in the backtrack unwind.
type trellisNode struct {
	prev  int8  // index of best predecessor node
	_     int8  // padding (was sign; folded into level)
	level int16 // signed quantized level (may be negative)
}

// maxCost is a sentinel "infinity" score for dead DP states.
const maxCost = int64(1) << 50

// zeroCostTable is a dummy all-zero level-cost table used as a sentinel for
// dead DP states or positions past n=15. Using a pointer to this instead of
// nil avoids nil-pointer branches in the hot inner loop — dead states are
// gated exclusively by score >= maxCost.
var zeroCostTable [maxVariableLevel + 1]int16

// trellisQuantize applies trellis quantization to a 4x4 DCT block.
//
// Parameters:
//   - in[16]: DCT coefficients in raster order; modified in-place to dequantized values
//   - out[16]: quantized levels in zigzag order (output)
//   - m: quantization matrix
//   - first: 0 for i4/UV (quantize all 16), 1 for i16-AC (skip DC slot 0)
//   - lambda: trellis lambda (e.g. (7*qI4*qI4)>>3 for i4)
//   - costs: precomputed level cost tables (from buildTrellisCostTables)
//   - probs: raw coefficient probability table for EOB cost computation
//   - ctx0: context for the first coefficient position (0, 1, or 2)
//
// Returns true if any non-zero level was produced.
// After return, in[j] = out[n]*m.q[j] for all positions (dequantized, ready for iDCT).
// Mirrors TrellisQuantizeBlock() in libwebp/src/enc/quant_enc.c.
func trellisQuantize(
	in []int16,
	out []int16,
	m *quantMatrix,
	first int,
	lambda int,
	costs *trellisCostTables,
	probs *[numBands][numCtx][numProbas]uint8,
	ctx0 int,
) bool {
	// scoreState holds the accumulated RD score and the cost table pointer
	// for the CURRENT position. When this becomes ss_prev, we use its table
	// to compute the cost of emitting the level at the next position.
	// costs is a pointer to a fixed-size [maxVariableLevel+1]int16 array inside
	// trellisCostTables; using *[N]int16 instead of []int16 shrinks the field
	// from 24 bytes (slice header) to 8 bytes (pointer), reducing memory traffic
	// in the Viterbi DP inner loop.
	type scoreState struct {
		score int64
		costs *[maxVariableLevel + 1]int16
	}

	var nodes [16][2]trellisNode
	var ss [2][2]scoreState
	curIdx := 0
	prevIdx := 1

	// Determine the last interesting coefficient position.
	// Use per-position zthresh: any trailing coeff with |coeff| <= zthresh[j]
	// will always quantize to zero regardless of trellis, so skip it.
	last := first - 1
	for n := 15; n >= first; n-- {
		j := int(kZigzag[n])
		v := int32(in[j])
		if v < 0 {
			v = -v
		}
		if uint32(v) > m.zthresh[j] {
			last = n
			break
		}
	}
	if last < 15 {
		last++
	}

	// Sparse-block fast path: no position above zthresh — fall through to greedy
	// quantization. The trellis in[] invariant is maintained by the explicit
	// dequantize loop below.
	if last < 1 {
		nz := quantizeBlock(in, out, m, first)
		for n := 0; n < 16; n++ {
			j := int(kZigzag[n])
			in[j] = int16(int32(out[n]) * int32(m.q[j]))
		}
		return nz
	}

	// "Skip all" baseline: cost of emitting EOB at the start (all-zero block).
	// costs.eob[band][ctx] = VP8BitCost(0, probs[band][ctx][0]) — precomputed at
	// segment setup time, so no cold 4D-array access here.
	firstBand := int(vp8EncBands[first])
	skipCost := int64(costs.eob[firstBand][ctx0]) * int64(lambda)
	bestScore := skipCost

	// Initialize source nodes.
	// ss[cur][m].costs = cost table for position `first` with context ctx0.
	// In libwebp: rate = (ctx0 == 0) ? VP8BitCost(1, last_proba) : 0
	// because for ctx0==0 the "non-zero" flag is not part of the level cost table.
	initTable := &costs.level[firstBand][ctx0]
	initRate := int64(0)
	if ctx0 == 0 {
		lastProba := int(probs[firstBand][ctx0][0])
		initRate = int64(vp8BitCost(1, lastProba)) * int64(lambda)
	}
	ss[curIdx][0].score = initRate
	ss[curIdx][0].costs = initTable
	ss[curIdx][1].score = initRate
	ss[curIdx][1].costs = initTable

	// bestPath[0]=best end pos, [1]=best node idx, [2]=best prev idx
	bestPath := [3]int{-1, -1, -1}

	// Precompute quantized level bounds for each coefficient position.
	const neutralBias = uint32(0)
	biasHalf := uint32(0x80) << (qfix - 8) // BIAS(0x80)
	var lvl0 [16]int   // floor level (no rounding bias)
	var lvlMax [16]int // threshold level (0.5 bias; candidates pruned above this)
	for n := first; n <= last; n++ {
		j := int(kZigzag[n])
		iQ := uint32(m.iq[j])
		raw := int32(in[j])
		if raw < 0 {
			raw = -raw
		}
		coeff0 := raw + int32(m.sharpen[j])
		if coeff0 < 0 {
			coeff0 = 0
		}
		l0 := int((uint32(coeff0)*iQ + neutralBias) >> qfix)
		if l0 > maxLevel {
			l0 = maxLevel
		}
		lMax := int((uint32(coeff0)*iQ + biasHalf) >> qfix)
		if lMax > maxLevel {
			lMax = maxLevel
		}
		lvl0[n] = l0
		lvlMax[n] = lMax
	}

	iLambda := int64(lambda)

	for n := first; n <= last; n++ {
		j := int(kZigzag[n])
		Q := int32(m.q[j])

		// Apply sharpening bias to get coeff0 for distortion computation.
		// Mirrors libwebp: coeff0 = (sign ? -in[j] : in[j]) + mtx->sharpen[j].
		raw := int32(in[j])
		sign := int8(0)
		if raw < 0 {
			sign = 1
			raw = -raw
		}
		coeff0 := raw + int32(m.sharpen[j])
		if coeff0 < 0 {
			coeff0 = 0
		}

		level0 := lvl0[n]
		threshLevel := lvlMax[n]

		// Precompute nextBand for this position — same for both m2 nodes.
		nextBand := int(vp8EncBands[n+1]) // sentinel at 16 is 0, harmless

		// Swap cur ↔ prev.
		curIdx, prevIdx = prevIdx, curIdx

		// Hoist all values shared by or derivable before the two m2 bodies.
		wt := int64(kWeightTrellis[j])
		coeff0sq := int64(coeff0) * int64(coeff0)
		prev0 := &ss[prevIdx][0]
		prev1 := &ss[prevIdx][1]

		// Precompute clamped context and nextCosts for both m2=0 (ctx0) and m2=1 (ctx1).
		// ctx = min(level, 2); for m2=1 ctx1 = min(level0+1, 2) which is simply
		// ctx0 when ctx0 == 2, and ctx0+1 otherwise — both computable without branches
		// once we know ctx0.
		ctx0 := level0
		if ctx0 > 2 {
			ctx0 = 2
		}
		ctx1 := level0 + 1
		if ctx1 > 2 {
			ctx1 = 2
		}
		// nextCosts: use zeroCostTable at n==15 (no successor position).
		var nextCosts0, nextCosts1 *[maxVariableLevel + 1]int16
		eobLC0, eobLC1 := int64(0), int64(0)
		if n < 15 {
			nextCosts0 = &costs.level[nextBand][ctx0]
			nextCosts1 = &costs.level[nextBand][ctx1]
			eobLC0 = int64(costs.eob[nextBand][ctx0]) * iLambda
			eobLC1 = int64(costs.eob[nextBand][ctx1]) * iLambda
		} else {
			nextCosts0 = &zeroCostTable
			nextCosts1 = &zeroCostTable
		}

		// ── m2 = 0: level = level0 ────────────────────────────────────────────
		{
			level := level0
			if level >= 0 && level <= threshLevel {
				ss[curIdx][0].costs = nextCosts0

				newErr := coeff0 - int32(level)*Q
				baseScore := wt*(int64(newErr)*int64(newErr)-coeff0sq)*rdDistoMult

				bestCurScore := maxCost
				bestPrev := 0
				if prev0.score < maxCost {
					s := prev0.score + int64(levelCostFromTable(prev0.costs, level))*iLambda
					if s < bestCurScore {
						bestCurScore = s
					}
				}
				if prev1.score < maxCost {
					s := prev1.score + int64(levelCostFromTable(prev1.costs, level))*iLambda
					if s < bestCurScore {
						bestCurScore = s
						bestPrev = 1
					}
				}
				bestCurScore += baseScore

				// Store signed level directly: avoids a branch in backtrack unwind.
				signedLvl := int16(level)
				if sign != 0 {
					signedLvl = -signedLvl
				}
				nodes[n][0].level = signedLvl
				nodes[n][0].prev = int8(bestPrev)
				ss[curIdx][0].score = bestCurScore

				if level != 0 && bestCurScore < maxCost {
					if tot := bestCurScore + eobLC0; tot < bestScore {
						bestScore = tot
						bestPath[0] = n
						bestPath[1] = 0
						bestPath[2] = bestPrev
					}
				}
			} else {
				ss[curIdx][0].score = maxCost
				ss[curIdx][0].costs = &zeroCostTable
			}
		}

		// ── m2 = 1: level = level0 + 1 ───────────────────────────────────────
		{
			level := level0 + 1
			if level >= 0 && level <= threshLevel {
				ss[curIdx][1].costs = nextCosts1

				newErr := coeff0 - int32(level)*Q
				baseScore := wt*(int64(newErr)*int64(newErr)-coeff0sq)*rdDistoMult

				bestCurScore := maxCost
				bestPrev := 0
				if prev0.score < maxCost {
					s := prev0.score + int64(levelCostFromTable(prev0.costs, level))*iLambda
					if s < bestCurScore {
						bestCurScore = s
					}
				}
				if prev1.score < maxCost {
					s := prev1.score + int64(levelCostFromTable(prev1.costs, level))*iLambda
					if s < bestCurScore {
						bestCurScore = s
						bestPrev = 1
					}
				}
				bestCurScore += baseScore

				signedLvl := int16(level)
				if sign != 0 {
					signedLvl = -signedLvl
				}
				nodes[n][1].level = signedLvl
				nodes[n][1].prev = int8(bestPrev)
				ss[curIdx][1].score = bestCurScore

				if level != 0 && bestCurScore < maxCost {
					if tot := bestCurScore + eobLC1; tot < bestScore {
						bestScore = tot
						bestPath[0] = n
						bestPath[1] = 1
						bestPath[2] = bestPrev
					}
				}
			} else {
				ss[curIdx][1].score = maxCost
				ss[curIdx][1].costs = &zeroCostTable
			}
		}
	}

	// Clear outputs.
	if first == 1 {
		// i16-AC: preserve DC slot (kZigzag[0]=0).
		clear(out[1:16])
		saved := in[0]
		clear(in[:16])
		in[0] = saved // kZigzag[0] = 0, so raster[0] = DC preserved
	} else {
		clear(out[:16])
		clear(in[:16])
	}

	if bestPath[0] == -1 {
		return false // block is all zeros
	}

	// Unwind best path, write quantized levels and dequantized raster values.
	// node.level is already signed — write directly to out[n], no branch needed.
	nz := false
	bestNode := bestPath[1]
	n := bestPath[0]
	nodes[n][bestNode].prev = int8(bestPath[2])

	for ; n >= first; n-- {
		node := &nodes[n][bestNode]
		j := int(kZigzag[n])
		sl := node.level // signed quantized level
		out[n] = sl
		if sl != 0 {
			nz = true
		}
		in[j] = int16(int32(sl) * int32(m.q[j]))
		bestNode = int(node.prev)
	}

	return nz
}
