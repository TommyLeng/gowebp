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
// table[band][ctx][v] = variable-cost of emitting level v in context (band, ctx).
// Total level cost = vp8LevelFixedCosts[level] + table[band][ctx][min(level,67)].
type trellisCostTables [numBands][numCtx][maxVariableLevel + 1]int16

// buildTrellisCostTables precomputes all level cost tables from a coefficient
// probability table. Mirrors VP8CalculateLevelCosts() in libwebp.
func buildTrellisCostTables(probs *[numBands][numCtx][numProbas]uint8) trellisCostTables {
	var tables trellisCostTables
	for band := 0; band < numBands; band++ {
		for ctx := 0; ctx < numCtx; ctx++ {
			p := probs[band][ctx][:]
			cost0 := 0
			if ctx > 0 {
				cost0 = vp8BitCost(1, int(p[0]))
			}
			// Level 0: emit the "non-zero = 0" bit.
			tables[band][ctx][0] = int16(vp8BitCost(0, int(p[1])) + cost0)
			// Levels 1..maxVariableLevel: non-zero + variable coding.
			costBase := vp8BitCost(1, int(p[1])) + cost0
			for v := 1; v <= maxVariableLevel; v++ {
				tables[band][ctx][v] = int16(costBase + variableLevelCost(v, p))
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
// Mirrors VP8LevelCost(): vp8LevelFixedCosts[level] + table[min(level,67)]
func levelCostFromTable(table []int16, level int) int {
	if level > maxLevel {
		level = maxLevel
	}
	vl := level
	if vl > maxVariableLevel {
		vl = maxVariableLevel
	}
	return int(vp8LevelFixedCosts[level]) + int(table[vl])
}

// trellisNode stores the Viterbi DP state for one trellis candidate.
type trellisNode struct {
	prev  int8  // index of best predecessor node
	sign  int8  // sign of coeff_i (0=positive, 1=negative)
	level int16 // quantized level magnitude
}

// maxCost is a sentinel "infinity" score for dead DP states.
const maxCost = int64(1) << 50

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
	// We test candidates: level0 and level0+1 (MIN_DELTA=0, MAX_DELTA=1 → 2 nodes).
	const numNodes = 2

	// scoreState holds the accumulated RD score and the cost table pointer
	// for the CURRENT position. When this becomes ss_prev, we use its table
	// to compute the cost of emitting the level at the next position.
	type scoreState struct {
		score int64
		costs []int16 // pointer into trellisCostTables[band][ctx][:], nil if dead
	}

	var nodes [16][numNodes]trellisNode
	var ss [2][numNodes]scoreState
	curIdx := 0
	prevIdx := 1

	// Determine the last interesting coefficient position.
	// Skip trailing positions where coeff^2 <= q[1]^2/4 (libwebp's early-exit).
	thresh := int32(m.q[1]) * int32(m.q[1]) / 4
	last := first - 1
	for n := 15; n >= first; n-- {
		j := int(kZigzag[n])
		err := int32(in[j]) * int32(in[j])
		if err > thresh {
			last = n
			break
		}
	}
	if last < 15 {
		last++
	}

	// "Skip all" baseline: cost of emitting EOB at the start (all-zero block).
	// = VP8BitCost(0, probs[firstBand][ctx0][0]) * lambda
	firstBand := int(vp8EncBands[first])
	lastProba := int(probs[firstBand][ctx0][0])
	skipCost := int64(vp8BitCost(0, lastProba)) * int64(lambda)
	bestScore := skipCost

	// Initialize source nodes.
	// ss[cur][m].costs = cost table for position `first` with context ctx0.
	// In libwebp: rate = (ctx0 == 0) ? VP8BitCost(1, last_proba) : 0
	// because for ctx0==0 the "non-zero" flag is not part of the level cost table.
	initTable := costs[firstBand][ctx0][:]
	initRate := int64(0)
	if ctx0 == 0 {
		initRate = int64(vp8BitCost(1, lastProba)) * int64(lambda)
	}
	for m2 := 0; m2 < numNodes; m2++ {
		ss[curIdx][m2].score = initRate
		ss[curIdx][m2].costs = initTable
	}

	// bestPath[0]=best end pos, [1]=best node idx, [2]=best prev idx
	bestPath := [3]int{-1, -1, -1}

	for n := first; n <= last; n++ {
		j := int(kZigzag[n])
		Q := int32(m.q[j])
		iQ := uint32(m.iq[j])

		// Neutral bias = BIAS(0x00) = 0.
		const neutralBias = uint32(0)

		// Apply sharpening bias to increase high-frequency coefficient magnitude.
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

		// level0: floor of coeff/q (no rounding bias).
		level0 := int((uint32(coeff0)*iQ + neutralBias) >> qfix)
		if level0 > maxLevel {
			level0 = maxLevel
		}
		// threshLevel: round with standard 0.5 bias; anything above is too costly.
		biasHalf := uint32(0x80) << (qfix - 8) // BIAS(0x80)
		threshLevel := int((uint32(coeff0)*iQ + biasHalf) >> qfix)
		if threshLevel > maxLevel {
			threshLevel = maxLevel
		}

		// Swap cur ↔ prev.
		curIdx, prevIdx = prevIdx, curIdx

		for m2 := 0; m2 < numNodes; m2++ {
			level := level0 + m2

			// Prune dead candidates.
			if level < 0 || level > threshLevel {
				ss[curIdx][m2].score = maxCost
				ss[curIdx][m2].costs = nil
				continue
			}

			// Context produced by this level for the next position.
			ctx := level
			if ctx > 2 {
				ctx = 2
			}
			nextBand := int(vp8EncBands[n+1]) // sentinel at 16 is 0, harmless
			if n+1 < 16 {
				ss[curIdx][m2].costs = costs[nextBand][ctx][:]
			} else {
				ss[curIdx][m2].costs = nil
			}

			// Distortion: kWeightTrellis[j] * ((coeff0 - level*Q)^2 - coeff0^2).
			newErr := coeff0 - int32(level)*Q
			deltaError := int64(kWeightTrellis[j]) * (int64(newErr)*int64(newErr) - int64(coeff0)*int64(coeff0))
			baseScore := deltaError * rdDistoMult

			// Find best predecessor: iterate over prev nodes, look up the cost
			// of emitting `level` using ss_prev[p].costs (= cost table for position n).
			bestCurScore := maxCost
			bestPrev := 0
			for p2 := 0; p2 < numNodes; p2++ {
				prev := &ss[prevIdx][p2]
				if prev.score >= maxCost || prev.costs == nil {
					continue
				}
				cost := levelCostFromTable(prev.costs, level)
				score := prev.score + int64(cost)*int64(lambda)
				if score < bestCurScore {
					bestCurScore = score
					bestPrev = p2
				}
			}
			bestCurScore += baseScore

			nodes[n][m2].sign = sign
			nodes[n][m2].level = int16(level)
			nodes[n][m2].prev = int8(bestPrev)
			ss[curIdx][m2].score = bestCurScore

			// Check if this is a better terminal state (this position is last non-zero).
			if level != 0 && bestCurScore < maxCost {
				// Cost of EOB after position n: VP8BitCost(0, probs[nextBand][ctx][0]).
				var lastPosCost int64
				if n < 15 {
					nextBand := int(vp8EncBands[n+1])
					eobProba := int(probs[nextBand][ctx][0])
					lastPosCost = int64(vp8BitCost(0, eobProba)) * int64(lambda)
				}
				totalScore := bestCurScore + lastPosCost
				if totalScore < bestScore {
					bestScore = totalScore
					bestPath[0] = n
					bestPath[1] = m2
					bestPath[2] = bestPrev
				}
			}
		}
	}

	// Clear outputs.
	if first == 1 {
		// i16-AC: preserve DC slot (kZigzag[0]=0).
		for i := 1; i < 16; i++ {
			out[i] = 0
		}
		saved := in[0]
		for k := 0; k < 16; k++ {
			in[k] = 0
		}
		in[0] = saved // kZigzag[0] = 0, so raster[0] = DC preserved
	} else {
		for i := 0; i < 16; i++ {
			out[i] = 0
			in[i] = 0
		}
	}

	if bestPath[0] == -1 {
		return false // block is all zeros
	}

	// Unwind best path, write quantized levels and dequantized raster values.
	nz := false
	bestNode := bestPath[1]
	n := bestPath[0]
	nodes[n][bestNode].prev = int8(bestPath[2])

	for ; n >= first; n-- {
		node := &nodes[n][bestNode]
		j := int(kZigzag[n])
		lvl := int(node.level)
		if node.sign != 0 {
			out[n] = -int16(lvl)
		} else {
			out[n] = int16(lvl)
		}
		if lvl != 0 {
			nz = true
		}
		in[j] = int16(int32(out[n]) * int32(m.q[j]))
		bestNode = int(node.prev)
	}

	return nz
}
