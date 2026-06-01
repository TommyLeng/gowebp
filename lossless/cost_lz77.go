package lossless

// Cost-based ("optimal-parse") LZ77 for VP8L, ported in spirit from libwebp's
// backward_references_cost_enc.c (BackwardReferencesHashChainDistanceOnly +
// TraceBackwards). The greedy parser in encodeImageDataGreedy always takes the
// longest match at each position with no regard to the bit cost of the distance
// code; on data where literals are cheap and matches are far (e.g. alpha masks)
// this picks expensive copies. Here we:
//
//  1. run the greedy parse once to gather a token histogram,
//  2. convert that histogram to per-symbol bit-cost estimates (the cost model),
//  3. run a forward shortest-path DP that, at every position, weighs a literal
//     (or color-cache reference) against every match length using the model,
//  4. trace the minimum-cost path back and re-emit tokens.
//
// The caller keeps whichever of greedy / optimal has the lower estimated cost,
// so this can never regress the output.

import (
	"image/color"
	"math"
)

// lz77Distances maps small 2D (x, y) pixel offsets to short distance plane
// codes per the WebP spec. Hoisted to package scope so the greedy and the
// cost-based parser share one mapping (previously a local in encodeImageData).
var lz77Distances = []int{
	96, 73, 55, 39, 23, 13, 5, 1, 255, 255, 255, 255, 255, 255, 255, 255,
	101, 78, 58, 42, 26, 16, 8, 2, 0, 3, 9, 17, 27, 43, 59, 79,
	102, 86, 62, 46, 32, 20, 10, 6, 4, 7, 11, 21, 33, 47, 63, 87,
	105, 90, 70, 52, 37, 28, 18, 14, 12, 15, 19, 29, 38, 53, 71, 91,
	110, 99, 82, 66, 48, 35, 30, 24, 22, 25, 31, 36, 49, 67, 83, 100,
	115, 108, 94, 76, 64, 50, 44, 40, 34, 41, 45, 51, 65, 77, 95, 109,
	118, 113, 103, 92, 80, 68, 60, 56, 54, 57, 61, 69, 81, 93, 104, 114,
	119, 116, 111, 106, 97, 88, 84, 74, 72, 75, 85, 89, 98, 107, 112, 117,
}

// distancePlaneCode maps a raw pixel distance to its WebP distance plane code.
// Small 2D-local offsets get short codes (1..120); larger distances use
// dis+120. This is the exact mapping the greedy emitter uses.
func distancePlaneCode(dis, width int) int {
	y := dis / width
	x := dis - y*width
	code := dis + 120
	if x <= 8 && y < 8 {
		code = lz77Distances[y*16+8-x] + 1
	} else if x > width-8 && y < 7 {
		code = lz77Distances[(y+1)*16+8+(width-x)] + 1
	}
	return code
}

// Empirical scaling factors copied from libwebp's AddSingleLiteralWithCostModel
// (DivRound(cost*82,100) for literals, *68 for cache hits). The cost model is
// built from the greedy parse, which under-uses literals relative to the
// optimal parse; biasing literals cheaper compensates for that feedback skew.
const (
	literalCostFactor = 0.82
	cacheCostFactor   = 0.68
)

// costModel holds per-symbol bit-cost estimates (in bits). Port of libwebp's
// CostModel: cost[s] = log2(total) - log2(count[s]).
type costModel struct {
	alpha    [256]float64
	red      [256]float64
	blue     [256]float64
	distance [40]float64
	literal  []float64 // 256 (green) + 24 (length codes) + colorCacheSize
}

// fastLog2 mirrors libwebp's VP8LFastLog2, which returns 0 for v==0 (so a
// never-seen symbol is estimated at log2(total) bits rather than +Inf).
func fastLog2(v int) float64 {
	if v == 0 {
		return 0
	}
	return math.Log2(float64(v))
}

// convertToBitEstimates is the port of ConvertPopulationCountTableToBitEstimates.
// When the alphabet has <=1 distinct symbol it costs 0 bits (only one option).
func convertToBitEstimates(counts []int, out []float64) {
	sum, nonzeros := 0, 0
	for _, c := range counts {
		sum += c
		if c > 0 {
			nonzeros++
		}
	}
	if nonzeros <= 1 {
		for i := range out {
			out[i] = 0
		}
		return
	}
	logsum := math.Log2(float64(sum))
	for i, c := range counts {
		out[i] = logsum - fastLog2(c)
	}
}

// buildCostModel converts the 5 token-stream histograms produced by
// computeHistograms into per-symbol bit-cost estimates. Histogram order is
// [green/length/cache, R, B, A, distance] (see computeHistograms); libwebp's
// GetLiteralCost reads alpha[A]+red[R]+literal[G]+blue[B].
func buildCostModel(histos [][]int) *costModel {
	m := &costModel{}
	m.literal = make([]float64, len(histos[0]))
	convertToBitEstimates(histos[0], m.literal)
	convertToBitEstimates(histos[1], m.red[:])
	convertToBitEstimates(histos[2], m.blue[:])
	convertToBitEstimates(histos[3], m.alpha[:])
	convertToBitEstimates(histos[4], m.distance[:])
	return m
}

func (m *costModel) literalCost(c color.NRGBA) float64 {
	return m.alpha[c.A] + m.red[c.R] + m.literal[c.G] + m.blue[c.B]
}

func (m *costModel) cacheCost(idx int) float64 {
	return m.literal[256+24+idx]
}

func (m *costModel) lengthCost(length int) float64 {
	sym, _ := prefixEncodeCode(length)
	return m.literal[256+sym] + float64(prefixEncodeBits(sym))
}

func (m *costModel) distanceCost(planeCode int) float64 {
	sym, _ := prefixEncodeCode(planeCode)
	return m.distance[sym] + float64(prefixEncodeBits(sym))
}

// streamExtraBits sums the raw length/distance extra bits in a token stream.
// These are emitted verbatim (not Huffman-coded) so dataCostBits omits them —
// but they dominate the cost of large-distance matches, so any greedy-vs-optimal
// size comparison MUST include them or it will systematically prefer the parse
// with more (expensive) copies.
func streamExtraBits(enc []int) float64 {
	e := 0.0
	for i := 0; i < len(enc); {
		s := enc[i]
		switch {
		case s < 256:
			i += 4
		case s < 256+24:
			e += float64(prefixEncodeBits(s-256)) + float64(prefixEncodeBits(enc[i+2]))
			i += 4
		default:
			i++
		}
	}
	return e
}

// matchLenAt returns the number of consecutive equal pixels starting at b and
// a (a < b), up to maxLen. Caller guarantees b+maxLen <= len(pixels).
func matchLenAt(pixels []color.NRGBA, a, b, maxLen int) int {
	l := 0
	for l < maxLen && pixels[b+l] == pixels[a+l] {
		l++
	}
	return l
}

// Match-finder tuning, ported from libwebp:
//   matchIterMax  = GetMaxItersForQuality(75) = 8 + 75*75/128 = 51 chain probes.
//   matchChainCap = libwebp's length_max: stop accepting chain matches once one
//                   reaches 256, so we don't chase huge-distance long matches
//                   when a closer (cheaper-distance) one is good enough.
const (
	matchIterMax  = 51
	matchChainCap = 256
	// matchIterMaxLow / matchWindowBitsLow are the reduced LZ77 search effort used
	// for the alpha plane, mirroring libwebp's low-quality alpha path
	// (alpha is encoded at quality 8*method ≈ 32): GetMaxItersForQuality(32) =
	// 8 + 32*32/128 = 16 hash-chain probes (vs 51 at quality 75), and
	// GetWindowSizeForHashChain(quality<=50) = xsize<<6 (vs the full 2^20 window).
	// Fewer probes + a smaller window cut the dominant fillMatches cost ~22% on a
	// 1 MP alpha with no size penalty (the cheap dist=1/dist=width heuristics still
	// catch the coherent runs that matter).
	matchIterMaxLow   = 16
	matchWindowBitsLow = 6
	// matchMaxLen is the maximum match length, matching the greedy parser and the
	// WebP length-code range. The DP stays fast at this length because it
	// propagates only the per-bucket-end lengths (lz77BucketEnds), and
	// fillMatches stays fast because dist=1/dist=width use O(1) run lengths and
	// the hash chain uses the bestArgb skip.
	matchMaxLen = 4096
)

// lz77BucketEnds is the largest match length within each length-prefix code
// bucket, up to matchMaxLen. Length codes are constant within a bucket and only
// the extra-bit *count* (also constant) is added, so GetLengthCost is constant
// across a bucket — a match to the bucket end covers the most pixels for that
// cost. The DP therefore only needs to consider these ~15 lengths per start
// (plus the exact match length), turning its O(matchMaxLen) inner loop into
// O(buckets) and removing the run-region blow-up.
var lz77BucketEnds = func() []int {
	const minMatch = 3
	var ends []int
	prevCode, _ := prefixEncodeCode(minMatch)
	for k := minMatch + 1; k <= matchMaxLen; k++ {
		if code, _ := prefixEncodeCode(k); code != prevCode {
			ends = append(ends, k-1)
			prevCode = code
		}
	}
	return append(ends, matchMaxLen)
}()

// fillMatches records, for every position, the backward match the cost-based
// parser will consider. Like libwebp's VP8LHashChainFill it prefers the two
// cheapest distance plane codes — distance=width (the pixel directly above,
// code 1) and distance=1 (the previous pixel, code 2) — and only lets a
// *strictly longer* hash-chain match displace them. So coherent regions (alpha
// masks especially) match at near-zero distance cost instead of the ~2000-pixel
// distances a raw hash3 chain returns.
//
// The dist=1 and dist=width match lengths are computed in O(1) per position via
// backward recurrences (a constant-distance match at i is one longer than the
// same-distance match at i+1 whenever the leading pixels are equal), so even
// long constant runs cost O(1) each instead of an O(runLength) rescan. This
// keeps the per-position search (good match quality) without the blow-up that
// forced libwebp to left-extend.
func fillMatches(pixels []color.NRGBA, width int, lowEffort bool) (matchLen, matchOff []int) {
	n := len(pixels)
	matchLen = make([]int, n)
	matchOff = make([]int, n)
	if n <= 2 {
		return matchLen, matchOff
	}

	// LZ77 search effort: full (quality-75) by default, reduced for the alpha
	// plane (lowEffort) to libwebp's quality-32 alpha settings.
	iterMax, windowSize := matchIterMax, 1<<20-120
	if lowEffort {
		iterMax = matchIterMaxLow
		if w := width << matchWindowBitsLow; w < windowSize {
			windowSize = w
		}
	}

	// d1[i] / dW[i]: full (uncapped) match length at distance 1 / width.
	d1 := make([]int, n)
	dW := make([]int, n)
	for i := n - 1; i >= 1; i-- {
		ext := 0
		if i+1 < n {
			ext = d1[i+1]
		}
		if pixels[i] == pixels[i-1] {
			d1[i] = 1 + ext
		}
	}
	for i := n - 1; i >= width; i-- {
		ext := 0
		if i+1 < n {
			ext = dW[i+1]
		}
		if pixels[i] == pixels[i-width] {
			dW[i] = 1 + ext
		}
	}

	hashBits := lz77HashBits(n)
	head := make([]int, 1<<hashBits)
	prev := make([]int, n)
	for i := 0; i+1 < n; i++ {
		h := hashPixPair(pixels, i, hashBits)
		cur := head[h] - 1
		prev[i] = head[h]
		head[h] = i + 1

		maxLen := n - i
		if maxLen > matchMaxLen {
			maxLen = matchMaxLen
		}
		lenCap := matchChainCap
		if maxLen < lenCap {
			lenCap = maxLen
		}

		bestLen, bestOff := 0, 0
		// Cheap-distance heuristics (O(1) via the precomputed run lengths).
		if l := dW[i]; l > 0 { // distance = width (plane code 1)
			if l > maxLen {
				l = maxLen
			}
			if l > bestLen {
				bestLen, bestOff = l, width
			}
		}
		if l := d1[i]; l > 0 { // distance = 1 (plane code 2)
			if l > maxLen {
				l = maxLen
			}
			if l > bestLen {
				bestLen, bestOff = l, 1
			}
		}
		// Hash chain: accept only strictly-longer matches so the cheap-distance
		// heuristics win ties; stop once a match reaches the cap. The bestArgb
		// guard (libwebp HashChainFill line 397) skips, in O(1), any candidate
		// whose pixel at offset bestLen differs from the incumbent's — it cannot
		// be longer — so most probes avoid the full matchLenAt scan.
		if bestLen < lenCap {
			bestArgb := pixels[i+bestLen]
			for iter := iterMax; cur != -1 && i-cur < windowSize && iter > 0; iter-- {
				if pixels[cur+bestLen] == bestArgb {
					if l := matchLenAt(pixels, cur, i, maxLen); l > bestLen {
						bestLen, bestOff = l, i-cur
						if bestLen >= lenCap {
							break
						}
						bestArgb = pixels[i+bestLen]
					}
				}
				cur = prev[cur] - 1
			}
		}
		matchLen[i] = bestLen
		matchOff[i] = bestOff
	}
	return matchLen, matchOff
}

// encodeImageDataOptimal runs the cost-based DP and emits the resulting token
// stream in the same flat format as encodeImageDataGreedy.
func encodeImageDataOptimal(pixels []color.NRGBA, width, colorCacheBits int,
	m *costModel, matchLen, matchOff []int) ([]int, []int) {
	const minMatch = 3
	n := len(pixels)
	useCache := colorCacheBits > 0

	// Per-position color-cache state. Every produced pixel updates the cache
	// regardless of how it is coded, so the cache content at position i depends
	// only on the pixel values before i — independent of the parse. We can
	// therefore precompute hit/index in one forward pass.
	cacheHit := make([]bool, n)
	cacheIdx := make([]int, n)
	if useCache {
		cache := make([]color.NRGBA, 1<<colorCacheBits)
		for i := 0; i < n; i++ {
			h := int(hash(pixels[i], colorCacheBits))
			if i > 0 && cache[h] == pixels[i] {
				cacheHit[i] = true
				cacheIdx[i] = h
			}
			cache[h] = pixels[i]
		}
	}

	litCost := func(i int) float64 {
		if useCache && cacheHit[i] {
			return m.cacheCost(cacheIdx[i]) * cacheCostFactor
		}
		return m.literalCost(pixels[i]) * literalCostFactor
	}

	// Precompute lengthCost[k] (constant per image) so the DP inner loop is a
	// table lookup instead of a prefix-code computation per (start, length).
	lenCostTab := make([]float64, matchMaxLen+1)
	for k := minMatch; k <= matchMaxLen; k++ {
		lenCostTab[k] = m.lengthCost(k)
	}

	// Forward shortest-path DP. cost[i] = min bits to encode pixels[0..i];
	// distArr[i] = length of the token ending at position i (1 = literal/cache).
	cost := make([]float64, n)
	distArr := make([]int, n)
	for i := range cost {
		cost[i] = math.Inf(1)
	}
	for i := 0; i < n; i++ {
		prev := 0.0
		if i > 0 {
			prev = cost[i-1]
		}
		if c := prev + litCost(i); c < cost[i] {
			cost[i] = c
			distArr[i] = 1
		}
		L := matchLen[i]
		if L >= minMatch {
			base := prev + m.distanceCost(distancePlaneCode(matchOff[i], width))
			// Only consider bucket-end lengths <= L, plus L itself. Within a
			// length-code bucket the cost is constant, so the longest length
			// dominates — this keeps the inner loop O(buckets) even when L is large.
			for _, be := range lz77BucketEnds {
				k := be
				if k >= L {
					k = L // clamp the straddling bucket to the exact match length
				}
				if c := base + lenCostTab[k]; c < cost[i+k-1] {
					cost[i+k-1] = c
					distArr[i+k-1] = k
				}
				if be >= L {
					break
				}
			}
		}
	}

	// Trace the chosen token lengths back (end → start) and reverse.
	var lengths []int
	for i := n - 1; i >= 0; {
		k := distArr[i]
		if k == 0 {
			k = 1
		}
		lengths = append(lengths, k)
		i -= k
	}
	for l, r := 0, len(lengths)-1; l < r; l, r = l+1, r-1 {
		lengths[l], lengths[r] = lengths[r], lengths[l]
	}

	// Re-emit. Replicate encodeImageDataGreedy's color-cache update rule exactly
	// (every covered pixel is inserted) so the stream round-trips bit-exact.
	encoded := make([]int, 0, n)
	tokenStart := make([]int, 0, len(lengths))
	var cache []color.NRGBA
	if useCache {
		cache = make([]color.NRGBA, 1<<colorCacheBits)
	}
	pos := 0
	for _, k := range lengths {
		tokenStart = append(tokenStart, pos)
		if k == 1 {
			p := pixels[pos]
			if useCache {
				h := int(hash(p, colorCacheBits))
				if pos > 0 && cache[h] == p {
					encoded = append(encoded, h+256+24)
					pos++
					continue
				}
				cache[h] = p
			}
			encoded = append(encoded, int(p.G), int(p.R), int(p.B), int(p.A))
			pos++
			continue
		}
		off := matchOff[pos]
		if useCache {
			for j := 0; j < k; j++ {
				h := int(hash(pixels[pos+j], colorCacheBits))
				cache[h] = pixels[pos+j]
			}
		}
		s, l := prefixEncodeCode(k)
		encoded = append(encoded, s+256, l)
		s, l = prefixEncodeCode(distancePlaneCode(off, width))
		encoded = append(encoded, s, l)
		pos += k
	}
	return encoded, tokenStart
}
