package lossless

import (
    "bytes"
    "image/color"
)

// Meta-Huffman support.
//
// VP8L lets the top-level ARGB image use multiple Huffman code groups, selected
// per tile via an "entropy image". A single global Huffman code is a poor fit
// for content whose local statistics vary (e.g. a flat region next to a
// detailed one); per-region codes adapt and shrink the stream. Decoder side is
// golang.org/x/image/vp8l decodeHuffmanGroups: meta_index = (R<<8)|G of the
// entropy-image tile, tile = (y>>hBits)*tilesPerRow + (x>>hBits).

const (
    metaTileBits = 4    // 16x16 tiles (raised for very large images to cap count)
    metaMaxTiles = 1024 // upper bound on tile count (memory / time)
    metaNumBins  = 16   // entropy bins used to cluster tiles into groups
)

func nTilesL(size, bits int) int { return (size + (1 << bits) - 1) >> bits }

// metaPlan is the result of clustering image tiles into Huffman groups.
type metaPlan struct {
    hBits       int
    bw, bh      int
    numGroups   int
    groupOfTile []int // len bw*bh, group index per tile
}

// newHistoSet allocates the 5 per-stream histograms (green+len+cache, R, B, A,
// distance) with the correct alphabet sizes.
func newHistoSet(cacheSize int) [][]int {
    return [][]int{
        make([]int, 256+24+cacheSize),
        make([]int, 256),
        make([]int, 256),
        make([]int, 256),
        make([]int, 40),
    }
}

// addToken accumulates one token's Huffman symbols into a 5-stream histogram set
// and returns the number of ints consumed from the encoded stream.
func addToken(h [][]int, encoded []int, i int) int {
    sym := encoded[i]
    h[0][sym]++
    if sym < 256 { // literal: G already counted, then R,B,A
        h[1][encoded[i+1]]++
        h[2][encoded[i+2]]++
        h[3][encoded[i+3]]++
        return 4
    } else if sym < 256+24 { // LZ77 copy: length sym (counted) + distance sym
        h[4][encoded[i+2]]++
        return 4
    }
    return 1 // color-cache reference: only the green-stream symbol
}

// planMetaHuffman builds per-tile green-stream histograms from the LZ77 token
// stream and clusters tiles into Huffman groups by entropy. It returns a plan
// with numGroups>1 when the tiles fall into more than one entropy bin; the
// caller still verifies meta-Huffman is actually smaller before using it.
func planMetaHuffman(encoded, tokenStart []int, width, height, cacheBits int) *metaPlan {
    hBits := metaTileBits
    for nTilesL(width, hBits)*nTilesL(height, hBits) > metaMaxTiles {
        hBits++
    }
    bw := nTilesL(width, hBits)
    bh := nTilesL(height, hBits)
    numTiles := bw * bh
    if numTiles < 2 {
        return nil
    }

    cacheSize := 0
    if cacheBits > 0 {
        cacheSize = 1 << cacheBits
    }
    greenLen := 256 + 24 + cacheSize

    // Per-tile green-stream histogram (the dominant cost; a cheap proxy that
    // separates flat from detailed regions).
    greenHisto := make([][]int, numTiles)
    for t := range greenHisto {
        greenHisto[t] = make([]int, greenLen)
    }
    for k, i := 0, 0; i < len(encoded); k++ {
        pi := tokenStart[k]
        t := (pi/width>>hBits)*bw + (pi%width >> hBits)
        sym := encoded[i]
        greenHisto[t][sym]++
        if sym < 256 || sym < 256+24 {
            i += 4
        } else {
            i++
        }
    }

    // Bin each tile by its green entropy (bits/symbol).
    groupOfTile := make([]int, numTiles)
    binUsed := make([]bool, metaNumBins)
    for t := 0; t < numTiles; t++ {
        total := 0
        for _, c := range greenHisto[t] {
            total += c
        }
        if total == 0 {
            groupOfTile[t] = 0 // empty tile: its group is never used for decoding
            continue
        }
        e := histogramEntropy(greenHisto[t], total) // 0..~log2(greenLen)
        b := int(e / 9.0 * float64(metaNumBins))
        if b < 0 {
            b = 0
        } else if b >= metaNumBins {
            b = metaNumBins - 1
        }
        groupOfTile[t] = b
        binUsed[b] = true
    }

    // Renumber used bins to a dense 0..numGroups-1.
    remap := make([]int, metaNumBins)
    numGroups := 0
    for b := 0; b < metaNumBins; b++ {
        if binUsed[b] {
            remap[b] = numGroups
            numGroups++
        }
    }
    if numGroups <= 1 {
        return nil
    }
    for t := 0; t < numTiles; t++ {
        groupOfTile[t] = remap[groupOfTile[t]]
    }

    // The entropy bins tend to over-split (each extra group costs ~5 Huffman
    // code tables). Greedily merge groups while a merge lowers total cost
    // (combined data entropy + the per-group table overhead saved). This uses
    // the full 5-stream histograms, not just the green proxy used for binning.
    grp := make([][][]int, numGroups)
    for g := range grp {
        grp[g] = newHistoSet(cacheSize)
    }
    for k, i := 0, 0; i < len(encoded); k++ {
        pi := tokenStart[k]
        t := (pi/width>>hBits)*bw + (pi%width >> hBits)
        i += addToken(grp[groupOfTile[t]], encoded, i)
    }
    parent := mergeGroupsGreedy(grp)

    // Renumber surviving roots to a dense 0..final-1 and apply to the tiles.
    rootID := make(map[int]int)
    final := 0
    for g := 0; g < numGroups; g++ {
        r := findRoot(parent, g)
        if _, ok := rootID[r]; !ok {
            rootID[r] = final
            final++
        }
    }
    for t := 0; t < numTiles; t++ {
        groupOfTile[t] = rootID[findRoot(parent, groupOfTile[t])]
    }
    if final <= 1 {
        return nil
    }
    return &metaPlan{hBits: hBits, bw: bw, bh: bh, numGroups: final, groupOfTile: groupOfTile}
}

// mergeOverheadBits estimates the bits needed to transmit one Huffman group's 5
// code tables. Merging two groups removes one such overhead, at the cost of the
// extra entropy of a combined (less specialised) histogram. Used only as a
// merge heuristic; the caller still verifies the meta stream is actually
// smaller, so an imperfect estimate can never cause a size regression.
const mergeOverheadBits = 720.0

// mergeGroupsGreedy repeatedly merges the pair of groups whose combination most
// reduces total cost, until no merge helps. Returns a union-find parent array
// mapping each input group to its surviving root.
func mergeGroupsGreedy(grp [][][]int) []int {
    n := len(grp)
    parent := make([]int, n)
    active := make([]bool, n)
    cost := make([]float64, n)
    for g := range grp {
        parent[g] = g
        active[g] = true
        cost[g] = dataCostBits(grp[g])
    }
    for {
        bestA, bestB, bestDelta := -1, -1, -1e-9
        for a := 0; a < n; a++ {
            if !active[a] {
                continue
            }
            for b := a + 1; b < n; b++ {
                if !active[b] {
                    continue
                }
                delta := dataCostBits(mergeHistos(grp[a], grp[b])) - cost[a] - cost[b] - mergeOverheadBits
                if delta < bestDelta {
                    bestDelta, bestA, bestB = delta, a, b
                }
            }
        }
        if bestA < 0 {
            break
        }
        grp[bestA] = mergeHistos(grp[bestA], grp[bestB])
        cost[bestA] = dataCostBits(grp[bestA])
        active[bestB] = false
        parent[bestB] = bestA
    }
    return parent
}

func findRoot(parent []int, g int) int {
    for parent[g] != g {
        parent[g] = parent[parent[g]]
        g = parent[g]
    }
    return g
}

// streamCost returns the entropy-coded bit cost of one symbol histogram.
func streamCost(h []int) float64 {
    total := 0
    for _, c := range h {
        total += c
    }
    if total == 0 {
        return 0
    }
    return histogramEntropy(h, total) * float64(total)
}

// dataCostBits returns the summed entropy bit cost of a 5-stream histogram set.
func dataCostBits(histos [][]int) float64 {
    s := 0.0
    for _, h := range histos {
        s += streamCost(h)
    }
    return s
}

// mergeHistos returns the element-wise sum of two 5-stream histogram sets.
func mergeHistos(a, b [][]int) [][]int {
    out := make([][]int, len(a))
    for s := range a {
        out[s] = make([]int, len(a[s]))
        for j := range a[s] {
            out[s][j] = a[s][j] + b[s][j]
        }
    }
    return out
}

// groupOf returns the Huffman group index for the tile containing the given
// pixel (matching the decoder's tile lookup).
func (p *metaPlan) groupOf(pixelIndex, width int) int {
    t := (pixelIndex/width>>p.hBits)*p.bw + (pixelIndex%width >> p.hBits)
    return p.groupOfTile[t]
}

// buildEntropyImage renders the plan's per-tile group indices as a bw*bh pixel
// slice where meta_index = (R<<8)|G, matching the decoder.
func buildEntropyImage(plan *metaPlan) []color.NRGBA {
    pix := make([]color.NRGBA, plan.bw*plan.bh)
    for t := range pix {
        g := plan.groupOfTile[t]
        pix[t] = color.NRGBA{R: uint8(g >> 8), G: uint8(g & 0xff), B: 0, A: 255}
    }
    return pix
}

// newTempWriter returns a standalone bitWriter backed by a fresh buffer.
func newTempWriter() *bitWriter { return &bitWriter{Buffer: &bytes.Buffer{}} }

// bitLen returns the number of bits written so far.
func (w *bitWriter) bitLen() int { return w.Buffer.Len()*8 + w.BitBufferSize }

// appendBits copies all bits from t into w (t need not be byte-aligned).
func appendBits(w *bitWriter, t *bitWriter) {
    for _, b := range t.Buffer.Bytes() {
        w.writeBits(uint64(b), 8)
    }
    if t.BitBufferSize > 0 {
        w.writeBits(t.BitBuffer&((1<<t.BitBufferSize)-1), t.BitBufferSize)
    }
}
