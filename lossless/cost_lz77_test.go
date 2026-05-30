package lossless

import (
	"image/color"
	"testing"
)

// fullCost is the order-0 token-stream cost INCLUDING the raw length/distance
// extra bits — the metric encodeImageData uses to choose greedy vs optimal.
func fullCost(enc []int, ccBits int) float64 {
	return dataCostBits(computeHistograms(enc, ccBits)) + streamExtraBits(enc)
}

// makeRunHeavy builds a single-channel run-heavy stream (alpha-mask shaped):
// large constant blocks separated by short ramps, carried as G with R=B=0,A=255.
func makeRunHeavy(w, h int) []color.NRGBA {
	px := make([]color.NRGBA, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var g uint8
			switch {
			case x < w/3:
				g = 0
			case x < 2*w/3:
				g = uint8((x - w/3) * 255 / (w / 3)) // ramp (antialiased-edge analogue)
			default:
				g = 255
			}
			px[y*w+x] = color.NRGBA{G: g, A: 255}
		}
	}
	return px
}

// TestFillMatchesValid checks that every recorded match is a real backward
// match: pixels[i..i+len) must equal pixels[i-off..).
func TestFillMatchesValid(t *testing.T) {
	w, h := 120, 90
	px := makeRunHeavy(w, h)
	matchLen, matchOff := fillMatches(px, w)
	for i := range matchLen {
		l, off := matchLen[i], matchOff[i]
		if l == 0 {
			continue
		}
		if off <= 0 || off > i {
			t.Fatalf("pos %d: bad offset %d (len %d)", i, off, l)
		}
		for k := 0; k < l; k++ {
			if px[i+k] != px[i-off+k] {
				t.Fatalf("pos %d: match len=%d off=%d invalid at k=%d", i, l, off, k)
			}
		}
	}
}

// TestOptimalNotWorseThanGreedy verifies the cost-based parse is selected only
// when it genuinely lowers the full cost (extra bits included), and that on
// run-heavy data it does win — the core value of the cost model.
func TestOptimalNotWorseThanGreedy(t *testing.T) {
	for _, ccBits := range []int{0, 4} {
		w, h := 200, 150
		px := makeRunHeavy(w, h)

		encG, _ := encodeImageDataGreedy(px, w, h, ccBits)
		matchLen, matchOff := fillMatches(px, w)
		model := buildCostModel(computeHistograms(encG, ccBits))
		encO, _ := encodeImageDataOptimal(px, w, ccBits, model, matchLen, matchOff)

		cg := fullCost(encG, ccBits)
		co := fullCost(encO, ccBits)
		t.Logf("cc=%d  greedy full-cost=%.0f B  optimal full-cost=%.0f B  (%.1f%%)",
			ccBits, cg/8, co/8, 100*(co-cg)/cg)
		if co > cg {
			t.Errorf("cc=%d: optimal full-cost %.0f > greedy %.0f — selection would regress", ccBits, co, cg)
		}
	}
}
