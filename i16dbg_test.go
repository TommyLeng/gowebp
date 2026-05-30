package gowebp

import (
	"bytes"
	"image/jpeg"
	"os"
	"testing"
)

// TestAlphaSubtractGreenRegression guards the lossless transform analysis
// (lossless/transform.go: useSubtractGreen). Subtract-green must be SKIPPED for
// single-channel data — an alpha plane carried in the green channel with R=B=0,
// where R-G/B-G turn two constant channels into copies of the alpha signal and
// inflate the stream — and KEPT for correlated RGB photos where it roughly
// halves the size.
func TestAlphaSubtractGreenRegression(t *testing.T) {
	// (1) i1-a carries a real alpha channel. Its ALPH (lossless VP8L) chunk was
	// ~46 KB when subtract-green was forced on the single-channel alpha, ~33 KB
	// after the transform analysis, and ~28 KB after the cost-based LZ77 parse
	// (cost_lz77.go) replaced the distance-cost-blind greedy match selection.
	if orig, err := loadPNG("test_data/original/i1-a.png"); err == nil {
		var buf bytes.Buffer
		if err := Encode(&buf, orig, &Options{Quality: 75}); err != nil {
			t.Fatal(err)
		}
		alph := webpChunkSize(buf.Bytes(), "ALPH")
		t.Logf("i1-a ALPH chunk: %d bytes", alph)
		if alph > 29000 {
			t.Errorf("ALPH chunk %d bytes — cost-based LZ77 / subtract-green regressed (want <29000)", alph)
		}
	}

	// (2) A no-alpha RGB photo encoded losslessly MUST keep subtract-green
	// (forcing it off ~doubles the size: 1.32 MB -> 2.03 MB on heidi). Guards
	// against the analysis wrongly disabling it.
	if hf, err := os.Open("test_data/original/jable-heidilau0905-003.jpg"); err == nil {
		defer hf.Close()
		if heidi, err := jpeg.Decode(hf); err == nil {
			var lb bytes.Buffer
			if err := Encode(&lb, heidi, &Options{Lossless: true}); err != nil {
				t.Fatal(err)
			}
			t.Logf("heidi lossless: %d bytes", lb.Len())
			if lb.Len() > 1600000 {
				t.Errorf("heidi lossless %d bytes — subtract-green wrongly disabled (want <1.6M)", lb.Len())
			}
		}
	}
}

// webpChunkSize returns the payload size of the named RIFF chunk, or -1.
func webpChunkSize(data []byte, fourcc string) int {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
		return -1
	}
	off := 12
	for off+8 <= len(data) {
		cc := string(data[off : off+4])
		sz := int(data[off+4]) | int(data[off+5])<<8 | int(data[off+6])<<16 | int(data[off+7])<<24
		if cc == fourcc {
			return sz
		}
		off += 8 + sz + (sz & 1)
	}
	return -1
}
