package gowebp

import (
	"bytes"
	"image"
	"image/jpeg"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestLZ77Baseline records lossless sizes (alpha ALPH chunk + general RGB
// lossless) and compares them against cwebp. It forks cwebp many times, so it
// is a measurement tool, not a routine gate: set GOWEBP_SIZES=1 to run it.
// Routine size protection lives in TestAlphaSubtractGreenRegression.
func TestLZ77Baseline(t *testing.T) {
	if os.Getenv("GOWEBP_SIZES") == "" {
		t.Skip("set GOWEBP_SIZES=1 to run the cwebp size comparison")
	}
	cwebp, hasCwebp := exec.LookPath("cwebp")
	if hasCwebp != nil {
		t.Log("cwebp not found — gowebp-only numbers")
	}
	tmp := t.TempDir()

	// cwebpLossless runs cwebp -lossless -m6 -exact and returns the output size.
	cwebpLossless := func(srcPNG string) int {
		if hasCwebp != nil {
			return -1
		}
		out := filepath.Join(tmp, filepath.Base(srcPNG)+".ll.webp")
		cmd := exec.Command(cwebp, "-lossless", "-m", "6", "-exact", "-quiet", srcPNG, "-o", out)
		if err := cmd.Run(); err != nil {
			return -1
		}
		fi, err := os.Stat(out)
		if err != nil {
			return -1
		}
		return int(fi.Size())
	}

	// cwebpALPH runs cwebp -q 75 (lossy VP8 + lossless ALPH chunk, the same dual
	// path gowebp's Quality:75 uses) and returns the size of the ALPH chunk only —
	// the apples-to-apples comparison for gowebp's alpha-plane encoder.
	cwebpALPH := func(srcPNG string) int {
		if hasCwebp != nil {
			return -1
		}
		out := filepath.Join(tmp, filepath.Base(srcPNG)+".q75.webp")
		cmd := exec.Command(cwebp, "-q", "75", "-exact", "-quiet", srcPNG, "-o", out)
		if err := cmd.Run(); err != nil {
			return -1
		}
		data, err := os.ReadFile(out)
		if err != nil {
			return -1
		}
		return webpChunkSize(data, "ALPH")
	}

	// --- Alpha planes (ALPH chunk size is the lossless-encoded alpha) ---
	alphaCases := []string{
		"test_data/original/i1-a.png",
		"test_data/original/i11-a.png",
		"test_data/original/i18-a.png",
		"test_data/original/jable-heidilau0905-004-a.png",
	}
	t.Log("=== ALPHA planes (gowebp ALPH chunk vs cwebp full-image lossless) ===")
	for _, p := range alphaCases {
		orig, err := loadPNG(p)
		if err != nil {
			continue
		}
		var buf bytes.Buffer
		if err := Encode(&buf, orig, &Options{Quality: 75}); err != nil {
			t.Errorf("%s: %v", p, err)
			continue
		}
		alph := webpChunkSize(buf.Bytes(), "ALPH")
		cw := cwebpALPH(p)
		pct := 0.0
		if cw > 0 {
			pct = 100 * float64(alph-cw) / float64(cw)
		}
		t.Logf("%-44s gowebp ALPH=%6d B   cwebp ALPH=%6d B   (%+.1f%%)", filepath.Base(p), alph, cw, pct)
	}

	// --- General RGB lossless ---
	t.Log("=== GENERAL lossless (gowebp full vs cwebp full) ===")
	rgbCases := []string{
		"test_data/original/jable-heidilau0905-003.jpg",
		"test_data/original/j3.png",
	}
	for _, p := range rgbCases {
		var img image.Image
		f, err := os.Open(p)
		if err != nil {
			continue
		}
		if filepath.Ext(p) == ".jpg" || filepath.Ext(p) == ".jpeg" {
			img, err = jpeg.Decode(f)
		} else {
			img, err = loadPNG(p)
		}
		f.Close()
		if err != nil || img == nil {
			continue
		}
		var buf bytes.Buffer
		if err := Encode(&buf, img, &Options{Lossless: true}); err != nil {
			t.Errorf("%s: %v", p, err)
			continue
		}
		cw := cwebpLossless(p)
		t.Logf("%-48s gowebp=%7d B   cwebp=%7d B", filepath.Base(p), buf.Len(), cw)
	}
}
