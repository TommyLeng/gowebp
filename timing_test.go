package gowebp

import (
	"bytes"
	"image/jpeg"
	"os"
	"testing"
	"time"
)

// TestLZ77Timing reports wall-clock for the lossless alpha and full-lossless
// paths (cost-based LZ77). Diagnostic only — set GOWEBP_SIZES=1 to run.
func TestLZ77Timing(t *testing.T) {
	if os.Getenv("GOWEBP_SIZES") == "" {
		t.Skip("set GOWEBP_SIZES=1 to run timing")
	}
	if orig, err := loadPNG("test_data/original/i1-a.png"); err == nil {
		st := time.Now()
		var buf bytes.Buffer
		Encode(&buf, orig, &Options{Quality: 75})
		t.Logf("i1-a (1.1MP, lossy+alpha): %v  -> %d B", time.Since(st), buf.Len())
	}
	if f, err := os.Open("test_data/original/jable-heidilau0905-003.jpg"); err == nil {
		defer f.Close()
		if heidi, err := jpeg.Decode(f); err == nil {
			st := time.Now()
			var buf bytes.Buffer
			Encode(&buf, heidi, &Options{Lossless: true})
			t.Logf("heidi (1.5MP, full lossless): %v  -> %d B", time.Since(st), buf.Len())
		}
	}
}
