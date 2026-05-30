package gowebp

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"testing"
)

// TestPSNRSingleSegment encodes with SNS forced to single segment (no
// per-segment quantizer override) to isolate whether the segment header
// signaling is responsible for the encoder/decoder mismatch.
func TestPSNRSingleSegment(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}

	for _, target := range []int{27, 28, 32} {
		origPath := fmt.Sprintf("test_data/frame_compare/original_frame%02d.png", target)
		orig, err := loadPNG(origPath)
		if err != nil {
			continue
		}

		debugForceSingleSegment = true
		var buf bytes.Buffer
		if err := Encode(&buf, orig, &Options{Quality: 75}); err != nil {
			t.Errorf("encode: %v", err)
			continue
		}
		debugForceSingleSegment = false

		tmpWeb := fmt.Sprintf("/tmp/single_seg_%d.webp", target)
		tmpPng := fmt.Sprintf("/tmp/single_seg_%d.png", target)
		os.WriteFile(tmpWeb, buf.Bytes(), 0644)
		exec.Command(dwebp, tmpWeb, "-o", tmpPng).Run()
		decoded, err := loadPNG(tmpPng)
		if err != nil {
			t.Errorf("decode: %v", err)
			continue
		}

		psnr := psnrRGBA(orig, decoded)
		t.Logf("frame %d (single segment): gowebp %.2f dB (%d B)", target, psnr, buf.Len())
	}
}
