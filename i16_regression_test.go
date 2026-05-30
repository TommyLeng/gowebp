package gowebp

import (
	"bytes"
	"os"
	"os/exec"
	"testing"
)

// TestI16MixedModeDecodes guards against the i16 Y2-DC NZ-context regression.
//
// Background: VP8's ParseResiduals leaves nz_dc UNCHANGED for i4 MBs (no Y2
// block), while gowebp's emission used to reset it to 0. Once mixed i4/i16
// mode selection was enabled, an i16 MB following an i4 neighbour computed a
// different Y2-DC context than the decoder, desyncing the bool decoder and
// collapsing PSNR (~9 dB) or producing an undecodable stream.
//
// This test encodes images that select a substantial fraction of i16 MBs,
// decodes them with C dwebp (the reference decoder), and asserts the round-trip
// PSNR stays well above the desync floor. It runs at the default GOMAXPROCS so
// the wave-front parallel path is exercised too.
func TestI16MixedModeDecodes(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}

	images := []string{
		"test_data/frame_compare/original_frame27.png",
		"test_data/original/i1-a.png",
	}

	for _, src := range images {
		orig, err := loadPNG(src)
		if err != nil {
			t.Logf("skip %s: %v", src, err)
			continue
		}

		var snap []mbInfo
		debugMBStats = &snap
		var buf bytes.Buffer
		if err := Encode(&buf, orig, &Options{Quality: 75}); err != nil {
			debugMBStats = nil
			t.Fatalf("encode %s: %v", src, err)
		}
		debugMBStats = nil

		var i16 int
		for _, mi := range snap {
			if !mi.isI4 {
				i16++
			}
		}

		tmp := "/tmp/i16_regression.webp"
		if err := os.WriteFile(tmp, buf.Bytes(), 0644); err != nil {
			t.Fatal(err)
		}
		dec, err := dwebpDecode(dwebp, tmp)
		if err != nil {
			t.Fatalf("%s: dwebp failed to decode (likely bitstream desync): %v", src, err)
		}
		psnr := psnrRGBA(orig, dec)
		t.Logf("%s: i16=%d/%d (%.0f%%)  round-trip PSNR=%.2f dB",
			src, i16, len(snap), 100*float64(i16)/float64(len(snap)), psnr)

		if i16 == 0 {
			t.Errorf("%s: expected some i16 MBs (mixed mode), got 0 — fix may be masked", src)
		}
		if psnr < 30 {
			t.Errorf("%s: round-trip PSNR %.2f dB below 30 dB floor — i16 decoder desync regressed", src, psnr)
		}
	}
}
