package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"os/exec"
	"testing"
)

// TestModeDistribution re-encodes frame 27 with mbInfos snapshotting enabled
// so we can count how often gowebp picks i4 vs i16.
func TestModeDistribution(t *testing.T) {
	origPath := "test_data/frame_compare/original_frame27.png"
	orig, err := loadPNG(origPath)
	if err != nil {
		t.Skipf("missing %s", origPath)
	}

	var snap []mbInfo
	debugMBStats = &snap
	defer func() { debugMBStats = nil }()

	var buf bytes.Buffer
	if err := Encode(&buf, orig, &Options{Quality: 75}); err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(snap) == 0 {
		t.Fatal("no mbInfos snapshot captured")
	}

	var i4, i16 int
	uvModes := [4]int{}
	i16Modes := [4]int{}
	for _, mi := range snap {
		if mi.isI4 {
			i4++
		} else {
			i16++
			i16Modes[mi.i16Mode]++
		}
		uvModes[mi.uvMode]++
	}
	t.Logf("MB total=%d  i4=%d (%.1f%%)  i16=%d (%.1f%%)",
		len(snap), i4, 100*float64(i4)/float64(len(snap)),
		i16, 100*float64(i16)/float64(len(snap)))
	t.Logf("i16 modes: DC=%d V=%d H=%d TM=%d",
		i16Modes[0], i16Modes[1], i16Modes[2], i16Modes[3])
	t.Logf("UV  modes: DC=%d V=%d H=%d TM=%d",
		uvModes[0], uvModes[1], uvModes[2], uvModes[3])
}

// TestPSNRBaseline encodes the worst frames with gowebp, decodes via C dwebp,
// and compares to the original PNG. Also counts MB modes by tapping a global
// counter inside encodeFrame (set via a debug env var or build tag — for now
// we just read mode counts from the static encode test artefacts).
//
// Goal: produce a quantitative baseline (gowebp PSNR vs libwebp PSNR) so we
// can measure improvements from the lambda fix.
func TestPSNRBaseline(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}

	for _, target := range []int{27, 28, 32} {
		origPath := fmt.Sprintf("test_data/frame_compare/original_frame%02d.png", target)
		goWebP := fmt.Sprintf("test_data/frame_compare/gowebp_lossy_frame%02d.webp", target)
		libWebP := fmt.Sprintf("test_data/frame_compare/libwebp_lossy_frame%02d.webp", target)

		orig, err := loadPNG(origPath)
		if err != nil {
			t.Logf("frame %d: missing %s, skipping", target, origPath)
			continue
		}

		goImg, err := dwebpDecode(dwebp, goWebP)
		if err != nil {
			t.Errorf("frame %d gowebp decode: %v", target, err)
			continue
		}
		libImg, err := dwebpDecode(dwebp, libWebP)
		if err != nil {
			t.Errorf("frame %d libwebp decode: %v", target, err)
			continue
		}

		goPSNR := psnrRGBA(orig, goImg)
		libPSNR := psnrRGBA(orig, libImg)
		goFI, _ := os.Stat(goWebP)
		libFI, _ := os.Stat(libWebP)
		t.Logf("frame %d: gowebp %.2f dB (%d B)  libwebp %.2f dB (%d B)  Δ=%.2f dB",
			target, goPSNR, goFI.Size(), libPSNR, libFI.Size(), libPSNR-goPSNR)

		_ = math.Pi // keep math import
	}
}

func loadPNG(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return png.Decode(f)
}

func dwebpDecode(dwebpBin, webpPath string) (image.Image, error) {
	cmd := exec.Command(dwebpBin, webpPath, "-o", "-")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = nil
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("dwebp: %w", err)
	}
	return png.Decode(&out)
}
