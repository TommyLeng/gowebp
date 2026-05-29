package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"
	"os/exec"
	"testing"
)

// TestEncoderSingleMBPattern: encode a small 16×16 NRGBA image (single MB),
// decode, compare. Identifies whether the encoder/decoder disagree for a
// single-MB case (no cross-MB cascade possible).
func TestEncoderSingleMBPattern(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}

	img := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	// Striped gradient: each row has slightly different content
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			v := uint8((y * 20) + x)
			img.Pix[(y*16+x)*4+0] = v
			img.Pix[(y*16+x)*4+1] = v
			img.Pix[(y*16+x)*4+2] = v
			img.Pix[(y*16+x)*4+3] = 255
		}
	}

	var snap []mbInfo
	debugMBStats = &snap
	var reconY []uint8
	var reconStride int
	debugReconCapture = func(r []uint8, stride, h int) {
		reconY = append([]uint8(nil), r...)
		reconStride = stride
	}
	debugDisableLoopFilter = true
	defer func() { debugMBStats = nil; debugReconCapture = nil; debugDisableLoopFilter = false }()

	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 75}); err != nil {
		t.Fatal(err)
	}
	os.WriteFile("/tmp/single.webp", buf.Bytes(), 0644)
	if out, err := exec.Command(dwebp, "-nofancy", "-pgm", "/tmp/single.webp", "-o", "/tmp/single.pgm").CombinedOutput(); err != nil {
		t.Fatalf("dwebp: %v %s", err, out)
	}
	pgmData, err := os.ReadFile("/tmp/single.pgm")
	if err != nil {
		t.Fatal(err)
	}
	hdrEnd := bytes.Index(pgmData, []byte("255\n"))
	raw := pgmData[hdrEnd+4:]

	t.Logf("single MB info: isI4=%v i16Mode=%d  segment=%d",
		snap[0].isI4, snap[0].i16Mode, snap[0].segment)

	var maxD int
	for y := 0; y < 16; y++ {
		var row string
		for x := 0; x < 16; x++ {
			yDec := int(raw[y*16+x])
			yEnc := int(reconY[y*reconStride+x])
			d := yDec - yEnc
			if d < 0 {
				d = -d
			}
			if d > maxD {
				maxD = d
			}
			row += fmt.Sprintf(" %3d/%3d", yEnc, yDec)
		}
		t.Logf("y=%2d:%s", y, row)
	}
	t.Logf("max Δ: %d", maxD)
}

// TestEncoderReconVsDecoder hooks into the encoder to capture the final recon
// buffer (Y plane, padded), then decodes the output with dwebp and compares
// pixel-by-pixel. If they differ, the encoder writes bytes that the decoder
// interprets differently from what the encoder expected.
func TestEncoderReconVsDecoder(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}

	origPath := "test_data/frame_compare/original_frame27.png"
	orig, err := loadPNG(origPath)
	if err != nil {
		t.Skipf("missing %s", origPath)
	}

	var snap []mbInfo
	debugMBStats = &snap
	defer func() { debugMBStats = nil }()

	// Hook into encoder: capture final recon Y plane.
	var reconYCapture []uint8
	var reconStrideCapture int
	debugReconCapture = func(r []uint8, stride, h int) {
		reconYCapture = make([]uint8, len(r))
		copy(reconYCapture, r)
		reconStrideCapture = stride
	}
	debugDisableLoopFilter = true
	defer func() { debugReconCapture = nil; debugDisableLoopFilter = false }()

	var buf bytes.Buffer
	if err := Encode(&buf, orig, &Options{Quality: 75}); err != nil {
		t.Fatal(err)
	}
	t.Logf("encoded %d bytes, mb=%d", buf.Len(), len(snap))

	// Decode with dwebp to raw YUV.
	os.WriteFile("/tmp/recon_diff.webp", buf.Bytes(), 0644)
	if out, err := exec.Command(dwebp, "-pgm", "/tmp/recon_diff.webp", "-o", "/tmp/recon_diff.pgm").CombinedOutput(); err != nil {
		t.Fatalf("dwebp: %v %s", err, out)
	}
	pgmData, err := os.ReadFile("/tmp/recon_diff.pgm")
	if err != nil {
		t.Fatal(err)
	}
	// Parse PGM header: "P5\n<w> <h>\n255\n" then raw bytes
	// PGM contains Y plane and 4:2:0 UV planes stacked vertically.
	hdrEnd := bytes.Index(pgmData, []byte("255\n"))
	if hdrEnd < 0 {
		t.Fatal("bad pgm")
	}
	raw := pgmData[hdrEnd+4:]
	// First imgW*imgH bytes = Y plane (using image size, padded)
	b := orig.Bounds()
	imgW, imgH := b.Max.X, b.Max.Y

	// PGM dimensions include 4:2:0 layout; we want just the Y plane (top part).
	// Each line is imgW pixels wide; first imgH lines are Y.
	mismatchCount := 0
	maxDiff := 0
	var sampleDiffs []string
	for y := 0; y < imgH; y++ {
		for x := 0; x < imgW; x++ {
			yDec := int(raw[y*imgW+x])
			yEnc := int(reconYCapture[y*reconStrideCapture+x])
			d := yDec - yEnc
			if d < 0 {
				d = -d
			}
			if d > 0 {
				mismatchCount++
				if d > maxDiff {
					maxDiff = d
				}
				if len(sampleDiffs) < 20 {
					sampleDiffs = append(sampleDiffs, fmt.Sprintf("(%d,%d) MB(%d,%d) enc=%d dec=%d Δ=%d", x, y, x/16, y/16, yEnc, yDec, d))
				}
			}
		}
	}
	decoded := image.NewGray(orig.Bounds())
	t.Logf("Y mismatches: %d / %d (%.1f%%), maxDiff=%d", mismatchCount, imgW*imgH, 100*float64(mismatchCount)/float64(imgW*imgH), maxDiff)
	for _, s := range sampleDiffs {
		t.Log(s)
	}

	t.Logf("MB(0,0) corner (first 8×8) enc/dec:")
	for y := 0; y < 8; y++ {
		var row string
		for x := 0; x < 8; x++ {
			yDec := int(raw[y*imgW+x])
			yEnc := int(reconYCapture[y*reconStrideCapture+x])
			row += fmt.Sprintf(" %3d/%3d", yEnc, yDec)
		}
		t.Logf(" y=%d:%s", y, row)
	}
	t.Logf("MB(0,0) info: isI4=%v i16Mode=%d uvMode=%d segment=%d",
		snap[0].isI4, snap[0].i16Mode, snap[0].uvMode, snap[0].segment)

	// Count mismatches per-MB to find which MBs disagree.
	mbMismatch := map[[2]int]int{}
	for y := 0; y < imgH; y++ {
		for x := 0; x < imgW; x++ {
			yDec := int(raw[y*imgW+x])
			yEnc := int(reconYCapture[y*reconStrideCapture+x])
			if yDec != yEnc {
				mbMismatch[[2]int{x / 16, y / 16}]++
			}
		}
	}
	// Show MBs with no mismatch (start) and worst mismatches.
	mbs := len(mbMismatch)
	t.Logf("MBs with at least one Y mismatch: %d / %d", mbs, ((imgW+15)/16)*((imgH+15)/16))
	// Find first MB that has mismatches.
	for ry := 0; ry < (imgH+15)/16; ry++ {
		for rx := 0; rx < (imgW+15)/16; rx++ {
			if mbMismatch[[2]int{rx, ry}] > 0 {
				m := snap[ry*((imgW+15)/16)+rx]
				t.Logf("first mismatched MB(%d,%d): %d pixels diff, isI4=%v i16Mode=%d seg=%d",
					rx, ry, mbMismatch[[2]int{rx, ry}], m.isI4, m.i16Mode, m.segment)
				goto done
			}
		}
	}
done:
	_ = decoded

	// Print first 4 rows of MB(2,0) for debugging
	t.Logf("MB(2,0) first 4 rows (x=32..47):")
	for y := 0; y < 4; y++ {
		var row string
		for x := 32; x < 48; x++ {
			yDec := int(raw[y*imgW+x])
			yEnc := int(reconYCapture[y*reconStrideCapture+x])
			row += fmt.Sprintf(" %3d/%3d", yEnc, yDec)
		}
		t.Logf(" y=%d:%s", y, row)
	}

	_ = image.Black
	_ = png.Decode
}
