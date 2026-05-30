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

// TestVEPREDDump: dump the i16 internal state for forced VE_PRED on all-127
// image. Expected: pred=127, all DC/AC levels=0, recon=127. Anything else
// is the bug.
func TestVEPREDDump(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}

	img := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+0] = 127
		img.Pix[i+1] = 127
		img.Pix[i+2] = 127
		img.Pix[i+3] = 255
	}

	dumps := map[[2]int]*debugI16Dump{}
	debugDumpI16Capture = &dumps
	debugDisableLoopFilter = true
	debugForceI16Mode = I16_VE_PRED // = 2 in gowebp (DC=0, TM=1, VE=2, HE=3)
	defer func() { debugDumpI16Capture = nil; debugDisableLoopFilter = false; debugForceI16Mode = -1 }()

	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 75}); err != nil {
		t.Fatal(err)
	}
	os.WriteFile("/tmp/dump.webp", buf.Bytes(), 0644)
	exec.Command(dwebp, "-nofancy", "-pgm", "/tmp/dump.webp", "-o", "/tmp/dump.pgm").Run()
	pgmData, _ := os.ReadFile("/tmp/dump.pgm")
	hdrEnd := bytes.Index(pgmData, []byte("255\n"))
	raw := pgmData[hdrEnd+4:]

	d, ok := dumps[[2]int{0, 0}]
	if !ok {
		t.Fatal("no dump captured")
	}

	if early, ok2 := dumps[[2]int{-1, -1}]; ok2 {
		t.Logf("EARLY: bestI16Mode=%d  pred[0..3]=%d %d %d %d", early.dcLevels[15], early.pred[0], early.pred[1], early.pred[2], early.pred[3])
	}

	// pred should be 127 for VE_PRED no-top
	t.Logf("pred[0..3]: %d %d %d %d  (expected 127 each)", d.pred[0], d.pred[1], d.pred[2], d.pred[3])
	t.Logf("yDcRaw[0..15]: %v  (expected all 0)", d.yDcRaw)
	t.Logf("whtOut[0..15]: %v  (expected all 0)", d.whtOut)
	t.Logf("dcLevels[0..3]: %d %d %d %d  (expected 0)", d.dcLevels[0], d.dcLevels[1], d.dcLevels[2], d.dcLevels[3])
	t.Logf("dcBlockCoeff[0..3]: %d %d %d %d  (expected 0)", d.dcBlockCoeff[0], d.dcBlockCoeff[1], d.dcBlockCoeff[2], d.dcBlockCoeff[3])
	t.Logf("acLevels[0][0..3]: %d %d %d %d (block 0)", d.acLevels[0][0], d.acLevels[0][1], d.acLevels[0][2], d.acLevels[0][3])
	t.Logf("recon[0..3]: %d %d %d %d  (expected 127)", d.recon[0], d.recon[1], d.recon[2], d.recon[3])
	t.Logf("decoder[0..3]: %d %d %d %d", raw[0], raw[1], raw[2], raw[3])
}

// TestSingleMBForceVEPRED: encode a 16x16 image where VE_PRED is the
// unambiguous winner (all pixels = 127, so VE_PRED no-top gives 0 residual).
func TestSingleMBForceVEPRED(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}

	img := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i+0] = 127
		img.Pix[i+1] = 127
		img.Pix[i+2] = 127
		img.Pix[i+3] = 255
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
	debugForceI16Mode = 1 // VE_PRED
	defer func() { debugMBStats = nil; debugReconCapture = nil; debugDisableLoopFilter = false; debugForceI16Mode = -1 }()

	var buf bytes.Buffer
	if err := Encode(&buf, img, &Options{Quality: 75}); err != nil {
		t.Fatal(err)
	}
	os.WriteFile("/tmp/ve.webp", buf.Bytes(), 0644)
	if out, err := exec.Command(dwebp, "-nofancy", "-pgm", "/tmp/ve.webp", "-o", "/tmp/ve.pgm").CombinedOutput(); err != nil {
		t.Fatalf("dwebp: %v %s", err, out)
	}
	pgmData, err := os.ReadFile("/tmp/ve.pgm")
	if err != nil {
		t.Fatal(err)
	}
	hdrEnd := bytes.Index(pgmData, []byte("255\n"))
	raw := pgmData[hdrEnd+4:]

	t.Logf("MB info: isI4=%v i16Mode=%d segment=%d", snap[0].isI4, snap[0].i16Mode, snap[0].segment)
	var maxD int
	for y := 0; y < 16; y++ {
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
		}
	}
	t.Logf("[0,0]: enc=%d dec=%d  max Δ: %d", reconY[0], raw[0], maxD)
}

// TestSingleMBVerticalGradient: encode a single 16x16 MB with content
// favouring VE_PRED (vertical, each row=const, rows differ). Compare encoder
// vs decoder. If bit-exact (max Δ=0), then VE_PRED for a single MB works
// fine and the face-frame MB(2,0) +1 offset must come from CROSS-MB state.
func TestSingleMBVerticalGradient(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}

	img := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	// Vertical gradient: each row uniform, rows differ. Favours VE_PRED-like content.
	// With no top neighbor, pred=127, residual = source - 127. Row-wise constant
	// residuals trigger high-amplitude DCT vertical-only signal.
	for y := 0; y < 16; y++ {
		v := uint8(80 + y*10)
		for x := 0; x < 16; x++ {
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
	os.WriteFile("/tmp/vert.webp", buf.Bytes(), 0644)
	if out, err := exec.Command(dwebp, "-nofancy", "-pgm", "/tmp/vert.webp", "-o", "/tmp/vert.pgm").CombinedOutput(); err != nil {
		t.Fatalf("dwebp: %v %s", err, out)
	}
	pgmData, err := os.ReadFile("/tmp/vert.pgm")
	if err != nil {
		t.Fatal(err)
	}
	hdrEnd := bytes.Index(pgmData, []byte("255\n"))
	raw := pgmData[hdrEnd+4:]

	t.Logf("MB info: isI4=%v i16Mode=%d segment=%d", snap[0].isI4, snap[0].i16Mode, snap[0].segment)

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
	dumps := map[[2]int]*debugI16Dump{}
	debugDumpI16Capture = &dumps
	defer func() { debugMBStats = nil; debugDumpI16Capture = nil }()

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

	// Dump MB(8,0) internal state
	if d8, ok := dumps[[2]int{8, 0}]; ok {
		t.Logf("MB(8,0): segQ=%d  y2.q[0]=%d", d8.segQ, d8.y2q0)
		t.Logf("MB(8,0): mbI16Pred[0..3]=%d %d %d %d", d8.pred[0], d8.pred[1], d8.pred[2], d8.pred[3])
		t.Logf("MB(8,0): dcLevels[0..3]=%d %d %d %d", d8.dcLevels[0], d8.dcLevels[1], d8.dcLevels[2], d8.dcLevels[3])
		t.Logf("MB(8,0): dcBlockCoeff[0..3]=%d %d %d %d", d8.dcBlockCoeff[0], d8.dcBlockCoeff[1], d8.dcBlockCoeff[2], d8.dcBlockCoeff[3])
		t.Logf("MB(8,0): yDcRaw[0..3]=%d %d %d %d", d8.yDcRaw[0], d8.yDcRaw[1], d8.yDcRaw[2], d8.yDcRaw[3])
		t.Logf("MB(8,0): whtOut[0..3]=%d %d %d %d", d8.whtOut[0], d8.whtOut[1], d8.whtOut[2], d8.whtOut[3])
		t.Logf("MB(8,0): recon (from buffer)[0..3]=%d %d %d %d", d8.recon[0], d8.recon[1], d8.recon[2], d8.recon[3])
	}

	// Show MB(7,0) right column (which MB(8,0) reads for DC pred).
	t.Logf("MB(7,0) right column (x=127):")
	for y := 0; y < 16; y++ {
		yDec := int(raw[y*imgW+127])
		yEnc := int(reconYCapture[y*reconStrideCapture+127])
		t.Logf(" y=%d: enc=%d dec=%d", y, yEnc, yDec)
	}

	// Show MB(8,0) — known to diverge
	t.Logf("MB(8,0) first 4 rows (x=128..143):")
	for y := 0; y < 4; y++ {
		var row string
		for x := 128; x < 144; x++ {
			yDec := int(raw[y*imgW+x])
			yEnc := int(reconYCapture[y*reconStrideCapture+x])
			row += fmt.Sprintf(" %3d/%3d", yEnc, yDec)
		}
		t.Logf(" y=%d:%s", y, row)
	}

	_ = image.Black
	_ = png.Decode
}
