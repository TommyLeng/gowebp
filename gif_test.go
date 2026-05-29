// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"bytes"
	"image"
	"image/color"
	"image/gif"
	"os"
	"path/filepath"
	"testing"
)

// TestConvertGIF decodes a real GIF from test_data/original/, converts it
// to a lossy WebP animation via ConvertGIF, then sanity-checks the output
// container layout.
func TestConvertGIF(t *testing.T) {
	candidates := []string{
		"test_data/original/baf1a2d038ad43b4bbe8b13799c0987d.gif",
		"test_data/original/ezgif-6beab9280acd98.gif",
		"test_data/original/c8e7b15f1e97f2b4f3854032ca608982.gif",
		"test_data/original/9710eca79d8749e498f95692395114da.gif",
	}
	var gifPath string
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			gifPath = p
			break
		}
	}
	if gifPath == "" {
		t.Skipf("no GIF test data in test_data/original/; skipping")
	}

	f, err := os.Open(gifPath)
	if err != nil {
		t.Fatalf("open %s: %v", gifPath, err)
	}
	defer f.Close()

	g, err := gif.DecodeAll(f)
	if err != nil {
		t.Fatalf("gif.DecodeAll(%s): %v", gifPath, err)
	}
	if len(g.Image) == 0 {
		t.Fatalf("decoded gif has no frames")
	}

	var buf bytes.Buffer
	if err := ConvertGIF(&buf, g, &Options{Quality: 75}); err != nil {
		t.Fatalf("ConvertGIF(%s): %v", filepath.Base(gifPath), err)
	}

	out := buf.Bytes()
	if len(out) == 0 {
		t.Fatalf("ConvertGIF produced 0-byte output")
	}

	// Outer RIFF/WEBP wrapper.
	if len(out) < 12 {
		t.Fatalf("output too short (%d bytes) to contain RIFF/WEBP header", len(out))
	}
	if string(out[0:4]) != "RIFF" {
		t.Fatalf("output does not start with RIFF: got %q", string(out[0:4]))
	}
	if string(out[8:12]) != "WEBP" {
		t.Fatalf("output bytes[8:12] = %q, want %q", string(out[8:12]), "WEBP")
	}

	// Must contain ANIM chunk somewhere.
	if !bytes.Contains(out, []byte("ANIM")) {
		t.Fatalf("output does not contain ANIM chunk; first 64 bytes = %x", out[:min(64, len(out))])
	}
	if !bytes.Contains(out, []byte("ANMF")) {
		t.Fatalf("output does not contain any ANMF chunk")
	}
	if !bytes.Contains(out, []byte("VP8X")) {
		t.Fatalf("output does not contain VP8X chunk")
	}

	// Parse chunks and verify we have one ANMF per GIF frame.
	chunks, err := parseWebPChunks(out)
	if err != nil {
		t.Fatalf("parseWebPChunks: %v", err)
	}
	anmfCount := 0
	for _, c := range chunks {
		if c.tag == "ANMF" {
			anmfCount++
		}
	}
	if anmfCount != len(g.Image) {
		t.Errorf("ANMF chunks = %d, want %d (one per GIF frame)", anmfCount, len(g.Image))
	}

	t.Logf("%s: %d frames → %d bytes WebP (input bytes: see file size)",
		filepath.Base(gifPath), len(g.Image), len(out))
}

// TestConvertGIF_NilGif verifies ConvertGIF rejects a nil GIF.
func TestConvertGIF_NilGif(t *testing.T) {
	var buf bytes.Buffer
	if err := ConvertGIF(&buf, nil, nil); err == nil {
		t.Fatalf("ConvertGIF(nil) returned no error")
	}
}

// TestConvertGIF_EmptyGif verifies ConvertGIF rejects a GIF with no frames.
func TestConvertGIF_EmptyGif(t *testing.T) {
	var buf bytes.Buffer
	if err := ConvertGIF(&buf, &gif.GIF{}, nil); err == nil {
		t.Fatalf("ConvertGIF(empty) returned no error")
	}
}

// TestConvertGIF_DeltaTransparent verifies that ConvertGIF produces
// alpha-bearing delta frames for regions that didn't change between
// consecutive GIF frames. This is the fix for the bottom-left
// "flickering brightness" artifact: unchanged regions must be marked
// transparent so the WebP decoder leaves the previously-decoded canvas
// pixel in place (avoiding per-frame VP8 quantisation noise on
// otherwise-static pixels).
//
// Strategy:
//
//  1. Build a 32×32 GIF with two frames:
//     - frame 0: solid red.
//     - frame 1: solid red with two *isolated* blue pixels at (4, 4)
//     and (28, 28). The dirty bounding rectangle spans most of the
//     canvas, but the vast majority of pixels inside the rect didn't
//     change — those must be marked alpha=0 in the delta frame.
//  2. Convert via ConvertGIF.
//  3. Parse the resulting WebP container, verify the second ANMF chunk
//     contains an ALPH chunk (delta encoding active) AND the top-level
//     VP8X chunk has the alpha flag set.
func TestConvertGIF_DeltaTransparent(t *testing.T) {
	pal := color.Palette{
		color.NRGBA{255, 0, 0, 255}, // index 0: red
		color.NRGBA{0, 0, 255, 255}, // index 1: blue
		color.NRGBA{0, 0, 0, 255},   // index 2: black (unused)
	}
	const W, H = 32, 32

	makeFrame := func(withDots bool) *image.Paletted {
		p := image.NewPaletted(image.Rect(0, 0, W, H), pal)
		for i := range p.Pix {
			p.Pix[i] = 0 // red
		}
		if withDots {
			// Two isolated blue pixels far apart so the dirty bounding
			// box covers most of the canvas while *most* of its
			// interior pixels are unchanged (still red).
			p.Pix[4*p.Stride+4] = 1
			p.Pix[28*p.Stride+28] = 1
		}
		return p
	}

	g := &gif.GIF{
		Image: []*image.Paletted{
			makeFrame(false),
			makeFrame(true),
		},
		Delay:    []int{4, 4},
		Disposal: []byte{gif.DisposalNone, gif.DisposalNone},
		Config: image.Config{
			ColorModel: pal,
			Width:      W,
			Height:     H,
		},
		BackgroundIndex: 0,
		LoopCount:       0,
	}

	var buf bytes.Buffer
	if err := ConvertGIF(&buf, g, &Options{Quality: 75}); err != nil {
		t.Fatalf("ConvertGIF: %v", err)
	}
	out := buf.Bytes()

	// Parse chunks. The output should be:
	//   RIFF .... WEBP
	//     VP8X (with alpha flag set, since at least one frame has alpha)
	//     ANIM
	//     ANMF 1 (keyframe — VP8 only, no ALPH)
	//     ANMF 2 (delta — ALPH + VP8)
	chunks, err := parseWebPChunks(out)
	if err != nil {
		t.Fatalf("parseWebPChunks: %v", err)
	}
	if len(chunks) < 4 {
		t.Fatalf("expected >=4 chunks (VP8X, ANIM, ANMF, ANMF), got %d", len(chunks))
	}

	// Check VP8X has alpha flag (bit 4) set.
	var vp8x []byte
	for _, c := range chunks {
		if c.tag == "VP8X" {
			vp8x = c.payload
			break
		}
	}
	if vp8x == nil {
		t.Fatalf("no VP8X chunk found")
	}
	if vp8x[0]&(1<<4) == 0 {
		t.Errorf("VP8X alpha flag (bit 4) not set: flags=0x%02x", vp8x[0])
	}

	// Count ANMF chunks with and without ALPH.
	anmfWithAlph := 0
	anmfWithoutAlph := 0
	for _, c := range chunks {
		if c.tag != "ANMF" {
			continue
		}
		if len(c.payload) <= 16 {
			continue
		}
		// ANMF payload layout: 16-byte header + inner sub-chunks
		// (either "VP8 ..." or "ALPH..." + "VP8 ...").
		inner := c.payload[16:]
		if bytes.Contains(inner, []byte("ALPH")) {
			anmfWithAlph++
		} else {
			anmfWithoutAlph++
		}
	}
	if anmfWithAlph == 0 {
		t.Errorf("expected at least one ANMF with ALPH chunk (delta encoding), got 0")
	}
	if anmfWithoutAlph == 0 {
		t.Errorf("expected first ANMF to be a keyframe without ALPH, got 0 keyframes")
	}
	t.Logf("ANMF chunks: %d with ALPH (delta), %d without (keyframe)", anmfWithAlph, anmfWithoutAlph)
}

// TestConvertGIF_NoChangeDirtyRect verifies that a 2-frame GIF whose
// second frame is *identical* to the first produces a minimal 2×2
// no-op sub-frame (rather than a full-canvas re-encode), preserving
// frame count and durations while avoiding unnecessary bitstream
// expansion.
func TestConvertGIF_NoChangeDirtyRect(t *testing.T) {
	pal := color.Palette{
		color.NRGBA{200, 100, 50, 255},
	}
	const W, H = 64, 64
	makeFrame := func() *image.Paletted {
		return image.NewPaletted(image.Rect(0, 0, W, H), pal)
	}
	g := &gif.GIF{
		Image:    []*image.Paletted{makeFrame(), makeFrame()},
		Delay:    []int{5, 5},
		Disposal: []byte{gif.DisposalNone, gif.DisposalNone},
		Config: image.Config{
			ColorModel: pal,
			Width:      W,
			Height:     H,
		},
	}
	var buf bytes.Buffer
	if err := ConvertGIF(&buf, g, &Options{Quality: 75}); err != nil {
		t.Fatalf("ConvertGIF: %v", err)
	}

	out := buf.Bytes()
	chunks, err := parseWebPChunks(out)
	if err != nil {
		t.Fatalf("parseWebPChunks: %v", err)
	}
	var anmfs [][]byte
	for _, c := range chunks {
		if c.tag == "ANMF" {
			anmfs = append(anmfs, c.payload)
		}
	}
	if len(anmfs) != 2 {
		t.Fatalf("expected 2 ANMF chunks, got %d", len(anmfs))
	}
	// ANMF payload bytes 6..8 = width-1 (24-bit LE), bytes 9..11 =
	// height-1. Header layout: x_off(3) + y_off(3) + w-1(3) + h-1(3) +
	// dur(3) + flags(1) = 16 bytes total.
	if len(anmfs[1]) < 16 {
		t.Fatalf("ANMF[1] too short: %d", len(anmfs[1]))
	}
	w := int(anmfs[1][6]) | int(anmfs[1][7])<<8 | int(anmfs[1][8])<<16
	h := int(anmfs[1][9]) | int(anmfs[1][10])<<8 | int(anmfs[1][11])<<16
	w++
	h++
	if w > 4 || h > 4 {
		t.Errorf("ANMF[1] dirty rect should be ~2×2 when nothing changed, got %d×%d", w, h)
	}
	t.Logf("ANMF[1] dirty rect: %d×%d (expected ~2×2)", w, h)
}
