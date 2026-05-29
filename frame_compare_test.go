package gowebp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/gif"
	"image/png"
	"io"
	"os"
	"os/exec"
	"sort"
	"testing"

	"golang.org/x/image/webp"
)

// gifAnimResult is a type alias so we don't conflict with the gif package name.
type gifAnimResult = gif.GIF

// decodeGIFFile decodes an animated GIF from r.
func decodeGIFFile(r io.Reader) (*gifAnimResult, error) {
	return gif.DecodeAll(r)
}

// drawOver composites src (a *image.Paletted GIF frame) over dst using draw.Over.
func drawOver(dst *image.NRGBA, r image.Rectangle, src *image.Paletted, sp image.Point) {
	draw.Draw(dst, r, src, sp, draw.Over)
}

// animFrame is a single decoded ANMF frame, with its bounding rectangle on
// the canvas and the blend/dispose flags as parsed from the ANMF header.
type animFrame struct {
	x, y     int
	w, h     int
	duration uint32
	dispose  bool // bit0: true → background, false → keep
	blend    bool // bit1: true → no-blend (overwrite), false → alpha-blend
	hasAlpha bool // ALPH sub-chunk present
	img      *image.NRGBA
}

// parsedAnim is a fully-decoded animation: canvas size, bgcolor, and per-frame
// composited images (as decoded, not yet composited).
type parsedAnim struct {
	canvasW, canvasH int
	bgcolor          uint32 // ANIM bgcolor, BGRA byte order in payload
	loopCount        uint16
	frames           []animFrame
	// alphSizes[i] = number of bytes in the ALPH inner sub-chunk for frame i,
	// or -1 if the frame has no ALPH chunk.
	alphSizes []int
}

func read24LE(p []byte) uint32 {
	return uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16
}

// parseAnimWebP parses an animated WebP file at path and returns the canvas
// metadata plus each ANMF frame already decoded into NRGBA at frame-local
// coordinates (Bounds = Rect(0,0,w,h)).
func parseAnimWebP(path string) (*parsedAnim, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WEBP" {
		return nil, fmt.Errorf("not a RIFF/WEBP: %s", path)
	}
	riffSize := binary.LittleEndian.Uint32(raw[4:8])
	if int(riffSize)+8 > len(raw) {
		return nil, fmt.Errorf("riff size %d overflows file %d", riffSize, len(raw))
	}
	body := raw[12 : 8+riffSize]

	out := &parsedAnim{}

	off := 0
	for off < len(body) {
		if off+8 > len(body) {
			return nil, fmt.Errorf("chunk header overruns body at off=%d", off)
		}
		tag := string(body[off : off+4])
		sz := binary.LittleEndian.Uint32(body[off+4 : off+8])
		if off+8+int(sz) > len(body) {
			return nil, fmt.Errorf("chunk %q size %d overruns body at off=%d", tag, sz, off)
		}
		payload := body[off+8 : off+8+int(sz)]
		switch tag {
		case "VP8X":
			if len(payload) < 10 {
				return nil, fmt.Errorf("short VP8X payload: %d", len(payload))
			}
			out.canvasW = int(read24LE(payload[4:7])) + 1
			out.canvasH = int(read24LE(payload[7:10])) + 1
		case "ANIM":
			if len(payload) < 6 {
				return nil, fmt.Errorf("short ANIM payload: %d", len(payload))
			}
			out.bgcolor = binary.LittleEndian.Uint32(payload[0:4])
			out.loopCount = binary.LittleEndian.Uint16(payload[4:6])
		case "ANMF":
			frame, alphSize, err := decodeANMF(payload)
			if err != nil {
				return nil, fmt.Errorf("decodeANMF[%d]: %w", len(out.frames), err)
			}
			out.frames = append(out.frames, frame)
			out.alphSizes = append(out.alphSizes, alphSize)
		}
		off += 8 + int(sz)
		if sz&1 == 1 {
			off++
		}
	}

	return out, nil
}

// decodeANMF decodes a single ANMF payload (the bytes after the 8-byte chunk
// header). Returns the parsed frame and the size of the ALPH sub-chunk (-1
// if absent).
func decodeANMF(payload []byte) (animFrame, int, error) {
	if len(payload) < 16 {
		return animFrame{}, 0, fmt.Errorf("ANMF payload too small: %d", len(payload))
	}
	var f animFrame
	f.x = int(read24LE(payload[0:3])) * 2
	f.y = int(read24LE(payload[3:6])) * 2
	f.w = int(read24LE(payload[6:9])) + 1
	f.h = int(read24LE(payload[9:12])) + 1
	f.duration = read24LE(payload[12:15])
	flags := payload[15]
	f.dispose = (flags & 0x01) != 0
	f.blend = (flags & 0x02) != 0
	// Iterate inner sub-chunks starting at offset 16.
	inner := payload[16:]
	var alphData []byte
	var vp8Data []byte
	var vp8LData []byte
	off := 0
	for off < len(inner) {
		if off+8 > len(inner) {
			return animFrame{}, 0, fmt.Errorf("inner chunk header overrun at off=%d (inner len=%d)", off, len(inner))
		}
		itag := string(inner[off : off+4])
		isz := binary.LittleEndian.Uint32(inner[off+4 : off+8])
		if off+8+int(isz) > len(inner) {
			return animFrame{}, 0, fmt.Errorf("inner chunk %q size %d overrun", itag, isz)
		}
		ip := inner[off+8 : off+8+int(isz)]
		switch itag {
		case "ALPH":
			alphData = ip
		case "VP8 ":
			vp8Data = ip
		case "VP8L":
			vp8LData = ip
		}
		off += 8 + int(isz)
		if isz&1 == 1 {
			off++
		}
	}

	alphSize := -1
	if alphData != nil {
		alphSize = len(alphData)
		f.hasAlpha = true
	}

	// Build a standalone WebP container and decode it.
	var built bytes.Buffer
	switch {
	case vp8LData != nil:
		// VP8L lossless can carry its own alpha; wrap as plain VP8L.
		writeRIFF(&built, [][2]string{{"VP8L", string(vp8LData)}})
	case alphData != nil && vp8Data != nil:
		// Build a VP8X Extended file so x/image/webp consumes ALPH+VP8.
		writeRIFFWithVP8X(&built, f.w, f.h, alphData, vp8Data)
	case vp8Data != nil:
		writeRIFF(&built, [][2]string{{"VP8 ", string(vp8Data)}})
	default:
		return animFrame{}, 0, fmt.Errorf("ANMF frame has no VP8/VP8L inner chunk")
	}

	img, err := webp.Decode(&built)
	if err != nil {
		return animFrame{}, 0, fmt.Errorf("webp.Decode inner frame: %w", err)
	}
	b := img.Bounds()
	if b.Dx() != f.w || b.Dy() != f.h {
		return animFrame{}, 0, fmt.Errorf("inner frame size mismatch: got %dx%d, want %dx%d",
			b.Dx(), b.Dy(), f.w, f.h)
	}
	// Normalise to NRGBA at origin (0,0).
	nrgba := image.NewNRGBA(image.Rect(0, 0, f.w, f.h))
	for yy := 0; yy < f.h; yy++ {
		for xx := 0; xx < f.w; xx++ {
			r, g, bl, a := img.At(b.Min.X+xx, b.Min.Y+yy).RGBA()
			// VP8-only frames return 16-bit fully-opaque values; convert to 8-bit.
			nrgba.SetNRGBA(xx, yy, color.NRGBA{
				R: uint8(r >> 8),
				G: uint8(g >> 8),
				B: uint8(bl >> 8),
				A: uint8(a >> 8),
			})
		}
	}
	f.img = nrgba
	return f, alphSize, nil
}

// writeRIFF writes a minimal RIFF/WEBP wrapper containing the listed chunks.
func writeRIFF(buf *bytes.Buffer, chunks [][2]string) {
	// Outer header reserved later.
	body := &bytes.Buffer{}
	for _, c := range chunks {
		body.WriteString(c[0])
		_ = binary.Write(body, binary.LittleEndian, uint32(len(c[1])))
		body.WriteString(c[1])
		if len(c[1])&1 == 1 {
			body.WriteByte(0)
		}
	}
	buf.WriteString("RIFF")
	_ = binary.Write(buf, binary.LittleEndian, uint32(4+body.Len()))
	buf.WriteString("WEBP")
	buf.Write(body.Bytes())
}

// writeRIFFWithVP8X writes an Extended-format WebP with VP8X (alpha bit set)
// + ALPH + VP8.
func writeRIFFWithVP8X(buf *bytes.Buffer, w, h int, alphData, vp8Data []byte) {
	// Build the VP8X payload: 10 bytes.
	var vp8x [10]byte
	vp8x[0] = 0x10 // alpha bit
	// width-1 (24-bit LE) at bytes 4..6, height-1 at 7..9.
	w1 := uint32(w - 1)
	h1 := uint32(h - 1)
	vp8x[4] = byte(w1)
	vp8x[5] = byte(w1 >> 8)
	vp8x[6] = byte(w1 >> 16)
	vp8x[7] = byte(h1)
	vp8x[8] = byte(h1 >> 8)
	vp8x[9] = byte(h1 >> 16)
	writeRIFF(buf, [][2]string{
		{"VP8X", string(vp8x[:])},
		{"ALPH", string(alphData)},
		{"VP8 ", string(vp8Data)},
	})
}

// fillRect fills the rectangle [rx,ry .. rx+rw,ry+rh) of nrgba with the
// background colour. bgcolor byte order in the ANIM payload is BGRA: byte 0
// = B, byte 1 = G, byte 2 = R, byte 3 = A.
func fillRect(nrgba *image.NRGBA, rx, ry, rw, rh int, bgcolor uint32) {
	bg := color.NRGBA{
		B: uint8(bgcolor),
		G: uint8(bgcolor >> 8),
		R: uint8(bgcolor >> 16),
		A: uint8(bgcolor >> 24),
	}
	// libwebp's mux convention: the canvas is always rendered as fully opaque,
	// so when alpha component is 0 we still want a black opaque background.
	// However, when no alpha is set in bgcolor the comparison should be
	// consistent between gowebp and libwebp because both produce the same
	// canvas before frame 0. We keep the byte order honest.
	for yy := ry; yy < ry+rh; yy++ {
		for xx := rx; xx < rx+rw; xx++ {
			nrgba.SetNRGBA(xx, yy, bg)
		}
	}
}

// composite paints a single frame onto the canvas with the appropriate blend
// rule, then returns nothing (mutates canvas). Dispose handling happens after
// the caller has compared the canvas.
func composite(canvas *image.NRGBA, f animFrame) {
	for yy := 0; yy < f.h; yy++ {
		for xx := 0; xx < f.w; xx++ {
			fr := f.img.NRGBAAt(xx, yy)
			cx := f.x + xx
			cy := f.y + yy
			if cx < 0 || cy < 0 || cx >= canvas.Bounds().Dx() || cy >= canvas.Bounds().Dy() {
				continue
			}
			if f.blend || !f.hasAlpha {
				// Blend=1 (no-blend, overwrite) — or no alpha channel at all.
				canvas.SetNRGBA(cx, cy, color.NRGBA{R: fr.R, G: fr.G, B: fr.B, A: 255})
				continue
			}
			// Alpha blend (Blend=0).
			a := uint32(fr.A)
			if a == 255 {
				canvas.SetNRGBA(cx, cy, color.NRGBA{R: fr.R, G: fr.G, B: fr.B, A: 255})
				continue
			}
			if a == 0 {
				// Fully transparent → canvas unchanged.
				continue
			}
			cc := canvas.NRGBAAt(cx, cy)
			inv := 255 - a
			r := (uint32(fr.R)*a + uint32(cc.R)*inv + 127) / 255
			g := (uint32(fr.G)*a + uint32(cc.G)*inv + 127) / 255
			b := (uint32(fr.B)*a + uint32(cc.B)*inv + 127) / 255
			canvas.SetNRGBA(cx, cy, color.NRGBA{R: uint8(r), G: uint8(g), B: uint8(b), A: 255})
		}
	}
}

// diffResult holds per-frame comparison statistics.
type diffResult struct {
	frame    int
	blendKey bool // gowebp frame's blend flag
	alphSize int
	totalPx  int
	maxDiff  int
	blPx     int
	blMax    int
}

// TestFrameComparison composites both the gowebp and libwebp animation
// outputs frame-by-frame and reports per-frame pixel deltas.
func TestFrameComparison(t *testing.T) {
	const (
		gowebpPath  = "test_data/gowebp/anim/baf1a2d038ad43b4bbe8b13799c0987d.webp"
		libwebpPath = "test_data/libwebp/anim/baf1a2d038ad43b4bbe8b13799c0987d.webp"
		// Bottom-left region.
		blMaxX = 80
		blMinY = 200
	)

	for _, p := range []string{gowebpPath, libwebpPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("test data missing: %s (%v)", p, err)
			return
		}
	}

	gowAnim, err := parseAnimWebP(gowebpPath)
	if err != nil {
		t.Fatalf("parse gowebp anim: %v", err)
	}
	libAnim, err := parseAnimWebP(libwebpPath)
	if err != nil {
		t.Fatalf("parse libwebp anim: %v", err)
	}

	if gowAnim.canvasW != libAnim.canvasW || gowAnim.canvasH != libAnim.canvasH {
		t.Fatalf("canvas mismatch: gowebp=%dx%d libwebp=%dx%d",
			gowAnim.canvasW, gowAnim.canvasH, libAnim.canvasW, libAnim.canvasH)
	}
	if len(gowAnim.frames) != len(libAnim.frames) {
		t.Logf("WARN: frame count differs gowebp=%d libwebp=%d",
			len(gowAnim.frames), len(libAnim.frames))
	}
	W := gowAnim.canvasW
	H := gowAnim.canvasH
	t.Logf("canvas: %dx%d, gowebp frames=%d libwebp frames=%d",
		W, H, len(gowAnim.frames), len(libAnim.frames))
	t.Logf("gowebp bgcolor=0x%08x  libwebp bgcolor=0x%08x", gowAnim.bgcolor, libAnim.bgcolor)

	// Initialise both canvases to the ANIM bgcolor. We use each animation's
	// own declared bgcolor for accurate reproduction of the decoder canvas.
	gowCanvas := image.NewNRGBA(image.Rect(0, 0, W, H))
	libCanvas := image.NewNRGBA(image.Rect(0, 0, W, H))
	fillRect(gowCanvas, 0, 0, W, H, gowAnim.bgcolor)
	fillRect(libCanvas, 0, 0, W, H, libAnim.bgcolor)

	totalPx := W * H
	blPxCount := blMaxX * (H - blMinY) // 80 × 82 = 6560 for 282×282
	if H <= blMinY {
		blPxCount = 0
	}

	nFrames := len(gowAnim.frames)
	if len(libAnim.frames) < nFrames {
		nFrames = len(libAnim.frames)
	}
	results := make([]diffResult, 0, nFrames)

	t.Logf("Frame |  Blend          | ALPH bytes |   diff_px / total |  max | BL_diff / BL_px | BL_max")
	t.Logf("------+-----------------+------------+-------------------+------+-----------------+--------")

	for i := 0; i < nFrames; i++ {
		gf := gowAnim.frames[i]
		lf := libAnim.frames[i]

		// Composite this frame onto each canvas.
		composite(gowCanvas, gf)
		composite(libCanvas, lf)

		// Compare canvases pixel-by-pixel.
		var diffPx, maxDiff int
		var blDiffPx, blMax int
		for yy := 0; yy < H; yy++ {
			for xx := 0; xx < W; xx++ {
				gp := gowCanvas.NRGBAAt(xx, yy)
				lp := libCanvas.NRGBAAt(xx, yy)
				dR := absDiff(int(gp.R), int(lp.R))
				dG := absDiff(int(gp.G), int(lp.G))
				dB := absDiff(int(gp.B), int(lp.B))
				dA := absDiff(int(gp.A), int(lp.A))
				m := dR
				if dG > m {
					m = dG
				}
				if dB > m {
					m = dB
				}
				if dA > m {
					m = dA
				}
				if m > 1 { // > 1 to ignore ±1 rounding
					diffPx++
					if m > maxDiff {
						maxDiff = m
					}
					if xx < blMaxX && yy > blMinY {
						blDiffPx++
						if m > blMax {
							blMax = m
						}
					}
				}
			}
		}

		blendKey := gf.blend
		alphSize := -1
		if i < len(gowAnim.alphSizes) {
			alphSize = gowAnim.alphSizes[i]
		}
		results = append(results, diffResult{
			frame:    i,
			blendKey: blendKey,
			alphSize: alphSize,
			totalPx:  diffPx,
			maxDiff:  maxDiff,
			blPx:     blDiffPx,
			blMax:    blMax,
		})

		var blendStr string
		if blendKey {
			blendStr = "KEY (B=1)"
		} else {
			blendStr = "delta"
		}
		alphStr := "        -"
		if alphSize >= 0 {
			alphStr = fmt.Sprintf("%9d", alphSize)
		}
		t.Logf("%5d | %-15s | %s | %7d / %6d | %4d | %5d / %6d | %4d",
			i, blendStr, alphStr, diffPx, totalPx, maxDiff, blDiffPx, blPxCount, blMax)

		// Apply dispose AFTER comparison so frame N's canvas reflects its
		// displayed state — disposal prepares canvas for frame N+1.
		applyDispose(gowCanvas, gf, gowAnim.bgcolor)
		applyDispose(libCanvas, lf, libAnim.bgcolor)
	}

	// Summary.
	t.Log("")
	t.Logf("=== Summary over %d frames ===", len(results))
	if len(results) == 0 {
		return
	}

	var sumDiff, sumBL int
	var worstByDiff, worstByMax, worstByBL []diffResult
	for _, r := range results {
		sumDiff += r.totalPx
		sumBL += r.blPx
	}
	avgDiff := float64(sumDiff) / float64(len(results))
	avgBL := float64(sumBL) / float64(len(results))
	t.Logf("average diff_px = %.1f / %d (%.2f%%)", avgDiff, totalPx, avgDiff*100/float64(totalPx))
	if blPxCount > 0 {
		t.Logf("average BL_diff = %.1f / %d (%.2f%%)", avgBL, blPxCount, avgBL*100/float64(blPxCount))
	}

	// Top-10 worst frames by diff_px, max_diff, and BL_diff.
	worstByDiff = topN(results, 10, func(a, b diffResult) bool { return a.totalPx > b.totalPx })
	worstByMax = topN(results, 10, func(a, b diffResult) bool { return a.maxDiff > b.maxDiff })
	worstByBL = topN(results, 10, func(a, b diffResult) bool { return a.blPx > b.blPx })

	t.Log("")
	t.Log("Top 10 frames by diff_px (worst first):")
	for _, r := range worstByDiff {
		t.Logf("  frame %3d  diff_px=%7d  max=%3d  BL_diff=%5d  BL_max=%3d  blend=%v",
			r.frame, r.totalPx, r.maxDiff, r.blPx, r.blMax, r.blendKey)
	}
	t.Log("")
	t.Log("Top 10 frames by max_diff:")
	for _, r := range worstByMax {
		t.Logf("  frame %3d  max=%3d  diff_px=%7d  BL_diff=%5d  BL_max=%3d  blend=%v",
			r.frame, r.maxDiff, r.totalPx, r.blPx, r.blMax, r.blendKey)
	}
	t.Log("")
	t.Log("Top 10 frames by BL_diff:")
	for _, r := range worstByBL {
		t.Logf("  frame %3d  BL_diff=%5d  BL_max=%3d  diff_px=%7d  max=%3d  blend=%v",
			r.frame, r.blPx, r.blMax, r.totalPx, r.maxDiff, r.blendKey)
	}
}

func absDiff(a, b int) int {
	if a < b {
		return b - a
	}
	return a - b
}

// applyDispose mutates the canvas for the post-display dispose action.
func applyDispose(canvas *image.NRGBA, f animFrame, bgcolor uint32) {
	if !f.dispose {
		return
	}
	rx := f.x
	ry := f.y
	rw := f.w
	rh := f.h
	if rx < 0 {
		rw += rx
		rx = 0
	}
	if ry < 0 {
		rh += ry
		ry = 0
	}
	cb := canvas.Bounds()
	if rx+rw > cb.Dx() {
		rw = cb.Dx() - rx
	}
	if ry+rh > cb.Dy() {
		rh = cb.Dy() - ry
	}
	if rw <= 0 || rh <= 0 {
		return
	}
	fillRect(canvas, rx, ry, rw, rh, bgcolor)
}

// topN returns a copy of in sorted descending by less() (which compares for
// "a should come first"), capped at n entries.
func topN(in []diffResult, n int, less func(a, b diffResult) bool) []diffResult {
	out := make([]diffResult, len(in))
	copy(out, in)
	sort.Slice(out, func(i, j int) bool { return less(out[i], out[j]) })
	if len(out) > n {
		out = out[:n]
	}
	return out
}

// TestFrameVsOriginalGIF performs a 3-way comparison:
//   original GIF canvas  vs  gowebp canvas
//   original GIF canvas  vs  libwebp canvas
//
// For each frame it reports the mean absolute error (MAE) in the bottom-left
// region (x<80, y>200) for both encoders.  If gowebp's MAE is consistently
// much larger than libwebp's MAE, gowebp is showing wrong (stale/ghosted)
// content there — not just normal VP8 quantisation noise, which would affect
// both encoders similarly.
func TestFrameVsOriginalGIF(t *testing.T) {
	const (
		gifPath     = "test_data/original/baf1a2d038ad43b4bbe8b13799c0987d.gif"
		gowebpPath  = "test_data/gowebp/anim/baf1a2d038ad43b4bbe8b13799c0987d.webp"
		libwebpPath = "test_data/libwebp/anim/baf1a2d038ad43b4bbe8b13799c0987d.webp"
		blMaxX      = 80
		blMinY      = 200
		// A frame where gowebp's BL MAE exceeds libwebp's BL MAE by more than
		// this margin is flagged as a problem: extra error beyond normal VP8
		// quantisation noise that affects both encoders equally.
		extraErrThresh = 5.0
	)
	for _, p := range []string{gifPath, gowebpPath, libwebpPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing: %s", p)
		}
	}

	// Decode original GIF.
	gf, err := os.Open(gifPath)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	gifAnim, err := gif.DecodeAll(gf)
	if err != nil {
		t.Fatalf("gif.DecodeAll: %v", err)
	}
	canvasW, canvasH := gifAnim.Config.Width, gifAnim.Config.Height
	canvasRect := image.Rect(0, 0, canvasW, canvasH)

	// Decode both WebP animations.
	gowAnim, err := parseAnimWebP(gowebpPath)
	if err != nil {
		t.Fatalf("parseAnimWebP gowebp: %v", err)
	}
	libAnim, err := parseAnimWebP(libwebpPath)
	if err != nil {
		t.Fatalf("parseAnimWebP libwebp: %v", err)
	}

	nFrames := len(gifAnim.Image)
	if n := len(gowAnim.frames); n < nFrames {
		nFrames = n
	}
	if n := len(libAnim.frames); n < nFrames {
		nFrames = n
	}
	blPxCount := blMaxX * (canvasH - blMinY)
	t.Logf("3-way GIF vs gowebp vs libwebp: %d frames, BL region x<%d y>%d (%d px)",
		nFrames, blMaxX, blMinY, blPxCount)
	t.Logf("Each BL_MAE = mean absolute error per channel in bottom-left vs original GIF.")
	t.Logf("If gowebp BL_MAE >> libwebp BL_MAE, gowebp is showing wrong (ghosted) content there.")

	// GIF background colour.
	gifBg := color.NRGBA{0, 0, 0, 255}
	if p, ok := gifAnim.Config.ColorModel.(color.Palette); ok && len(p) > int(gifAnim.BackgroundIndex) {
		if r16, g16, b16, a16 := p[gifAnim.BackgroundIndex].RGBA(); a16 != 0 {
			gifBg = color.NRGBA{uint8(r16 >> 8), uint8(g16 >> 8), uint8(b16 >> 8), 255}
		}
	}

	// Three compositor canvases.
	gifCanvas := image.NewNRGBA(canvasRect)
	fillNRGBA(gifCanvas, gifBg)
	gowCanvas := image.NewNRGBA(canvasRect)
	fillNRGBA(gowCanvas, gifBg)
	libCanvas := image.NewNRGBA(canvasRect)
	fillNRGBA(libCanvas, gifBg)

	// blMAE returns the mean absolute error per channel in the BL region.
	blMAE := func(ref, enc *image.NRGBA) float64 {
		var sum int
		for yy := blMinY + 1; yy < canvasH; yy++ {
			for xx := 0; xx < blMaxX; xx++ {
				rp := ref.NRGBAAt(xx, yy)
				ep := enc.NRGBAAt(xx, yy)
				sum += absDiff(int(rp.R), int(ep.R))
				sum += absDiff(int(rp.G), int(ep.G))
				sum += absDiff(int(rp.B), int(ep.B))
			}
		}
		return float64(sum) / float64(blPxCount*3)
	}

	t.Logf("Frame | gowKey | gowBL_MAE | libBL_MAE | extra(gow-lib) | verdict")
	t.Logf("------+--------+-----------+-----------+----------------+--------")

	var prevGIFDisposal byte
	var prevGIFBounds image.Rectangle
	var sumGow, sumLib float64
	var worseCnt int

	for i := 0; i < nFrames; i++ {
		// Composite GIF frame i.
		switch prevGIFDisposal {
		case 2: // DisposalBackground
			if !prevGIFBounds.Empty() {
				fillRectNRGBA(gifCanvas, prevGIFBounds.Intersect(canvasRect), gifBg)
			}
		}
		fr := gifAnim.Image[i]
		fb := fr.Bounds()
		if dst := fb.Intersect(canvasRect); !dst.Empty() {
			drawOver(gifCanvas, dst, fr, fb.Min)
		}
		var thisDisp byte
		if gifAnim.Disposal != nil && i < len(gifAnim.Disposal) {
			thisDisp = gifAnim.Disposal[i]
		}
		prevGIFDisposal = thisDisp
		prevGIFBounds = fb

		// Composite gowebp and libwebp frames.
		composite(gowCanvas, gowAnim.frames[i])
		composite(libCanvas, libAnim.frames[i])

		// Measure BL MAE vs original GIF for each encoder.
		gowMAE := blMAE(gifCanvas, gowCanvas)
		libMAE := blMAE(gifCanvas, libCanvas)
		extra := gowMAE - libMAE
		sumGow += gowMAE
		sumLib += libMAE

		verdict := "ok"
		if extra > extraErrThresh {
			verdict = "GOW WORSE"
			worseCnt++
		} else if extra < -extraErrThresh {
			verdict = "lib worse"
		}

		isKey := gowAnim.frames[i].blend
		t.Logf("%5d |  %-5v | %9.2f | %9.2f | %14.2f | %s",
			i, isKey, gowMAE, libMAE, extra, verdict)

		// Apply dispose AFTER comparison (prepares canvas for frame i+1).
		applyDispose(gowCanvas, gowAnim.frames[i], gowAnim.bgcolor)
		applyDispose(libCanvas, libAnim.frames[i], libAnim.bgcolor)
	}

	t.Logf("")
	t.Logf("=== 3-way summary (%d frames) ===", nFrames)
	t.Logf("avg gowebp  BL MAE vs GIF: %.2f", sumGow/float64(nFrames))
	t.Logf("avg libwebp BL MAE vs GIF: %.2f", sumLib/float64(nFrames))
	t.Logf("avg extra error (gow-lib): %.2f", (sumGow-sumLib)/float64(nFrames))
	t.Logf("frames gowebp worse by >%.0f: %d / %d", extraErrThresh, worseCnt, nFrames)
	if worseCnt > nFrames/4 {
		t.Errorf("FAIL: gowebp bottom-left MAE exceeds libwebp by >%.0f in %d/%d frames (ghosting)",
			extraErrThresh, worseCnt, nFrames)
	}
}

// gifDecodeAll wraps image/gif DecodeAll to get a *gif.GIF.
func gifDecodeAll(r interface{ Read([]byte) (int, error) }) (*gifAnimResult, error) {
	type gifIface interface {
		Read([]byte) (int, error)
	}
	return decodeGIFFile(r.(gifIface))
}

// TestCompositorAccuracy checks whether gowebp's GIF compositor (the canvas
// passed to VP8 encoding) matches the original GIF compositor canvas. If
// the compositor canvas already diverges from the original, the problem is
// in ConvertGIF's compositing logic — NOT VP8 encoding quality.
// If the compositor matches (MAE ≈ 0), then the difference is purely VP8
// encoding quality (a separate, harder problem to fix).
func TestCompositorAccuracy(t *testing.T) {
	const (
		gifPath = "test_data/original/baf1a2d038ad43b4bbe8b13799c0987d.gif"
		blMaxX  = 80
		blMinY  = 200
	)
	if _, err := os.Stat(gifPath); err != nil {
		t.Skipf("missing: %s", gifPath)
	}

	gf, err := os.Open(gifPath)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	gifAnim, err := gif.DecodeAll(gf)
	if err != nil {
		t.Fatalf("gif.DecodeAll: %v", err)
	}
	canvasW, canvasH := gifAnim.Config.Width, gifAnim.Config.Height
	canvasRect := image.Rect(0, 0, canvasW, canvasH)

	// Reference GIF compositor (same logic as Go's image/draw).
	gifBg := color.NRGBA{0, 0, 0, 255}
	if p, ok := gifAnim.Config.ColorModel.(color.Palette); ok && len(p) > int(gifAnim.BackgroundIndex) {
		if r16, g16, b16, a16 := p[gifAnim.BackgroundIndex].RGBA(); a16 != 0 {
			gifBg = color.NRGBA{uint8(r16 >> 8), uint8(g16 >> 8), uint8(b16 >> 8), 255}
		}
	}
	refCanvas := image.NewNRGBA(canvasRect)
	fillNRGBA(refCanvas, gifBg)

	// gowebp internal compositor — replicate ConvertGIF's canvas logic exactly.
	gow := image.NewNRGBA(canvasRect)
	fillNRGBA(gow, gifBg)
	// prevSnapshot for DisposalPrevious (not used in this GIF but needed for completeness).
	prevSnap := image.NewNRGBA(canvasRect)

	blPxCount := blMaxX * (canvasH - blMinY)

	t.Logf("Frame | refBL[0] | gowBL[0] | BL_MAE | verdict")
	t.Logf("------+----------+----------+--------+--------")

	var prevRefDisp, prevGowDisp byte
	var prevRefBounds, prevGowBounds image.Rectangle

	for i := 0; i < len(gifAnim.Image); i++ {
		// --- Reference compositor (same as image/gif semantics) ---
		switch prevRefDisp {
		case 2:
			if !prevRefBounds.Empty() {
				fillRectNRGBA(refCanvas, prevRefBounds.Intersect(canvasRect), gifBg)
			}
		case 3:
			// DisposalPrevious: restore snapshot (simplified: not tracked for ref)
		}
		fr := gifAnim.Image[i]
		fb := fr.Bounds()
		if dst := fb.Intersect(canvasRect); !dst.Empty() {
			drawOver(refCanvas, dst, fr, fb.Min)
		}
		var refDisp byte
		if gifAnim.Disposal != nil && i < len(gifAnim.Disposal) {
			refDisp = gifAnim.Disposal[i]
		}
		prevRefDisp = refDisp
		prevRefBounds = fb

		// --- gowebp ConvertGIF compositor (identical logic from gif.go) ---
		switch prevGowDisp {
		case gif.DisposalBackground:
			if !prevGowBounds.Empty() {
				fillRectNRGBA(gow, prevGowBounds.Intersect(canvasRect), gifBg)
			}
		case gif.DisposalPrevious:
			copy(gow.Pix, prevSnap.Pix)
		}
		var thisDisp byte
		if gifAnim.Disposal != nil && i < len(gifAnim.Disposal) {
			thisDisp = gifAnim.Disposal[i]
		}
		if thisDisp == gif.DisposalPrevious {
			copy(prevSnap.Pix, gow.Pix)
		}
		frb := fr.Bounds()
		if dst := frb.Intersect(canvasRect); !dst.Empty() {
			drawOver(gow, dst, fr, frb.Min)
		}
		prevGowDisp = thisDisp
		prevGowBounds = frb

		// Compare BL region of ref vs gowebp compositor canvas.
		var blSum int
		for yy := blMinY + 1; yy < canvasH; yy++ {
			for xx := 0; xx < blMaxX; xx++ {
				rp := refCanvas.NRGBAAt(xx, yy)
				gp := gow.NRGBAAt(xx, yy)
				blSum += absDiff(int(rp.R), int(gp.R))
				blSum += absDiff(int(rp.G), int(gp.G))
				blSum += absDiff(int(rp.B), int(gp.B))
			}
		}
		blMAE := float64(blSum) / float64(blPxCount*3)

		// Sample pixel at bottom-left corner for quick check.
		refPx := refCanvas.NRGBAAt(0, canvasH-1)
		gowPx := gow.NRGBAAt(0, canvasH-1)
		verdict := "ok"
		if blMAE > 1 {
			verdict = "MISMATCH"
		}
		t.Logf("%5d | (%3d,%3d,%3d) | (%3d,%3d,%3d) | %6.3f | %s",
			i, refPx.R, refPx.G, refPx.B, gowPx.R, gowPx.G, gowPx.B, blMAE, verdict)
	}
}

// TestEncodeWorstFramesAsStatic composites the original GIF to frames 27, 28,
// and 32 (the frames with the largest gowebp vs original BL error), then
// encodes each composited canvas as a plain lossy WebP at quality 75 using
// both gowebp's Encode() and cwebp (libwebp), saving:
//   gowebp_lossy_frameXX.webp  — gowebp static encode
//   libwebp_lossy_frameXX.webp — cwebp static encode
//
// Comparing these static encodes against each other and against the animation
// output shows whether the quality gap is in the VP8 encoder itself or in the
// animation delta-encoding layer.
func TestEncodeWorstFramesAsStatic(t *testing.T) {
	const (
		gifPath = "test_data/original/baf1a2d038ad43b4bbe8b13799c0987d.gif"
		outDir  = "test_data/frame_compare"
	)
	targets := []int{27, 28, 32}

	if _, err := os.Stat(gifPath); err != nil {
		t.Skipf("missing %s", gifPath)
	}
	if err := os.MkdirAll(outDir, 0755); err != nil {
		t.Fatalf("mkdir %s: %v", outDir, err)
	}

	// Decode and composite the GIF up to each target frame.
	gf, err := os.Open(gifPath)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	g, err := gif.DecodeAll(gf)
	if err != nil {
		t.Fatalf("gif.DecodeAll: %v", err)
	}

	canvasW, canvasH := g.Config.Width, g.Config.Height
	canvasRect := image.Rect(0, 0, canvasW, canvasH)
	bg := color.NRGBA{0, 0, 0, 255}
	canvas := image.NewNRGBA(canvasRect)
	fillNRGBA(canvas, bg)

	var prevDisp byte
	var prevBounds image.Rectangle

	for i, fr := range g.Image {
		switch prevDisp {
		case 2: // DisposalBackground
			if !prevBounds.Empty() {
				fillRectNRGBA(canvas, prevBounds.Intersect(canvasRect), bg)
			}
		}
		fb := fr.Bounds()
		if dst := fb.Intersect(canvasRect); !dst.Empty() {
			drawOver(canvas, dst, fr, fb.Min)
		}
		var thisDisp byte
		if g.Disposal != nil && i < len(g.Disposal) {
			thisDisp = g.Disposal[i]
		}
		prevDisp = thisDisp
		prevBounds = fb

		for _, target := range targets {
			if i != target {
				continue
			}
			// Snapshot the composited canvas at this frame.
			snap := image.NewNRGBA(canvasRect)
			copy(snap.Pix, canvas.Pix)

			// Encode as lossy WebP quality 75 (same as the animation).
			outPath := fmt.Sprintf("%s/gowebp_lossy_frame%02d.webp", outDir, target)
			wf, err := os.Create(outPath)
			if err != nil {
				t.Errorf("create %s: %v", outPath, err)
				continue
			}
			if err := Encode(wf, snap, &Options{Quality: 75}); err != nil {
				wf.Close()
				t.Errorf("Encode frame %d: %v", target, err)
				continue
			}
			wf.Close()
			fi, _ := os.Stat(outPath)
			t.Logf("gowebp frame %d → %s (%d bytes)", target, outPath, fi.Size())

			// Also encode with cwebp (libwebp) for direct comparison.
			cwebpPath, lookErr := exec.LookPath("cwebp")
			if lookErr != nil {
				t.Logf("cwebp not found, skipping libwebp encode for frame %d", target)
				continue
			}
			// Write snap as a temp PNG for cwebp input.
			tmpPNG := fmt.Sprintf("%s/_tmp_frame%02d.png", outDir, target)
			if pf, err := os.Create(tmpPNG); err == nil {
				png.Encode(pf, snap)
				pf.Close()
			}
			libOut := fmt.Sprintf("%s/libwebp_lossy_frame%02d.webp", outDir, target)
			cmd := exec.Command(cwebpPath, "-q", "75", tmpPNG, "-o", libOut)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Logf("cwebp frame %d failed: %v\n%s", target, err, out)
			} else {
				os.Remove(tmpPNG)
				fi2, _ := os.Stat(libOut)
				t.Logf("libwebp frame %d → %s (%d bytes)", target, libOut, fi2.Size())
			}
		}
	}
}
