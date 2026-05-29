// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/gif"
	"io"
	"math"
)

// ConvertGIF encodes an animated GIF as a lossy WebP animation.
//
// Each GIF frame may be a sub-rectangle of the full canvas. ConvertGIF
// maintains an internal canvas the size of g.Config (or, if zero-valued,
// the bounds of g.Image[0]) and composites every GIF frame onto it,
// honouring GIF disposal methods (DoNotDispose, RestoreBackground,
// RestorePrevious).
//
// For GIFs where every frame uses DisposalBackground, ConvertGIF takes a
// specialised path that maps the GIF frame's own palette transparency
// directly to the ANMF alpha channel and sets the WebP per-frame disposal
// to 1 (clear-to-background). This mirrors the GIF compositor exactly:
//
//   - Opaque GIF pixels → alpha=255 in the ANMF (freshly VP8-encoded).
//   - Transparent GIF pixels → alpha=0 (inherit the bgcol-cleared canvas).
//
// The result is a compact, contiguous alpha bitmap whose shape matches the
// animation region rather than the pixel-level diff between composite
// canvases. VP8L compresses it far more efficiently, and the uniform alpha
// boundary eliminates the edge-flickering that arises when scattered
// transparent/opaque boundaries cut through VP8 macroblocks.
//
// For all other disposal methods the function falls back to the generic
// path: compute the dirty bounding rectangle of changed pixels, mark
// identical pixels as transparent (alpha=0) so the WebP decoder inherits
// the previously-decoded value, and use WebP per-frame disposal=0 (keep).
//
// The canvas is initialised to an opaque background colour derived from
// g.BackgroundIndex + the first frame's palette; if the indexed entry is
// transparent (or out of range) the canvas is filled with opaque black.
// This keeps the composited canvas fully opaque at all times — alpha
// channels in the emitted ANMF frames are used only as a transparency
// mask, never to encode partial translucency.
//
// Per-frame durations are converted from GIF (1/100 s) to WebP (1 ms) by
// multiplying by 10. The loop count is preserved (gif's -1 "show once"
// is mapped to 1 loop; 0 = infinite is kept as 0).
//
// Pass nil options for lossy at quality 90.
func ConvertGIF(w io.Writer, g *gif.GIF, o *Options) error {
	if g == nil {
		return errors.New("gowebp: nil gif")
	}
	if len(g.Image) == 0 {
		return errors.New("gowebp: gif has no frames")
	}

	// Encoding quality — needed for qualityToMaxDiff.
	quality := 90
	if o != nil && o.Quality > 0 {
		quality = o.Quality
	}
	// maxDiff is the per-channel RGB tolerance below which a pixel change is
	// considered "similar enough to inherit" rather than re-encode. A value of
	// 0 disables block-level flattening (lossless or very high quality).
	maxDiff := qualityToMaxDiff(quality)

	// Canvas size: prefer g.Config; if zero-valued, fall back to first
	// frame's Bounds.Max as documented by image/gif.
	canvasW := g.Config.Width
	canvasH := g.Config.Height
	if canvasW <= 0 || canvasH <= 0 {
		b := g.Image[0].Bounds()
		canvasW = b.Max.X
		canvasH = b.Max.Y
	}
	if canvasW <= 0 || canvasH <= 0 {
		return errors.New("gowebp: invalid gif canvas size")
	}
	canvasRect := image.Rect(0, 0, canvasW, canvasH)

	// Determine an opaque background colour. GIF's BackgroundIndex refers
	// to the global palette; if the global palette is absent, fall back to
	// the first frame's palette. If the indexed entry is transparent or
	// the index is out of range, use opaque black.
	bgCol := color.NRGBA{0, 0, 0, 255}
	{
		var palette color.Palette
		if p, ok := g.Config.ColorModel.(color.Palette); ok && len(p) > 0 {
			palette = p
		} else if len(g.Image[0].Palette) > 0 {
			palette = g.Image[0].Palette
		}
		if idx := int(g.BackgroundIndex); palette != nil && idx >= 0 && idx < len(palette) {
			r, gn, b, a := palette[idx].RGBA()
			if a != 0 {
				bgCol = color.NRGBA{
					R: uint8(r >> 8),
					G: uint8(gn >> 8),
					B: uint8(b >> 8),
					A: 255,
				}
			}
		}
	}

	// Initialise the canvas with the background colour.
	canvas := image.NewNRGBA(canvasRect)
	fillNRGBA(canvas, bgCol)

	// Snapshot used by DisposalPrevious — restored before drawing each
	// frame that *requested* prev-frame disposal (i.e. captured at the
	// moment before this frame is drawn, only when the *previous* frame's
	// disposal is DisposalPrevious). Per the spec, DisposalPrevious means
	// "restore to what was there before *this* frame was drawn", so we
	// snapshot just before draw and rewind after.
	prevSnapshot := image.NewNRGBA(canvasRect)

	// prevCanvas remembers the canvas *as it was sent to the encoder* for
	// the previous frame. Comparing it to the current canvas yields the
	// dirty bounding rectangle (the only region we need to re-encode).
	// Initialised to a colour that cannot match a legal NRGBA byte so
	// that the very first frame's diff covers the entire canvas (we want
	// the first ANMF to be a full keyframe regardless of background).
	prevCanvas := image.NewNRGBA(canvasRect)
	for i := range prevCanvas.Pix {
		prevCanvas.Pix[i] = 0xFF
	}
	// Mark "no previous frame yet" so frame 0 is forced to be a full
	// keyframe even if every byte of the initial bgCol fill happens to
	// equal the placeholder sentinel.
	havePrev := false

	// Pre-allocate per-frame slices for Animation.
	images := make([]image.Image, 0, len(g.Image))
	durations := make([]uint, 0, len(g.Image))
	disposals := make([]uint, 0, len(g.Image))
	blends := make([]uint, 0, len(g.Image))

	// Keyframe parameters. Matching libwebp's gif2webp defaults for lossy
	// (-kmin 3 -kmax 5): a keyframe is inserted no later than every kmax
	// frames. Keyframes overwrite the entire canvas (Blend=1, no ALPH chunk)
	// which resets accumulated VP8 quantisation noise that would otherwise
	// "ghost" into pixels that flip between opaque-fresh and transparent-inherited
	// across the delta chain.
	const (
		kmax = 5
	)
	countSinceKeyframe := 0

	// Detect whether every GIF frame uses DisposalBackground. If so we can
	// use the faster, flicker-free GIF-transparency-as-alpha path (see the
	// doc comment above).
	allDisposalBg := g.Disposal != nil && len(g.Disposal) == len(g.Image)
	if allDisposalBg {
		for _, d := range g.Disposal {
			if d != gif.DisposalBackground {
				allDisposalBg = false
				break
			}
		}
	}

	// Track the previous GIF frame's disposal so we know how to prepare
	// the canvas before drawing the current frame. We also track the
	// bounds of the previously-drawn frame for DisposalBackground.
	var prevDisposal byte
	var prevBounds image.Rectangle
	for i, frame := range g.Image {
		// 1. Apply *previous* frame's disposal before drawing the current frame.
		switch prevDisposal {
		case gif.DisposalBackground:
			// Clear the previously-drawn region to the background colour.
			if !prevBounds.Empty() {
				fillRectNRGBA(canvas, prevBounds.Intersect(canvasRect), bgCol)
			}
		case gif.DisposalPrevious:
			// Restore the canvas to the snapshot taken before the previous
			// frame was drawn.
			copy(canvas.Pix, prevSnapshot.Pix)
		}

		// 2. If this frame requests DisposalPrevious for *its* disposal,
		// snapshot the canvas now (before drawing) so we can restore later.
		var thisDisposal byte
		if g.Disposal != nil && i < len(g.Disposal) {
			thisDisposal = g.Disposal[i]
		}
		if thisDisposal == gif.DisposalPrevious {
			copy(prevSnapshot.Pix, canvas.Pix)
		}

		// 3. Composite the frame onto the canvas. Use draw.Over so
		// transparent palette entries leave the canvas pixel intact.
		// Note: we mutate `canvas` here in place; the canvas retains
		// whatever alpha values draw.Over produces. We deliberately do
		// not run forceOpaqueNRGBA on the compositing canvas itself —
		// that would corrupt blending state for later frames if any
		// pixel were partially transparent. Instead, we flatten to
		// opaque only on the *snapshot* we hand to the encoder below.
		fb := frame.Bounds()
		dst := fb.Intersect(canvasRect)
		if !dst.Empty() {
			draw.Draw(canvas, dst, frame, fb.Min, draw.Over)
		}

		// 4. Decide whether this frame is a keyframe or a delta frame.
		//
		// Frame 0 is always a keyframe (forced). For subsequent frames,
		// libwebp's gif2webp inserts a keyframe at least every kmax frames
		// (default 5 for lossy). A keyframe overwrites the whole canvas
		// with no alpha and is decoded into the WebP canvas as a single
		// fresh VP8 reference — this breaks the cumulative VP8 quantisation
		// chain that otherwise builds up across long sequences of delta
		// frames and causes "ghosting" / residual content in regions that
		// flip between opaque (freshly encoded) and transparent (inherited).
		//
		// The DisposalBackground path doesn't suffer from the inheritance
		// chain (it resets to bgcol every frame), so keyframes are
		// suppressed there to keep its output compact.
		isKeyframe := !havePrev
		if havePrev && !allDisposalBg && countSinceKeyframe >= kmax {
			isKeyframe = true
		}

		// 5. Compute the dirty bounding rectangle.
		// Keyframes always cover the full canvas. For delta frames the
		// dirty rect depends on the disposal path:
		//   - DisposalBackground path: bounding box of this frame's opaque
		//     GIF pixels (the animation region that needs fresh encoding).
		//   - Generic path: bounding box of pixels that changed vs prevCanvas.
		var dirty image.Rectangle
		if isKeyframe {
			dirty = canvasRect
		} else if allDisposalBg {
			dirty = computeGIFDirtyRect(frame, canvasRect)
		} else {
			dirty = computeDirtyRect(prevCanvas, canvas)
		}
		// Every GIF frame must produce one ANMF chunk to preserve frame
		// count and timing. Align to even offsets (WebP stores x/y ÷ 2)
		// and pad to at least 2×2.
		dirty = alignAndPadDirty(dirty, canvasRect)

		// 6. Extract the dirty sub-rect into a fresh NRGBA snapshot and
		// set per-pixel alpha.
		//
		// Keyframe path:
		//   All pixels are forced opaque so the WebP encoder skips ALPH and
		//   emits plain VP8. Combined with Blend=1 on the ANMF flags, this
		//   overwrites the entire decoder canvas with a clean VP8 image,
		//   resetting accumulated quantisation noise.
		//
		// DisposalBackground path (allDisposalBg):
		//   Alpha is taken directly from the GIF frame's palette
		//   transparency. Opaque GIF pixels → alpha=255 (VP8-encode fresh);
		//   transparent GIF pixels → alpha=0 (inherit bgcol from the
		//   WebP canvas cleared by the previous frame's Dispose=1).
		//   This maps the GIF compositor's "clear then draw" semantics into
		//   WebP with no per-canvas diffing — the result is a compact,
		//   contiguous alpha bitmap that VP8L compresses far better than
		//   the scattered pixel-diff mask.
		//
		// Generic delta path:
		//   Mark pixels identical to the previous canvas as transparent so
		//   the WebP decoder inherits the previously-decoded value, avoiding
		//   VP8 quantisation noise on static regions.
		sub := image.NewNRGBA(dirty)
		copyRectNRGBA(sub, canvas, dirty)
		if isKeyframe {
			forceOpaqueNRGBA(sub, bgCol)
		} else if allDisposalBg {
			markAlphaFromGIFFrame(sub, frame, dirty)
		} else {
			// Per-pixel fuzzy delta: pixels whose per-channel diff vs the
			// previous canvas is ≤ maxDiff are marked transparent (inherit
			// from the WebP decoder's canvas). Pixels that changed by more
			// than maxDiff are re-encoded fresh (opaque).
			//
			// Block-level approaches (FlattenSimilarBlocks) produce more
			// compressible alpha masks but create visible 8×8 grid artifacts
			// in frames where some blocks change and adjacent blocks inherit
			// a stale/noisy keyframe value. Per-pixel decisions follow the
			// actual change boundary at pixel resolution — no block-aligned
			// artifacts.
			markUnchangedTransparent(sub, prevCanvas, dirty, maxDiff)
		}
		images = append(images, sub)

		// 7. Save the current canvas for the next iteration's diff
		// (used by the generic path; harmless for the DisposalBackground path).
		copy(prevCanvas.Pix, canvas.Pix)
		havePrev = true

		// Reset / advance the keyframe counter. A keyframe resets the chain;
		// a delta frame extends it.
		if isKeyframe {
			countSinceKeyframe = 0
		} else {
			countSinceKeyframe++
		}

		// 8. Duration: 100ths of a second → milliseconds.
		var d int
		if i < len(g.Delay) {
			d = g.Delay[i]
		}
		if d < 0 {
			d = 0
		}
		durations = append(durations, uint(d)*10)

		// 9. Per-frame WebP disposal and blend.
		// DisposalBackground path: Dispose=1 clears the WebP canvas to bgcol
		// after each frame, mirroring the GIF compositor. The next frame's
		// alpha=0 pixels inherit bgcol exactly as intended.
		// Generic / keyframe path: Dispose=0 (keep) — pixels outside the
		// dirty rect remain visible from the previously-decoded frame.
		//
		// Blend=1 (no alpha blending) on keyframes overwrites the canvas
		// outright; Blend=0 on delta frames lets alpha=0 pixels pass through
		// to the previously decoded canvas value.
		if allDisposalBg {
			disposals = append(disposals, 1)
		} else {
			disposals = append(disposals, 0)
		}
		if isKeyframe {
			blends = append(blends, 1)
		} else {
			blends = append(blends, 0)
		}

		prevDisposal = thisDisposal
		prevBounds = fb
	}

	// LoopCount mapping: GIF -1 = show once → WebP loop count 1. GIF 0
	// = infinite → WebP 0. Otherwise pass through (GIF n loops the
	// animation n+1 times, which is also the WebP semantic).
	var loopCount uint16
	switch {
	case g.LoopCount < 0:
		loopCount = 1
	case g.LoopCount > 0xFFFF:
		loopCount = 0xFFFF
	default:
		loopCount = uint16(g.LoopCount)
	}

	ani := &Animation{
		Images:    images,
		Durations: durations,
		Disposals: disposals,
		Blends:    blends,
		LoopCount: loopCount,
		// BackgroundColor in WebP ANIM is BGRA. Pack bgCol into a uint32.
		BackgroundColor: uint32(bgCol.B) |
			uint32(bgCol.G)<<8 |
			uint32(bgCol.R)<<16 |
			uint32(bgCol.A)<<24,
	}
	return EncodeAll(w, ani, o)
}

// fillNRGBA fills the entire image with a solid opaque colour. The image
// must be aligned with the origin (Rect.Min == (0,0)).
func fillNRGBA(m *image.NRGBA, c color.NRGBA) {
	if len(m.Pix) == 0 {
		return
	}
	// Fill first row, then copy onto subsequent rows for speed.
	w := m.Rect.Dx()
	row := m.Pix[0 : w*4]
	for x := 0; x < w; x++ {
		row[x*4+0] = c.R
		row[x*4+1] = c.G
		row[x*4+2] = c.B
		row[x*4+3] = c.A
	}
	for y := 1; y < m.Rect.Dy(); y++ {
		copy(m.Pix[y*m.Stride:y*m.Stride+w*4], row)
	}
}

// fillRectNRGBA fills the given rectangle with the colour c. The rect
// must be already clipped to m.Bounds().
func fillRectNRGBA(m *image.NRGBA, r image.Rectangle, c color.NRGBA) {
	if r.Empty() {
		return
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		row := m.Pix[y*m.Stride+r.Min.X*4 : y*m.Stride+r.Max.X*4]
		for x := 0; x < r.Dx(); x++ {
			row[x*4+0] = c.R
			row[x*4+1] = c.G
			row[x*4+2] = c.B
			row[x*4+3] = c.A
		}
	}
}

// computeDirtyRect returns the smallest rectangle covering every pixel
// at which prev and curr differ in any RGBA byte. Both images must have
// identical Pix layout (same Rect, Stride, len(Pix)). The returned rect
// is in absolute canvas coords (matching curr.Rect). If the images are
// identical the returned rect has Min.X == Max.X (an empty rect).
func computeDirtyRect(prev, curr *image.NRGBA) image.Rectangle {
	// Both images are origin-anchored canvases of identical size; their
	// Pix slices line up byte-for-byte.
	w := curr.Rect.Dx()
	h := curr.Rect.Dy()
	stride := curr.Stride

	// Scan rows top-to-bottom to find first changed row.
	minY := h
	for y := 0; y < h; y++ {
		row := y * stride
		// Compare row-aligned 4-byte pixels.
		end := row + w*4
		// Fast path: bulk-compare the row. Go doesn't ship a memcmp but
		// `bytes.Equal` lowers to one on amd64/arm64.
		if !bytesEqualU8(prev.Pix[row:end], curr.Pix[row:end]) {
			minY = y
			break
		}
	}
	if minY == h {
		// No differences anywhere.
		return image.Rect(0, 0, 0, 0)
	}

	// Scan rows bottom-to-top for last changed row.
	maxY := minY
	for y := h - 1; y > minY; y-- {
		row := y * stride
		end := row + w*4
		if !bytesEqualU8(prev.Pix[row:end], curr.Pix[row:end]) {
			maxY = y
			break
		}
	}

	// Scan columns. Restrict the search to rows [minY, maxY].
	minX := w
	maxX := 0
	for y := minY; y <= maxY; y++ {
		base := y * stride
		// Left edge: smallest x with a diff.
		for x := 0; x < minX; x++ {
			i := base + x*4
			if prev.Pix[i] != curr.Pix[i] ||
				prev.Pix[i+1] != curr.Pix[i+1] ||
				prev.Pix[i+2] != curr.Pix[i+2] ||
				prev.Pix[i+3] != curr.Pix[i+3] {
				minX = x
				break
			}
		}
		// Right edge: largest x+1 with a diff.
		for x := w - 1; x > maxX; x-- {
			i := base + x*4
			if prev.Pix[i] != curr.Pix[i] ||
				prev.Pix[i+1] != curr.Pix[i+1] ||
				prev.Pix[i+2] != curr.Pix[i+2] ||
				prev.Pix[i+3] != curr.Pix[i+3] {
				maxX = x
				break
			}
		}
	}

	// Convert to half-open [Min, Max).
	return image.Rect(minX, minY, maxX+1, maxY+1)
}

// bytesEqualU8 is a tiny shim so callers don't need to import bytes.
// The compiler inlines this trivially.
func bytesEqualU8(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// alignAndPadDirty aligns r so it's safe to emit as a WebP ANMF sub-rect:
//
//   - Min.X and Min.Y are rounded down to even (WebP stores frame
//     offsets divided by 2, so the offset must be even).
//   - Max.X and Max.Y are rounded up to even and clamped to canvas.
//   - If r is empty, a 2×2 patch anchored at canvas origin is returned
//     (every GIF frame must contribute one ANMF chunk to preserve
//     frame timing, so we can never emit a zero-area frame).
//   - The returned rect is intersected with canvas.
//
// The canvas itself is assumed to start at (0, 0) — ConvertGIF constructs
// it that way.
func alignAndPadDirty(r, canvas image.Rectangle) image.Rectangle {
	if r.Empty() {
		// Emit a minimal 2×2 patch — visually a no-op (the patch
		// contains the canvas's existing pixels) but keeps frame
		// count and durations consistent.
		maxX := 2
		if maxX > canvas.Max.X {
			maxX = canvas.Max.X
		}
		maxY := 2
		if maxY > canvas.Max.Y {
			maxY = canvas.Max.Y
		}
		return image.Rect(0, 0, maxX, maxY)
	}
	// Round Min down to even.
	minX := r.Min.X &^ 1
	minY := r.Min.Y &^ 1
	// Round Max up to even.
	maxX := (r.Max.X + 1) &^ 1
	maxY := (r.Max.Y + 1) &^ 1
	// Clamp to canvas.
	if minX < canvas.Min.X {
		minX = canvas.Min.X
	}
	if minY < canvas.Min.Y {
		minY = canvas.Min.Y
	}
	if maxX > canvas.Max.X {
		maxX = canvas.Max.X
	}
	if maxY > canvas.Max.Y {
		maxY = canvas.Max.Y
	}
	// Ensure ≥ 2×2 to satisfy the even-offset / non-empty invariant.
	if maxX-minX < 2 {
		if minX+2 <= canvas.Max.X {
			maxX = minX + 2
		} else if maxX-2 >= canvas.Min.X {
			minX = maxX - 2
		}
	}
	if maxY-minY < 2 {
		if minY+2 <= canvas.Max.Y {
			maxY = minY + 2
		} else if maxY-2 >= canvas.Min.Y {
			minY = maxY - 2
		}
	}
	return image.Rect(minX, minY, maxX, maxY)
}

// copyRectNRGBA copies the pixels of src inside r into dst (dst.Rect ==
// r). Both images share the same coordinate space; r must lie within
// src.Bounds().
func copyRectNRGBA(dst, src *image.NRGBA, r image.Rectangle) {
	rw := r.Dx() * 4
	for y := r.Min.Y; y < r.Max.Y; y++ {
		dstOff := (y - dst.Rect.Min.Y) * dst.Stride
		srcOff := (y-src.Rect.Min.Y)*src.Stride + (r.Min.X-src.Rect.Min.X)*4
		copy(dst.Pix[dstOff:dstOff+rw], src.Pix[srcOff:srcOff+rw])
	}
}

// markUnchangedTransparent compares each pixel in sub (which holds the
// current canvas sub-rect at sub.Rect) against the same canvas location
// in prev. Pixels whose per-channel RGB difference is at most maxDiff in
// all three channels have their alpha set to 0 so the WebP decoder leaves
// the previous frame's pixel in place during alpha blending. Pixels that
// differ by more than maxDiff in any channel keep alpha = 255.
//
// maxDiff = 0 means exact comparison (only identical pixels are transparent).
// For lossy encoding pass qualityToMaxDiff(quality) so that "barely changed"
// pixels within VP8 quantisation tolerance are treated as unchanged —
// this eliminates flicker on pixels whose change is too small to survive
// lossy compression anyway. This is a per-pixel decision, so no block-level
// grid artifacts are introduced.
//
// The RGB component of an alpha-0 pixel is irrelevant on decode (the
// blend formula multiplies it by alpha/255 = 0), but it *does* affect
// the lossy VP8 encoder's quality: zeroing it would create artificial
// high-contrast edges along the boundary between unchanged and changed
// regions, which a DCT-based codec spreads as ringing into the RGB of
// nearby opaque pixels. We therefore leave the RGB at the *current
// canvas* value — i.e. the same colour as the previous frame — so the
// lossy VP8 stream sees a single smooth field across the boundary and
// can spend bits more efficiently on the actually-changed pixels.
//
// sub must be aligned: sub.Rect equals the dirty rectangle r passed to
// copyRectNRGBA, and prev is a full-canvas NRGBA with Rect.Min == (0,0).
func markUnchangedTransparent(sub *image.NRGBA, prev *image.NRGBA, r image.Rectangle, maxDiff int) {
	for y := r.Min.Y; y < r.Max.Y; y++ {
		subRowOff := (y - sub.Rect.Min.Y) * sub.Stride
		prevRowOff := y * prev.Stride
		for x := r.Min.X; x < r.Max.X; x++ {
			si := subRowOff + (x-sub.Rect.Min.X)*4
			pi := prevRowOff + x*4
			// Compare only the RGB bytes with per-channel tolerance.
			dr := int(sub.Pix[si]) - int(prev.Pix[pi])
			if dr < 0 {
				dr = -dr
			}
			dg := int(sub.Pix[si+1]) - int(prev.Pix[pi+1])
			if dg < 0 {
				dg = -dg
			}
			db := int(sub.Pix[si+2]) - int(prev.Pix[pi+2])
			if db < 0 {
				db = -db
			}
			if dr <= maxDiff && dg <= maxDiff && db <= maxDiff {
				// Within tolerance — mark transparent; keep RGB as-is.
				sub.Pix[si+3] = 0
			} else {
				sub.Pix[si+3] = 255
			}
		}
	}
}

// qualityToMaxDiff maps the encoding quality (0–100) to a per-channel RGB
// tolerance used for "similar enough to inherit from previous frame" decisions.
// Ported directly from libwebp's QualityToMaxDiff in mux/anim_encode.c:
//
//	val = sqrt(quality / 100)
//	maxDiff = round(31*(1-val) + val)
//
// At quality=90 → maxDiff=3; quality=75 → 5; quality=50 → 10; quality=0 → 31.
func qualityToMaxDiff(quality int) int {
	val := math.Sqrt(float64(quality) / 100.0)
	return int(31*(1-val) + val + 0.5)
}

// flattenSimilarBlocks refines the per-pixel alpha mask produced by
// markUnchangedTransparent by applying block-level decisions. For each
// aligned 8×8 block entirely within r: if every pixel in the block
// satisfies abs(sub.R-prev.R) ≤ maxDiff && abs(sub.G-prev.G) ≤ maxDiff &&
// abs(sub.B-prev.B) ≤ maxDiff (i.e. the visible change is within VP8
// quantisation tolerance), the entire block is forced to alpha=0
// (transparent / inherit-from-canvas). Partial blocks at the boundary of r
// are left untouched (the per-pixel step already handles them).
//
// This is a port of libwebp's FlattenSimilarBlocks in mux/anim_encode.c.
// The block-aligned result compresses far better with VP8L (contiguous opaque
// and transparent regions replace scattered per-pixel flags) and eliminates
// the edge-flickering that occurs when a jagged alpha boundary bisects VP8
// macroblocks.
func flattenSimilarBlocks(sub *image.NRGBA, prev *image.NRGBA, r image.Rectangle, maxDiff int) {
	if maxDiff <= 0 {
		return
	}
	const bs = 8
	// Process only complete 8×8 blocks, starting from the second block
	// boundary inside r (matching libwebp's skip of the first partial block).
	startY := (r.Min.Y + bs) &^ (bs - 1)
	startX := (r.Min.X + bs) &^ (bs - 1)
	endY := r.Max.Y &^ (bs - 1)
	endX := r.Max.X &^ (bs - 1)

	for by := startY; by < endY; by += bs {
		for bx := startX; bx < endX; bx += bs {
			// Check whether every pixel in the block is within maxDiff.
			ok := true
		outer:
			for dy := 0; dy < bs; dy++ {
				si := (by+dy-sub.Rect.Min.Y)*sub.Stride + (bx-sub.Rect.Min.X)*4
				pi := (by+dy)*prev.Stride + bx*4
				for dx := 0; dx < bs; dx++ {
					dr := int(sub.Pix[si]) - int(prev.Pix[pi])
					if dr < 0 {
						dr = -dr
					}
					if dr > maxDiff {
						ok = false
						break outer
					}
					dg := int(sub.Pix[si+1]) - int(prev.Pix[pi+1])
					if dg < 0 {
						dg = -dg
					}
					if dg > maxDiff {
						ok = false
						break outer
					}
					db := int(sub.Pix[si+2]) - int(prev.Pix[pi+2])
					if db < 0 {
						db = -db
					}
					if db > maxDiff {
						ok = false
						break outer
					}
					si += 4
					pi += 4
				}
			}
			if ok {
				for dy := 0; dy < bs; dy++ {
					si := (by+dy-sub.Rect.Min.Y)*sub.Stride + (bx-sub.Rect.Min.X)*4
					for dx := 0; dx < bs; dx++ {
						sub.Pix[si+3] = 0
						si += 4
					}
				}
			}
		}
	}
}

// gifTransparentIndex returns the palette index of the transparent colour
// entry, or -1 if none exists.
func gifTransparentIndex(frame *image.Paletted) int {
	for i, c := range frame.Palette {
		if _, _, _, a := c.RGBA(); a == 0 {
			return i
		}
	}
	return -1
}

// computeGIFDirtyRect returns the smallest rectangle that contains every
// non-transparent pixel in frame, clipped to canvasRect. Used by the
// DisposalBackground path to find the ANMF sub-rect that needs encoding
// (only the animation content, not the background the WebP decoder inherits).
func computeGIFDirtyRect(frame *image.Paletted, canvasRect image.Rectangle) image.Rectangle {
	transpIdx := gifTransparentIndex(frame)
	fb := frame.Bounds().Intersect(canvasRect)
	if fb.Empty() {
		return image.Rectangle{}
	}
	bfb := frame.Bounds() // unclipped, for Pix offset arithmetic

	minX, minY := fb.Max.X, fb.Max.Y
	maxX, maxY := fb.Min.X-1, fb.Min.Y-1
	for y := fb.Min.Y; y < fb.Max.Y; y++ {
		row := frame.Pix[(y-bfb.Min.Y)*frame.Stride:]
		for x := fb.Min.X; x < fb.Max.X; x++ {
			if int(row[x-bfb.Min.X]) != transpIdx {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < minX || maxY < minY {
		return image.Rectangle{}
	}
	return image.Rect(minX, minY, maxX+1, maxY+1)
}

// markAlphaFromGIFFrame sets the alpha channel of sub using the GIF frame's
// own palette transparency within r. Opaque GIF pixels → alpha=255 (VP8 will
// encode fresh colour values); transparent GIF pixels → alpha=0 (the WebP
// decoder inherits bgcol from the canvas cleared by the previous ANMF's
// Dispose=1). Pixels in r that lie outside the GIF frame bounds are treated
// as transparent (they contain bgcol from the disposal step).
func markAlphaFromGIFFrame(sub *image.NRGBA, frame *image.Paletted, r image.Rectangle) {
	transpIdx := gifTransparentIndex(frame)
	fb := frame.Bounds()
	for y := r.Min.Y; y < r.Max.Y; y++ {
		subRowOff := (y - sub.Rect.Min.Y) * sub.Stride
		for x := r.Min.X; x < r.Max.X; x++ {
			si := subRowOff + (x-sub.Rect.Min.X)*4
			if x < fb.Min.X || x >= fb.Max.X || y < fb.Min.Y || y >= fb.Max.Y {
				sub.Pix[si+3] = 0
				continue
			}
			if int(frame.Pix[(y-fb.Min.Y)*frame.Stride+(x-fb.Min.X)]) == transpIdx {
				sub.Pix[si+3] = 0
			} else {
				sub.Pix[si+3] = 255
			}
		}
	}
}

// forceOpaqueNRGBA replaces any pixel with alpha < 255 by alpha-blending
// it against bg, then forcing alpha to 255. After this call every pixel
// is fully opaque so the image can be safely passed to VP8 lossy encode.
func forceOpaqueNRGBA(m *image.NRGBA, bg color.NRGBA) {
	if len(m.Pix) == 0 {
		return
	}
	w := m.Rect.Dx()
	h := m.Rect.Dy()
	for y := 0; y < h; y++ {
		row := m.Pix[y*m.Stride : y*m.Stride+w*4]
		for x := 0; x < w; x++ {
			a := row[x*4+3]
			if a == 255 {
				continue
			}
			if a == 0 {
				row[x*4+0] = bg.R
				row[x*4+1] = bg.G
				row[x*4+2] = bg.B
			} else {
				// NRGBA blend over opaque background: out = src*α + bg*(1-α).
				af := uint16(a)
				naf := 255 - af
				row[x*4+0] = uint8((uint16(row[x*4+0])*af + uint16(bg.R)*naf) / 255)
				row[x*4+1] = uint8((uint16(row[x*4+1])*af + uint16(bg.G)*naf) / 255)
				row[x*4+2] = uint8((uint16(row[x*4+2])*af + uint16(bg.B)*naf) / 255)
			}
			row[x*4+3] = 255
		}
	}
}
