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
)

// ConvertGIF encodes an animated GIF as a lossy WebP animation.
//
// Each GIF frame may be a sub-rectangle of the full canvas. ConvertGIF
// maintains a canvas the size of g.Config (or, if zero-valued, the bounds
// of g.Image[0]) and composites every frame onto it before encoding the
// full canvas as a VP8 lossy frame. GIF disposal methods are handled by
// updating the in-memory canvas between frames, so every emitted ANMF
// chunk covers the entire canvas with WebP disposal=keep (0).
//
// The canvas is initialised to an opaque background colour derived from
// g.BackgroundIndex + the first frame's palette; if the indexed entry is
// transparent (or out of range) the canvas is filled with opaque black.
// This keeps the composited canvas fully opaque at all times — VP8 lossy
// cannot carry alpha and EncodeAll will reject any frame with a non-trivial
// alpha channel.
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

	// Pre-allocate per-frame slices for Animation.
	images := make([]image.Image, 0, len(g.Image))
	durations := make([]uint, 0, len(g.Image))
	disposals := make([]uint, 0, len(g.Image))

	// Track the previous frame's disposal so we know how to prepare the
	// canvas before drawing the current frame. We also track the bounds
	// of the previously-drawn frame for DisposalBackground.
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
		fb := frame.Bounds()
		dst := fb.Intersect(canvasRect)
		if !dst.Empty() {
			draw.Draw(canvas, dst, frame, fb.Min, draw.Over)
		}

		// 4. Ensure the canvas is fully opaque before encoding. After
		// draw.Over on top of an opaque canvas the result is opaque, but
		// be defensive: any pixel that somehow ended up with alpha<255 is
		// alpha-flattened against the background.
		forceOpaqueNRGBA(canvas, bgCol)

		// 5. Snapshot the composited frame into the output slice. We
		// have to clone because canvas is mutated in place between frames.
		frameOut := image.NewNRGBA(canvasRect)
		copy(frameOut.Pix, canvas.Pix)
		images = append(images, frameOut)

		// 6. Duration: 100ths of a second → milliseconds.
		var d int
		if i < len(g.Delay) {
			d = g.Delay[i]
		}
		if d < 0 {
			d = 0
		}
		durations = append(durations, uint(d)*10)

		// 7. Per-frame WebP disposal: every frame already covers the full
		// canvas (pre-composited), so the decoder doesn't need to clear
		// anything — keep the frame in place (disposal 0).
		disposals = append(disposals, 0)

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
