// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

// Package gowebp implements a pure-Go WebP encoder supporting both
// lossy (VP8) and lossless (VP8L) output.
package gowebp

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"io"
	"runtime"

	"github.com/TommyLeng/gowebp/lossless"
)

// Options controls encoding behaviour.
type Options struct {
	Lossless bool // true = VP8L lossless, false = VP8 lossy (default)
	Quality  int  // 0–100, only used when Lossless=false (default: 90)
}

// Animation holds configuration for an animated WebP sequence.
//
// Fields:
//   - Images: frames to be displayed in sequence (must be non-empty).
//   - Durations: per-frame display time in milliseconds (must match Images length).
//   - Disposals: per-frame disposal method after display: 0 = keep, 1 = clear to background.
//   - Blends: per-frame blend method. Optional; if nil or empty all frames are
//     treated as 0 = alpha-blend (the WebP decoder composites a frame's pixels
//     over the existing canvas honouring per-pixel alpha). A value of 1 means
//     "no blending" — the frame's pixels overwrite the canvas regardless of
//     alpha. Keyframes use Blends=1. Length, when non-nil, must match Images.
//   - LoopCount: number of times the animation repeats; 0 = infinite.
//   - BackgroundColor: canvas background colour in BGRA byte order, used when a
//     frame's disposal == 1. Stored as a little-endian uint32 in the ANIM chunk.
type Animation struct {
	Images          []image.Image
	Durations       []uint
	Disposals       []uint
	Blends          []uint
	LoopCount       uint16
	BackgroundColor uint32
}

// Encode encodes img as a WebP image and writes the result to w.
// Pass nil options for lossy at quality 90.
func Encode(w io.Writer, img image.Image, o *Options) error {
	if o != nil && o.Lossless {
		return lossless.Encode(w, img, nil)
	}
	quality := 90
	if o != nil && o.Quality > 0 {
		quality = o.Quality
	}
	return encodeLossy(w, img, quality)
}

// encodeLossy encodes img as a lossy VP8 WebP image at the given quality (0..100).
func encodeLossy(w io.Writer, img image.Image, quality int) error {
	if quality < 0 {
		quality = 0
	}
	if quality > 100 {
		quality = 100
	}

	// If image has a non-trivial alpha channel, use Extended format (VP8 RGB + VP8L alpha).
	if imageHasAlpha(img) {
		return encodeLossyWithAlpha(w, img, quality)
	}

	internalQuality := quality

	// Acquire a reusable frame arena; return it when done.
	arena := arenaPool.Get().(*frameArena)
	defer arenaPool.Put(arena)

	// Convert to YUV 4:2:0
	yuv := rgbaToYUV420(img, arena)

	// Compute base quantizer index; per-MB quantizers are determined inside encodeFrame
	// via the SNS two-segment scheme (computeSNSSegmentQualities).
	baseQ := qualityToLevel(internalQuality)

	// Encode the VP8 frame — use wave-front parallel encoding for large images.
	mbCount := (yuv.mbW / 16) * (yuv.mbH / 16)
	var vp8Data []byte
	if mbCount > parallelThreshold && runtime.GOMAXPROCS(0) > 1 {
		vp8Data = encodeFrameParallel(yuv, baseQ, arena)
	} else {
		vp8Data = encodeFrame(yuv, baseQ, arena)
	}

	// Write WebP container
	return writeWebPHeader(w, vp8Data)
}

// EncodeAll writes ani as an animated WebP to w.
//
// The container layout is:
//
//	RIFF .... WEBP
//	  VP8X    (canvas size + animation flag)
//	  ANIM    (background colour + loop count)
//	  ANMF .. (per-frame: offset/size/duration/flags + inner "VP8 " chunk)
//	  ANMF ..
//	  ...
//
// Each frame is encoded using the existing lossy VP8 pipeline at o.Quality
// (default 90). Alpha is dropped — VP8 lossy does not encode alpha and this
// implementation does not emit ALPH chunks per frame. Pass nil options for
// lossy at quality 90.
//
// Errors are returned for empty Images, mismatched length slices, or any
// underlying encoder failure.
func EncodeAll(w io.Writer, ani *Animation, o *Options) error {
	if ani == nil {
		return errors.New("gowebp: nil animation")
	}
	if len(ani.Images) == 0 {
		return errors.New("gowebp: must provide at least one image")
	}
	if len(ani.Images) != len(ani.Durations) {
		return errors.New("gowebp: mismatched image and durations lengths")
	}
	if len(ani.Images) != len(ani.Disposals) {
		return errors.New("gowebp: mismatched image and disposals lengths")
	}
	// Blends is optional. When provided, its length must match Images.
	if ani.Blends != nil && len(ani.Blends) != len(ani.Images) {
		return errors.New("gowebp: mismatched image and blends lengths")
	}

	quality := 90
	if o != nil && o.Quality > 0 {
		quality = o.Quality
	}

	// Canvas = max bounds across all frames (matches lossless behaviour).
	var canvas image.Rectangle
	for _, img := range ani.Images {
		if img == nil {
			return errors.New("gowebp: nil image in animation")
		}
		b := img.Bounds()
		if b.Max.X > canvas.Max.X {
			canvas.Max.X = b.Max.X
		}
		if b.Max.Y > canvas.Max.Y {
			canvas.Max.Y = b.Max.Y
		}
	}
	if canvas.Dx() < 1 || canvas.Dy() < 1 {
		return errors.New("gowebp: invalid canvas size")
	}
	// VP8X canvas size is encoded as (width-1)/(height-1) in 24 bits each, so
	// the maximum addressable canvas is 2^24 = 16,777,216 in either dim.
	if canvas.Dx() > 1<<24 || canvas.Dy() > 1<<24 {
		return errors.New("gowebp: canvas dimensions exceed 24-bit limit")
	}

	// Encode every frame ahead of time so we know all chunk sizes before
	// emitting the outer RIFF header (which carries the total payload size).
	frames := &bytes.Buffer{}
	const maxU24 = uint(1<<24 - 1)
	// anyAlpha is set to true if any frame ends up encoded with an ALPH
	// chunk; we need to reflect that in the top-level VP8X flags.
	anyAlpha := false
	for i, img := range ani.Images {
		// Encode this frame via the standard lossy pipeline (which itself
		// chooses plain VP8 vs VP8X+ALPH+VP8 based on alpha content), then
		// strip the RIFF/WEBP wrapper and extract the chunks suitable for
		// ANMF embedding. ANMF embeds either "VP8 " alone or "ALPH" + "VP8 "
		// — never "VP8X" (the per-canvas VP8X header lives at the top of
		// the WebP container).
		var frameBuf bytes.Buffer
		if err := encodeLossy(&frameBuf, img, quality); err != nil {
			return err
		}
		frameBytes := frameBuf.Bytes()
		// Outer wrapper is 12 bytes: "RIFF" + uint32 size + "WEBP".
		if len(frameBytes) < 12 ||
			string(frameBytes[0:4]) != "RIFF" ||
			string(frameBytes[8:12]) != "WEBP" {
			return errors.New("gowebp: unexpected frame container layout")
		}
		// Parse the inner chunks. For plain lossy we expect:
		//   "VP8 "(4) + size(4) + data + pad
		// For lossy + alpha we expect:
		//   "VP8X"(4) + size(4) + 10-byte VP8X payload
		//   "ALPH"(4) + size(4) + alpha payload + pad
		//   "VP8 "(4) + size(4) + data + pad
		// We need to capture the ALPH chunk (if present) and the VP8 chunk
		// and concatenate them for ANMF embedding (without the outer VP8X).
		innerChunk, frameHasAlpha, err := extractANMFInner(frameBytes[12:])
		if err != nil {
			return err
		}
		if frameHasAlpha {
			anyAlpha = true
		}

		b := img.Bounds()
		dur := uint(ani.Durations[i])
		if dur > maxU24 {
			dur = maxU24
		}
		disp := uint(ani.Disposals[i])
		if disp > 1 {
			disp = 1
		}
		var blend uint
		if ani.Blends != nil {
			blend = uint(ani.Blends[i])
			if blend > 1 {
				blend = 1
			}
		}

		// ANMF payload = 16 bytes of frame params + inner chunks.
		anmfPayloadSize := uint32(16 + len(innerChunk))

		frames.Write([]byte("ANMF"))
		_ = binary.Write(frames, binary.LittleEndian, anmfPayloadSize)

		// Frame x/y offset: 24-bit LE, stored as value/2 (the spec divides
		// by 2). Callers passing odd offsets is a bug — round-down here.
		writeU24LE(frames, uint32(b.Min.X/2))
		writeU24LE(frames, uint32(b.Min.Y/2))
		// Frame width-1 / height-1: 24-bit LE.
		writeU24LE(frames, uint32(b.Dx()-1))
		writeU24LE(frames, uint32(b.Dy()-1))
		// Duration: 24-bit LE milliseconds.
		writeU24LE(frames, uint32(dur))
		// Flags byte: bit 0 = disposal, bit 1 = blending.
		//   - bit 1 = 0: use alpha blending (the previous canvas pixel
		//     shows through transparent areas of this frame). This is
		//     what we want for delta-encoded frames where unchanged
		//     pixels carry alpha=0.
		//   - bit 1 = 1: do not blend (overwrite canvas pixel regardless
		//     of frame alpha). Used for keyframes (full-canvas refresh
		//     that resets accumulated VP8 quantisation noise).
		// Default is alpha blending; the caller can request no-blend on a
		// per-frame basis via Animation.Blends.
		var flags byte
		if disp == 1 {
			flags |= 1 << 0
		}
		if blend == 1 {
			flags |= 1 << 1
		}
		frames.WriteByte(flags)

		// Inner chunks (ALPH + VP8 or just VP8). Pre-padded.
		frames.Write(innerChunk)

		// Pad ANMF payload to even length (RIFF alignment).
		if anmfPayloadSize&1 == 1 {
			frames.WriteByte(0)
		}
	}

	// Build the inner-payload buffer (everything after "WEBP").
	body := &bytes.Buffer{}
	writeAnimVP8XChunk(body, canvas.Dx(), canvas.Dy(), anyAlpha)

	body.Write([]byte("ANIM"))
	_ = binary.Write(body, binary.LittleEndian, uint32(6))
	_ = binary.Write(body, binary.LittleEndian, uint32(ani.BackgroundColor))
	_ = binary.Write(body, binary.LittleEndian, uint16(ani.LoopCount))

	body.Write(frames.Bytes())

	// Outer RIFF/WEBP wrapper.
	if _, err := w.Write([]byte("RIFF")); err != nil {
		return err
	}
	if err := writeLE32(w, uint32(4+body.Len())); err != nil {
		return err
	}
	if _, err := w.Write([]byte("WEBP")); err != nil {
		return err
	}
	if _, err := w.Write(body.Bytes()); err != nil {
		return err
	}
	return nil
}

// extractANMFInner scans the chunks of a WebP container's inner body
// (the bytes after "WEBP") and returns the bytes that should be embedded
// inside an ANMF chunk: either "VP8 "(...) alone (no alpha), or
// "ALPH"(...) + "VP8 "(...) (lossy frame carrying alpha). The
// per-canvas VP8X chunk (if present) is stripped — ANMF must not embed
// it. hasAlpha reports whether an ALPH chunk was present.
//
// Each returned chunk includes its 4-byte tag, 4-byte little-endian
// size, payload, and the optional 1-byte odd-size pad — exactly the
// on-disk format ANMF expects.
func extractANMFInner(inner []byte) (out []byte, hasAlpha bool, err error) {
	var alphChunk, vp8Chunk []byte
	for off := 0; off+8 <= len(inner); {
		tag := string(inner[off : off+4])
		size := uint32(inner[off+4]) |
			uint32(inner[off+5])<<8 |
			uint32(inner[off+6])<<16 |
			uint32(inner[off+7])<<24
		// Chunks are padded to even length on disk.
		end := off + 8 + int(size)
		padded := end
		if size&1 == 1 {
			padded++
		}
		if padded > len(inner) {
			return nil, false, errors.New("gowebp: malformed inner chunk size")
		}
		chunk := inner[off:padded] // includes tag+size+payload(+pad)
		switch tag {
		case "VP8X":
			// Top-level VP8X — never embedded in ANMF.
		case "ALPH":
			alphChunk = chunk
		case "VP8 ":
			vp8Chunk = chunk
		default:
			// Unknown chunk: ignore (forward-compatible with future
			// chunks emitted by encodeLossy variants).
		}
		off = padded
	}
	if vp8Chunk == nil {
		return nil, false, errors.New("gowebp: encoded frame missing VP8 chunk")
	}
	if alphChunk == nil {
		return vp8Chunk, false, nil
	}
	// ANMF embedded order: ALPH must precede VP8 (spec, Sec. Animation).
	combined := make([]byte, 0, len(alphChunk)+len(vp8Chunk))
	combined = append(combined, alphChunk...)
	combined = append(combined, vp8Chunk...)
	return combined, true, nil
}

// writeAnimVP8XChunk writes a VP8X chunk for an animated WebP. The
// animation flag is always set; the alpha flag is set iff any frame in
// the animation carries an ALPH chunk (so decoders know to look for
// per-frame transparency).
func writeAnimVP8XChunk(buf *bytes.Buffer, width, height int, hasAlpha bool) {
	buf.Write([]byte("VP8X"))
	_ = binary.Write(buf, binary.LittleEndian, uint32(10))

	// Flags byte: bit 1 = animation, bit 4 = alpha.
	var flags byte
	flags |= 1 << 1
	if hasAlpha {
		flags |= 1 << 4
	}
	buf.WriteByte(flags)
	// 3 reserved bytes.
	buf.Write([]byte{0x00, 0x00, 0x00})

	dx := uint32(width - 1)
	dy := uint32(height - 1)
	writeU24LE(buf, dx)
	writeU24LE(buf, dy)
}

// writeU24LE writes v as a 24-bit little-endian value (3 bytes). Caller
// must ensure v fits in 24 bits.
func writeU24LE(buf *bytes.Buffer, v uint32) {
	buf.WriteByte(byte(v))
	buf.WriteByte(byte(v >> 8))
	buf.WriteByte(byte(v >> 16))
}
