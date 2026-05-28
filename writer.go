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
//   - LoopCount: number of times the animation repeats; 0 = infinite.
//   - BackgroundColor: canvas background colour in BGRA byte order, used when a
//     frame's disposal == 1. Stored as a little-endian uint32 in the ANIM chunk.
type Animation struct {
	Images          []image.Image
	Durations       []uint
	Disposals       []uint
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
	for i, img := range ani.Images {
		// Encode this frame via the standard lossy pipeline, then strip the
		// outer RIFF/WEBP wrapper. After stripping, frameBuf starts at the
		// inner "VP8 " chunk (tag + size + payload), which is exactly what
		// ANMF expects to embed.
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
		// The inner chunk starts at byte 12. For plain lossy (no alpha,
		// no VP8X) the first inner chunk is "VP8 "; if the frame contained
		// alpha then encodeLossy emits an Extended container (VP8X + VP8 +
		// ALPH) which we cannot embed directly into ANMF. Reject that here
		// — alpha support is explicitly out of scope.
		if string(frameBytes[12:16]) != "VP8 " {
			return errors.New("gowebp: lossy animation does not support alpha frames")
		}
		innerChunk := frameBytes[12:] // "VP8 " + size + data (+ pad)

		b := img.Bounds()
		dur := uint(ani.Durations[i])
		if dur > maxU24 {
			dur = maxU24
		}
		disp := uint(ani.Disposals[i])
		if disp > 1 {
			disp = 1
		}

		// ANMF payload = 16 bytes of frame params + inner VP8 chunk.
		anmfPayloadSize := uint32(16 + len(innerChunk))

		frames.Write([]byte("ANMF"))
		_ = binary.Write(frames, binary.LittleEndian, anmfPayloadSize)

		// Frame x/y offset: 24-bit LE, must be even (spec divides by 2).
		writeU24LE(frames, uint32(b.Min.X/2))
		writeU24LE(frames, uint32(b.Min.Y/2))
		// Frame width-1 / height-1: 24-bit LE.
		writeU24LE(frames, uint32(b.Dx()-1))
		writeU24LE(frames, uint32(b.Dy()-1))
		// Duration: 24-bit LE milliseconds.
		writeU24LE(frames, uint32(dur))
		// Flags byte: bit 0 = disposal, bit 1 = blending (0 = use alpha
		// blending; we have no alpha so the bit is irrelevant, leave 0).
		var flags byte
		if disp == 1 {
			flags |= 1 << 0
		}
		frames.WriteByte(flags)

		// Inner VP8 chunk (tag + size + data + optional pad byte).
		frames.Write(innerChunk)

		// Pad ANMF payload to even length (RIFF alignment).
		if anmfPayloadSize&1 == 1 {
			frames.WriteByte(0)
		}
	}

	// Build the inner-payload buffer (everything after "WEBP").
	body := &bytes.Buffer{}
	writeAnimVP8XChunk(body, canvas.Dx(), canvas.Dy())

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

// writeAnimVP8XChunk writes a VP8X chunk for an animated WebP with the
// animation flag set, alpha flag clear, canvas width-1 / height-1 in 24-bit LE.
func writeAnimVP8XChunk(buf *bytes.Buffer, width, height int) {
	buf.Write([]byte("VP8X"))
	_ = binary.Write(buf, binary.LittleEndian, uint32(10))

	// Flags byte: bit 1 = animation, bit 4 = alpha (cleared).
	var flags byte
	flags |= 1 << 1
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
