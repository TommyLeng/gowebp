// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"io"
	"runtime"

	"github.com/TommyLeng/gowebp/lossless"
)

// imageHasAlpha reports whether img contains any pixel with alpha < 255.
// Only *image.NRGBA and *image.RGBA are inspected; all other types are opaque.
func imageHasAlpha(img image.Image) bool {
	switch m := img.(type) {
	case *image.NRGBA:
		w := m.Rect.Dx()
		for y, h := 0, m.Rect.Dy(); y < h; y++ {
			row := m.Pix[y*m.Stride : y*m.Stride+w*4]
			for i := 3; i < len(row); i += 4 {
				if row[i] < 255 {
					return true
				}
			}
		}
	case *image.RGBA:
		w := m.Rect.Dx()
		for y, h := 0, m.Rect.Dy(); y < h; y++ {
			row := m.Pix[y*m.Stride : y*m.Stride+w*4]
			for i := 3; i < len(row); i += 4 {
				if row[i] < 255 {
					return true
				}
			}
		}
	}
	return false
}

// encodeAlphaChunk encodes the alpha plane of img as VP8L lossless.
// Returns the ALPH chunk payload: flags byte (0x01 = VP8L) + VP8L bitstream.
// The WebP spec requires alpha to be stored in the "green" channel of a VP8L image.
func encodeAlphaChunk(img image.Image, w, h int) ([]byte, error) {
	alphaImg := image.NewNRGBA(image.Rect(0, 0, w, h))
	switch m := img.(type) {
	case *image.NRGBA:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				a := m.Pix[y*m.Stride+x*4+3]
				i := y*alphaImg.Stride + x*4
				alphaImg.Pix[i+1] = a   // G = alpha value
				alphaImg.Pix[i+3] = 255 // A = opaque so VP8L treats as RGB
			}
		}
	case *image.RGBA:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				a := m.Pix[y*m.Stride+x*4+3]
				i := y*alphaImg.Stride + x*4
				alphaImg.Pix[i+1] = a
				alphaImg.Pix[i+3] = 255
			}
		}
	}

	var buf bytes.Buffer
	if err := lossless.Encode(&buf, alphaImg, nil); err != nil {
		return nil, fmt.Errorf("alpha VP8L encode: %w", err)
	}

	// Strip RIFF container header and VP8L frame header to get the raw entropy-coded data.
	//
	// golang.org/x/image/webp readAlpha(compression=1) synthesizes its own 5-byte VP8L
	// frame header (magic 0x2f + 14-bit w-1 + 14-bit h-1 + 1-bit hasAlpha + 3-bit version)
	// and prepends it via io.MultiReader before calling vp8l.Decode. So the ALPH payload
	// must NOT include those 5 bytes — only the Huffman/entropy-coded pixel data.
	//
	// lossless.Encode output layout:
	//   "RIFF"(4) + size(4) + "WEBP"(4) + "VP8L"(4) + vp8l_size(4) = 20-byte RIFF header
	//   0x2f(1) + w-1(14bits) + h-1(14bits) + hasAlpha(1bit) + version(3bits) = 5-byte VP8L frame header
	// Total to skip: 25 bytes.
	b := buf.Bytes()
	const skipLen = 25 // 20-byte RIFF container + 5-byte VP8L frame header
	if len(b) < skipLen {
		return nil, fmt.Errorf("alpha VP8L encode: output too short (%d bytes)", len(b))
	}
	vp8lStream := b[skipLen:]

	// ALPH payload: flags byte (0x01 = VP8L, no filtering, no preprocessing) + bitstream.
	payload := make([]byte, 1+len(vp8lStream))
	payload[0] = 0x01
	copy(payload[1:], vp8lStream)
	return payload, nil
}

// writeLE24 writes v as a 24-bit little-endian integer.
func writeLE24(w io.Writer, v uint32) error {
	b := [3]byte{byte(v), byte(v >> 8), byte(v >> 16)}
	_, err := w.Write(b[:])
	return err
}

// writeWebPExtended writes the WebP Extended format: RIFF/WEBP with VP8X + ALPH + VP8 chunks.
func writeWebPExtended(w io.Writer, vp8Data, alphData []byte, width, height int) error {
	vp8Size := uint32(len(vp8Data))
	alphSize := uint32(len(alphData))
	vp8Pad := vp8Size & 1
	alphPad := alphSize & 1

	const vp8xPayload = 10
	riffSize := uint32(4) + // "WEBP"
		uint32(8+vp8xPayload) + // VP8X chunk
		uint32(8)+alphSize+alphPad + // ALPH chunk
		uint32(8)+vp8Size+vp8Pad // VP8  chunk

	// RIFF header.
	if _, err := w.Write([]byte{'R', 'I', 'F', 'F'}); err != nil {
		return err
	}
	if err := writeLE32(w, riffSize); err != nil {
		return err
	}
	if _, err := w.Write([]byte{'W', 'E', 'B', 'P'}); err != nil {
		return err
	}

	// VP8X chunk (10 bytes payload: flags + canvas w-1 + canvas h-1).
	if _, err := w.Write([]byte{'V', 'P', '8', 'X'}); err != nil {
		return err
	}
	if err := writeLE32(w, vp8xPayload); err != nil {
		return err
	}
	if err := writeLE32(w, 1<<4); err != nil { // bit 4 = has_alpha
		return err
	}
	if err := writeLE24(w, uint32(width-1)); err != nil {
		return err
	}
	if err := writeLE24(w, uint32(height-1)); err != nil {
		return err
	}

	// ALPH chunk.
	if _, err := w.Write([]byte{'A', 'L', 'P', 'H'}); err != nil {
		return err
	}
	if err := writeLE32(w, alphSize); err != nil {
		return err
	}
	if _, err := w.Write(alphData); err != nil {
		return err
	}
	if alphPad == 1 {
		if _, err := w.Write([]byte{0}); err != nil {
			return err
		}
	}

	// VP8  chunk.
	if _, err := w.Write([]byte{'V', 'P', '8', ' '}); err != nil {
		return err
	}
	if err := writeLE32(w, vp8Size); err != nil {
		return err
	}
	if _, err := w.Write(vp8Data); err != nil {
		return err
	}
	if vp8Pad == 1 {
		if _, err := w.Write([]byte{0}); err != nil {
			return err
		}
	}

	return nil
}

// encodeLossyWithAlpha encodes img as lossy VP8 (RGB) + lossless alpha in WebP Extended format.
func encodeLossyWithAlpha(out io.Writer, img image.Image, quality int) error {
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	arena := arenaPool.Get().(*frameArena)
	defer arenaPool.Put(arena)
	yuv := rgbaToYUV420(img, arena)
	baseQ := qualityToLevel(quality)
	mbCount := (yuv.mbW / 16) * (yuv.mbH / 16)
	var vp8Data []byte
	// debugDumpI16Capture writes a shared map per-MB; force serial when active.
	if mbCount > parallelThreshold && runtime.GOMAXPROCS(0) > 1 && debugDumpI16Capture == nil {
		vp8Data = encodeFrameParallel(yuv, baseQ, arena)
	} else {
		vp8Data = encodeFrame(yuv, baseQ, arena)
	}

	alphData, err := encodeAlphaChunk(img, w, h)
	if err != nil {
		return err
	}

	return writeWebPExtended(out, vp8Data, alphData, w, h)
}
