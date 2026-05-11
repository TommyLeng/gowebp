// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

// Package gowebp implements a pure-Go WebP encoder supporting both
// lossy (VP8) and lossless (VP8L) output.
package gowebp

import (
	"image"
	"io"

	"github.com/TommyLeng/gowebp/lossless"
)

// Options controls encoding behaviour.
type Options struct {
	Lossless bool // true = VP8L lossless, false = VP8 lossy (default)
	Quality  int  // 0–100, only used when Lossless=false (default: 90)
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

	internalQuality := quality

	// Convert to YUV 4:2:0
	yuv := rgbaToYUV420(img)

	// Compute base quantizer index; per-MB quantizers are determined inside encodeFrame
	// via the SNS two-segment scheme (computeSNSSegmentQualities).
	baseQ := qualityToLevel(internalQuality)

	// Encode the VP8 frame — use wave-front parallel encoding for large images.
	mbCount := (yuv.mbW / 16) * (yuv.mbH / 16)
	var vp8Data []byte
	if mbCount > parallelThreshold {
		vp8Data = encodeFrameParallel(yuv, baseQ)
	} else {
		vp8Data = encodeFrame(yuv, baseQ)
	}

	// Write WebP container
	return writeWebPHeader(w, vp8Data)
}
