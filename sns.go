// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// Spatial Noise Shaping (SNS) — 2-segment implementation.
//
// Two segments are used to redistribute bits between smooth and textured MBs:
//   segment 0: smooth/flat MBs — coarser quantizer (saves bits on flat areas)
//   segment 1: textured/busy MBs — base quantizer (preserves quality on edges)
//
// Segment quantizer indices are signaled in the VP8 partition 0 segment header.
// Per-MB segment IDs are emitted in partition 0 (update_map=1).
//
// Calibrated for snsStrength=50 (cwebp default) at quality=90:
//   smooth  MBs: q = baseQ + 12 (coarser by ~12 q-index steps)
//   textured MBs: q = baseQ     (unchanged)
//
// This gives ~5-6% file size reduction with ~1.5 dB PSNR loss on natural photos.

// snsQDelta is the quantizer-index offset applied to smooth MBs.
// Increasing this value saves more bits but reduces PSNR on flat regions.
const snsQDelta = 12

// computeMBAlpha computes the luma activity score for a 16×16 macroblock.
// Returns a value in [0, 255] where 0 = flat/smooth, 255 = highly textured.
//
// Uses mean-absolute-deviation of luma samples as an activity metric.
// Mirrors the spirit of GetAlpha() in libwebp/src/enc/analysis_enc.c.
func computeMBAlpha(yuv *yuvImage, mbX, mbY int) int {
	px := mbX * 16
	py := mbY * 16

	// Compute mean of the 16×16 luma block.
	var sum int
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			sx := px + x
			sy := py + y
			if sx >= yuv.mbW {
				sx = yuv.mbW - 1
			}
			if sy >= yuv.mbH {
				sy = yuv.mbH - 1
			}
			sum += int(yuv.y[sy*yuv.yStride+sx])
		}
	}
	mean := (sum + 128) / 256 // divide by 16×16=256 pixels

	// Compute mean absolute deviation from the mean.
	var mad int
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			sx := px + x
			sy := py + y
			if sx >= yuv.mbW {
				sx = yuv.mbW - 1
			}
			if sy >= yuv.mbH {
				sy = yuv.mbH - 1
			}
			d := int(yuv.y[sy*yuv.yStride+sx]) - mean
			if d < 0 {
				d = -d
			}
			mad += d
		}
	}
	// Scale MAD (0..~10240 for 256 pixels) to [0, 255].
	// Natural images: per-pixel MAD typically 0..40, so total 0..10240.
	// >>5 = divide by 32 maps 0..8160 → 0..255.
	alpha := mad >> 5
	if alpha > 255 {
		alpha = 255
	}
	return alpha
}

// computeAlphaThreshold returns the mean alpha across all MBs.
// MBs with alpha <= threshold → segment 0 (smooth/coarse).
// MBs with alpha > threshold  → segment 1 (textured/fine).
func computeAlphaThreshold(mbAlpha []int) int {
	if len(mbAlpha) == 0 {
		return 128
	}
	sum := 0
	for _, a := range mbAlpha {
		sum += a
	}
	return sum / len(mbAlpha)
}

// computeSNSSegmentQualities returns the quality levels for SNS segments 0 and 1.
//
// seg0 (smooth): baseQ + snsQDelta → coarser quantizer → fewer bits on flat areas
// seg1 (textured): baseQ           → unchanged quality
//
// Returns (qual0, qual1) as quality levels [0..100] for use with makeSegmentParams.
func computeSNSSegmentQualities(baseQ int, mbAlpha []int) (int, int) {
	_ = mbAlpha // reserved for future adaptive delta tuning

	q0 := clipQ(baseQ+snsQDelta, 0, 127)
	q1 := baseQ

	return qIndexToQuality(q0), qIndexToQuality(q1)
}

// qIndexToQuality finds the quality value [0..100] whose qualityToLevel output
// is closest to the given target q-index.
func qIndexToQuality(targetQ int) int {
	bestQual := 90
	bestDiff := 1000
	for qual := 0; qual <= 100; qual++ {
		q := qualityToLevel(qual)
		d := q - targetQ
		if d < 0 {
			d = -d
		}
		if d < bestDiff {
			bestDiff = d
			bestQual = qual
		}
	}
	return bestQual
}
