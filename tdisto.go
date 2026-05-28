// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// This file holds the port of libwebp's Hadamard-distortion machinery used
// for the `tlambda_` SD term in mode RD scoring (PickBestIntra16 /
// PickBestIntra4 in libwebp/src/enc/quant_enc.c).
//
// The functions below are correctness-checked ports but are NOT currently
// wired into the gowebp mode selectors. Enabling them on top of the existing
// trellis-quantized RD path caused a ~20% serial slowdown with no measurable
// quality benefit at quality=90 (the dominant operating point) — file sizes
// and PSNR were within noise on all tested images. They are kept here so a
// future cost-model rework (e.g. dropping trellis in favour of greedy quant
// like libwebp at method=4) can re-enable tlambda without re-porting.

// kWeightY is the per-coefficient weight applied during the Walsh-Hadamard
// distortion measurement. Mirrors libwebp's `kWeightY[16]` in
// src/enc/quant_enc.c:489. Layout: row-major 4×4 (raster).
var kWeightY = [16]uint16{
	38, 32, 20, 9,
	32, 28, 17, 7,
	20, 17, 10, 4,
	9, 7, 4, 2,
}

// ttransform computes the Hadamard transform of a 4×4 block, returning the
// sum of `w[i] * |H(i)|` over the 16 coefficients. The block is read
// row-major from `in` (16 entries). Mirrors `TTransform` in
// libwebp/src/dsp/enc.c:615 — the C version reads uint8 with BPS stride; we
// take int16 inputs so the same routine works for both source (uint8 widened)
// and reconstructed (uint8 widened) blocks.
func ttransform(in []int16, w *[16]uint16) int {
	var tmp [16]int
	// Horizontal pass.
	for i := 0; i < 4; i++ {
		base := i * 4
		a0 := int(in[base+0]) + int(in[base+2])
		a1 := int(in[base+1]) + int(in[base+3])
		a2 := int(in[base+1]) - int(in[base+3])
		a3 := int(in[base+0]) - int(in[base+2])
		tmp[0+i*4] = a0 + a1
		tmp[1+i*4] = a3 + a2
		tmp[2+i*4] = a3 - a2
		tmp[3+i*4] = a0 - a1
	}
	// Vertical pass.
	sum := 0
	for i := 0; i < 4; i++ {
		a0 := tmp[0+i] + tmp[8+i]
		a1 := tmp[4+i] + tmp[12+i]
		a2 := tmp[4+i] - tmp[12+i]
		a3 := tmp[0+i] - tmp[8+i]
		b0 := a0 + a1
		b1 := a3 + a2
		b2 := a3 - a2
		b3 := a0 - a1
		sum += int(w[i+0]) * iabsInt(b0)
		sum += int(w[i+4]) * iabsInt(b1)
		sum += int(w[i+8]) * iabsInt(b2)
		sum += int(w[i+12]) * iabsInt(b3)
	}
	return sum
}

func iabsInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// tDisto4x4 computes the Hadamard-domain weighted distortion between source
// and reconstructed 4×4 blocks. Mirrors libwebp's `Disto4x4_C` in
// src/dsp/enc.c:650: `abs(TTransform(b) - TTransform(a)) >> 5`.
func tDisto4x4(srcU8 []int16, refU8 []int16) int {
	s1 := ttransform(srcU8, &kWeightY)
	s2 := ttransform(refU8, &kWeightY)
	d := s2 - s1
	if d < 0 {
		d = -d
	}
	return d >> 5
}

// tDisto16x16 computes the Hadamard-domain weighted distortion for a 16×16
// block, by summing tDisto4x4 over all 16 sub-blocks. Mirrors libwebp's
// `Disto16x16_C` in src/dsp/enc.c:658.
func tDisto16x16(src []int16, ref []int16) int {
	var subSrc, subRef [16]int16
	d := 0
	for by := 0; by < 4; by++ {
		for bx := 0; bx < 4; bx++ {
			for y := 0; y < 4; y++ {
				for x := 0; x < 4; x++ {
					subSrc[y*4+x] = src[(by*4+y)*16+(bx*4+x)]
					subRef[y*4+x] = ref[(by*4+y)*16+(bx*4+x)]
				}
			}
			d += tDisto4x4(subSrc[:], subRef[:])
		}
	}
	return d
}

// mult8B computes (a * b + 128) >> 8, mirroring MULT_8B from libwebp's
// quant_enc.c. Used to scale the Hadamard distortion by per-segment tlambda.
func mult8B(a, b int) int {
	return (a*b + 128) >> 8
}
