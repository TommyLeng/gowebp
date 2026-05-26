// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// Dual-block transform variants: fTransform2 / iTransform4x4x2.
//
// These wrappers mirror libwebp's FTransform2 and ITransform (do_two=1) API:
// both process TWO adjacent 4x4 blocks per call, with the second block's
// data concatenated after the first (block A in [0..15], block B in [16..31]).
//
// They are kept as thin Go forwarders because, on the platforms gowebp ships
// SIMD for (arm64 NEON and amd64 SSE2), the single-block fTransform /
// iTransform4x4 are already fully vectorised across all 4 columns/rows
// (lane = column). There is no spare SIMD bandwidth to gain by widening
// the inner kernel to int16x8:
//
//   - arm64 NEON has 128-bit vectors; the current fTransform already uses
//     int32x4 (one lane per column) and saturates the issue width.
//   - amd64 SSE2 fTransform uses PMADDWD with int16-input rotation constants
//     and produces 4 int32 results per pass, again saturating XMM lanes.
//   - libwebp's enc_neon.c follows the same pattern (FTransform / ITransform
//     wrap two single-block kernels back-to-back).
//
// Only libwebp's SSE2 path (FTransform2_SSE2 / ITransform_Two_SSE2) achieves
// real 2x throughput by packing the two blocks' rows into a single int16x8
// vector and operating in int16 throughout (using PMULHW with the K-65536
// trick to stay in 16-bit arithmetic). That requires the input to arrive
// as side-by-side 8-pixel-wide rows (uint8 buffer with BPS=32 stride), not
// the [block A | block B] layout these wrappers expect. Implementing it would
// require both:
//   (a) a different input layout (caller hands over 8 pixels per row),
//   (b) a hand-rolled SSE2 kernel that does not exist on arm64.
//
// Microbenchmarks of the wrapper-vs-two-single-calls show the wrapper is
// neutral or marginally slower (Go ABI overhead is non-zero), so we do NOT
// route the encoder loops through these wrappers — the call sites continue
// to issue single-block transforms directly. The dual API is provided so
// that future contributors who add a true SSE2 fused kernel can route the
// i16/UV loops through it without changing the function signatures.

// fTransform2 computes the 4x4 forward DCT for TWO adjacent 4x4 blocks.
// Layout (each slice is exactly 32 int16):
//   src[0..15]  / ref[0..15]  / out[0..15]  = block A
//   src[16..31] / ref[16..31] / out[16..31] = block B
// Mirrors libwebp's FTransform2_C (src/dsp/enc.c:193).
func fTransform2(src []int16, ref []int16, out []int16) {
	fTransform(src[0:16], ref[0:16], out[0:16])
	fTransform(src[16:32], ref[16:32], out[16:32])
}

// iTransform4x4x2 computes the inverse 4x4 DCT for TWO adjacent 4x4 blocks.
// Layout matches fTransform2.
// Mirrors libwebp's ITransform_C with do_two=1 (src/dsp/enc.c:152).
func iTransform4x4x2(coeffs []int16, pred []int16, out []int16) {
	iTransform4x4(coeffs[0:16], pred[0:16], out[0:16])
	iTransform4x4(coeffs[16:32], pred[16:32], out[16:32])
}
