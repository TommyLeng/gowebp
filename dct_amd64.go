// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// fTransform computes the 4x4 forward DCT of (src - ref), storing into out[16].
// Both passes are fully vectorised with SSE2: the horizontal pass uses PMADDWD
// for rotation terms; the vertical pass processes all 4 columns in parallel
// using PMADDWD with int16 narrowing (PACKSSLW/PACKSSDW + PUNPCKLWD) — no PMULLD
// (SSE4.1) required. The branchless (a3!=0) correction for out[4..7] is
// implemented with PCMPEQD + PANDN + PSRLD.
//
// Implemented in dct_amd64.s. Only XMM registers are used — R14 (goroutine
// pointer) is never touched, so signal-based preemption cannot corrupt g.
//
//go:noescape
func fTransform(src []int16, ref []int16, out []int16)

// fTransform2Plane computes the 4×4 forward DCT for two horizontally-adjacent
// blocks, reading directly from the pixel plane (uint8) and int16 prediction.
//
// srcPlane: the full Y (or U/V) plane as uint8
// srcStride: row stride of srcPlane (in bytes)
// srcX, srcY: top-left pixel coordinates of the FIRST block (second is at srcX+4)
// pred: int16 prediction buffer, row-major, predStride int16 elements per row
// out: [32]int16, out[0..15]=DCT of first block, out[16..31]=DCT of second block
//
// SSE2 implementation in dct_amd64.s. Loads 8 uint8 bytes per row (MOVQ +
// PUNPCKLBW zero-extend), subtracts 8 int16 pred (MOVOU + PSUBW), then uses the
// existing FTransformPass1/Pass2 structure on the two 4-wide halves simultaneously.
//
//go:noescape
func fTransform2Plane(srcPlane []byte, srcStride, srcX, srcY int, pred []int16, predStride int, out []int16)

// fTransformWHT computes the 4x4 Walsh-Hadamard Transform on the 16 DC values.
// in[16] are the DC values from each 4x4 block's DCT output (one per block),
// laid out row-major (in[r*4+c] = DC of block at row r, col c).
// out[16] receives the WHT coefficients in the same layout.
//
// SSE2 implementation in dct_amd64.s. Mirrors FTransformWHT_SSE2 in libwebp.
// Strategy: load each row as int16x4, accumulate via PUNPCK + PADDSW/PSUBSW
// rotations, then narrow with PACKSSDW and PSRAW #1.
//
//go:noescape
func fTransformWHT(in []int16, out []int16)
