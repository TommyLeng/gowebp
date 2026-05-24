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
