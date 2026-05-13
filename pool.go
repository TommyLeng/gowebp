// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import "sync"

// frameArena holds all large slice buffers for one Encode() call.
// It is stored in arenaPool and reused across calls to avoid per-encode
// allocation churn (which was causing 20%+ CPU in runtime.madvise).
type frameArena struct {
	// YUV planes — fully overwritten by rgbaToYUV420, no need to zero.
	yPlane []uint8
	uPlane []uint8
	vPlane []uint8

	// Reconstruction buffers — must be zeroed / filled to 128 before use.
	recon  []uint8
	reconU []uint8
	reconV []uint8

	// MB metadata — must be zeroed before use.
	mbInfos  []mbInfo
	mbCoeffs []mbCoeffData

	// NZ context arrays for the serial encoder.
	topNzY     []int
	topNzU     []int
	topNzV     []int
	topNzDC    []int
	topI4Modes []int

	// NZ context arrays for the parallel encoder.
	topNzYShared     []int
	topNzUShared     []int
	topNzVShared     []int
	topNzDCShared    []int
	topI4ModesShared []int
}

// arenaPool is the global pool of frameArena objects.
var arenaPool = sync.Pool{New: func() any { return &frameArena{} }}

// growSliceU8 returns a []uint8 of length n, reusing s if cap(s) >= n.
func growSliceU8(s []uint8, n int) []uint8 {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]uint8, n)
}

// growSliceMBInfo returns a []mbInfo of length n, reusing s if cap(s) >= n.
func growSliceMBInfo(s []mbInfo, n int) []mbInfo {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]mbInfo, n)
}

// growSliceMBCoeff returns a []mbCoeffData of length n, reusing s if cap(s) >= n.
func growSliceMBCoeff(s []mbCoeffData, n int) []mbCoeffData {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]mbCoeffData, n)
}

// growSliceInt returns a []int of length n, reusing s if cap(s) >= n.
func growSliceInt(s []int, n int) []int {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]int, n)
}
