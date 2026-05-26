// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"math/rand"
	"testing"
)

// TestFTransform2PlaneMatchesScalar verifies that fTransform2Plane produces
// bit-identical output to calling fTransformScalar twice (once per block).
func TestFTransform2PlaneMatchesScalar(t *testing.T) {
	// Build a fake srcPlane (256×256 uint8) and pred buffer.
	const W, H = 256, 256
	plane := make([]byte, W*H)
	rng := rand.New(rand.NewSource(1234))
	for i := range plane {
		plane[i] = byte(rng.Intn(256))
	}

	for trial := 0; trial < 500; trial++ {
		srcX := rng.Intn(W - 10) // ensure srcX+7 is in bounds
		srcY := rng.Intn(H - 5)  // ensure srcY+3 is in bounds
		predStride := 16
		var pred [4 * 16]int16 // 4 rows × 16 int16 (wide enough)
		for i := range pred {
			pred[i] = int16(rng.Intn(256))
		}

		// Reference: extract src and pred into flat [16]int16 per block, call fTransform.
		var src0, src1, p0, p1 [16]int16
		for r := 0; r < 4; r++ {
			row := plane[(srcY+r)*W:]
			pr := pred[r*predStride:]
			for c := 0; c < 4; c++ {
				src0[r*4+c] = int16(row[srcX+c])
				p0[r*4+c] = pr[c]
				src1[r*4+c] = int16(row[srcX+4+c])
				p1[r*4+c] = pr[4+c]
			}
		}
		var want [32]int16
		fTransform(src0[:], p0[:], want[:16])
		fTransform(src1[:], p1[:], want[16:])

		// SIMD path under test
		var got [32]int16
		fTransform2Plane(plane, W, srcX, srcY, pred[:], predStride, got[:])

		for i := 0; i < 32; i++ {
			if got[i] != want[i] {
				t.Errorf("trial %d (srcX=%d srcY=%d): out[%d] got=%d want=%d",
					trial, srcX, srcY, i, got[i], want[i])
				break
			}
		}
	}
}

func BenchmarkFTransform2Plane(b *testing.B) {
	const W, H = 256, 256
	plane := make([]byte, W*H)
	for i := range plane {
		plane[i] = byte(i & 0xFF)
	}
	var pred [4 * 16]int16
	for i := range pred {
		pred[i] = int16(i & 0xFF)
	}
	var out [32]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fTransform2Plane(plane, W, 4, 4, pred[:], 16, out[:])
	}
}

// TestFTransform2PlaneSimple tests with a single known-value pixel block.
func TestFTransform2PlaneSimple(t *testing.T) {
	// Fill a simple plane: all zeros for block0, constant 128 for block1
	const W = 64
	plane := make([]byte, W*32)
	// Row 0: block0=[1,2,3,4], block1=[10,20,30,40]
	// Row 1: block0=[5,6,7,8], block1=[50,60,70,80]
	// Row 2: block0=[9,10,11,12], block1=[100,110,120,130]
	// Row 3: block0=[13,14,15,16], block1=[140,150,160,170]
	vals := [4][8]byte{
		{1, 2, 3, 4, 10, 20, 30, 40},
		{5, 6, 7, 8, 50, 60, 70, 80},
		{9, 10, 11, 12, 100, 110, 120, 130},
		{13, 14, 15, 16, 140, 150, 160, 170},
	}
	for r := 0; r < 4; r++ {
		for c := 0; c < 8; c++ {
			plane[r*W+c] = vals[r][c]
		}
	}

	var pred [4 * 8]int16 // all zeros

	// Reference
	var src0, src1, p0, p1 [16]int16
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			src0[r*4+c] = int16(vals[r][c])
			p0[r*4+c] = 0
			src1[r*4+c] = int16(vals[r][4+c])
			p1[r*4+c] = 0
		}
	}
	var want [32]int16
	fTransform(src0[:], p0[:], want[:16])
	fTransform(src1[:], p1[:], want[16:])

	var got [32]int16
	fTransform2Plane(plane, W, 0, 0, pred[:], 8, got[:])

	t.Logf("Block0 reference: %v", want[:16])
	t.Logf("Block0 got:       %v", got[:16])
	t.Logf("Block1 reference: %v", want[16:])
	t.Logf("Block1 got:       %v", got[16:])

	for i := 0; i < 32; i++ {
		if got[i] != want[i] {
			t.Errorf("out[%d]: got=%d want=%d", i, got[i], want[i])
		}
	}
}

// TestFTransform2PlaneAllSame tests with identical pixels in both blocks.
func TestFTransform2PlaneAllSame(t *testing.T) {
	// Both blocks have same pixel pattern - block1 result should equal block0 result
	const W = 64
	plane := make([]byte, W*32)
	// Use same pixels for both blocks
	vals := [4]byte{100, 120, 140, 160}
	for r := 0; r < 4; r++ {
		for c := 0; c < 8; c++ {
			plane[r*W+c] = vals[c%4]
		}
	}
	var pred [4 * 8]int16 // all zeros
	var got [32]int16
	fTransform2Plane(plane, W, 0, 0, pred[:], 8, got[:])
	// Block0 and Block1 should produce identical results
	for i := 0; i < 16; i++ {
		if got[i] != got[16+i] {
			t.Errorf("out[%d]=%d != out[%d]=%d (same pixels, should match)", i, got[i], 16+i, got[16+i])
		}
	}
}

// TestFTransform2PlaneDebug prints intermediate values for a specific trial.
func TestFTransform2PlaneDebug(t *testing.T) {
	const W = 256
	plane := make([]byte, W*W)
	rng := rand.New(rand.NewSource(1234))
	for i := range plane { plane[i] = byte(rng.Intn(256)) }

	// Trial 0: srcX=97 srcY=142
	srcX, srcY := 97, 142
	predStride := 16
	var pred [4 * 16]int16
	rng2 := rand.New(rand.NewSource(1234 + 0))
	// Skip 97+142*256 bytes first from plane generation, then generate pred
	// Actually let me just use zero pred to simplify
	for i := range pred { pred[i] = 0 }

	// Compute reference with zero pred
	var src0, src1, p0, p1 [16]int16
	for r := 0; r < 4; r++ {
		row := plane[(srcY+r)*W:]
		for c := 0; c < 4; c++ {
			src0[r*4+c] = int16(row[srcX+c])
			p0[r*4+c] = 0
			src1[r*4+c] = int16(row[srcX+4+c])
			p1[r*4+c] = 0
		}
	}
	var want [32]int16
	fTransform(src0[:], p0[:], want[:16])
	fTransform(src1[:], p1[:], want[16:])

	var got [32]int16
	fTransform2Plane(plane, W, srcX, srcY, pred[:], predStride, got[:])

	t.Logf("Block0 reference: %v", want[:16])
	t.Logf("Block0 got:       %v", got[:16])
	t.Logf("Block1 reference: %v", want[16:])
	t.Logf("Block1 got:       %v", got[16:])
	t.Logf("")
	// Also print the pixel values
	for r := 0; r < 4; r++ {
		row := plane[(srcY+r)*W:]
		t.Logf("row%d: b0=[%d %d %d %d] b1=[%d %d %d %d]", r,
			row[srcX], row[srcX+1], row[srcX+2], row[srcX+3],
			row[srcX+4], row[srcX+5], row[srcX+6], row[srcX+7])
	}
	// Manually compute horizontal pass for block1
	for r := 0; r < 4; r++ {
		row := plane[(srcY+r)*W:]
		d0 := int32(row[srcX+4])
		d1 := int32(row[srcX+5])
		d2 := int32(row[srcX+6])
		d3 := int32(row[srcX+7])
		a0 := d0 + d3; a1 := d1 + d2; a2 := d1 - d2; a3 := d0 - d3
		t1 := (a0+a1)*8; t3 := (a0-a1)*8
		rot1 := (a2*2217 + a3*5352 + 1812) >> 9
		rot3 := (a3*2217 - a2*5352 + 937) >> 9
		t.Logf("b1 row%d: a0=%d a1=%d a2=%d a3=%d → t0=%d t1=%d t2=%d t3=%d", r, a0,a1,a2,a3,t1,rot1,t3,rot3)
	}
	_ = rng2
}

// BenchmarkFTransform2PlaneVsManual compares fTransform2Plane against the
// old manual approach: extract src/pred into flat arrays, then call fTransform twice.
func BenchmarkFTransform2PlaneVsManual(b *testing.B) {
	const W = 256
	plane := make([]byte, W*W)
	for i := range plane { plane[i] = byte(i & 0xFF) }
	var pred [4 * 16]int16
	for i := range pred { pred[i] = int16(i & 0xFF) }
	var out [32]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fTransform2Plane(plane, W, 4, 4, pred[:], 16, out[:])
	}
}

func BenchmarkFTransform2PlaneManualPath(b *testing.B) {
	const W = 256
	plane := make([]byte, W*W)
	for i := range plane { plane[i] = byte(i & 0xFF) }
	var pred [4 * 16]int16
	for i := range pred { pred[i] = int16(i & 0xFF) }
	var src0, src1, p0, p1, out0, out1 [16]int16
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srcX, srcY := 4, 4
		for r := 0; r < 4; r++ {
			row := plane[(srcY+r)*W:]
			pr := pred[r*16:]
			for c := 0; c < 4; c++ {
				src0[r*4+c] = int16(row[srcX+c])
				p0[r*4+c] = pr[c]
				src1[r*4+c] = int16(row[srcX+4+c])
				p1[r*4+c] = pr[4+c]
			}
		}
		fTransform(src0[:], p0[:], out0[:])
		fTransform(src1[:], p1[:], out1[:])
	}
}
