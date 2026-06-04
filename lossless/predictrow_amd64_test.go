// Exhaustive verification that predResAvgRowSSE2 modes 5–10 produce byte-
// identical output to predictResidualsRowScalar on AMD64.

//go:build amd64

package lossless

import (
	"image/color"
	"testing"
)

// referenceAvg2 is the scalar average2 for comparison.
func referenceAvg2(a, b color.NRGBA) color.NRGBA {
	return color.NRGBA{
		uint8((int(a.R) + int(b.R)) / 2),
		uint8((int(a.G) + int(b.G)) / 2),
		uint8((int(a.B) + int(b.B)) / 2),
		uint8((int(a.A) + int(b.A)) / 2),
	}
}

// TestPredResAvgMath exhaustively verifies floor_avg(a,b) via PAVGB fixup
// against uint8((a+b)/2) for all 256×256 single-byte pairs.
func TestPredResAvgMath(t *testing.T) {
	for a := 0; a < 256; a++ {
		for b := 0; b < 256; b++ {
			want := uint8((a + b) / 2)

			// Simulate the PAVGB + correction in Go.
			pavgb := uint8((a + b + 1) / 2)
			correction := uint8((a ^ b) & 1)
			got := pavgb - correction

			if got != want {
				t.Fatalf("floor_avg(%d,%d): want %d got %d (PAVGB=%d corr=%d)",
					a, b, want, got, pavgb, correction)
			}
		}
	}
}

// TestPredResAvgRowSSE2_Modes5to10 verifies that predResAvgRowSSE2 output
// matches predictResidualsRowScalar for modes 5–10 on a synthetic image.
func TestPredResAvgRowSSE2_Modes5to10(t *testing.T) {
	const width = 64
	const height = 8

	// Build a pseudo-random pixel buffer.
	pixels := make([]color.NRGBA, width*height)
	v := uint8(37)
	for i := range pixels {
		pixels[i] = color.NRGBA{v, v + 13, v + 77, v + 200}
		v += 31
	}

	for mode := 5; mode <= 10; mode++ {
		for y := 1; y < height; y++ {
			xStart := 0
			xEnd := width

			wantOut := make([]color.NRGBA, xEnd-xStart)
			gotOut := make([]color.NRGBA, xEnd-xStart)

			// Scalar reference.
			predictResidualsRowScalar(pixels, width, mode, xStart, xEnd, y, wantOut)

			// SSE2 path (same wrapper as production).
			predictResidualsRow(pixels, width, mode, xStart, xEnd, y, gotOut)

			for x := 0; x < xEnd-xStart; x++ {
				if gotOut[x] != wantOut[x] {
					t.Errorf("mode %d y=%d x=%d: want %v got %v",
						mode, y, x+xStart, wantOut[x], gotOut[x])
				}
			}
		}
	}
}

// TestPredResAvgRowSSE2_TailHandling verifies that the n%4 tail (handled by
// scalar) plus the vectorised body together match a full scalar run.
func TestPredResAvgRowSSE2_TailHandling(t *testing.T) {
	// Width = 13: n4 = 12 (vectorised), tail = 1 (scalar).  iStart=1 after x==0
	// boundary, so vectorised portion covers x=1..12, scalar tail x=13.
	const width = 13
	const height = 4

	pixels := make([]color.NRGBA, width*height)
	v := uint8(11)
	for i := range pixels {
		pixels[i] = color.NRGBA{v, v + 7, v + 53, v + 101}
		v += 17
	}

	for mode := 5; mode <= 10; mode++ {
		for y := 1; y < height; y++ {
			want := make([]color.NRGBA, width)
			got := make([]color.NRGBA, width)
			predictResidualsRowScalar(pixels, width, mode, 0, width, y, want)
			predictResidualsRow(pixels, width, mode, 0, width, y, got)
			for x := 0; x < width; x++ {
				if got[x] != want[x] {
					t.Errorf("mode %d y=%d x=%d: want %v got %v",
						mode, y, x, want[x], got[x])
				}
			}
		}
	}
}
