//go:build amd64

package gowebp

import (
	"image"
	"testing"
)

// scalarY computes the Y value for one pixel using the scalar formula from
// rgbaToYUV420, so we can compare against the SSE2 result.
func scalarY(r, g, b int) uint8 {
	const yuvFix = 16
	const yuvHalf = 1 << (yuvFix - 1)
	luma := 16839*r + 33059*g + 6420*b
	y := (luma + yuvHalf + (16 << yuvFix)) >> yuvFix
	if y > 235 {
		y = 235
	}
	if y < 16 {
		y = 16
	}
	return uint8(y)
}

// TestYuvYRowNRGBASSE2_ExhaustiveMath verifies the PMADDWD coefficient split:
// (16839·R + 32767·G) + (292·G + 6420·B) == 16839·R + 33059·G + 6420·B
// for all (R,G,B) ∈ {0,255}³ (corner cases) and a representative interior.
func TestYuvYRowNRGBASSE2_ExhaustiveMath(t *testing.T) {
	// Corner and edge cases.
	vals := []int{0, 1, 127, 128, 254, 255}
	for _, r := range vals {
		for _, g := range vals {
			for _, b := range vals {
				// Build a 4-pixel slice with this (r,g,b,255) value.
				pix := make([]uint8, 16)
				for i := 0; i < 4; i++ {
					pix[i*4+0] = uint8(r)
					pix[i*4+1] = uint8(g)
					pix[i*4+2] = uint8(b)
					pix[i*4+3] = 255
				}
				dst := make([]uint8, 4)
				yuvYRowNRGBASSE2(pix, 0, 4, dst, 0)
				want := scalarY(r, g, b)
				for i := 0; i < 4; i++ {
					if dst[i] != want {
						t.Fatalf("R=%d G=%d B=%d pixel %d: want %d got %d",
							r, g, b, i, want, dst[i])
					}
				}
			}
		}
	}
}

// TestYuvYRowNRGBASSE2_BatchVsScalar verifies that yuvYRowNRGBASSE2 is
// byte-identical to the scalar formula for a pseudo-random batch of n4 pixels.
func TestYuvYRowNRGBASSE2_BatchVsScalar(t *testing.T) {
	const n = 256 // test 256 pixels = 64 SSE2 batches
	pix := make([]uint8, n*4)
	v := uint8(7)
	for i := range pix {
		pix[i] = v
		v = v*31 + 13 // cheap LCG, covers [0,255] well
	}

	wantY := make([]uint8, n)
	for i := 0; i < n; i++ {
		wantY[i] = scalarY(int(pix[i*4]), int(pix[i*4+1]), int(pix[i*4+2]))
	}

	gotY := make([]uint8, n)
	yuvYRowNRGBASSE2(pix, 0, n, gotY, 0)

	for i := 0; i < n; i++ {
		if gotY[i] != wantY[i] {
			t.Fatalf("pixel %d (R=%d G=%d B=%d): want %d got %d",
				i, pix[i*4], pix[i*4+1], pix[i*4+2], wantY[i], gotY[i])
		}
	}
}

// TestYuvYRowNRGBASSE2_SrcOffDstOff verifies that the srcOff and dstOff
// parameters correctly offset into the pix and dst slices.
func TestYuvYRowNRGBASSE2_SrcOffDstOff(t *testing.T) {
	const n = 8
	pix := make([]uint8, (n+2)*4) // 2 pixels of padding at the front
	for i := range pix {
		pix[i] = uint8(i * 17)
	}
	srcOff := 2 * 4 // skip first 2 pixels

	wantY := make([]uint8, n)
	for i := 0; i < n; i++ {
		r := int(pix[srcOff+i*4])
		g := int(pix[srcOff+i*4+1])
		b := int(pix[srcOff+i*4+2])
		wantY[i] = scalarY(r, g, b)
	}

	dst := make([]uint8, n+3) // 3 bytes of padding at the front
	dstOff := 3
	yuvYRowNRGBASSE2(pix, srcOff, n, dst, dstOff)

	for i := 0; i < n; i++ {
		if dst[dstOff+i] != wantY[i] {
			t.Fatalf("pixel %d: want %d got %d", i, wantY[i], dst[dstOff+i])
		}
	}
}

// TestRgbaToYUV420_NRGBAFastPath verifies that the full rgbaToYUV420 path for
// *image.NRGBA produces bit-identical Y values to the scalar formula.
func TestRgbaToYUV420_NRGBAFastPath(t *testing.T) {
	const w, h = 13, 7 // non-multiples of 4 to exercise tail handling
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	v := uint8(3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i+0] = v
			img.Pix[i+1] = v + 50
			img.Pix[i+2] = v + 100
			img.Pix[i+3] = 255
			v += 17
		}
	}

	arena := &frameArena{}
	yuv := rgbaToYUV420(img, arena)

	mbW := (w + 15) &^ 15
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			want := scalarY(int(img.Pix[i]), int(img.Pix[i+1]), int(img.Pix[i+2]))
			got := yuv.y[y*mbW+x]
			if got != want {
				t.Fatalf("pixel (%d,%d): want Y=%d got Y=%d", x, y, want, got)
			}
		}
	}
}
