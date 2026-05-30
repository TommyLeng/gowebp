package gowebp

import (
	"bytes"
	"image"
	"testing"

	xwebp "golang.org/x/image/webp"
)

// makeRegionalImage builds an image with strong regional statistical variation:
// the top band is a smooth vertical gradient (low local entropy) and the bottom
// band is a deterministic high-frequency pattern (high local entropy). A single
// global Huffman code is a poor fit for both halves, so this is exactly the case
// meta-Huffman (per-region codes) should win on — while still round-tripping
// bit-exact because lossless.
func makeRegionalImage(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*img.Stride + x*4
			var r, g, b uint8
			if y < h/2 {
				v := uint8((y * 255) / h)
				r, g, b = v, v, v
			} else {
				// pseudo-random but deterministic high-frequency content
				n := (x*1103515245 + y*12345 + x*y*7) & 0xff
				r = uint8(n)
				g = uint8((n * 3) & 0xff)
				b = uint8((n * 7) & 0xff)
			}
			img.Pix[i+0] = r
			img.Pix[i+1] = g
			img.Pix[i+2] = b
			img.Pix[i+3] = 255
		}
	}
	return img
}

// TestLosslessRoundTrip verifies that gowebp's lossless encoder produces output
// that decodes bit-exact via the reference golang.org/x/image/webp decoder.
// This is the safety net for the meta-Huffman work.
func TestLosslessRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		img  *image.NRGBA
	}{
		{"regional_256x256", makeRegionalImage(256, 256)},
		{"regional_300x200", makeRegionalImage(300, 200)},
		{"regional_64x64", makeRegionalImage(64, 64)},
	}
	for _, tc := range cases {
		var buf bytes.Buffer
		if err := Encode(&buf, tc.img, &Options{Lossless: true}); err != nil {
			t.Errorf("%s: encode: %v", tc.name, err)
			continue
		}
		dec, err := xwebp.Decode(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Errorf("%s: decode: %v (size=%d)", tc.name, err, buf.Len())
			continue
		}
		mism := comparePixels(tc.img, dec)
		t.Logf("%s: %d bytes, %d pixel mismatches", tc.name, buf.Len(), mism)
		if mism != 0 {
			t.Errorf("%s: lossless round-trip not bit-exact (%d mismatches)", tc.name, mism)
		}
	}
}

func comparePixels(want *image.NRGBA, got image.Image) int {
	b := want.Bounds()
	mism := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			wr, wg, wb, wa := want.At(x, y).RGBA()
			gr, gg, gb, ga := got.At(x, y).RGBA()
			if wr != gr || wg != gg || wb != gb || wa != ga {
				mism++
			}
		}
	}
	return mism
}
