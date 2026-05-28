package gowebp

import (
	"bytes"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
)

// perfImages is the fixed set of 20 images used for performance benchmarking.
// 13 JPEGs (decode as *image.YCbCr) + 7 PNGs (decode as *image.NRGBA/*image.RGBA).
var perfImages = []string{
	"test_data/original/i1-a.png",
	"test_data/original/i11-a.png",
	"test_data/original/i18-a.png",
	"test_data/original/i30.jpeg",
	"test_data/original/ishikawa-06.jpg",
	"test_data/original/j3.png",
	"test_data/original/jable-heidilau0905-003.jpg",
	"test_data/original/jable-heidilau0905-004-a.png",
	"test_data/original/jable-heidilau0905-004.jpg",
	"test_data/original/jable-natabcde-0020.jpg",
	"test_data/original/jable-snexxxxxxx-fantia-july-143.jpg",
	"test_data/original/jable-snexxxxxxx-fantia-sep-067.jpg",
	"test_data/original/jable-twy_jacinta-001.jpg",
	"test_data/original/jable-twy_jacinta-008.jpg",
	"test_data/original/jable-twy_jacinta-022.jpg",
	"test_data/original/jablehk_snexxxxxxx_0029.jpg",
	"test_data/original/jablehk_snexxxxxxx_0040.jpg",
	"test_data/original/jablehk_snexxxxxxx_0055.jpg",
	"test_data/original/jablehk_snexxxxxxx_0081.jpg",
	"test_data/original/kodak/kodim05.png",
}

// BenchmarkPerfSuite encodes 20 images at quality=90.
// Run with -cpu 1,10 to compare serial vs wave-front parallel paths.
func BenchmarkPerfSuite(b *testing.B) {
	for _, path := range perfImages {
		f, err := os.Open(path)
		if err != nil {
			b.Logf("skip %s: %v", path, err)
			continue
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			b.Logf("skip %s: decode: %v", path, err)
			continue
		}
		name := filepath.Base(path)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var buf bytes.Buffer
				if err := Encode(&buf, img, &Options{Quality: 90}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
