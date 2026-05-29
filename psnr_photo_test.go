package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestPSNRPhotos encodes a few real photos and reports gowebp PSNR vs libwebp.
func TestPSNRPhotos(t *testing.T) {
	dwebp, err := exec.LookPath("dwebp")
	if err != nil {
		t.Skip("dwebp not installed")
	}
	cwebp, _ := exec.LookPath("cwebp")

	images := []string{
		"test_data/original/i1-a.png",
		"test_data/original/jable-heidilau0905-003.jpg",
	}

	for _, src := range images {
		f, err := os.Open(src)
		if err != nil {
			t.Logf("skip %s: %v", src, err)
			continue
		}
		var orig image.Image
		if strings.HasSuffix(src, ".png") {
			orig, err = png.Decode(f)
		} else {
			orig, err = jpeg.Decode(f)
		}
		f.Close()
		if err != nil {
			t.Logf("decode %s: %v", src, err)
			continue
		}

		// resize? no, original
		var snap []mbInfo
		debugMBStats = &snap

		var buf bytes.Buffer
		if err := Encode(&buf, orig, &Options{Quality: 75}); err != nil {
			t.Errorf("encode: %v", err)
			continue
		}
		debugMBStats = nil

		var i4, i16 int
		for _, mi := range snap {
			if mi.isI4 {
				i4++
			} else {
				i16++
			}
		}

		tmpWeb := "/tmp/photo.webp"
		tmpPng := "/tmp/photo.png"
		os.WriteFile(tmpWeb, buf.Bytes(), 0644)
		if out, err := exec.Command(dwebp, tmpWeb, "-o", tmpPng).CombinedOutput(); err != nil {
			t.Errorf("dwebp %s: %v %s", src, err, out)
			continue
		}
		decoded, err := loadPNG(tmpPng)
		if err != nil {
			t.Errorf("loadPNG: %v", err)
			continue
		}
		goPSNR := psnrRGBA(orig, decoded)
		goSize := buf.Len()

		// cwebp baseline
		libPSNR := -1.0
		libSize := 0
		if cwebp != "" {
			tmpSrcPng := "/tmp/photo_src.png"
			f, _ := os.Create(tmpSrcPng)
			png.Encode(f, orig)
			f.Close()
			tmpLib := "/tmp/photo_lib.webp"
			exec.Command(cwebp, "-q", "75", tmpSrcPng, "-o", tmpLib).Run()
			fi, _ := os.Stat(tmpLib)
			libSize = int(fi.Size())
			tmpLibPng := "/tmp/photo_lib.png"
			exec.Command(dwebp, tmpLib, "-o", tmpLibPng).Run()
			libDecoded, _ := loadPNG(tmpLibPng)
			if libDecoded != nil {
				libPSNR = psnrRGBA(orig, libDecoded)
			}
		}

		t.Logf("%s  i4=%d (%.0f%%) i16=%d (%.0f%%)  gowebp %.2f dB (%d B)  libwebp %.2f dB (%d B)",
			src,
			i4, 100*float64(i4)/float64(i4+i16),
			i16, 100*float64(i16)/float64(i4+i16),
			goPSNR, goSize, libPSNR, libSize)
		fmt.Println()
	}
}
