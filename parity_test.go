package gowebp

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"runtime"
	"testing"
)

func TestP1vsP10Parity(t *testing.T) {
	for _, path := range perfImages {
		f, err := os.Open(path)
		if err != nil {
			t.Logf("skip %s: %v", path, err)
			continue
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Fatalf("%s: decode: %v", path, err)
		}

		// Encode at P=1
		runtime.GOMAXPROCS(1)
		var buf1 bytes.Buffer
		if err := Encode(&buf1, img, &Options{Quality: 90}); err != nil {
			t.Fatalf("%s: encode P=1: %v", path, err)
		}

		// Encode at P=10
		runtime.GOMAXPROCS(10)
		var buf10 bytes.Buffer
		if err := Encode(&buf10, img, &Options{Quality: 90}); err != nil {
			t.Fatalf("%s: encode P=10: %v", path, err)
		}

		// Restore
		runtime.GOMAXPROCS(runtime.NumCPU())

		size1 := buf1.Len()
		size10 := buf10.Len()
		match := bytes.Equal(buf1.Bytes(), buf10.Bytes())

		status := "MATCH"
		if !match {
			status = fmt.Sprintf("MISMATCH  P=1=%d bytes  P=10=%d bytes  diff=%+d", size1, size10, size10-size1)
		}
		t.Logf("%-52s  P=1=%6d B  P=10=%6d B  %s", path, size1, size10, status)

		if !match {
			t.Errorf("%s: P=1 and P=10 outputs differ", path)
		}
	}
}
