package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

func TestCompareWithCwebp(t *testing.T) {
	originalDir := "test_data/original"
	libwebpLossyDir := "test_data/libwebp/lossy"
	libwebpLossyM6Dir := "test_data/libwebp/lossy_m6"
	libwebpLosslessDir := "test_data/libwebp/lossless"
	gowebpLossyDir := "test_data/gowebp/lossy"
	gowebpLosslessDir := "test_data/gowebp/lossless"

	entries, err := os.ReadDir(originalDir)
	if err != nil {
		t.Skipf("original/ not found or empty: %v", err)
	}

	type imgEntry struct {
		relPath string
	}
	var images []imgEntry

	scanDir := func(subdir string) {
		dir := originalDir
		if subdir != "" {
			dir = filepath.Join(originalDir, subdir)
		}
		es, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		for _, e := range es {
			if e.IsDir() {
				continue
			}
			ext := strings.ToLower(filepath.Ext(e.Name()))
			if ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp" {
				rel := e.Name()
				if subdir != "" {
					rel = subdir + "/" + e.Name()
				}
				images = append(images, imgEntry{rel})
			}
		}
	}

	scanDir("")
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			scanDir(e.Name())
		}
	}

	if len(images) == 0 {
		t.Skip("no images in test_data/original/")
	}

	_, cwebpErr := exec.LookPath("cwebp")
	hasCwebp := cwebpErr == nil

	type row struct {
		name              string
		srcKB             float64
		libLossyKB        string
		libLossyMs        string
		libM6LossyKB      string
		libM6LossyMs      string
		libLosslessKB     string
		libLosslessMs     string
		goLossyKB         float64
		goLossyMs         float64
		goLossyPSNR       string
		goLosslessKB      float64
		goLosslessMs      float64
	}
	var rows []row

	for _, name := range images {
		rel := name.relPath
		srcPath := filepath.Join(originalDir, rel)
		stem := strings.TrimSuffix(rel, filepath.Ext(rel))

		// ensure output dirs exist
		for _, d := range []string{libwebpLossyDir, libwebpLossyM6Dir, libwebpLosslessDir, gowebpLossyDir, gowebpLosslessDir} {
			os.MkdirAll(filepath.Join(d, filepath.Dir(stem)), 0755)
		}

		libLossyPath := filepath.Join(libwebpLossyDir, stem+".webp")
		libM6LossyPath := filepath.Join(libwebpLossyM6Dir, stem+".webp")
		libLosslessPath := filepath.Join(libwebpLosslessDir, stem+".webp")
		goLossyPath := filepath.Join(gowebpLossyDir, stem+".webp")
		goLosslessPath := filepath.Join(gowebpLosslessDir, stem+".webp")

		f, err := os.Open(srcPath)
		if err != nil {
			t.Errorf("open %s: %v", rel, err)
			continue
		}
		src, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Errorf("decode %s: %v", rel, err)
			continue
		}

		isHidden := strings.HasPrefix(rel, "hidden/")
		var dst *image.NRGBA
		if isHidden {
			dst = image.NewNRGBA(image.Rect(0, 0, 300, 300))
			xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Src, nil)
		} else {
			dst = image.NewNRGBA(src.Bounds())
			draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)
		}

		srcInfo, _ := os.Stat(srcPath)
		srcKB := float64(srcInfo.Size()) / 1024

		// --- cwebp lossy ---
		libLossyKBStr, libLossyMsStr := "N/A", "N/A"
		if hasCwebp {
			args := []string{"-q", "90", "-m", "4"}
			if isHidden {
				args = append(args, "-resize", "300", "300")
			}
			args = append(args, srcPath, "-o", libLossyPath)
			cmd := exec.Command("cwebp", args...)
			cmd.Stderr = nil
			t0 := time.Now()
			if cmd.Run() == nil {
				elapsed := time.Since(t0)
				if fi, fiErr := os.Stat(libLossyPath); fiErr == nil {
					libLossyKBStr = fmt.Sprintf("%.1f kb", float64(fi.Size())/1024)
					libLossyMsStr = fmt.Sprintf("%.0f ms", float64(elapsed.Milliseconds()))
				}
			}
		}

		// --- cwebp lossy -m 6 ---
		libM6LossyKBStr, libM6LossyMsStr := "N/A", "N/A"
		if hasCwebp {
			args := []string{"-q", "90", "-m", "6"}
			if isHidden {
				args = append(args, "-resize", "300", "300")
			}
			args = append(args, srcPath, "-o", libM6LossyPath)
			cmd := exec.Command("cwebp", args...)
			cmd.Stderr = nil
			t0 := time.Now()
			if cmd.Run() == nil {
				elapsed := time.Since(t0)
				if fi, fiErr := os.Stat(libM6LossyPath); fiErr == nil {
					libM6LossyKBStr = fmt.Sprintf("%.1f kb", float64(fi.Size())/1024)
					libM6LossyMsStr = fmt.Sprintf("%.0f ms", float64(elapsed.Milliseconds()))
				}
			}
		}

		// --- cwebp lossless ---
		libLosslessKBStr, libLosslessMsStr := "N/A", "N/A"
		if hasCwebp {
			args := []string{"-lossless", "-q", "90", "-m", "4"}
			if isHidden {
				args = append(args, "-resize", "300", "300")
			}
			args = append(args, srcPath, "-o", libLosslessPath)
			cmd := exec.Command("cwebp", args...)
			cmd.Stderr = nil
			t0 := time.Now()
			if cmd.Run() == nil {
				elapsed := time.Since(t0)
				if fi, fiErr := os.Stat(libLosslessPath); fiErr == nil {
					libLosslessKBStr = fmt.Sprintf("%.1f kb", float64(fi.Size())/1024)
					libLosslessMsStr = fmt.Sprintf("%.0f ms", float64(elapsed.Milliseconds()))
				}
			}
		}

		// --- gowebp lossy ---
		var bufLossy bytes.Buffer
		t0 := time.Now()
		if encErr := Encode(&bufLossy, dst, &Options{Quality: 90}); encErr != nil {
			t.Errorf("lossy encode %s: %v", rel, encErr)
			continue
		}
		goLossyMs := float64(time.Since(t0).Milliseconds())
		os.WriteFile(goLossyPath, bufLossy.Bytes(), 0644)
		goLossyKB := float64(bufLossy.Len()) / 1024

		goLossyPSNR := "err"
		if decoded, decErr := webp.Decode(bytes.NewReader(bufLossy.Bytes())); decErr == nil {
			goLossyPSNR = fmt.Sprintf("%.1f dB", psnrRGBA(dst, decoded))
		}

		// --- gowebp lossless ---
		var bufLossless bytes.Buffer
		t0 = time.Now()
		if encErr := Encode(&bufLossless, dst, &Options{Lossless: true}); encErr != nil {
			t.Errorf("lossless encode %s: %v", rel, encErr)
			continue
		}
		goLosslessMs := float64(time.Since(t0).Milliseconds())
		os.WriteFile(goLosslessPath, bufLossless.Bytes(), 0644)
		goLosslessKB := float64(bufLossless.Len()) / 1024

		rows = append(rows, row{
			rel, srcKB,
			libLossyKBStr, libLossyMsStr,
			libM6LossyKBStr, libM6LossyMsStr,
			libLosslessKBStr, libLosslessMsStr,
			goLossyKB, goLossyMs, goLossyPSNR,
			goLosslessKB, goLosslessMs,
		})
	}

	// print to console
	fmt.Printf("\n%-40s %9s | %11s %7s | %11s %7s | %12s %7s | %9s %7s %10s | %11s %7s\n",
		"File", "Original",
		"lib -m4", "time",
		"lib -m6", "time",
		"lib lossless", "time",
		"go lossy", "time", "PSNR",
		"go lossless", "time")
	fmt.Println(strings.Repeat("-", 155))
	for _, r := range rows {
		fmt.Printf("%-40s %8.1fkb | %11s %7s | %11s %7s | %12s %7s | %8.1fkb %7.0fms %10s | %10.1fkb %7.0fms\n",
			r.name, r.srcKB,
			r.libLossyKB, r.libLossyMs,
			r.libM6LossyKB, r.libM6LossyMs,
			r.libLosslessKB, r.libLosslessMs,
			r.goLossyKB, r.goLossyMs, r.goLossyPSNR,
			r.goLosslessKB, r.goLosslessMs)
	}
	fmt.Println()

	// write markdown
	var md strings.Builder
	md.WriteString("# WebP Conversion Comparison\n\n")
	md.WriteString("Parameters: quality=90. `hidden/` images resized to 300×300.\n\n")
	md.WriteString("| File | Original | cwebp -m4 | time | cwebp -m6 | time | lib lossless | lib lossless time | go lossy | go lossy time | PSNR (go) | go lossless | go lossless time |\n")
	md.WriteString("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rows {
		md.WriteString(fmt.Sprintf("| %s | %.1f kb | %s | %s | %s | %s | %s | %s | %.1f kb | %.0f ms | %s | %.1f kb | %.0f ms |\n",
			r.name, r.srcKB,
			r.libLossyKB, r.libLossyMs,
			r.libM6LossyKB, r.libM6LossyMs,
			r.libLosslessKB, r.libLosslessMs,
			r.goLossyKB, r.goLossyMs, r.goLossyPSNR,
			r.goLosslessKB, r.goLosslessMs))
	}
	mdPath := "test_data/compare_results.md"
	os.WriteFile(mdPath, []byte(md.String()), 0644)
	t.Logf("results saved to %s", mdPath)
}

func psnrRGBA(a, b image.Image) float64 {
	bounds := a.Bounds()
	var mse float64
	n := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r1, g1, b1, _ := a.At(x, y).RGBA()
			r2, g2, b2, _ := b.At(x, y).RGBA()
			dr := float64(r1>>8) - float64(r2>>8)
			dg := float64(g1>>8) - float64(g2>>8)
			db := float64(b1>>8) - float64(b2>>8)
			mse += dr*dr + dg*dg + db*db
			n++
		}
	}
	if n == 0 {
		return 0
	}
	mse /= float64(n * 3)
	if mse == 0 {
		return 100
	}
	return 10 * math.Log10(255*255/mse)
}
