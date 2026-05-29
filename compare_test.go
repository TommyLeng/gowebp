package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	"image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
		hasAlpha          bool
		alphaTransPct     string // % of transparent pixels correctly preserved
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

		hasAlpha := imageHasAlpha(dst)
		goLossyPSNR := "err"
		alphaTransPct := "-"
		if decoded, decErr := webp.Decode(bytes.NewReader(bufLossy.Bytes())); decErr == nil {
			goLossyPSNR = fmt.Sprintf("%.1f dB", psnrRGBA(dst, decoded))
			if hasAlpha {
				b := dst.Bounds()
				total, ok := 0, 0
				for y := b.Min.Y; y < b.Max.Y; y++ {
					for x := b.Min.X; x < b.Max.X; x++ {
						_, _, _, srcA := dst.At(x, y).RGBA()
						if srcA == 0 {
							total++
							_, _, _, gotA := decoded.At(x, y).RGBA()
							if gotA == 0 {
								ok++
							}
						}
					}
				}
				if total > 0 {
					alphaTransPct = fmt.Sprintf("%.1f%%", float64(ok)/float64(total)*100)
				}
			}
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
			name:          rel,
			srcKB:         srcKB,
			libLossyKB:    libLossyKBStr, libLossyMs: libLossyMsStr,
			libM6LossyKB:  libM6LossyKBStr, libM6LossyMs: libM6LossyMsStr,
			libLosslessKB: libLosslessKBStr, libLosslessMs: libLosslessMsStr,
			goLossyKB:     goLossyKB, goLossyMs: goLossyMs, goLossyPSNR: goLossyPSNR,
			goLosslessKB:  goLosslessKB, goLosslessMs: goLosslessMs,
			hasAlpha:      hasAlpha, alphaTransPct: alphaTransPct,
		})
	}

	// print to console
	fmt.Printf("\n%-40s %9s | %11s %7s | %11s %7s | %12s %7s | %9s %7s %10s | %11s %7s | %s\n",
		"File", "Original",
		"lib -m4", "time",
		"lib -m6", "time",
		"lib lossless", "time",
		"go lossy", "time", "PSNR",
		"go lossless", "time", "alpha trans%")
	fmt.Println(strings.Repeat("-", 170))
	for _, r := range rows {
		fmt.Printf("%-40s %8.1fkb | %11s %7s | %11s %7s | %12s %7s | %8.1fkb %7.0fms %10s | %10.1fkb %7.0fms | %s\n",
			r.name, r.srcKB,
			r.libLossyKB, r.libLossyMs,
			r.libM6LossyKB, r.libM6LossyMs,
			r.libLosslessKB, r.libLosslessMs,
			r.goLossyKB, r.goLossyMs, r.goLossyPSNR,
			r.goLosslessKB, r.goLosslessMs,
			r.alphaTransPct)
	}
	fmt.Println()

	// --- Animated GIF → WebP comparison ---
	type gifRow struct {
		name        string
		gifKB       float64
		frames      int
		durationMs  int
		gowebpKB    string
		gowebpMs    string
		gif2webpKB  string
		gif2webpMs  string
	}
	var gifRows []gifRow

	_, gif2webpErr := exec.LookPath("gif2webp")
	hasGif2webp := gif2webpErr == nil

	// Top-level GIFs only (non-recursive — top entries we listed earlier).
	var gifFiles []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.ToLower(filepath.Ext(e.Name())) == ".gif" {
			gifFiles = append(gifFiles, e.Name())
		}
	}

	gowebpAnimDir := filepath.Join("test_data", "gowebp", "anim")
	libwebpAnimDir := filepath.Join("test_data", "libwebp", "anim")
	os.MkdirAll(gowebpAnimDir, 0755)
	os.MkdirAll(libwebpAnimDir, 0755)

	for _, name := range gifFiles {
		srcPath := filepath.Join(originalDir, name)
		stem := strings.TrimSuffix(name, filepath.Ext(name))

		srcInfo, statErr := os.Stat(srcPath)
		if statErr != nil {
			t.Errorf("stat %s: %v", name, statErr)
			continue
		}
		gifKB := float64(srcInfo.Size()) / 1024

		f, err := os.Open(srcPath)
		if err != nil {
			t.Errorf("open %s: %v", name, err)
			continue
		}
		g, err := gif.DecodeAll(f)
		f.Close()
		if err != nil {
			t.Errorf("gif.DecodeAll %s: %v", name, err)
			continue
		}
		nFrames := len(g.Image)
		totalDur := 0
		for _, d := range g.Delay {
			totalDur += d
		}
		// totalDur is in 100ths of a second; convert to ms.
		durationMs := totalDur * 10

		// --- gowebp ConvertGIF ---
		gowebpKBStr, gowebpMsStr := "err", "err"
		var anim bytes.Buffer
		t0 := time.Now()
		if encErr := ConvertGIF(&anim, g, &Options{Quality: 75}); encErr != nil {
			t.Errorf("ConvertGIF %s: %v", name, encErr)
		} else {
			elapsed := time.Since(t0)
			outPath := filepath.Join(gowebpAnimDir, stem+".webp")
			os.WriteFile(outPath, anim.Bytes(), 0644)
			gowebpKBStr = fmt.Sprintf("%.1f kb", float64(anim.Len())/1024)
			gowebpMsStr = fmt.Sprintf("%.0f ms", float64(elapsed.Milliseconds()))
		}

		// --- gif2webp (libwebp companion tool) ---
		// Use -lossy for a fair comparison with gowebp's VP8 lossy output.
		// (gif2webp defaults to lossless, which produces much larger files
		// than the lossy VP8 encoder we use here.)
		gif2webpKBStr, gif2webpMsStr := "-", "-"
		if hasGif2webp {
			outPath := filepath.Join(libwebpAnimDir, stem+".webp")
			cmd := exec.Command("gif2webp", "-lossy", "-q", "75", srcPath, "-o", outPath)
			cmd.Stderr = nil
			t0 := time.Now()
			if cmd.Run() == nil {
				elapsed := time.Since(t0)
				if fi, fiErr := os.Stat(outPath); fiErr == nil {
					gif2webpKBStr = fmt.Sprintf("%.1f kb", float64(fi.Size())/1024)
					gif2webpMsStr = fmt.Sprintf("%.0f ms", float64(elapsed.Milliseconds()))
				}
			}
		}

		gifRows = append(gifRows, gifRow{
			name:       name,
			gifKB:      gifKB,
			frames:     nFrames,
			durationMs: durationMs,
			gowebpKB:   gowebpKBStr,
			gowebpMs:   gowebpMsStr,
			gif2webpKB: gif2webpKBStr,
			gif2webpMs: gif2webpMsStr,
		})
	}

	// Print GIF results to console.
	if len(gifRows) > 0 {
		fmt.Printf("\n%-45s %9s %7s %9s | %12s %9s | %12s %9s\n",
			"GIF File", "Original", "Frames", "Duration",
			"gowebp", "go time", "gif2webp", "g2w time")
		fmt.Println(strings.Repeat("-", 130))
		for _, r := range gifRows {
			fmt.Printf("%-45s %8.1fkb %7d %7dms | %12s %9s | %12s %9s\n",
				r.name, r.gifKB, r.frames, r.durationMs,
				r.gowebpKB, r.gowebpMs,
				r.gif2webpKB, r.gif2webpMs)
		}
		fmt.Println()
	}

	// write markdown
	var md strings.Builder
	md.WriteString("# WebP Conversion Comparison\n\n")
	md.WriteString("Parameters: quality=90. `hidden/` images resized to 300×300.\n\n")
	md.WriteString("| File | Original | cwebp -m4 | time | cwebp -m6 | time | lib lossless | lib lossless time | go lossy | go lossy time | PSNR (go) | go lossless | go lossless time | alpha trans% |\n")
	md.WriteString("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rows {
		md.WriteString(fmt.Sprintf("| %s | %.1f kb | %s | %s | %s | %s | %s | %s | %.1f kb | %.0f ms | %s | %.1f kb | %.0f ms | %s |\n",
			r.name, r.srcKB,
			r.libLossyKB, r.libLossyMs,
			r.libM6LossyKB, r.libM6LossyMs,
			r.libLosslessKB, r.libLosslessMs,
			r.goLossyKB, r.goLossyMs, r.goLossyPSNR,
			r.goLosslessKB, r.goLosslessMs,
			r.alphaTransPct))
	}

	if len(gifRows) > 0 {
		md.WriteString("\n## Animated GIF → WebP\n\n")
		md.WriteString("Parameters: quality=75. Per-frame gowebp uses VP8 lossy; gif2webp uses lossy mode for fair comparison.\n\n")
		md.WriteString("| File | GIF size | Frames | Duration | gowebp size | gowebp time | gif2webp size | gif2webp time |\n")
		md.WriteString("|---|---|---|---|---|---|---|---|\n")
		for _, r := range gifRows {
			md.WriteString(fmt.Sprintf("| %s | %.1f kb | %d | %d ms | %s | %s | %s | %s |\n",
				r.name, r.gifKB, r.frames, r.durationMs,
				r.gowebpKB, r.gowebpMs,
				r.gif2webpKB, r.gif2webpMs))
		}
	}

	ts := time.Now().Format("20060102-150405")
	mdPath := fmt.Sprintf("test_data/compare_results-%s-p%d.md", ts, runtime.GOMAXPROCS(0))
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
