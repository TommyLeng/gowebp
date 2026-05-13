package gowebp

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
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

// TestCompareGOMAXPROCS runs the lossy benchmark at GOMAXPROCS = 1, 2, 4, 10
// and saves a separate markdown for each run.
//
// Environment variables (optional):
//   GOWEBP_IMAGES_DIR  — directory to scan for images (default: test_data/original)
//   GOWEBP_RESULTS_DIR — directory to write markdown files (default: test_data)
func TestCompareGOMAXPROCS(t *testing.T) {
	originalDir := "test_data/original"
	if d := os.Getenv("GOWEBP_IMAGES_DIR"); d != "" {
		originalDir = d
	}
	resultsDir := "test_data"
	if d := os.Getenv("GOWEBP_RESULTS_DIR"); d != "" {
		resultsDir = d
	}

	// Scan directory recursively (one level deep) for image files.
	var images []string
	entries, err := os.ReadDir(originalDir)
	if err != nil {
		t.Skipf("%s not found: %v", originalDir, err)
	}
	for _, e := range entries {
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if !e.IsDir() && (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp") {
			images = append(images, e.Name())
		}
		if e.IsDir() {
			sub, _ := os.ReadDir(filepath.Join(originalDir, e.Name()))
			for _, s := range sub {
				ext2 := strings.ToLower(filepath.Ext(s.Name()))
				if !s.IsDir() && (ext2 == ".jpg" || ext2 == ".jpeg" || ext2 == ".png" || ext2 == ".webp") {
					images = append(images, filepath.Join(e.Name(), s.Name()))
				}
			}
		}
	}
	if len(images) == 0 {
		t.Skipf("no images in %s", originalDir)
	}

	_, hasCwebp := exec.LookPath("cwebp")

	type imgData struct {
		name      string
		src       *image.NRGBA
		srcKB     float64
		cwebp4KB  string
		cwebp4Ms  string
		cwebp6KB  string
		cwebp6Ms  string
	}

	runCwebp := func(srcPath, outPath string, method int, isHidden bool) (sizeKB, ms string) {
		args := []string{"-q", "90", "-m", fmt.Sprintf("%d", method)}
		if isHidden {
			args = append(args, "-resize", "300", "300")
		}
		args = append(args, srcPath, "-o", outPath)
		cmd := exec.Command("cwebp", args...)
		cmd.Stderr = nil
		t0 := time.Now()
		if cmd.Run() == nil {
			ms = fmt.Sprintf("%.0f ms", float64(time.Since(t0).Milliseconds()))
			if fi, err := os.Stat(outPath); err == nil {
				sizeKB = fmt.Sprintf("%.1f kb", float64(fi.Size())/1024)
			}
		}
		return
	}

	// Load all images and run cwebp once (cwebp is unaffected by GOMAXPROCS).
	var imgs []imgData
	for _, name := range images {
		srcPath := filepath.Join(originalDir, name)
		f, err := os.Open(srcPath)
		if err != nil {
			continue
		}
		src, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			continue
		}

		isHidden := strings.HasPrefix(name, "hidden/")
		var dst *image.NRGBA
		if isHidden {
			dst = image.NewNRGBA(image.Rect(0, 0, 300, 300))
			xdraw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Src, nil)
		} else {
			dst = image.NewNRGBA(src.Bounds())
			draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)
		}

		fi, _ := os.Stat(srcPath)
		d := imgData{
			name:     name,
			src:      dst,
			srcKB:    float64(fi.Size()) / 1024,
			cwebp4KB: "N/A",
			cwebp4Ms: "N/A",
			cwebp6KB: "N/A",
			cwebp6Ms: "N/A",
		}

		if hasCwebp == nil {
			tmp := t.TempDir()
			base := filepath.Base(name)
			d.cwebp4KB, d.cwebp4Ms = runCwebp(srcPath, filepath.Join(tmp, base+".m4.webp"), 4, isHidden)
			d.cwebp6KB, d.cwebp6Ms = runCwebp(srcPath, filepath.Join(tmp, base+".m6.webp"), 6, isHidden)
		}
		imgs = append(imgs, d)
	}

	if len(imgs) == 0 {
		t.Skip("no images decoded")
	}

	origProcs := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(origProcs)

	procsList := []int{1, 2, 4, 10}

	type row struct {
		name     string
		srcKB    float64
		cwebp4KB string
		cwebp4Ms string
		cwebp6KB string
		cwebp6Ms string
		goKB     float64
		goPSNR   string
		goMs     [4]float64 // indexed by position in procsList
	}
	rows := make([]row, len(imgs))
	for i, d := range imgs {
		rows[i] = row{
			name:     d.name,
			srcKB:    d.srcKB,
			cwebp4KB: d.cwebp4KB,
			cwebp4Ms: d.cwebp4Ms,
			cwebp6KB: d.cwebp6KB,
			cwebp6Ms: d.cwebp6Ms,
		}
	}

	for pi, procs := range procsList {
		runtime.GOMAXPROCS(procs)
		fmt.Printf("\n=== GOMAXPROCS=%d ===\n", procs)

		outDir := filepath.Join(resultsDir, "gowebp", fmt.Sprintf("gomaxprocs_%d", procs))
		if err := os.MkdirAll(outDir, 0755); err != nil {
			t.Fatalf("mkdir %s: %v", outDir, err)
		}

		for i, d := range imgs {
			var buf bytes.Buffer
			t0 := time.Now()
			if err := Encode(&buf, d.src, &Options{Quality: 90}); err != nil {
				t.Errorf("encode %s: %v", d.name, err)
				continue
			}
			rows[i].goMs[pi] = float64(time.Since(t0).Milliseconds())

			// Write encoded WebP to disk.
			outPath := filepath.Join(outDir, strings.TrimSuffix(d.name, filepath.Ext(d.name))+".webp")
			if err := os.MkdirAll(filepath.Dir(outPath), 0755); err == nil {
				os.WriteFile(outPath, buf.Bytes(), 0644)
			}

			// Only compute size and PSNR on first pass (deterministic).
			if pi == 0 {
				rows[i].goKB = float64(buf.Len()) / 1024
				if decoded, err := webp.Decode(bytes.NewReader(buf.Bytes())); err == nil {
					rows[i].goPSNR = fmt.Sprintf("%.1f dB", psnrRGBA(d.src, decoded))
				} else {
					rows[i].goPSNR = "-"
				}
			}
		}
	}

	// Combined markdown — one row per image, timing columns for each GOMAXPROCS.
	var md strings.Builder
	md.WriteString("# gowebp vs cwebp — GOMAXPROCS comparison\n\n")
	md.WriteString("quality=90, Apple M1 Max, Go 1.25\n\n")
	md.WriteString("| File | Original | cwebp -m4 | m4 time | cwebp -m6 | m6 time | go size | PSNR | P=1 ms | P=2 ms | P=4 ms | P=10 ms |\n")
	md.WriteString("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rows {
		md.WriteString(fmt.Sprintf("| %s | %.1f kb | %s | %s | %s | %s | %.1f kb | %s | %.0f | %.0f | %.0f | %.0f |\n",
			r.name, r.srcKB, r.cwebp4KB, r.cwebp4Ms, r.cwebp6KB, r.cwebp6Ms, r.goKB, r.goPSNR,
			r.goMs[0], r.goMs[1], r.goMs[2], r.goMs[3]))
	}
	combinedPath := filepath.Join(resultsDir, "gomaxprocs_combined.md")
	os.WriteFile(combinedPath, []byte(md.String()), 0644)
	t.Logf("Combined results saved to %s", combinedPath)
}
