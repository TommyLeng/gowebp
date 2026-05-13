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
		name       string
		src        *image.NRGBA
		srcKB      float64
		cwebpKB    string
		cwebpMs    string
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
		dst := image.NewNRGBA(src.Bounds())
		draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)

		fi, _ := os.Stat(srcPath)
		d := imgData{
			name:    name,
			src:     dst,
			srcKB:   float64(fi.Size()) / 1024,
			cwebpKB: "N/A",
			cwebpMs: "N/A",
		}

		if hasCwebp == nil {
			out := filepath.Join(t.TempDir(), filepath.Base(name)+".webp")
			cmd := exec.Command("cwebp", "-q", "90", "-m", "4", srcPath, "-o", out)
			cmd.Stderr = nil
			t0 := time.Now()
			if cmd.Run() == nil {
				d.cwebpMs = fmt.Sprintf("%.0f ms", float64(time.Since(t0).Milliseconds()))
				if fi2, err2 := os.Stat(out); err2 == nil {
					d.cwebpKB = fmt.Sprintf("%.1f kb", float64(fi2.Size())/1024)
				}
			}
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
		name    string
		srcKB   float64
		cwebpKB string
		cwebpMs string
		goKB    float64
		goPSNR  string
		goMs    [4]float64 // indexed by position in procsList
	}
	rows := make([]row, len(imgs))
	for i, d := range imgs {
		rows[i] = row{
			name:    d.name,
			srcKB:   d.srcKB,
			cwebpKB: d.cwebpKB,
			cwebpMs: d.cwebpMs,
		}
	}

	for pi, procs := range procsList {
		pi := pi
		t.Run(fmt.Sprintf("GOMAXPROCS=%d", procs), func(t *testing.T) {
			runtime.GOMAXPROCS(procs)

			for i, d := range imgs {
				var buf bytes.Buffer
				t0 := time.Now()
				if err := Encode(&buf, d.src, &Options{Quality: 90}); err != nil {
					t.Errorf("encode %s: %v", d.name, err)
					continue
				}
				rows[i].goMs[pi] = float64(time.Since(t0).Milliseconds())

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

			// Per-GOMAXPROCS console output.
			fmt.Printf("\n=== GOMAXPROCS=%d ===\n", procs)
			fmt.Printf("%-40s %8s | %11s %7s | %9s %7s %10s\n",
				"File", "Original", "cwebp lossy", "time", "go lossy", "time", "PSNR")
			fmt.Println(strings.Repeat("-", 100))
			for _, r := range rows {
				fmt.Printf("%-40s %7.1fkb | %11s %7s | %8.1fkb %6.0fms %10s\n",
					r.name, r.srcKB, r.cwebpKB, r.cwebpMs, r.goKB, r.goMs[pi], r.goPSNR)
			}

			// Per-GOMAXPROCS markdown.
			var md strings.Builder
			md.WriteString(fmt.Sprintf("# gowebp vs cwebp — GOMAXPROCS=%d\n\n", procs))
			md.WriteString(fmt.Sprintf("quality=90, cwebp -m 4, Apple M1 Max, Go 1.25, GOMAXPROCS=%d\n\n", procs))
			md.WriteString("| File | Original | cwebp lossy | cwebp time | go lossy | go time | PSNR |\n")
			md.WriteString("|---|---|---|---|---|---|---|\n")
			for _, r := range rows {
				md.WriteString(fmt.Sprintf("| %s | %.1f kb | %s | %s | %.1f kb | %.0f ms | %s |\n",
					r.name, r.srcKB, r.cwebpKB, r.cwebpMs, r.goKB, r.goMs[pi], r.goPSNR))
			}
			mdPath := filepath.Join(resultsDir, fmt.Sprintf("gomaxprocs_%d.md", procs))
			os.WriteFile(mdPath, []byte(md.String()), 0644)
			t.Logf("GOMAXPROCS=%d results saved to %s", procs, mdPath)
		})
	}

	// Combined markdown — one row per image, timing columns for each GOMAXPROCS.
	var md strings.Builder
	md.WriteString("# gowebp vs cwebp — GOMAXPROCS comparison\n\n")
	md.WriteString("quality=90, cwebp -m 4, Apple M1 Max, Go 1.25\n\n")
	md.WriteString("| File | Original | cwebp | cwebp time | go size | PSNR | P=1 ms | P=2 ms | P=4 ms | P=10 ms |\n")
	md.WriteString("|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rows {
		md.WriteString(fmt.Sprintf("| %s | %.1f kb | %s | %s | %.1f kb | %s | %.0f | %.0f | %.0f | %.0f |\n",
			r.name, r.srcKB, r.cwebpKB, r.cwebpMs, r.goKB, r.goPSNR,
			r.goMs[0], r.goMs[1], r.goMs[2], r.goMs[3]))
	}
	combinedPath := filepath.Join(resultsDir, "gomaxprocs_combined.md")
	os.WriteFile(combinedPath, []byte(md.String()), 0644)
	t.Logf("Combined results saved to %s", combinedPath)
}
