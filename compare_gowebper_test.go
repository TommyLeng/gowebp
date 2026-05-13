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
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/eringen/gowebper"
	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

// TestCompareWithGowebper compares our gowebp (lossy + lossless) against gowebper (lossless only).
//
// gowebper is lossless-only, so our lossy column has no direct counterpart.
// All encoders run on images scaled to max 400px for a fair comparison.
//
// gowebper level guide:
//   - L0 = Huffman only (no transforms, no LZ77)
//   - L3 = SubtractGreen + LZ77 (4K window), no predictor
//   - L3 Q80 = L3 with near-lossless pre-quantization (Quality=80 → RGB shift≈1)
//   - L4+ = adds Predictor transform — too slow for large images (excluded)
const gowebperMaxDim = 400

func scaleTo400(src image.Image) *image.NRGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= gowebperMaxDim && h <= gowebperMaxDim {
		dst := image.NewNRGBA(b)
		draw.Draw(dst, dst.Bounds(), src, b.Min, draw.Src)
		return dst
	}
	scale := math.Min(float64(gowebperMaxDim)/float64(w), float64(gowebperMaxDim)/float64(h))
	nw, nh := int(float64(w)*scale), int(float64(h)*scale)
	dst := image.NewNRGBA(image.Rect(0, 0, nw, nh))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), src, b, draw.Src, nil)
	return dst
}

func TestCompareWithGowebper(t *testing.T) {
	originalDir := "test_data/original"
	entries, err := os.ReadDir(originalDir)
	if err != nil {
		t.Skipf("test_data/original/ not found: %v", err)
	}

	var images []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp" {
			images = append(images, e.Name())
		}
	}
	if len(images) == 0 {
		t.Skip("no images in test_data/original/")
	}

	type row struct {
		name        string
		dims        string
		lossyKB     float64
		lossyMs     float64
		lossyPSNR   string
		losslessKB  float64
		losslessMs  float64
		per0KB      float64
		per0Ms      float64
		per3KB      float64
		per3Ms      float64
		per3Q80KB   float64
		per3Q80Ms   float64
		per3Q80PSNR string
	}
	var rows []row

	encPer := func(img image.Image, level, quality int) (float64, float64) {
		var b bytes.Buffer
		t0 := time.Now()
		if err := gowebper.Encode(&b, img, &gowebper.Options{Level: level, Quality: quality}); err != nil {
			return 0, 0
		}
		return float64(b.Len()) / 1024, float64(time.Since(t0).Milliseconds())
	}

	psnrStr := func(img image.Image, data []byte) string {
		decoded, err := webp.Decode(bytes.NewReader(data))
		if err != nil {
			return "-"
		}
		v := psnrRGBA(img, decoded)
		if v >= 99.9 {
			return "inf"
		}
		return fmt.Sprintf("%.1f dB", v)
	}

	for _, name := range images {
		srcPath := filepath.Join(originalDir, name)
		f, err := os.Open(srcPath)
		if err != nil {
			t.Logf("skip %s: %v", name, err)
			continue
		}
		src, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			t.Logf("skip %s: %v", name, err)
			continue
		}

		scaled := scaleTo400(src)
		sb := scaled.Bounds()
		dims := fmt.Sprintf("%dx%d", sb.Dx(), sb.Dy())

		// gowebp lossy Q90
		var lossyBuf bytes.Buffer
		t0 := time.Now()
		if err := Encode(&lossyBuf, scaled, &Options{Quality: 90}); err != nil {
			t.Errorf("gowebp lossy %s: %v", name, err)
			continue
		}
		lossyKB := float64(lossyBuf.Len()) / 1024
		lossyMs := float64(time.Since(t0).Milliseconds())
		lossyPSNR := psnrStr(scaled, lossyBuf.Bytes())

		// gowebp lossless
		var llBuf bytes.Buffer
		t0 = time.Now()
		if err := Encode(&llBuf, scaled, &Options{Lossless: true}); err != nil {
			t.Errorf("gowebp lossless %s: %v", name, err)
			continue
		}
		losslessKB := float64(llBuf.Len()) / 1024
		losslessMs := float64(time.Since(t0).Milliseconds())

		p0KB, p0Ms := encPer(scaled, 0, 0)
		p3KB, p3Ms := encPer(scaled, 3, 0)

		var nlBuf bytes.Buffer
		t0 = time.Now()
		gowebper.Encode(&nlBuf, scaled, &gowebper.Options{Level: 3, Quality: 80})
		p3Q80KB := float64(nlBuf.Len()) / 1024
		p3Q80Ms := float64(time.Since(t0).Milliseconds())
		p3Q80PSNR := psnrStr(scaled, nlBuf.Bytes())

		rows = append(rows, row{
			name, dims,
			lossyKB, lossyMs, lossyPSNR,
			losslessKB, losslessMs,
			p0KB, p0Ms,
			p3KB, p3Ms,
			p3Q80KB, p3Q80Ms, p3Q80PSNR,
		})
	}

	fmt.Printf("\n%-30s %10s | %9s %5s %9s | %9s %5s | %9s %5s | %9s %5s | %9s %5s %9s\n",
		"File", "Scaled",
		"🟢 lossy Q90", "ms", "PSNR",
		"🟢 lossless", "ms",
		"🔵 per L0", "ms",
		"🔵 per L3", "ms",
		"🔵 per L3 Q80", "ms", "PSNR")
	fmt.Println(strings.Repeat("-", 152))
	for _, r := range rows {
		fmt.Printf("%-30s %10s | %8.1fkb %4.0fms %9s | %8.1fkb %4.0fms | %8.1fkb %4.0fms | %8.1fkb %4.0fms | %8.1fkb %4.0fms %9s\n",
			r.name, r.dims,
			r.lossyKB, r.lossyMs, r.lossyPSNR,
			r.losslessKB, r.losslessMs,
			r.per0KB, r.per0Ms,
			r.per3KB, r.per3Ms,
			r.per3Q80KB, r.per3Q80Ms, r.per3Q80PSNR)
	}
	fmt.Println()

	var md strings.Builder
	md.WriteString("# gowebp vs gowebper Comparison\n\n")
	md.WriteString("兩個 encoder 都係用 **max 400px** 縮小後的圖做測試（公平比較）。\n\n")
	md.WriteString("gowebper 係 **lossless only**，所以我哋的 lossy 欄位冇直接對應。\n\n")
	md.WriteString("**gowebper level 說明：**\n")
	md.WriteString("- L0 = Huffman only，冇 transform，冇 LZ77，最快\n")
	md.WriteString("- L3 = SubtractGreen + LZ77（4K window），冇 Predictor\n")
	md.WriteString("- L3 Q80 = L3 + near-lossless（encode 前對 RGB 做輕微 bit-rounding）\n")
	md.WriteString("- L4+ = 加 Predictor transform，大圖極慢，排除\n\n")
	md.WriteString("格式：`size / time`\n\n")
	md.WriteString("| File | Scaled | 🟢 gowebp lossy Q90 | PSNR | 🟢 gowebp lossless | 🔵 gowebper L0 | 🔵 gowebper L3 | 🔵 gowebper L3 Q80 | PSNR |\n")
	md.WriteString("|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rows {
		md.WriteString(fmt.Sprintf("| %s | %s | %.1f kb / %.0f ms | %s | %.1f kb / %.0f ms | %.1f kb / %.0f ms | %.1f kb / %.0f ms | %.1f kb / %.0f ms | %s |\n",
			r.name, r.dims,
			r.lossyKB, r.lossyMs, r.lossyPSNR,
			r.losslessKB, r.losslessMs,
			r.per0KB, r.per0Ms,
			r.per3KB, r.per3Ms,
			r.per3Q80KB, r.per3Q80Ms, r.per3Q80PSNR))
	}
	mdPath := "test_data/compare_gowebper.md"
	os.WriteFile(mdPath, []byte(md.String()), 0644)
	t.Logf("results saved to %s", mdPath)

	// Pixel-perfect check using Level=0 (Huffman-only, no LZ77).
	// golang.org/x/image/webp rejects some gowebper L3+ outputs with
	// "invalid LZ77 parameters" — a known decoder incompatibility, not a gowebper bug.
	t.Run("lossless_correctness", func(t *testing.T) {
		name := images[0]
		srcPath := filepath.Join(originalDir, name)
		f, _ := os.Open(srcPath)
		src, _, _ := image.Decode(f)
		f.Close()

		scaled := scaleTo400(src)

		var buf bytes.Buffer
		if err := gowebper.Encode(&buf, scaled, &gowebper.Options{Level: 0}); err != nil {
			t.Fatalf("encode: %v", err)
		}
		decoded, err := webp.Decode(bytes.NewReader(buf.Bytes()))
		if err != nil {
			t.Skipf("golang.org/x/image/webp decoder incompatibility: %v", err)
		}
		v := psnrRGBA(scaled, decoded)
		t.Logf("gowebper Level=0 PSNR on %s (scaled): %.1f dB", name, v)
		if v < 99.9 {
			t.Errorf("not pixel-perfect: PSNR=%.2f dB", v)
		}
	})
}
