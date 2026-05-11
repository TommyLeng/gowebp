# gowebp

A pure-Go WebP encoder supporting both **lossy (VP8)** and **lossless (VP8L)** output. No cgo, no external binaries.

Ported from [libwebp](https://github.com/webmproject/libwebp) (BSD 3-Clause).

## Installation

```bash
go get github.com/TommyLeng/gowebp
```

## Usage

```go
import "github.com/TommyLeng/gowebp"

// Lossy (VP8) — smaller files, recommended for photos
err := gowebp.Encode(w, img, &gowebp.Options{Quality: 90})

// Lossless (VP8L) — pixel-perfect, recommended for graphics/screenshots
err := gowebp.Encode(w, img, &gowebp.Options{Lossless: true})

// nil options = lossy quality 90
err := gowebp.Encode(w, img, nil)
```

## Performance

Benchmarked on Apple M1 Max, 300×300 portrait photo:

| | cwebp (C + SIMD) | gowebp (pure Go) |
|---|---|---|
| File size | 12.0 kb | 12.3 kb |
| Luma PSNR | 47.42 dB | 45.70 dB |
| Encode time | ~20ms\* | ~16ms |

\* cwebp time includes process fork/exec overhead (~5ms). Pure encode time is comparable.

For larger images (~1080×1350), gowebp takes ~290ms vs cwebp's ~160ms. The gap is due to missing SIMD — see [DESIGN.md](DESIGN.md) for optimization directions.

## Options

```go
type Options struct {
    Lossless bool // true = VP8L lossless, false = VP8 lossy (default)
    Quality  int  // 0–100, lossy only (default: 90)
}
```

## Comparison Test

`TestCompareWithCwebp` encodes every image in `test_data/original/` with both gowebp and cwebp, then writes the results to `test_data/compare_results.md`.

**Setup:**

```bash
# 1. Install cwebp (macOS)
brew install webp

# 2. Add your source images
mkdir -p test_data/original
cp /your/images/*.jpg test_data/original/

# 3. Create output folders
mkdir -p test_data/libwebp/lossy test_data/libwebp/lossless
mkdir -p test_data/gowebp/lossy  test_data/gowebp/lossless
```

**Run:**

```bash
go test -v -run TestCompareWithCwebp -timeout 300s
```

**Output folders:**

| Folder | Contents |
|---|---|
| `test_data/libwebp/lossy/` | cwebp lossy output (`-q 90 -m 4`) |
| `test_data/libwebp/lossless/` | cwebp lossless output (`-lossless -q 90`) |
| `test_data/gowebp/lossy/` | gowebp lossy output (quality=90) |
| `test_data/gowebp/lossless/` | gowebp lossless output |
| `test_data/compare_results.md` | Full comparison table |

Images placed in a `hidden/` subfolder inside `original/` are automatically resized to 300×300 before encoding.

## Licence

gowebp is MIT licensed.

Portions ported from [libwebp](https://github.com/webmproject/libwebp):
Copyright 2011 Google Inc. All Rights Reserved. BSD 3-Clause License.
