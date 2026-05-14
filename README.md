# gowebp

純 Go 實現的 WebP 編碼器，支援有損（VP8）和無損（VP8L）輸出。無 cgo，無外部二進制依賴。參考 [libwebp](https://github.com/webmproject/libwebp)（BSD 3-Clause）移植而來。

A pure-Go WebP encoder supporting both lossy (VP8) and lossless (VP8L) output. No cgo, no external binaries. Ported from [libwebp](https://github.com/webmproject/libwebp) (BSD 3-Clause).

---

## 安裝 / Installation

```bash
go get github.com/TommyLeng/gowebp
```

---

## 使用方法 / Usage

```go
import "github.com/TommyLeng/gowebp"

// 有損編碼（推薦用於照片）/ Lossy encoding (recommended for photos)
err := gowebp.Encode(w, img, &gowebp.Options{Quality: 90})

// 無損編碼（推薦用於截圖、圖標）/ Lossless encoding (recommended for screenshots, icons)
err := gowebp.Encode(w, img, &gowebp.Options{Lossless: true})

// 使用預設值（有損 quality=90）/ Use defaults (lossy quality=90)
err := gowebp.Encode(w, img, nil)
```

### Options

| 欄位 / Field | 類型 / Type | 說明 / Description |
|---|---|---|
| `Lossless` | `bool` | `true` = VP8L 無損，`false` = VP8 有損（預設）/ `true` = VP8L lossless, `false` = VP8 lossy (default) |
| `Quality` | `int` | 0–100，僅有損模式有效（預設 90）/ 0–100, lossy only (default 90) |

---

## 效能測試 / Benchmark Results

測試環境：Apple M1 Max，Go 1.25，cwebp 1.4.0（quality=90）

Test environment: Apple M1 Max, Go 1.25, cwebp 1.4.0 (quality=90)

### 速度 / Speed (Apple M1 Max, quality=90)

不使用 gowebp 時，Go 程式需透過 subprocess 調用 cwebp binary：

Without gowebp, Go programs must invoke cwebp via subprocess:

```go
cmd := exec.Command("cwebp", "-q", "90", "input.jpg", "-o", "output.webp")
if err := cmd.Run(); err != nil {
    log.Fatal(err)
}
```

| 圖片 / Image | P=1 | P=2 | P=4 | P=10 | cwebp (exec) |
|---|---|---|---|---|---|
| 300×300 | 11.5 ms | 7.2 ms | 4.6 ms | **4.3 ms** | ~21 ms |
| 768×512 (Kodak) | 98 ms | 69 ms | 42 ms | **27 ms** | ~55 ms |
| 1536×2048 | 419 ms | 278 ms | 190 ms | **131 ms** | ~250 ms |

cwebp 時間包含 subprocess fork 開銷（約 15 ms）。

cwebp timings include subprocess fork overhead (~15 ms).

### 檔案大小 / File Size (quality=90)

| 圖片 / Image | cwebp -m4 | cwebp -m6 | gowebp | Δ vs m4 | Δ vs m6 |
|---|---|---|---|---|---|
| 人像 300×300 / Portrait 300×300 | ~16 kb | ~15 kb | **~13 kb** | **−20%** | **~−13%** |
| Kodak 768×512 (kodim05) | 138 kb | 135 kb | **131 kb** | **−5.4%** | **−3.4%** |
| 大圖 1536×2048 / Large | 304 kb | ~296 kb | **287 kb** | **−5.6%** | **~−3%** |
| 大圖 1096×1600 / Large | 312 kb | ~300 kb | **259 kb** | **−17%** | **~−14%** |

gowebp 在 `-m 4` 及 `-m 6`（cwebp 最高壓縮）下均輸出更小的檔案。詳細數據見 [gowebp-testdata](https://github.com/TommyLeng/gowebp-testdata)。

gowebp produces smaller files than both `cwebp -m 4` and `cwebp -m 6` (maximum compression). See [gowebp-testdata](https://github.com/TommyLeng/gowebp-testdata) for full data.

詳細測試數據、對比圖片及說明見：

For detailed benchmark data, comparison images and analysis, see:

👉 **[gowebp-testdata](https://github.com/TommyLeng/gowebp-testdata)**

---

## 與 libwebp 的異同 / Differences from libwebp

gowebp 參考 libwebp 的核心算法移植，主要組件行為一致，但並非百分百相同。

gowebp ports libwebp's core algorithms and matches most behaviors, but is not a 100% identical reimplementation.

### 不同之處 / Differences

| 項目 / Item | libwebp | gowebp |
|---|---|---|
| 依賴 / Dependencies | C library，需要 cgo | 純 Go，無 cgo / Pure Go, no cgo |
| 並行 / Parallelism | 單執行緒 / Single-threaded | Wave-front goroutine 並行 / parallel |

---

## 執行測試 / Running Tests

```bash
# 單元測試 / Unit tests
go test ./...

# 效能基準 / Benchmarks
go test -bench=BenchmarkEncode -benchtime=5s

# 與 cwebp 對比（需安裝 cwebp，圖片放於 test_data/original/）
# Compare vs cwebp (requires cwebp, images in test_data/original/)
go test -run TestCompareWithCwebp -v -timeout 300s
```

---

## 授權 / License

gowebp 採用 MIT License。

部分代碼移植自 [libwebp](https://github.com/webmproject/libwebp)，保留其原始 BSD 3-Clause License 聲明（見各源文件頭部）。

gowebp is MIT licensed.

Portions ported from [libwebp](https://github.com/webmproject/libwebp) retain the original BSD 3-Clause License notice (see individual source file headers).
