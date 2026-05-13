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

### 速度 / Speed (GOMAXPROCS=10, Apple M1 Max)

| 圖片尺寸 / Image Size | cwebp | gowebp | 加速 / Speedup |
|---|---|---|---|
| 300×300 | ~21 ms | **~6 ms** | **3.4×** |
| 768×512 (Kodak) | ~55 ms | **~35 ms** | **1.6×** |
| 1536×2048 | ~250 ms | **~102 ms** | **2.5×** |

速度優勢來自 wave-front goroutine 並行編碼（每行 MB 一條 goroutine）及無 subprocess fork 開銷。單核（GOMAXPROCS=1）下 gowebp 因每個 MB 做更多優化（trellis、SNS）而比 cwebp 慢約 2×。

Speed advantage comes from wave-front goroutine parallel encoding and no subprocess fork overhead. At GOMAXPROCS=1, gowebp is ~2× slower than cwebp due to heavier per-MB work (trellis, SNS). See [gowebp-testdata](https://github.com/TommyLeng/gowebp-testdata) for full GOMAXPROCS breakdown.

### 檔案大小 / File Size (quality=90)

| 圖片 / Image | cwebp | gowebp | Δ |
|---|---|---|---|
| 人像 300×300 / Portrait 300×300 | ~16 kb | **~13 kb** | **−20%** |
| Kodak 768×512 (kodim05) | 138 kb | **131 kb** | **−5.4%** |
| 大圖 1536×2048 / Large | 304 kb | **287 kb** | **−5.6%** |
| 大圖 1096×1600 / Large | 312 kb | **259 kb** | **−17%** |

測試條件：`cwebp -q 90 -m 4`（非最高壓縮，`-m 6` 結果會不同）。

Comparison is against `cwebp -q 90 -m 4` (not maximum compression; `-m 6` results would differ).

詳細測試數據、對比圖片及說明見：

For detailed benchmark data, comparison images and analysis, see:

👉 **[gowebp-testdata](https://github.com/TommyLeng/gowebp-testdata)**

---

## 與 libwebp 的異同 / Differences from libwebp

gowebp 參考 libwebp 的核心算法移植，主要組件行為一致，但並非百分百相同。

gowebp ports libwebp's core algorithms and matches most behaviors, but is not a 100% identical reimplementation.

### 已實現 / Implemented

| 組件 / Component | 說明 / Description |
|---|---|
| Boolean arithmetic coder | VP8 布林算術編碼器 / VP8 boolean arithmetic coder |
| Forward / inverse DCT + WHT | 4×4 DCT 及 Walsh-Hadamard Transform |
| Intra4×4 + Intra16×16 | 全部 14 個預測模式 / All 14 intra prediction modes |
| RD mode selection | SSD + λ × bits，含 SAD top-4 預篩選 / with SAD top-4 pre-screening |
| Trellis quantization | Viterbi DP，移植自 `TrellisQuantizeBlock()` |
| Coeff probability adaptation | 兩次掃描自適應係數概率 / Two-pass adaptive coefficient probabilities |
| UV chroma RD prediction | DC / VE / HE / TM 四模式選擇 / Four-mode RD selection |
| SNS analysis | DCT-histogram alpha + K-means，精確移植 `VP8SetSegmentParams` |
| VP8L lossless | 完整無損編碼，支援動畫 `EncodeAll` / Full lossless with animation |

### 不同之處 / Differences

| 項目 / Item | libwebp | gowebp |
|---|---|---|
| 依賴 / Dependencies | C library，需要 cgo | 純 Go，無 cgo / Pure Go, no cgo |
| 並行 / Parallelism | 單執行緒 / Single-threaded | Wave-front goroutine 並行 / parallel |
| 量化分段 / Quant segments | 所有尺寸均 4 段 / Always 4 | 4 段（所有尺寸）/ 4 (all sizes) |
| per-MB lambda scaling | `tlambda_` 按局部紋理縮放 | 未實現，使用固定 lambda |
| Token partitions | 最多 8 個 / Up to 8 | 固定 1 個 / Fixed 1 |

### PSNR 差異說明 / Note on PSNR

PSNR 略低約 1 dB，主要來自 SNS 設計取捨：平滑區域（背景、皮膚）接受略多失真，紋理區域（頭髮、邊緣）保留更多細節。視覺差異極小。

PSNR is ~1 dB lower, mainly because SNS trades flat-area accuracy for perceptual quality — smooth regions accept slightly more distortion while textured regions are better preserved. The visual difference is barely noticeable.

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
