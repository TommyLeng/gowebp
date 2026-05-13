# gowebp — Pure Go VP8 Lossy WebP Encoder

## Goal

A pure-Go VP8 lossy WebP encoder with no cgo and no external binaries.
Ported from libwebp (BSD 3-Clause). All SIMD paths replaced with scalar Go.

Reference: `/Users/bayshark/-projects/self/libwebp`  
Target: quality=90, method=4 — matching cwebp's default parameters

---

## Current Status (as of 2026-05-13)

### Achieved

| Metric | cwebp | gowebp | Notes |
|---|---|---|---|
| CD15 300×300 size | 11.8 kb | **9.5 kb** | −19% vs cwebp -m 4 |
| portrait_1 300×300 size | 16.2 kb | **13.0 kb** | −20% |
| kodim05 768×512 size | 138 kb | **131 kb** | −5.4% |
| jablehk 1536×2048 size | 304 kb | **287 kb** | −5.6% |
| 300×300 speed | ~21ms | **~6ms** | in-process, wave-front parallel |
| Bitstream validity | ✅ | ✅ | golang.org/x/image/webp decodes |
| Color correctness | ✅ | ✅ | UV recon buffers fix chroma DC drift |

All Kodak test images are slightly smaller than `cwebp -q 90 -m 4` (avg −9.2%).
Note: `-m 4` is not maximum compression; results vs `-m 6` are untested.

### Implemented Components

- **Boolean arithmetic coder** — ported from `src/utils/bit_writer_utils.c`
- **Color space conversion** — RGBA → YUV 4:2:0, integer arithmetic matching libwebp
- **Forward DCT + WHT** — ported from `src/dsp/enc.c`
- **Inverse DCT + WHT** — matches `golang.org/x/image/vp8` decoder exactly
- **Quantization** — kDcTable/kAcTable from libwebp, quality→level mapping
- **Intra16 prediction** — all 4 modes: DC, V, H, TM
- **Intra4 prediction** — all 10 modes: DC, V, H, TM, LD, RD, VR, VL, HD, HU
- **RD mode selection** — `score = 256*D + λ*(H+R+flatPenalty)`, matches libwebp `SetRDScore`
- **Trellis quantization** — `TrellisQuantizeBlock()` port from `quant_enc.c`
- **Coefficient probability adaptation** — two-pass, updated probs in partition 0
- **Exact VP8EntropyCost table** — ported from `libwebp/src/dsp/cost.c`
- **Coefficient bit cost R** — `coeffBitCost()` mirrors `GetResidualCost_C`
- **Flatness penalty** — `FLATNESS_PENALTY=140` for flat i4 blocks (≤3 non-zero AC)
- **SNS (Spatial Noise Shaping)** — DCT-histogram alpha + 4-segment K-means, `VP8SetSegmentParams`
- **UV chroma RD prediction** — DC/VE/HE/TM four-mode RD selection
- **Wave-front parallel encoding** — goroutine pipeline across MB rows
- **RIFF/WEBP container** — correct chunk layout
- **VP8L lossless** — full lossless encoder with animation `EncodeAll`

### Known Bugs Fixed During Development

1. UV conversion used `>>20` shift instead of `>>18` → image appeared grayscale
2. Intra16 prediction read source pixels instead of reconstructed → cascade error across MBs
3. Recon buffer not updated after quantization → decoder prediction mismatch
4. `inverseWHT16` had transposed row/col indexing
5. `iTransform4x4` used wrong constant (`20091` vs `85627`)
6. 5 diagonal i4 modes (RD, VR, VL, HD, HU) had left/top pixel swapped
7. Edge MBs used wrong bounds for prediction → blocking at right/bottom

---

## Architecture

### Package Structure

```
gowebp/
  encoder.go      ← macroblock loop, RD mode selection
  writer.go       ← public Encode(w, img, quality) API + RIFF container
  bitstream.go    ← VP8 frame header, partition 0, coefficient encoding
  bool_encoder.go ← VP8 boolean arithmetic coder
  dct.go          ← forward/inverse DCT, WHT
  intra4.go       ← all 14 intra prediction modes (i16 + i4)
  quant.go        ← quantization tables, quality→level mapping
  probs.go        ← hardcoded VP8 probability tables
  yuv.go          ← RGBA→YUV 4:2:0 with edge padding
  go.mod
```

### Public API

```go
type Options struct {
    Lossless bool  // true = VP8L lossless, false = VP8 lossy (default)
    Quality  int   // 0–100, lossy only (default 90)
}
func Encode(w io.Writer, img image.Image, o *Options) error
```

### Encoding Pipeline

```
Encode()
  → RGBA → YUV 4:2:0 (padded to 16px multiples)
  → for each 16×16 macroblock:
      try i16 (4 modes) + i4 (10 modes × 16 sub-blocks)
      pick winner by: SSD(original, reconstructed) + λ × bits
      update recon buffer (decoder-consistent reconstruction)
  → write VP8 bitstream:
      partition 0: frame header + per-MB mode data (bool encoded)
      partition 1: DCT coefficients (bool encoded)
  → wrap in RIFF/WEBP container
```

---

## VP8 Encoder Components

### Boolean Arithmetic Coder (`bool_encoder.go`)
libwebp ref: `src/utils/bit_writer_utils.c`

VP8's entropy coder. Range coding with 8-bit probability model.
Every syntax element goes through this — it is the foundation.

### Color Space Conversion (`yuv.go`)
libwebp ref: `src/enc/picture_csp_enc.c`

RGBA → YUV 4:2:0. Integer arithmetic with `>>18` shift (not float).
Y at full resolution; U/V at half resolution (2×2 average).
YUV planes padded to next multiple of 16 by edge pixel replication.

### DCT + Quantization (`dct.go`, `quant.go`)
libwebp ref: `src/dsp/enc.c`, `src/enc/quant_enc.c`

- 4×4 forward DCT on residuals
- DC coefficients → WHT across 16 blocks per MB
- Quantize with step sizes from kDcTable/kAcTable
- Inverse transforms must match `golang.org/x/image/vp8` decoder exactly

### Intra Prediction (`intra4.go`)
libwebp ref: `src/enc/predictor_enc.c`

i16 modes (4): DC, V (Vertical), H (Horizontal), TM (TrueMotion)  
i4 modes (10): DC, V, H, TM, LD, RD, VR, VL, HD, HU

Critical: each i4 sub-block predicts from reconstructed (not source) neighbors.
After encoding each sub-block: dequantize → iDCT → store in recon buffer.

### RD Mode Selection (`encoder.go`)
libwebp ref: `src/enc/quant_enc.c` — `VP8Decimate`, `SetRDScore`

For each MB: `score = 256*D + λ*(H + R + flatPenalty)`
- D = SSD distortion (post-quant reconstruction vs source)
- H = mode header bits (VP8FixedCostsI4/I16)
- R = coefficient bit cost (`coeffBitCost`, mirrors `GetResidualCost_C`)
- flatPenalty = `FLATNESS_PENALTY=140` if block has ≤3 non-zero AC and mode != DC
- λ = `lambdaI4` or `lambdaI16` from `SetupMatrices()`

Try all i16 modes; try all i4 modes per sub-block with SAD top-4 pre-screening; pick overall winner.

### Coefficient Encoding (`bitstream.go`, `probs.go`)
libwebp ref: `src/enc/token_enc.c`, `src/enc/tree_enc.c`

Coefficients encoded via bool encoder using `default_coeff_probs[4][8][3][11]`.
Lookup context: plane (Y-DC/Y-AC/U/V) × band × neighbor non-zero count.

---

## Optimization Status

### 1. SAD Pre-screening ✅ implemented
Top-4 SAD candidates per i4 sub-block (`encoder.go` — `sad4x4`, `sadTopN=4`).
~2.5× speedup for i4 path.
libwebp ref: `src/enc/quant_enc.c` — uses `VP8SSE4x4` / `VP8SAD4x4`

### 2. Buffer Reuse ✅ implemented
Per-goroutine `mbWorkspace` struct (`encoder_parallel.go`) reused across all MBs.
Eliminates per-MB allocation GC pressure.

### 3. Wave-front Parallel Encoding ✅ implemented
Goroutine pipeline across MB rows (`encoder_parallel.go`).
Each row depends only on the row above; rows are pipelined with channel sync.
~2.5× speedup on multi-core for large images.

### 4. i4 Early Exit
Skip i4 search entirely if i16 score is already below a threshold.
Flat regions (low variance) rarely benefit from i4.
Estimated speedup: 20–30% for photos. Not yet implemented.

---

## Verification

```bash
# Single image test
go test -v -run TestEncodePhase2FinalI4Fixed

# Full comparison vs cwebp (requires cwebp installed)
go test -v -run TestCompareWithCwebp -timeout 300s

# Benchmark
go test -bench=BenchmarkEncode300x300 -benchtime=5s

# 1 CPU Benchmark
GOMAXPROCS=1 go test -bench=BenchmarkEncode300x300 -benchtime=10s -cpuprofile=cpu_gomaxprocs1.prof -run='^$' .

# Gomaxprocs test
go test -v -run TestCompareGOMAXPROCS -timeout 600s

# 

```

Reference outputs in `test_data/`:
- `CD15_cwebp_300x300.webp` — 12kb cwebp reference
- `CD15_lossy_i4_fixed.webp` — 12.3kb gowebp output

---

## Licence

Code ported from libwebp carries:
```
// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License
```
