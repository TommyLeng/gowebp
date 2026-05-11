# gowebp — Pure Go VP8 Lossy WebP Encoder

## Goal

A pure-Go VP8 lossy WebP encoder with no cgo and no external binaries.
Ported from libwebp (BSD 3-Clause). All SIMD paths replaced with scalar Go.

Reference: `/Users/bayshark/-projects/self/libwebp`  
Target: quality=90, method=4 — matching cwebp's default parameters

---

## Current Status (as of 2026-05-11)

### Achieved

| Metric | cwebp | gowebp | Notes |
|---|---|---|---|
| 300×300 photo size | 12kb | **13.5kb** | +12.5%, honest quality=90 |
| 300×300 luma PSNR | 47.42 dB | **47.59 dB** | +0.17 dB |
| 300×300 speed | ~1ms | **~13ms** | includes fork overhead for cwebp |
| Bitstream validity | ✅ | ✅ | golang.org/x/image/webp decodes |
| Color correctness | ✅ | ✅ | UV recon buffers fix chroma DC drift |
| Edge handling | ✅ | ✅ | YUV padded to 16px multiples |

### Implemented Components

- **Boolean arithmetic coder** — ported from `src/utils/bit_writer_utils.c`
- **Color space conversion** — RGBA → YUV 4:2:0, integer arithmetic matching libwebp
- **Forward DCT + WHT** — ported from `src/dsp/enc.c`
- **Inverse DCT + WHT** — matches `golang.org/x/image/vp8` decoder exactly
- **Quantization** — kDcTable/kAcTable from libwebp, quality→level mapping
- **Intra16 prediction** — all 4 modes: DC, V, H, TM
- **Intra4 prediction** — all 10 modes: DC, V, H, TM, LD, RD, VR, VL, HD, HU
- **RD mode selection** — SSD + lambda×bits per MB, chooses i16 vs i4
- **Coefficient tokenization** — default_coeff_probs from VP8 spec
- **VP8 bitstream** — frame header, partition 0, coefficient partitions
- **RIFF/WEBP container** — correct chunk layout

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
func Encode(w io.Writer, img image.Image, quality int) error
// quality: 0–100, matches cwebp -q flag
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
libwebp ref: `src/enc/quant_enc.c` — `VP8Decimate`

For each MB: score = SSD(source, reconstructed) + λ × mode_bits  
λ = quantizer_step² / 2  
Try all i16 modes; try all i4 modes per sub-block; pick overall winner.

### Coefficient Encoding (`bitstream.go`, `probs.go`)
libwebp ref: `src/enc/token_enc.c`, `src/enc/tree_enc.c`

Coefficients encoded via bool encoder using `default_coeff_probs[4][8][3][11]`.
Lookup context: plane (Y-DC/Y-AC/U/V) × band × neighbor non-zero count.

---

## Optimization Opportunities (Next Steps)

Current bottleneck: ~0.05ms per macroblock.  
For 300×300: 361 MBs → 16ms ✅  
For 1080×1350: 5,780 MBs → 290ms ⚠️

Per-MB work: up to 224 DCT operations (64 for i16 + 160 for i4).

### 1. SAD Pre-screening ✅ implemented
Top-4 SAD candidates per i4 sub-block (`encoder.go` — `sad4x4`, `sadTopN=4`).
Measured ~2.5× speedup for i4 path; 300×300 dropped from ~16ms to ~13ms.
libwebp ref: `src/enc/quant_enc.c` — uses `VP8SSE4x4` / `VP8SAD4x4`

### 2. i4 Early Exit
Skip i4 search entirely if i16 score is already below a threshold.
Flat regions (low variance) rarely benefit from i4.
Estimated speedup: 20-30% for photos.

### 3. Buffer Reuse
Currently allocates per-MB buffers. Pre-allocate a single workspace struct
and reuse across all MBs to reduce GC pressure.

### 4. Parallel MB Encoding (advanced)
VP8 allows encoding multiple rows in parallel with care for recon buffer dependencies.
Each row depends only on the row above — rows can be pipelined.

---

## Verification

```bash
# Single image test
go test -v -run TestEncodePhase2FinalI4Fixed

# Full comparison vs cwebp (requires cwebp installed)
go test -v -run TestCompareWithCwebp -timeout 300s

# Benchmark
go test -bench=BenchmarkEncode300x300 -benchtime=5s
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
