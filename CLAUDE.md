# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
go test ./...

# Run a single test
go test -run TestName ./...

# Run benchmarks
go test -bench=BenchmarkEncode300x300 -benchtime=5s

# Run comparison vs cwebp (requires cwebp installed, images in test_data/original/)
go test -v -run TestCompareWithCwebp -timeout 300s

# Tidy dependencies
go mod tidy
```

## Architecture

`gowebp` is a pure-Go WebP encoder supporting both lossy (VP8) and lossless (VP8L). No cgo, no external binaries. Ported from libwebp (BSD 3-Clause).

**Public API** (`writer.go`):
```go
type Options struct {
    Lossless bool  // true = VP8L, false = VP8 lossy (default)
    Quality  int   // 0–100, lossy only (default: 90)
}
func Encode(w io.Writer, img image.Image, o *Options) error
```

**Package structure:**
- Root package `gowebp` — VP8 lossy encoder + unified `Encode` API
- `lossless/` subpackage — VP8L lossless encoder (ported from nativewebp)

**VP8 lossy encoding pipeline:**
```
Encode()
  → encodeLossy()
      → rgbaToYUV420()        [RGBA → YUV 4:2:0, padded to 16px multiples]
      → encodeFrame()         [macroblock loop]
           for each 16×16 MB:
             try i16 (4 modes) + i4 (10 modes × 16 sub-blocks)
             pick winner: SSD + λ × bits
             update recon buffer (decoder-consistent)
      → writeWebPHeader()     [RIFF/WEBP container]
```

**Key files:**
- `encoder.go` — macroblock loop, RD mode selection
- `writer.go` — public API, RIFF container
- `bitstream.go` — VP8 frame header, partition 0, coefficient encoding
- `bool_encoder.go` — VP8 boolean arithmetic coder (range coding)
- `dct.go` — forward/inverse DCT, Walsh-Hadamard Transform
- `intra4.go` — all 14 prediction modes (4 i16 + 10 i4)
- `quant.go` — quantization tables, quality→level mapping
- `probs.go` — hardcoded VP8 probability tables (from spec)
- `yuv.go` — RGBA→YUV 4:2:0 with edge padding

## Key invariants

**YUV conversion**: uses integer arithmetic with `>>18` shift matching libwebp exactly. Deviation causes color cast.

**Recon buffer**: after encoding each MB, the reconstruction must match the decoder's output exactly. i4 sub-blocks predict from reconstructed (not source) neighbors — errors cascade across the entire image.

**iTransform4x4 constant**: uses `85627` (not `20091`). Must match `golang.org/x/image/vp8`'s `inverseDCT4` exactly.

**Edge padding**: YUV planes padded to next multiple of 16 by edge pixel replication before encoding. Partial MBs at right/bottom edge use padded data.

**i4 diagonal modes**: B_RD, B_VR, B_VL, B_HD, B_HU must use top-row pixels for top context and left-column for left context — swapping these causes cascade errors.

**Boolean arithmetic coder**: output is MSB-first. Partition 0 must emit all 1056 probability update bits after the coefficient refresh flag.

## Known hacks & tech debt

### Size comparison vs cwebp (`encoder.go`)
gowebp **beats** cwebp at quality=90 on most images:

| Image | cwebp | gowebp | Δ |
|---|---|---|---|
| CD15 300×300 | 11.8 kb | 9.5 kb | −19% |
| portrait_1 300×300 | 16.2 kb | 13.0 kb | −20% |
| kodim05 768×512 | 138 kb | 131 kb | −5.4% |
| jablehk 1536×2048 | 304 kb | 287 kb | −5.6% |
| i30 1096×1600 | 312 kb | 259 kb | −17% |

All Kodak test set images beat cwebp as of the RD_DISTO_MULT=256 fix.

**Implemented optimisations:**
1. ✅ **Coefficient probability adaptation** — two-pass encoding adapts `default_coeff_probs`
   to actual statistics; updated probs are signalled in partition 0.
2. ✅ **Trellis quantization** — `trellis.go` implements `TrellisQuantizeBlock()` ported from
   libwebp. Applied to i4-AC (lambda=(7*qI4^2)>>3), i16-AC (lambda=(qI16^2)>>2), and
   UV (lambda=qUV^2<<1). NZ context tracked per block. Frequency sharpening included.
3. ✅ **Exact VP8EntropyCost table** — ported directly from `libwebp/src/dsp/cost.c`.
   Previous table had off-by-one indexing and zeroed-out entries from index ~60 onward,
   causing 13–20% file-size regression on Kodak suite.
4. ✅ **Coefficient bit cost R in i4 scoring** — `coeffBitCost()` mirrors `GetResidualCost_C`
   from libwebp; added `(lambdaI4 * rCost) >> 8` to i4 block RD score so mode selection
   accounts for actual bitstream cost of quantized levels.
5. ✅ **RD scoring fix (RD_DISTO_MULT=256)** — `score = 256*D + λ*(H + R + flatPenalty)`.
   Previous formula had distortion at 1× scale, making rate dominate (92% weight) and
   causing excess non-zero coefficients. Now matches libwebp's `SetRDScore` exactly.
6. ✅ **Flatness penalty for i4** — `FLATNESS_PENALTY=140` added to RD score when a 4×4
   block has ≤3 non-zero AC coefficients and mode != DC. Port of `PickBestIntra4`/`IsFlat`
   from `libwebp/src/enc/quant_enc.c`.
7. ✅ **SNS (Spatial Noise Shaping)** — DCT-histogram alpha + 4-segment K-means, exact port
   of `VP8SetSegmentParams`. Runs in parallel with YUV conversion.

**Remaining known gaps:**
- `tlambda_` per-MB lambda scaling (local texture complexity): not implemented; libwebp
  uses this to reduce distortion weight in flat regions.
- Token partitions: fixed 1 partition; libwebp supports up to 8.

### Debug test files (safe to delete)
21 test files exist. Most were created to diagnose bugs during development and are
no longer needed. The ones worth keeping:

| File | Keep? |
|---|---|
| `encoder_test.go` | ✅ core tests |
| `compare_test.go` | ✅ cwebp comparison |
| `diag7_test.go` | ✅ luma PSNR check |
| `diag_firstmb_test.go` | ✅ MB reconstruction check |
| `verify_i4_test.go` | ✅ i4 correctness |
| `diag2_test.go` … `diag6_test.go` | ❌ one-off debugging |
| `debug_scores_test.go` | ❌ very large, one-off |
| `cascade_trace_test.go` | ❌ one-off |
| `pixel_trace_test.go` | ❌ one-off |
| `recon_export_test.go` | ❌ one-off |
| `recon_match_test.go` | ❌ one-off |
| `i4_enable_test.go` | ❌ one-off |
| `i4_mismatch_test.go` | ❌ one-off |
| `force_i4_test.go` | ❌ one-off |
| `diag_mb66_test.go` | ❌ one-off |
| `diag_recon_test.go` | ❌ one-off |
| `diag_test.go` | ❌ one-off |

## Suggested next steps

1. ~~**Update Go version to 1.25.10**~~ ✅ done
2. ~~**Fix red colour bias**~~ ✅ fixed: UV DC prediction now uses reconstructed UV, not original samples
3. ~~**Clean up debug test files**~~ ✅ done: 16 one-off debug test files removed
4. ~~**Speed optimisation** — SAD pre-screening~~ ✅ done: top-4 SAD candidates per i4 block, ~2.5× i4 speedup
5. ~~**Fix quality remapping**~~ ✅ done: quality-4 hack removed; quality=90 now honest
6. ~~**Reduce size gap vs libwebp** — trellis quantization, entropy cost, RD scoring~~ ✅ done: gowebp now beats cwebp on most images (−5% to −20%)
7. **EncodeAll / animation** — lossless subpackage has it; lossy does not
8. **Decode support** — currently only encode; could wrap `golang.org/x/image/webp`

## Performance (Apple M1 Max)

| Image | cwebp | gowebp | Notes |
|---|---|---|---|
| 300×300 photo | ~12kb / ~21ms | **~9.5kb / ~6ms** | includes fork overhead for cwebp |
| 768×512 (Kodak) | ~138kb / ~50ms | **~131kb / ~30ms** | wave-front parallel encoding |
| 1536×2048 | ~304kb / ~250ms | **~287kb / ~102ms** | 2.5× faster than cwebp |

Bottleneck (remaining): i4 early exit for flat regions.
