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

### Remaining size gap vs libwebp (`encoder.go`)
gowebp produces ~12.7kb vs cwebp ~12kb (+5.8%) at quality=90 on 300×300 photo.

Lambda values are correct — `lambdaI4`, `lambdaI16`, `lambdaMode` all match
libwebp's `SetupMatrices()` exactly. The i16/i4 comparison intentionally biases
toward i4 (because lambdaI16 inflates i16Score), which improves compression for
natural images (verified: changing to lambdaMode scale makes files larger, not smaller).

**Implemented optimisations:**
1. ✅ **Coefficient probability adaptation** — two-pass encoding adapts `default_coeff_probs`
   to actual statistics; updated probs are signalled in partition 0.
2. ✅ **Trellis quantization** — `trellis.go` implements `TrellisQuantizeBlock()` ported from
   libwebp. Applied to i4-AC (lambda=(7*qI4^2)>>3), i16-AC (lambda=(qI16^2)>>2), and
   UV ((lambda=qUV^2<<1). NZ context tracked per block. Frequency sharpening included.
   Saves ~216 bytes (1.6%) vs standard quantization.

**Residual gap causes:**
- Our `vp8EntropyCost` table differs slightly from libwebp's for low probabilities
  (p=1..3 range), causing slightly suboptimal trellis and adaptation decisions.
- No SNS (Spatial Noise Shaping): libwebp adjusts quantizer per MB based on local
  complexity; we use a single quality level for the whole frame.
- Single segment: libwebp uses up to 4 quantizer segments; we always use 1.

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
6. ~~**Reduce size gap vs libwebp** — trellis quantization~~ ✅ done: `trellis.go` saves ~216 bytes (1.6%), gap now ~5.8% vs cwebp
7. **EncodeAll / animation** — lossless subpackage has it; lossy does not
8. **Decode support** — currently only encode; could wrap `golang.org/x/image/webp`

## Performance (Apple M1 Max)

| Image | cwebp | gowebp | Notes |
|---|---|---|---|
| 300×300 photo | ~12kb / ~20ms | ~12.7kb / ~15ms | includes fork overhead for cwebp; trellis adds ~2ms |
| 1080×1350 photo | — / ~160ms | — / ~120ms est. | ~0.02ms per macroblock with SAD pre-screening |

Bottleneck (remaining): i4 early exit for flat regions (see DESIGN.md §2).
