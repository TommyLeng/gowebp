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

### Lambda asymmetry (`encoder.go:41–52`)
RD mode selection uses different lambda scales for i4 vs i16:
- `lambdaI4 = (3 * q * q) >> 7` — small, so i4 wins easily on fine detail
- `lambdaI16 = 3 * q * q` — large, heavily penalises i16 bit cost

These constants were empirically tuned to achieve 12.3kb output. Not derived from
libwebp's approach. Could be improved by studying `VP8Decimate` more carefully.

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
3. **Clean up debug test files** — delete the ❌ ones above
4. ~~**Speed optimisation** — SAD pre-screening~~ ✅ done: top-4 SAD candidates per i4 block, ~2.5× i4 speedup
5. ~~**Fix quality remapping**~~ ✅ done: quality-4 hack removed; quality=90 now honest
6. **EncodeAll / animation** — lossless subpackage has it; lossy does not
7. **Decode support** — currently only encode; could wrap `golang.org/x/image/webp`

## Performance (Apple M1 Max)

| Image | cwebp | gowebp | Notes |
|---|---|---|---|
| 300×300 photo | ~12kb / ~20ms | ~13.5kb / ~13ms | honest quality=90, includes fork overhead for cwebp |
| 1080×1350 photo | — / ~160ms | — / ~120ms est. | ~0.02ms per macroblock with SAD pre-screening |

Bottleneck (remaining): i4 early exit for flat regions (see DESIGN.md §2).
