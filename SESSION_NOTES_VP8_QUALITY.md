# VP8 Quality Investigation — 2026-05-29/30 Session Notes

## Goal
Close the 11 dB PSNR gap: gowebp 24.5 dB vs cwebp 35.5 dB at quality 75
on natural images (face frames from `baf1a2d038ad43b4bbe8b13799c0987d.gif`).
Measured via C `dwebp` (Go vp8 decoder is unreliable for PSNR).

## RESOLVED (commit 7ab5cd4)

**Two boundary bugs** in the intra-prediction context — once fixed, gowebp
matches libwebp within ~1 dB on every tested image.

### Bug #1: i4 topLeft fallbacks were SWAPPED
In `buildPred4Context` (intra4.go) and `buildPred4ContextFromPatch` (encoder.go):

```go
// WRONG (original):
} else if hasTop {
    ctx.topLeft = 127  // off — should be 129
} else if hasLeft {
    ctx.topLeft = 129  // off — should be 127
} else {
    ctx.topLeft = 128  // off — should be 127
}
```

VP8 border convention (per `intra4.go` `get()`):
- `x < 0` (left border) → 129
- `y < 0` (top border) → 127

For `hasTop && !hasLeft` (bpx=0, bpy>0), `topLeft` is at (-1, bpy-1) → x<0 → 129.
For `hasLeft && !hasTop` (bpy=0, bpx>0), `topLeft` is at (bpx-1, -1) → y<0 → 127.

The values were exactly swapped. Affected every i4 block in the leftmost column
and top row of every MB. Errors cascade since each i4 block's recon feeds the
next block's prediction.

### Bug #2: i16 TM_PRED used hardcoded topLeft=128
In both `intra16Predict` and `intra16PredictFromRecon` (intra4.go):

```go
// WRONG (original):
topLeft := 128
if hasTop && hasLeft {
    topLeft = getR(px-1, py-1)
}
```

libwebp `iterator_enc.c:27`:
```c
it->y_left[-1] = ... = (it->y > 0) ? 129 : 127;
```

Correct fallback is **127 for !hasTop** and **129 for hasTop && !hasLeft**.
TM_PRED formula is `pred = top + left - topLeft`, applied to every pixel of
the MB — an off-by-one on `topLeft` produces a systematic +1/-1 on all 256
pixels.

## Measured impact (q=75, via C dwebp)

| Test image | Before | After | libwebp | Gap (before→after) |
|---|---|---|---|---|
| face frame 27 | 24.49 dB | 34.58 dB | 35.53 dB | 11 → 0.94 dB |
| face frame 28 | 24.25 dB | 34.51 dB | 35.55 dB | 11 → 1.04 dB |
| face frame 32 | 25.42 dB | 34.54 dB | 35.49 dB | 10 → 0.95 dB |
| i1-a.png | 12.41 dB* | 40.47 dB | 42.12 dB | 30 → 1.65 dB |
| heidi photo | 12.11 dB* | 35.16 dB | 35.69 dB | 24 → 0.53 dB |

*Pre-fix measurements were taken with the lambda fix attempted but
the boundary bugs amplified the lambda regression — the lambda-only
"baseline" was 12 dB for natural images.

## Why lambda fix attempt collapsed PSNR earlier

The lambda_mode fix (use `lambda_mode` for cross-category i4-vs-i16 RD
comparison, matching libwebp `quant_enc.c:1029, 1121`) is conceptually
correct, but applying it on top of the boundary bugs amplified the
damage. The lambda fix flipped the encoder from 100% i4 (where the i4
topLeft bug cost 10 dB but didn't get worse) to ~50/50 i4/i16 (where
BOTH the i4 AND i16 TM_PRED corner bugs combined for 14+ dB collapse).

Now that the boundary bugs are fixed, the lambda fix can probably be
revisited safely in a future session — it should buy the remaining
~1 dB to close the gap fully.

## Diagnostic infrastructure left in repo

### Debug hooks (encoder.go, encoder_parallel.go, coeff_adapt.go, sns.go)
- `debugMBStats *[]mbInfo`: snapshot per-MB mode decisions
- `debugReconCapture func(reconY, stride, h)`: capture final Y-plane recon
- `debugDisableLoopFilter bool`: write filter_level=0 and per-segment
  fstrength=0 so decoder loop filter is bypassed (encoder.recon =
  decoder.recon bit-exact, useful for verification)
- `debugForceI16Mode int`: force a specific i16 mode (≥0 = mode, -1 = off)
- `debugForceSingleSegment bool`: collapse SNS to 1 segment
- `debugDumpI16Capture *map[[2]int]*debugI16Dump`: capture per-MB i16
  internal state (pred, dcLevels, AC levels, yDcRaw, whtOut, recon)

All hooks are inert in production (nil/false by default).

### Diagnostic tests
- `recon_diff_test.go`: encoder.recon vs dwebp's decoded recon, pixel-level.
  Includes `TestEncoderSingleMBPattern`, `TestSingleMBForceVEPRED`,
  `TestVEPREDDump`, `TestEncoderReconVsDecoder`.
- `psnr_diag_test.go`: `TestModeDistribution`, `TestPSNRBaseline`.
- `psnr_q95_test.go`: `TestI16FlatGray`.
- `psnr_photo_test.go`: `TestPSNRPhotos` on real photos.
- `loopfilter_test.go`: PSNR with loop filter disabled.
- `single_segment_test.go`: PSNR with single SNS segment.
- `idct_unit_test.go`: unit tests for `iTransform4x4` correctness.

## Next-session plan

1. **Re-attempt the lambda_mode fix** for cross-category i4-vs-i16
   comparison. Now that boundary bugs are fixed, mixed-mode encoding
   should produce correct output. Expected: closes the remaining
   ~1 dB gap to libwebp.

2. **Further libwebp parity items** (lower priority):
   - i16 per-mode selection uses pre-quant SSE from source-based pred;
     libwebp uses post-quant SSE from recon-based pred via
     `ReconstructIntra16`. Small effect at high quality.
   - i16 final score is missing `R` (coefficient rate) and `SD`
     (Hadamard texture distortion); libwebp includes both.
   - `tlambda_` per-MB texture activity weighting: not implemented.

## Quick reproduction

```bash
# Baseline (post-fix, current HEAD):
go test -run "TestPSNRBaseline|TestPSNRPhotos" -v
# Expected: gap ~1 dB on all images
```
