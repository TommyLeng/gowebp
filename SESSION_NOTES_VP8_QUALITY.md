# VP8 Quality Investigation — 2026-05-29/30 Session Notes

## RESOLVED (2026-05-30) — mixed i4/i16 mode enabled (lambda_mode + R)

Re-applied the `lambda_mode` cross-category RD fix. It immediately collapsed
PSNR (~9 dB) / produced undecodable streams — the long-feared "mixed-mode
collapse". Root-caused with `TestEncoderReconVsDecoder` (GOMAXPROCS=1,
`debugDisableLoopFilter=true`): first diverging MB was MB(8,0), an i16
DC_PRED MB whose Y2-DC arithmetic-coding context differed from the decoder.

### Root cause: i16 Y2-DC NZ-context reset on i4 MBs
libwebp `ParseResiduals` (vp8_dec.c:549-566) leaves `nz_dc` **unchanged** for
i4 MBs (they have no Y2 block). gowebp's emission (`encodeTokenPartition`) and
stats pass (`collectCoeffStats`) RESET `topNzDC[mbX]`/`leftNzY[4]` to 0 for i4
MBs. When an i16 MB followed an i4 neighbour, the Y2-DC context diverged from
the decoder → bool-decoder desync from that MB onward. Invisible while the
encoder was 100% i4 (Y2 plane never used). **Fix: don't touch the DC context
on i4 MBs** (coeff_adapt.go). Verified bit-exact: `TestEncoderReconVsDecoder`
0/79524 mismatch.

### lambda_mode + R cross-category scoring (follows libwebp exactly)
Both i16 and i4 cross-category scores now use SetRDScore's form
`RD_DISTO_MULT*D + lambda_mode*(H + R)`:
- i16 R = port of `VP8GetCostLuma16` (Y2 type-1 + 16 luma-AC type-0 blocks),
  using the same NZ contexts the emission/decoder use. Needed a new
  `trellisY2Costs` table (coeff type 1).
- i4 R = accumulated per-block `coeffBitCost` + flatness penalty of the
  winning mode (`VP8GetCostLuma4`).
Omitting R (first attempt) over-selected i16 and bloated files (+16% on
frame27). With R, i16 is picked only when RD-beneficial.

### Results (vs old 100%-i4; via C dwebp)
| Image | old 100% i4 | mixed + R | libwebp |
|---|---|---|---|
| i1-a.png | 40.47 dB / 84772 B | 40.48 dB / 82482 B (−2.7%) | 42.12 dB / 62494 B |
| heidi | 35.16 dB / 228302 B | 35.18 dB / 226248 B (−0.9%) | 35.69 dB / 231394 B |
| frame27 modes | 100% i4 | 20% i16 | (~50% i16) |

Net: smaller files + tiny PSNR gain; cannot RD-regress (i16 only chosen when
its score wins). serial==parallel bit-exact. Also added: serial fallback when
`debugDumpI16Capture != nil` (parallel raced on the shared map), and
`TestI16MixedModeDecodes` regression guard.

### Still open — the bulk of the ~1 dB gap remains
PSNR only moved +0.01–0.02 dB. gowebp picks far less i16 than libwebp
(20% vs ~50% on frame27). Prime suspect: **i16 per-mode selection still uses
pre-quant prediction SSD** (`intra16Predict` source-based), while libwebp uses
full `ReconstructIntra16` (post-quant recon SSE + R) per mode. That is the next
libwebp-parity step.

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

1. ~~**Re-attempt the lambda_mode fix**~~ ✅ done 2026-05-30 (see top section):
   mixed mode enabled, Y2-DC NZ-context desync fixed, R term added. RD
   improved but the bulk of the ~1 dB gap remains.

2. **i16 per-mode selection via recon (highest remaining priority).**
   gowebp picks the i16 mode by pre-quant SSD of the source-based
   prediction (`intra16Predict`); libwebp's `PickBestIntra16` reconstructs
   each mode (`ReconstructIntra16`: quant+dequant+iDCT) and scores
   post-recon `D + lambda_i16*(H+R)`. This likely explains why gowebp
   selects ~20% i16 vs libwebp's ~50% and leaves PSNR on the table.

3. **Further libwebp parity items** (lower priority):
   - `SD` (Hadamard texture distortion) + `tlambda_` per-MB texture
     weighting: not implemented (SD=0 in both i16 and i4 scores). libwebp
     also doubles D/SD for flat i16 blocks (`IsFlat` refinement).

## Quick reproduction

```bash
# Baseline (post-fix, current HEAD):
go test -run "TestPSNRBaseline|TestPSNRPhotos" -v
# Expected: gap ~1 dB on all images
```
