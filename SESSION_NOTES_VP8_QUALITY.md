# VP8 Quality Investigation — 2026-05-29 Session Notes

## Goal
Close the 11 dB PSNR gap: gowebp 24.5 dB vs cwebp 35.5 dB at quality 75
on natural images (face frames from `baf1a2d038ad43b4bbe8b13799c0987d.gif`).
Measured via C `dwebp` (Go vp8 decoder is unreliable for PSNR).

## What we found

### Confirmed root cause #1: lambda mismatch at MB-level i4-vs-i16 comparison

In libwebp's `quant_enc.c`:
- `PickBestIntra16` selects the best i16 mode using `lambda_i16` internally,
  then **re-scores the winner with `lambda_mode`** at line 1029.
- `PickBestIntra4` selects the best i4 mode per-block using `lambda_i4`
  internally, then **re-scores the winner with `lambda_mode`** at line 1121
  before accumulating into rd_best.
- The final i4-vs-i16 comparison at line 1123 (`if (rd_best.score >= rd->score)`)
  has both scores at the same `lambda_mode` scale and `RD_DISTO_MULT=256` weighting
  on distortion.

`lambda_i4 = (3 * q_i4²) >> 7`,
`lambda_i16 = 3 * q_i16²` (NOT shifted — 128× larger than `lambda_i4`),
`lambda_mode = (q_i4²) >> 7` (≈ `lambda_i4 / 3`).

**gowebp** (encoder.go:319-324, encoder_parallel.go:380-384) compares:
```
i16Score = i16PostQuantDistortion + lambdaI16 * i16ModeBitCost(mode)
i4Score  = Σ_block (D + lambdaI4 * modeBits)   +   lambdaMode * 211
```
i16 gets `lambdaI16` weighting on rate (huge), i4 gets `lambdaI4` (small).
This systematically biases the comparison toward i4. **Result: 100% i4
selection on natural content** (verified by `TestModeDistribution` —
324/324 MBs are i4 on face frame 27).

### Confirmed root cause #2: i16 reconstruction goes wrong in MIXED-mode encoding

Applying the lambda fix (use `lambdaMode` for both sides, with `RD_DISTO_MULT*256`
on D) produced the CORRECT mode distribution (~47% i4 / 53% i16 on face frame,
similar to libwebp). **But PSNR collapsed from 24.49 dB → 10.51 dB.**

Decoded output shows recognizable face structure but with scattered 16×16
block-shaped corruption.

**Verified facts:**
- Force all-i4 (with lambda fix in place): PSNR = 24.49 dB ✓ (no regression).
- Force all-i16: PSNR = 22.03 dB ✓ (close to all-i4, slightly worse).
- Real comparison (47/53 mix): PSNR = 10.51 dB ✗ (catastrophic).
- Disabling loop filter: same PSNR (not the cause).
- Disabling per-block early-out: same PSNR (not the cause).
- Forcing serial encoder (`parallelThreshold = 1<<30`): same PSNR (not parallelism).
- Per-mode i16 selection switched from source-based to recon-based prediction:
  marginal change (~1 dB), did not resolve issue and broke decoding on some images.

**Encoder.recon vs decoder.recon comparison** (`TestEncoderReconVsDecoder`,
loop filter disabled):
- Single 16×16 MB i16 HE_PRED with no neighbors: **bit-exact** (max Δ=0).
- Face frame mixed-mode: MB(0,0) and MB(1,0) match; **MB(2,0) onwards diverge**.
- MB(2,0) is i16 VE_PRED (no top neighbor): all 256 pixels off by exactly +1.

A systematic +1 offset for an entire MB strongly suggests **a single
constant-pixel mismatch in either the prediction or the dequantized residual**
that the encoder applies but the decoder produces differently. The fact that
this only appears in MIXED mode rules out a pure prediction bug (the same
VE_PRED with no-top works for the single-MB test).

**Best hypothesis:** When an i4 MB is followed by an i16 MB (or some specific
inter-mode pattern), some state propagation differs between encoder and decoder.
Candidates not yet ruled out:
1. `topNzDC` propagation across i4 (sets it to 0) → i16 (reads it for DC ctx).
2. `leftNzY[4]` (DC NZ) propagation similarly.
3. Coefficient probability adaptation: stats collected across mixed-type MBs
   could differ from what the decoder reads. Worth checking by disabling
   prob adaptation entirely.
4. The encoder's `mbI16AcLevels[n][0]` slot (DC slot, unused for emission)
   contains STALE data from previous MB (trellis with first=1 doesn't clear
   it). It's not emitted, but it might be read accidentally somewhere.
5. The encoder's i16 reconstruction at line 738-742 of encoder.go uses
   `ws.mbI16AcLevels[n]` — verify this is exactly what gets emitted.

## Diagnostic infrastructure left in repo (HEAD)

### Debug hooks (encoder.go, encoder_parallel.go)
- `debugMBStats *[]mbInfo`: set non-nil to capture per-MB mode decisions.
- `debugReconCapture func(reconY, stride, h)`: set non-nil to capture the
  final Y-plane recon buffer.
- `debugDisableLoopFilter bool`: when true, encoder writes `filter_level=0`
  in partition 0 so the decoder skips loop filtering (encoder.recon =
  decoder.recon bit-exact).

These hooks are inert in production (nil/false by default).

### Diagnostic tests
- `frame_compare_test.go` — 3-way frame comparison (gowebp vs libwebp vs original GIF).
- `psnr_diag_test.go` — `TestModeDistribution`, `TestPSNRBaseline`.
- `psnr_q95_test.go` — `TestI16FlatGray` (verifies i16 works on flat content, 100 dB).
- `psnr_photo_test.go` — `TestPSNRPhotos` (other natural photos).
- `recon_diff_test.go` — `TestEncoderReconVsDecoder`, `TestEncoderSingleMBPattern`
  (pixel-level encoder/decoder match check using `dwebp -pgm` Y plane).
- `loopfilter_test.go` — `TestPSNRWithLoopFilterDisabled`.

## Next-session plan

1. **Pinpoint root cause #2.** Add a per-MB dump that records the encoder's
   `cd.i16AC`, `cd.i16DC`, `mbI16Pred`, and final `recon` slice, plus the
   chosen `(isI4, mode)`. Run on face frame 27 with the lambda fix applied
   and loop filter disabled. Compare encoder's per-MB recon to dwebp's
   PGM output MB-by-MB. The FIRST MB that diverges (we know it's MB(2,0))
   is the smoking gun — dump every state used to compute its recon and
   chase the +1.

2. Likely targets: (a) coefficient probability adaptation interaction
   between types when both i4 and i16 are emitted in the same frame;
   (b) `topNzDC` / `leftNzY[4]` / `topI4Modes` propagation between modes;
   (c) the chroma path, which I haven't checked at all.

3. **Once root cause #2 is fixed**, re-apply the lambda fix and verify PSNR
   moves toward libwebp's 35 dB. Expected gain: 5–10 dB based on libwebp's
   mode-distribution-driven savings.

4. **Other gaps vs libwebp (lower priority):**
   - gowebp i16 per-mode selection uses pre-quant SSE from source-based
     prediction; libwebp uses post-quant SSE from recon-based prediction
     (full `ReconstructIntra16` per mode). For high quality this is a
     small effect; might matter more at lower quality.
   - gowebp i16 final score is missing `R` (coefficient rate) and `SD`
     (Hadamard texture distortion); libwebp includes both.
   - `tlambda_` per-MB texture activity weighting: not implemented in gowebp.

## Files modified (no functional change to production encoder; all reverts in place)

```
encoder.go              # added debugMBStats, debugReconCapture, debugDisableLoopFilter
encoder_parallel.go     # added the same debug hooks
coeff_adapt.go          # respect debugDisableLoopFilter (writes filter_level=0)
```

Diagnostic test files added (untracked → will be committed):
```
frame_compare_test.go
psnr_diag_test.go
psnr_q95_test.go
psnr_photo_test.go
recon_diff_test.go
loopfilter_test.go
bl_perp_test.go
correctness_hash_test.go
```

## Quick reproduction

```bash
# Baseline (post-revert, current HEAD):
go test -run "TestPSNRBaseline|TestModeDistribution" -v
# Expected: 100% i4, frame 27 = 24.49 dB

# To reproduce the failing fix attempt, look at the git log message body
# for commit 4eaa11c~..HEAD and the diff in encoder.go around line 319-565.
# Set i16Score = rdDistoMult*D + lambdaMode*H, similarly for i4Score
# (tracking D and H separately, not combined as bestBlkOldScore).
```
