# Lossless (VP8L) — session notes & deferred candidates

2026-05-31. GIF→WebP alpha + standalone lossless efficiency/speed work, all
"跟足 libwebp" where it pays. This file records what landed and, more
importantly, the **deferred optimisations** so they can be reconsidered later.

## Landed this session (committed, all round-trip-exact, libwebp-faithful)

| Commit | What | libwebp source | Effect |
|---|---|---|---|
| `0b65c88` | arm64 NEON predictor-residual kernel (modes 0–10) + row-major restructure | predictor_enc.c shape | GIF single-core −15%, byte-identical |
| `639961b` | incremental Gini cost-scan (running accumSum/accumSumSq, touched-bin tileHisto) | — (own metric kept) | GIF −7%, byte-identical |
| `ad6d50f` | 2-pixel / 18-bit LZ77 hash | `GetPixPairHash64` | GIF speed −2.5%; size ~neutral |
| `ea488dc` | color-cache-**size search** 0..10 | `CalculateBestCacheSize` | heidi lossless now **< cwebp**; i1-a lossless −5%; i1-a alpha −3.6% |
| `7a0a9b4` | auto-palette: ≤16 palette-only / 17–256 try-both / >256 predictor | `EncoderAnalyze`/`AnalyzeEntropy` vp8l_enc.c:109 | 4 anims −1.4..−4.3%; 9710 ALPH 27194→12320; **single-core −42%** (2.338→1.354 s) |
| `7f3e5db` | single-channel-aware 17–256: alpha→predictor-only, RGB→try-both | own (channel test) | soft-alpha `-a` encode **−43%** (642→367 ms), no size change; fixes auto-palette's alpha slowdown |
| `75eba41` | low-effort alpha: `EncodeFast` (reduced LZ77 iters/window) for alpha plane | `GetMaxItersForQuality`/`GetWindowSizeForHashChain` (q32) | i1-a alpha **−23%** (283→218 ms), ALPH −0.4%; 9710 GIF **−11%** (1.354→1.201 s); full-lossless unchanged |

Cumulative GIF single-core ≈ 2819 → **~1354 ms (−52%)** (the last −42% is the
auto-palette commit: binary masks now palette-packed, not predictor-coded).
heidi full lossless beats cwebp -lossless (1158 < 1180 kb). i1-a alpha gap to
cwebp +21% → +15%.

## LANDED — auto-palette for few-colour images / alpha masks (lossless/writer.go)

**The 9710 GIF gap vs gif2webp was the ALPHA, not the colour.** webpinfo of the
9710 output: gowebp ALPH 27194 B vs gif2webp 9958 B (**2.7×**); VP8 colour only
+1.4%. Cause: delta-animation alpha masks are binary (alpha 0/255 = 2 colours),
and gowebp only enabled the VP8L **color-indexing (palette)** transform for
`*image.Paletted` inputs — the alpha is `*image.NRGBA`, so the 2-colour mask was
coded as full VP8L instead of packed to ~1 bit/px.

**What libwebp actually does (verified in vp8l_enc.c, not assumed):**
`EncoderAnalyze` counts distinct colours via `GetColorPalette`; `use_palette =
(palette_size <= MAX_PALETTE_SIZE=256)`. Then `AnalyzeEntropy` (vp8l_enc.c:109):
- **≤16 colours → commit palette** outright ("In practice, small palettes are
  better than any other transform"). NOT a try-both.
- **17..256 → entropy-estimate** among {direct, spatial, subgreen,
  spatialSubGreen, palette} and pick the cheapest estimate (single encode).
- **>256 → no palette**, estimate among the spatial variants.
- (Only method 6 + quality 100 brute-forces a real try-all-keep-smallest.)
gowebp's packing matches libwebp exactly: ≤2→1bit, ≤4→2bit, ≤16→4bit, >16→8bit
(no pack) — so ≤16 is precisely where the index packs and dominates.

**Change (writer.go):** `countColors(rgba)` (early-exit at 257) drives a 3-band
choice in `writeBitStream`, via the `encodeVP8L(rgba, hasAlpha, usePalette)`
helper:
- **≤16 → palette only** (single encode; faithful to vp8l_enc.c:109; binary
  masks live here → the win + the speedup).
- **17..256 → try-both keep-smaller** (libwebp estimates this band; gowebp lacks
  the entropy estimator, so try-both is used — strictly more accurate than the
  estimate and never regresses. i1-a's ~256-value gradient alpha lands here and
  still prefers the predictor → stays 26452. Cost is bounded: these images are
  rare and never per-GIF-frame).
- **>256 → predictor only** (palette impossible; today's behaviour).

Replaces the earlier blind try-both-on-all-≤256 draft, which was correct on size
but **+19% slower** (every binary mask encoded twice). The 3-band version is the
fix: binary masks (≤16) are single palette encodes.

**Result (measured, same machine, GOMAXPROCS=1, apples-to-apples vs HEAD):**
| anim | HEAD predictor-only | landed | Δ size |
|---|---|---|---|
| 9710 | 430336 / 2.338 s | 415462 / **1.354 s** | −3.46% / **−42% time** |
| baf1a2d | 574346 | 549586 | −4.31% |
| c8e7b15 | 199886 | 197116 | −1.39% |
| ezgif | 216640 | 213250 | −1.57% |
heidi/i1-a lossless unchanged (>256 / palette-rejected); i1-a ALPH stays 26452.
**Not just the +19% removed — single-core GIF is −42% vs baseline**, because every
frame's binary mask drops from the predictor path (NEON search + full-res LZ77 +
0..10 cache search) to a 1-bit palette pack (1/8 the main-image data, no predictor
search). Round-trip exact; TestLosslessRoundTrip + TestAlphaSubtractGreenRegression
+ anim tests pass; amd64 generic fallback builds/vets.

**Deviation from libwebp (flagged):** the 17..256 band uses try-both instead of
libwebp's cheap entropy estimate. It's never worse on size (try-both ⊇ estimate)
and aligns with libwebp's max-effort (m6 q100) try-all; the only cost is one extra
encode for rare 17..256 images. Faithful upgrade if ever wanted: port
`AnalyzeEntropy`'s per-mode entropy estimate (≈80 lines) and pick by estimate.

Throwaway in tree to rm: zz_gifmeasure_test.go.

## FOLLOW-UP — alpha-plane encode speed + single-channel refinement (2026-06-01)

User noticed `-a` (transparent-background) images encode slowly and that the
auto-palette commit made them SLOWER. Investigated; two findings:

**1. The auto-palette 17..256 try-both was a soft-alpha SPEED regression.**
Soft-alpha photos (i1-a/i11-a/i18-a) carry a gradient alpha → ~183 distinct
values → 17..256 band → the committed try-both encoded the alpha plane TWICE.
Measured (GOMAXPROCS=1): alpha chunk 564 ms, full lossy encode 642 ms — and the
alpha plane is **83–88% of the whole `-a` encode** (colour VP8-lossy is only
~78 ms). So the double-encode added ~+45% to every soft-alpha image.

**2. The "faithful" AnalyzeEntropy estimate (spike) FAILS for alpha.** Ported it
(float port of VP8LBitsEntropy + BitsEntropyRefine + the 5-mode estimate, was in
analyze.go). It mis-picks palette for i1-a's alpha (estimate: palette 209183 <
spatial 215169 bits — within 3%) and regressed it 26452 → 33384 B. Root cause:
the estimate ignores LZ77, so it can't tell a gradient's tiny predictor residuals
(great LZ77) from its palette indices (poor LZ77). Worse, libwebp's own alpha path
runs the SAME estimate (alpha_enc.c → VP8LEncodeStream, quality = 8·method = 32),
so "faithful" would mean matching libwebp's *suboptimal* palette pick — gowebp's
26452 already BEATS that. Spike reverted.

**Fix that landed — single-channel-aware band (the "scheme 4"):** `analyzeColors`
returns the colour count AND whether the image is effectively single-channel (R and
B both constant — an alpha plane is R=B=0). The 17..256 band then splits:
- **single-channel (alpha) → predictor only** (one encode). Palette can NEVER win
  on single-channel data: the predictor's residual distribution is more peaked than
  the raw values, so its entropy is always ≤ the raw index entropy. Verified across
  gradient / random-noise / real alpha — predictor wins every time, even pure noise.
- **multi-channel (RGB few-colour graphics) → try-both keep-smaller** (unchanged).
  Here palette CAN win by collapsing R/G/B into one index; measured on synthetic
  few-colour RGB it beats predictor by 21–91%, so the try-both safety net stays.
Result: soft-alpha alpha chunk 564 → 285 ms (**−49%**), full `-a` encode 642 →
367 ms (**−43%**); i1-a ALPH back to 26452; GIF (≤16) and photos (>256) unchanged;
round-trip exact. Net vs the committed try-both: alpha is faster, nothing regresses.

**LANDED — low-effort alpha path (reduced LZ77 search).** CPU profile of the alpha
encode (i1-a): fillMatches (optimal-LZ77 match finder) **38%**, applyPredictTransform
30.5%, optimal-DP 9.8%, cache-search 6.9%, greedy 5%, meta-Huffman ~0% (alpha too
uniform). libwebp encodes alpha at quality 32, which gates the LZ77 search:
GetMaxItersForQuality(32)=16 probes (vs 51 at q75) and GetWindowSizeForHashChain
=xsize<<6 (vs the full 2^20 window). Ported that as `fillMatches(..., lowEffort)`
with matchIterMaxLow=16 / matchWindowBitsLow=6, threaded `lowEffort` through
writeBitStream→encodeVP8L→writeBitStreamData→writeImageData→encodeImageData, and
added `lossless.EncodeFast` (= Encode with lowEffort) which encodeAlphaChunk now
uses. Full-image lossless (`lossless.Encode`) keeps full effort — heidi unchanged
at 1186132. Cache-search was NOT cut (libwebp keeps it at q32: CalculateBestCacheSize
only disables at q≤25) and predictor search was left alone (30% but size-sensitive;
libwebp's predictor effort comes from tile size, a separate deferred lever).

Measured (GOMAXPROCS=1): i1-a alpha 283.7 → 218 ms (**−23%**), ALPH 26452 → 26354 B
(slightly *smaller*); 9710 GIF 1.354 → 1.201 s (**−11%**, its binary masks also
take the low-effort palette path) with all 4 anims −2..−32 B (no regression);
heidi/full-lossless unchanged; round-trip exact. The reduced search only changes
WHICH matches are found (any match is valid), so output stays decodable.

Still open if alpha needs to be even faster: applyPredictTransform (30%) — would
need a larger predictor tile size (deferred candidate #1) — and the wasted work on
the 3 constant channels of a single-channel alpha plane (would need a true
single-channel lossless path, a bigger refactor). Also untried: libwebp's
WebPCleanupTransparentArea (flatten transparent-region RGB so VP8 colour compresses
better) — gowebp may not do it for standalone `-a`.

Throwaway to rm: lossless/zz_prof_test.go, zz_alpha_test.go, zz_gifsize_test.go.

## Quality (RD) sanity-check — GIF→WebP vs gif2webp (2026-05-31)

Prompted by "quality 不太確定". Measured each encoder's decoded+composited
canvas frames (Pillow, time-aligned by frame midpoint — gif2webp DROPS duplicate
frames, e.g. 9710 48/50, so index-alignment is wrong and gives bogus 16 dB;
time-alignment fixes it) against the original GIF frames as ground truth.
**This is orthogonal to the auto-palette commit** — that change is lossless alpha
(0 quality impact, round-trip exact). The quality lives in the VP8-lossy colour
path + flattenSimilarBlocks, untouched this session.

Single-point (Q75): gowebp PSNR 34.1–37.6 dB / SSIM 0.89–0.95 — good in absolute
terms, but **0.7–1.8 dB below gif2webp** at the same nominal Q75, usually smaller.

RD curve (sweep Q50..95, PSNR interpolated to MATCHED SIZE — the honest test):
| GIF | gowebp deficit @ matched size |
|---|---|
| c8e7b15 | **−0.5 dB** (151k→700k sweep; gap ~0.5–0.8 dB at every Q) |
| ezgif | **−1.7 dB** (169k→514k sweep; gap ~0.9–1.4 dB at every Q) |

So gowebp's "smaller at Q75" is **not a free lunch** — normalised for size, gif2webp
delivers higher quality; gowebp sits ~0.5–1.7 dB below the RD curve. (gif2webp ran
at default -m4; -m6 would WIDEN the gap, so this is the friendly comparison.)
This matches [[project-vp8-quality-gap]]'s "~0.5–0.9 dB general RD"; ezgif's larger
1.7 dB may add a flattenSimilarBlocks-aggressiveness component (worth probing if
the GIF colour path is ever revisited). Lever = VP8-lossy RD work (tlambda per-MB
lambda, token partitions), NOT lossless. Throwaway: zz_qdump_test.go,
zz_gifmeasure_test.go; /tmp/qcompare2.py + /tmp/qrd.py.

## Deferred candidates (NOT done — decide later whether to add)

Ordered by how clean/faithful/impactful they look.

### 1. Predictor tile-bits — measured, deferred (size win but not faithful + slower)
- gowebp hardcodes the predictor-transform tile size to **4 (16px)** in
  `applyPredictTransform` (lossless/transform.go).
- **Probe (tbits sweep):** the optimum is **3 (8px)**, consistently:
  | tbits | i1-a ALPH | i1-a lossless | heidi lossless | 9710 anim |
  |---|---|---|---|---|
  | 3 | 26234 (−0.8%) | 528744 (−0.4%) | 1178192 (−0.7%) | **422798 (−1.75%)** |
  | 4 (current) | 26452 | 530682 | 1186132 | 430336 |
  | 5 | 26448 | 532576 | 1195734 | — |
- **Why deferred:** (a) **+3% GIF encode time** (8px = 4× the tiles → more
  predictor-search overhead); (b) **not libwebp-faithful** — libwebp
  *size-derives* transform_bits via `GetTransformBits(method, histo_bits)`
  (4 for 350×622, 5 for ~1 MP), it does NOT use a flat 3. libwebp's own formula
  is ≈ neutral here. User chose faithful + no slowdown.
- **To add:** either flat `tileBits := 3` in applyPredictTransform (empirical,
  −1.75% GIF size, +3% speed), or port `GetHistoBits`/`GetTransformBits`
  (faithful, ≈ neutral here, helps small images).

### 2. Cross-color transform — gowebp DISABLES it (a real "didn't follow" gap)
- `writeBitStreamData` sets `transforms[transformColor] = false` always;
  `applyColorTransform` exists but uses a hardcoded CTE. libwebp searches the
  best per-tile colour-transform element (`GetBestColorTransformForTile`).
- Helps **RGB** lossless (decorrelates R/B from G further); **useless for a
  single-channel alpha** (R=B=0 → would add noise, libwebp's CTE search picks
  0 there too, so safe/neutral for alpha).
- **Mixed signal:** gowebp is already *smaller* than cwebp -lossless on several
  JPEG photos (e.g. jable-twy-001 −17%) without it, so the payoff is uncertain.
- **To add:** implement the per-tile CTE search, then enable transformColor.
  Medium effort (~80–120 lines).

### 3. Meta-Huffman merge metric — PopulationCost vs pure Shannon
- gowebp's `mergeGroupsGreedy` (lossless/metahuffman.go) merges Huffman groups
  on `dataCostBits` (pure Shannon entropy × count) + a flat `mergeOverheadBits =
  720` constant.
- libwebp uses **PopulationCost = refined-Shannon (`BitsEntropyRefine`) +
  `FinalHuffmanCost`** (models the actual Huffman code-length table via run/
  streak stats) — `histogram_enc.c:313`.
- Likely helps **varied RGB** content (more, better-separated Huffman groups);
  memory says meta-Huffman is **alpha-neutral** (alpha too uniform), so probably
  not the i1-a alpha lever.
- **To add:** port `PopulationCost` and use it as the merge metric.

### 4. Explicit RLE parse (`kLZ77RLE`) — probably redundant
- libwebp tries an RLE-only parse (dist=1 + dist=width runs) as a separate
  backward-reference candidate and keeps the cheapest.
- gowebp's cost-based `fillMatches` already offers dist=1 / dist=width matches to
  the DP, so the optimal parse should already find the RLE solution when best.
- **Verdict:** likely no win; low priority. Measure before implementing.

### 5. Predictor selection bias — tested, NOT worth it
- libwebp's `GetBestPredictorForTile` adds `PredictionCostBias` (favour residuals
  near 0) + a spatial bias (favour the neighbour tile's mode). gowebp uses Gini
  with neither.
- A faithful port was tried in a prior session: **neutral size (±0.3%)** and
  slower (needs `math.Log2`). **Reverted.** Don't re-chase.

### 6. stride-2 cache search — faster but not exhaustive
- The committed cache search sweeps all 0..10 (faithful, +6% GIF). A stride-2
  sweep {0,2,4,6,8,10} gave **identical** results on the tested images at **+3%**
  GIF. Reverted to full 0..10 for faithfulness; revisit if the cache-search cost
  ever matters.

## Known floor
The remaining per-core gap to libwebp (and the residual i1-a alpha +15%) is
largely **C vs Go scalar codegen** + libwebp doing *less* work at its default
method — not a single portable bug. On ARM, libwebp's lossless DSP is itself
mostly scalar (only SubtractGreen + ColorTransform are NEON), so SIMD is not the
lever; algorithm/effort parity is.
