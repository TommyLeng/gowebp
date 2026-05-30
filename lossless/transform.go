package lossless

import (
    //------------------------------
    //general
    //------------------------------
    "math"
    "slices"
    //------------------------------
    //imaging
    //------------------------------
    "image/color"
    //------------------------------
    //errors
    //------------------------------
    //"log"
    "errors"
)

type transform int

const (
    transformPredict        = transform(0)
    transformColor          = transform(1)
    transformSubGreen       = transform(2)
    transformColorIndexing  = transform(3)     
)

func applyPredictTransform(pixels []color.NRGBA, width, height int) (int, int, int, []color.NRGBA) {
    tileBits := 4
    tileSize := 1 << tileBits
    bw := (width + tileSize - 1) / tileSize
    bh := (height + tileSize - 1) / tileSize

    blocks := make([]color.NRGBA, bw * bh)
    deltas := make([]color.NRGBA, width * height)
    
    accum := [][]int{
        make([]int, 256),
        make([]int, 256),
        make([]int, 256),
        make([]int, 256),
        make([]int, 40),
    }

    histos := make([][]int, len(accum))
    for i := range accum {
        histos[i] = make([]int, len(accum[i]))
    }

    // Residuals for one tile row, reused across modes/rows. The predictor
    // search runs row-by-row (rather than the old column-major per-pixel loop)
    // so the interior of each row is a contiguous run that a NEON kernel can
    // process; histogram counts and written deltas are order-independent, so
    // the bitstream is byte-identical to the per-pixel version.
    rowBuf := make([]color.NRGBA, tileSize)

    for y := 0; y < bh; y++ {
        for x := 0; x < bw; x++ {
            sx := x << tileBits
            sy := y << tileBits
            mx := min((x + 1) << tileBits, width)
            my := min((y + 1) << tileBits, height)
            n := mx - sx

            var best int
            var bestEntropy float64
            for i := 0; i < 14; i++ {
                for j := range accum {
                    copy(histos[j], accum[j])
                }

                for ty := sy; ty < my; ty++ {
                    predictResidualsRow(pixels, width, i, sx, mx, ty, rowBuf)
                    for k := 0; k < n; k++ {
                        histos[0][rowBuf[k].R]++
                        histos[1][rowBuf[k].G]++
                        histos[2][rowBuf[k].B]++
                        histos[3][rowBuf[k].A]++
                    }
                }

                var total float64
                for _, histo := range histos {
                    sum := 0
                    sumSquares := 0

                    for _, count := range histo {
                        sum += count
                        sumSquares += count * count
                    }

                    if sum == 0 {
                        continue
                    }

                    total += 1.0 - float64(sumSquares) / (float64(sum) * float64(sum))
                }

                if i == 0 || total < bestEntropy {
                    bestEntropy = total
                    best = i
                }
            }

            for ty := sy; ty < my; ty++ {
                predictResidualsRow(pixels, width, best, sx, mx, ty, rowBuf)
                base := ty * width + sx
                for k := 0; k < n; k++ {
                    deltas[base + k] = rowBuf[k]
                    accum[0][rowBuf[k].R]++
                    accum[1][rowBuf[k].G]++
                    accum[2][rowBuf[k].B]++
                    accum[3][rowBuf[k].A]++
                }
            }

            blocks[y * bw + x] = color.NRGBA{0, byte(best), 0, 255}
        }
    }

    copy(pixels, deltas)

    return tileBits, bw, bh, blocks
}

// predictResidualsRowScalar computes residual = current − predictor(mode) for
// each pixel in row y, columns [xStart, xEnd), writing the result to
// out[0:xEnd-xStart]. The per-pixel output is identical to applyFilter (same
// x==0 / y==0 boundary handling), so it leaves the bitstream unchanged. It is
// the portable reference: predictResidualsRow is this on non-arm64 builds, and
// the boundary / tail / not-yet-vectorised fallback for the arm64 NEON path
// (see predictrow_arm64.{go,s}).
func predictResidualsRowScalar(pixels []color.NRGBA, width, mode, xStart, xEnd, y int, out []color.NRGBA) {
    base := y * width
    for x := xStart; x < xEnd; x++ {
        d := applyFilter(pixels, width, x, y, mode)
        p := pixels[base + x]
        out[x - xStart] = color.NRGBA{
            R: p.R - d.R,
            G: p.G - d.G,
            B: p.B - d.B,
            A: p.A - d.A,
        }
    }
}

// average2 returns the per-channel arithmetic mean of two pixels.
func average2(a, b color.NRGBA) color.NRGBA {
    return color.NRGBA{
        uint8((int(a.R) + int(b.R)) / 2),
        uint8((int(a.G) + int(b.G)) / 2),
        uint8((int(a.B) + int(b.B)) / 2),
        uint8((int(a.A) + int(b.A)) / 2),
    }
}

// applyFilter returns the predicted pixel for position (x,y) under VP8L spatial
// predictor `prediction` (0..13).
//
// Previously this allocated a 14-element slice of predictor closures (plus an
// average closure) on every call and dispatched through a function pointer. The
// tile search calls it ~15× per pixel, so that was millions of per-pixel slice
// constructions and indirect calls. The switch below computes only the
// requested predictor directly (no allocation, inlinable) with bit-identical
// output — worth ~10% of single-core GIF-alpha encode time.
func applyFilter(pixels []color.NRGBA, width, x, y, prediction int) color.NRGBA {
    if x == 0 && y == 0 {
        return color.NRGBA{0, 0, 0, 255}
    } else if x == 0 {
        return pixels[(y - 1) * width + x]
    } else if y == 0 {
        return pixels[y * width + (x - 1)]
    }

    t := pixels[(y - 1) * width + x]
    l := pixels[y * width + (x - 1)]

    tl := pixels[(y - 1) * width + (x - 1)]
    tr := pixels[(y - 1) * width + (x + 1)]

    switch prediction {
    case 0:
        return color.NRGBA{0, 0, 0, 255}
    case 1:
        return l
    case 2:
        return t
    case 3:
        return tr
    case 4:
        return tl
    case 5:
        return average2(average2(l, tr), t)
    case 6:
        return average2(l, tl)
    case 7:
        return average2(l, t)
    case 8:
        return average2(tl, t)
    case 9:
        return average2(t, tr)
    case 10:
        return average2(average2(l, tl), average2(t, tr))
    case 11:
        pr := float64(l.R) + float64(t.R) - float64(tl.R)
        pg := float64(l.G) + float64(t.G) - float64(tl.G)
        pb := float64(l.B) + float64(t.B) - float64(tl.B)
        pa := float64(l.A) + float64(t.A) - float64(tl.A)

        // Manhattan distances to estimates for left and top pixels.
        pl := math.Abs(pa - float64(l.A)) + math.Abs(pr - float64(l.R)) +
              math.Abs(pg - float64(l.G)) + math.Abs(pb - float64(l.B))
        pt := math.Abs(pa - float64(t.A)) + math.Abs(pr - float64(t.R)) +
              math.Abs(pg - float64(t.G)) + math.Abs(pb - float64(t.B))

        if pl < pt {
            return l
        }
        return t
    case 12:
        return color.NRGBA{
            uint8(max(min(int(l.R) + int(t.R) - int(tl.R), 255), 0)),
            uint8(max(min(int(l.G) + int(t.G) - int(tl.G), 255), 0)),
            uint8(max(min(int(l.B) + int(t.B) - int(tl.B), 255), 0)),
            uint8(max(min(int(l.A) + int(t.A) - int(tl.A), 255), 0)),
        }
    case 13:
        a := average2(l, t)
        return color.NRGBA{
            uint8(max(min(int(a.R) + (int(a.R) - int(tl.R)) / 2, 255), 0)),
            uint8(max(min(int(a.G) + (int(a.G) - int(tl.G)) / 2, 255), 0)),
            uint8(max(min(int(a.B) + (int(a.B) - int(tl.B)) / 2, 255), 0)),
            uint8(max(min(int(a.A) + (int(a.A) - int(tl.A)) / 2, 255), 0)),
        }
    }
    return color.NRGBA{0, 0, 0, 255}
}

func applyColorTransform(pixels []color.NRGBA, width, height int) (int, int, int, []color.NRGBA) {
    tileBits := 4
    tileSize := 1 << tileBits
    bw := (width + tileSize - 1) / tileSize
    bh := (height + tileSize - 1) / tileSize

    blocks := make([]color.NRGBA, bw * bh)
    deltas := make([]color.NRGBA, width * height)
    
    //TODO: analyze block and pick best Color transform Element (CTE)
    cte := color.NRGBA {
        R: 1,   //red to blue
        G: 2,   //green to blue
        B: 3,   //green to red
        A: 255,
    }
    
    for y := 0; y < bh; y++ {
        for x := 0; x < bw; x++ {
            mx := min((x + 1) << tileBits, width)
            my := min((y + 1) << tileBits, height)

            for tx := x << tileBits; tx < mx; tx++ {
                for ty := y << tileBits; ty < my; ty++ {
                    off := ty * width + tx

                    r := int(int8(pixels[off].R))
                    g := int(int8(pixels[off].G))
                    b := int(int8(pixels[off].B))
                
                    b -= int(int8((int16(int8(cte.G)) * int16(g)) >> 5))
                    b -= int(int8((int16(int8(cte.R)) * int16(r)) >> 5))
                    r -= int(int8((int16(int8(cte.B)) * int16(g)) >> 5))
                    
                    pixels[off].R = uint8(r & 0xff)
                    pixels[off].B = uint8(b & 0xff)

                    deltas[off] = pixels[off]
                }
            }

            blocks[y * bw + x] = cte
        }
    }
    
    copy(pixels, deltas)
    
    return tileBits, bw, bh, blocks
}

func applySubtractGreenTransform(pixels []color.NRGBA) {
    for i, _ := range pixels {
        pixels[i].R = pixels[i].R - pixels[i].G
        pixels[i].B = pixels[i].B - pixels[i].G
    }
}

// useSubtractGreen reports whether the subtract-green transform would reduce the
// combined Shannon entropy of the R and B channels. It is a win for RGB photos
// (R/B correlate with G) but a loss for single-channel data such as an alpha
// plane carried in the green channel with R=B=0, where R-G/B-G become copies of
// the alpha signal. Mirrors the transform analysis in libwebp's AnalyzeAndInit.
func useSubtractGreen(pixels []color.NRGBA) bool {
    var hR, hB, hRG, hBG [256]int
    for _, p := range pixels {
        hR[p.R]++
        hB[p.B]++
        hRG[uint8(p.R-p.G)]++
        hBG[uint8(p.B-p.G)]++
    }
    n := len(pixels)
    before := histogramEntropy(hR[:], n) + histogramEntropy(hB[:], n)
    after := histogramEntropy(hRG[:], n) + histogramEntropy(hBG[:], n)
    return after < before
}

// histogramEntropy returns the Shannon entropy (in bits) of a symbol histogram
// with the given total count.
func histogramEntropy(hist []int, total int) float64 {
    if total == 0 {
        return 0
    }
    inv := 1.0 / float64(total)
    var e float64
    for _, c := range hist {
        if c == 0 {
            continue
        }
        p := float64(c) * inv
        e -= p * math.Log2(p)
    }
    return e
}

func applyPaletteTransform(pixels *[]color.NRGBA, width, height int) ([]color.NRGBA, int, error) {
    var pal []color.NRGBA
    for _, p := range (*pixels) {
        if !slices.Contains(pal, p) {
            pal = append(pal, p)
        }
   
        if len(pal) > 256 {
            return nil, 0, errors.New("palette exceeds 256 colors")
        }
    }

    size := 1
    if len(pal) <= 2 {
        size = 8
    } else if len(pal) <= 4 {
        size = 4
    } else if len(pal) <= 16 {
        size = 2
    }
    
    pw := (width + size - 1) / size

    packed := make([]color.NRGBA, pw * height)
    for y := 0; y < height; y++ {
        for x := 0; x < pw; x++ {
            pack := 0
            for i := 0; i < size; i++ {
                px := x * size + i
                if px >= width {
                    break
                }

                idx := slices.Index(pal, (*pixels)[y * width + px])
                pack |= int(idx) << (i * (8 / size))
            }

            packed[y * pw + x] = color.NRGBA{G: uint8(pack), A: 255}
        }
    }

    *pixels = packed
    
    for i := len(pal) - 1; i > 0; i-- {
        pal[i] = color.NRGBA{
            R: pal[i].R - pal[i - 1].R,
            G: pal[i].G - pal[i - 1].G,
            B: pal[i].B - pal[i - 1].B,
            A: pal[i].A - pal[i - 1].A,
        }
    }

    return pal, pw, nil
}
