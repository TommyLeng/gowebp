//go:build arm64

package lossless

import "image/color"

// predResSubRow computes out[k] = pixels[dstOff+k] - pixels[srcOff+k] (8-bit
// wrap) for k in 0..n-1, n a multiple of 4. NEON, implemented in
// predictrow_arm64.s. Covers predictors 1/2/3/4 (see the .s file).
//
//go:noescape
func predResSubRow(pixels []color.NRGBA, dstOff, srcOff int, out []color.NRGBA, n int)

// predResBlackRow computes out[k] = pixels[dstOff+k] - {0,0,0,255} (predictor
// 0). NEON, implemented in predictrow_arm64.s.
//
//go:noescape
func predResBlackRow(pixels []color.NRGBA, dstOff int, out []color.NRGBA, n int)

// predResAvgRow computes the averaging predictors 5..10 (avg2 == UHADD) and
// writes out[k] = pixels[curOff+k] - predictor. NEON, in predictrow_arm64.s.
//
//go:noescape
func predResAvgRow(pixels []color.NRGBA, mode, curOff, upOff int, out []color.NRGBA, n int)

// predictResidualsRow (arm64): NEON for the contiguous interior of the row,
// scalar for the boundary pixel (x==0), the n&3 tail, and the modes not yet
// vectorised. Output is byte-identical to predictResidualsRowScalar.
func predictResidualsRow(pixels []color.NRGBA, width, mode, xStart, xEnd, y int, out []color.NRGBA) {
    // Row 0 has no top context — every mode collapses to the left/black
    // predictor there (a single row for the whole image). Modes 11–13 are not
    // vectorised. Both go to the scalar reference.
    if y == 0 || mode > 10 {
        predictResidualsRowScalar(pixels, width, mode, xStart, xEnd, y, out)
        return
    }

    base := y * width
    iStart := xStart
    if iStart == 0 {
        // x==0, y>0: predictor is the pixel directly above (applyFilter rule).
        p := pixels[base]
        d := pixels[base-width]
        out[0] = color.NRGBA{p.R - d.R, p.G - d.G, p.B - d.B, p.A - d.A}
        iStart = 1
    }

    n := xEnd - iStart
    if n <= 0 {
        return
    }

    curOff := base + iStart // pixels index of cur[0] (x == iStart), > 0
    upOff := curOff - width  // pixels index of t[0] (pixel above), >= 0
    n4 := n &^ 3             // whole 4-pixel (16-byte) NEON vectors
    o := out[iStart-xStart:]

    switch {
    case n4 == 0:
        // Too short to vectorise.
        predictResidualsRowScalar(pixels, width, mode, iStart, xEnd, y, o)
        return
    case mode == 0:
        predResBlackRow(pixels, curOff, o, n4) // {0,0,0,255}
    case mode == 1:
        predResSubRow(pixels, curOff, curOff-1, o, n4) // left
    case mode == 2:
        predResSubRow(pixels, curOff, upOff, o, n4) // top
    case mode == 3:
        predResSubRow(pixels, curOff, upOff+1, o, n4) // top-right (wraps at edge)
    case mode == 4:
        predResSubRow(pixels, curOff, upOff-1, o, n4) // top-left
    default:
        // modes 5..10: averaging predictors (avg2 == UHADD).
        predResAvgRow(pixels, mode, curOff, upOff, o, n4)
    }

    // Scalar tail (n & 3 leftover pixels) for the vectorised modes.
    if iStart+n4 < xEnd {
        predictResidualsRowScalar(pixels, width, mode, iStart+n4, xEnd, y, out[iStart+n4-xStart:])
    }
}
