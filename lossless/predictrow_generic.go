//go:build !arm64

package lossless

import "image/color"

// predictResidualsRow (portable): no NEON kernel available, so it is exactly
// the scalar reference. See predictResidualsRowScalar in transform.go.
func predictResidualsRow(pixels []color.NRGBA, width, mode, xStart, xEnd, y int, out []color.NRGBA) {
    predictResidualsRowScalar(pixels, width, mode, xStart, xEnd, y, out)
}
