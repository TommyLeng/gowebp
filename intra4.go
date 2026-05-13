// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// intra4 prediction mode constants (VP8I4PredMode).
// These match the kBModesProba table index order in libwebp/src/enc/tree_enc.c.
const (
	B_DC_PRED = 0
	B_TM_PRED = 1
	B_VE_PRED = 2
	B_HE_PRED = 3
	B_RD_PRED = 4
	B_VR_PRED = 5
	B_LD_PRED = 6
	B_VL_PRED = 7
	B_HD_PRED = 8
	B_HU_PRED = 9
	numI4Modes = 10
)

// intra16 mode constants (VP8I16PredMode).
const (
	I16_DC_PRED = 0
	I16_TM_PRED = 1
	I16_VE_PRED = 2
	I16_HE_PRED = 3
	numI16Modes = 4
)

// avg3 returns (a + 2*b + c + 2) >> 2.
func avg3(a, b, c int) uint8 {
	return uint8((a + 2*b + c + 2) >> 2)
}

// avg2 returns (a + b + 1) >> 1.
func avg2(a, b int) uint8 {
	return uint8((a + b + 1) >> 1)
}

// clip8i clamps an int to [0,255].
func clip8i(v int) int {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return v
}

// pred4x4Context holds the 4+1+4 neighborhood pixels for a 4x4 block.
// The convention (from libwebp) is:
//   top[-5..top[-1]: left column pixels L[3]..L[0] then top-left (X)
//   top[0..7]:       top row then 4 right-of-top pixels
//
// We store it as a flat [13]int array with indexing:
//   ctx[0] = L[3] (top-left-left, 4 rows down)
//   ctx[1] = L[2]
//   ctx[2] = L[1]
//   ctx[3] = L[0] = leftmost of top row neighbors
//   ctx[4] = X    = top-left corner (top[-1] in libwebp)
//   ctx[5] = top[0]
//   ctx[6] = top[1]
//   ctx[7] = top[2]
//   ctx[8] = top[3]
//   ctx[9..12] = top[4..7] (right of top — needed by LD, VL)
//
// Access helpers to match libwebp naming inside prediction functions.

type pred4Context struct {
	// left[0..3] = rows 0..3 of left column (top to bottom)
	left [4]int
	// topLeft = pixel at (-1,-1) relative to block
	topLeft int
	// top[0..7] = top row[0..3] + right-of-top[0..3]
	top [8]int
}

// buildPred4Context extracts the prediction context for a 4x4 block at
// (bpx, bpy) within the luma plane.
// bpx, bpy are pixel coords within the 16x16 macroblock in global image coords.
// We fill with 128 for missing (border) neighbors.
func buildPred4Context(yuv *yuvImage, recon []uint8, reconStride int, bpx, bpy int) pred4Context {
	var ctx pred4Context

	// Helper to get reconstructed pixel with clamping to image bounds.
	// recon is the in-progress reconstructed macroblock buffer (16x16).
	// We look up from the YUV source for pixels outside the current MB.
	get := func(x, y int) int {
		if x < 0 || y < 0 || x >= yuv.width || y >= yuv.height {
			return 128
		}
		// If pixel is within [bpx..bpx+3, bpy..bpy+3] block, it's not yet reconstructed
		// but if it's above or left it's either from recon or yuv.
		mbStartX := (bpx / 16) * 16
		mbStartY := (bpy / 16) * 16
		if y >= mbStartY && x >= mbStartX && y < mbStartY+16 && x < mbStartX+16 {
			// within current MB — use recon buffer
			rx := x - mbStartX
			ry := y - mbStartY
			return int(recon[ry*reconStride+rx])
		}
		// outside current MB — use the source yuv
		if x >= yuv.width {
			x = yuv.width - 1
		}
		if y >= yuv.height {
			y = yuv.height - 1
		}
		return int(yuv.y[y*yuv.yStride+x])
	}

	hasLeft := bpx > 0
	hasTop := bpy > 0

	// left column: x = bpx-1, rows = bpy..bpy+3
	for i := 0; i < 4; i++ {
		if hasLeft {
			ctx.left[i] = get(bpx-1, bpy+i)
		} else {
			ctx.left[i] = 129
		}
	}

	// top-left corner
	if hasTop && hasLeft {
		ctx.topLeft = get(bpx-1, bpy-1)
	} else if hasTop {
		ctx.topLeft = 127
	} else if hasLeft {
		ctx.topLeft = 129
	} else {
		ctx.topLeft = 128
	}

	// top row[0..3]
	for i := 0; i < 4; i++ {
		if hasTop {
			ctx.top[i] = get(bpx+i, bpy-1)
		} else {
			ctx.top[i] = 127
		}
	}

	// right-of-top[0..3] (top[4..7])
	// Used by LD and VL. If the block is at the right edge of the image,
	// replicate top[3].
	for i := 0; i < 4; i++ {
		if hasTop {
			x := bpx + 4 + i
			if x >= yuv.width {
				x = yuv.width - 1
			}
			ctx.top[4+i] = get(x, bpy-1)
		} else {
			ctx.top[4+i] = 127
		}
	}

	return ctx
}

// intra4Predict fills pred[16] with the 4x4 prediction for the given mode.
// Uses pred4Context for neighbor pixels.
func intra4Predict(mode int, ctx pred4Context, pred []int16) {
	X := ctx.topLeft
	// left pixels: I=left[0]=row0, J=left[1], K=left[2], L=left[3]
	I := ctx.left[0]
	J := ctx.left[1]
	K := ctx.left[2]
	L := ctx.left[3]
	A := ctx.top[0]
	B := ctx.top[1]
	C := ctx.top[2]
	D := ctx.top[3]
	E := ctx.top[4]
	F := ctx.top[5]
	G := ctx.top[6]
	H := ctx.top[7]

	switch mode {
	case B_DC_PRED:
		// DC = (sum of top[0..3] + left[0..3] + 4) >> 3
		dc := A + B + C + D + I + J + K + L + 4
		v := uint8(dc >> 3)
		for i := range pred {
			pred[i] = int16(v)
		}

	case B_TM_PRED:
		// TrueMotion: pred[x,y] = clip(top[x] + left[y] - topLeft)
		// left[y] = I,J,K,L for y=0,1,2,3
		leftPx := [4]int{I, J, K, L}
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				v := ctx.top[x] + leftPx[y] - X
				pred[y*4+x] = int16(clip8i(v))
			}
		}

	case B_VE_PRED:
		// Vertical: pred[x,y] = AVG3(top[-1+x], top[x], top[x+1])
		// top[-1] = X, top[0..3], top[4] = E
		topEx := [6]int{X, A, B, C, D, E}
		vals := [4]int16{
			int16(avg3(topEx[0], topEx[1], topEx[2])),
			int16(avg3(topEx[1], topEx[2], topEx[3])),
			int16(avg3(topEx[2], topEx[3], topEx[4])),
			int16(avg3(topEx[3], topEx[4], topEx[5])),
		}
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				pred[y*4+x] = vals[x]
			}
		}

	case B_HE_PRED:
		// Horizontal: pred[x,y] = AVG3 of left column
		// left[-1] = X (topLeft), left[0..3] = I,J,K,L
		leftEx := [5]int{X, I, J, K, L}
		vals := [4]int16{
			int16(avg3(leftEx[0], leftEx[1], leftEx[2])),
			int16(avg3(leftEx[1], leftEx[2], leftEx[3])),
			int16(avg3(leftEx[2], leftEx[3], leftEx[4])),
			int16(avg3(leftEx[3], leftEx[4], leftEx[4])),
		}
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				pred[y*4+x] = vals[y]
			}
		}

	case B_LD_PRED:
		// Left-Down diagonal
		// Uses top[0..7]: A,B,C,D,E,F,G,H
		pred[0*4+0] = int16(avg3(A, B, C))
		pred[1*4+0] = int16(avg3(B, C, D))
		pred[0*4+1] = int16(avg3(B, C, D))
		pred[2*4+0] = int16(avg3(C, D, E))
		pred[1*4+1] = int16(avg3(C, D, E))
		pred[0*4+2] = int16(avg3(C, D, E))
		pred[3*4+0] = int16(avg3(D, E, F))
		pred[2*4+1] = int16(avg3(D, E, F))
		pred[1*4+2] = int16(avg3(D, E, F))
		pred[0*4+3] = int16(avg3(D, E, F))
		pred[3*4+1] = int16(avg3(E, F, G))
		pred[2*4+2] = int16(avg3(E, F, G))
		pred[1*4+3] = int16(avg3(E, F, G))
		pred[3*4+2] = int16(avg3(F, G, H))
		pred[2*4+3] = int16(avg3(F, G, H))
		pred[3*4+3] = int16(avg3(G, H, H))

	case B_RD_PRED:
		// Right-Down diagonal.
		// Ported directly from predFunc4RD in golang.org/x/image/vp8/predfunc.go.
		// pab=avg3(I,X,A), abc=avg3(X,A,B), bcd=avg3(A,B,C), cde=avg3(B,C,D)
		// qpa=avg3(J,I,X), rqp=avg3(K,J,I), srq=avg3(L,K,J)
		pab := avg3(I, X, A)
		abc := avg3(X, A, B)
		bcd := avg3(A, B, C)
		cde := avg3(B, C, D)
		qpa := avg3(J, I, X)
		rqp := avg3(K, J, I)
		srq := avg3(L, K, J)
		pred[0*4+0] = int16(pab)
		pred[0*4+1] = int16(abc)
		pred[0*4+2] = int16(bcd)
		pred[0*4+3] = int16(cde)
		pred[1*4+0] = int16(qpa)
		pred[1*4+1] = int16(pab)
		pred[1*4+2] = int16(abc)
		pred[1*4+3] = int16(bcd)
		pred[2*4+0] = int16(rqp)
		pred[2*4+1] = int16(qpa)
		pred[2*4+2] = int16(pab)
		pred[2*4+3] = int16(abc)
		pred[3*4+0] = int16(srq)
		pred[3*4+1] = int16(rqp)
		pred[3*4+2] = int16(qpa)
		pred[3*4+3] = int16(pab)

	case B_VR_PRED:
		// Vertical-Right.
		// Ported from predFunc4VR in golang.org/x/image/vp8/predfunc.go.
		// Even rows use avg2, odd rows use avg3.
		ab := avg2(X, A)
		bc := avg2(A, B)
		cd := avg2(B, C)
		de := avg2(C, D)
		pab := avg3(I, X, A)
		abc := avg3(X, A, B)
		bcd := avg3(A, B, C)
		cde := avg3(B, C, D)
		qpa := avg3(J, I, X)
		rqp := avg3(K, J, I)
		pred[0*4+0] = int16(ab)
		pred[0*4+1] = int16(bc)
		pred[0*4+2] = int16(cd)
		pred[0*4+3] = int16(de)
		pred[1*4+0] = int16(pab)
		pred[1*4+1] = int16(abc)
		pred[1*4+2] = int16(bcd)
		pred[1*4+3] = int16(cde)
		pred[2*4+0] = int16(qpa)
		pred[2*4+1] = int16(ab)
		pred[2*4+2] = int16(bc)
		pred[2*4+3] = int16(cd)
		pred[3*4+0] = int16(rqp)
		pred[3*4+1] = int16(pab)
		pred[3*4+2] = int16(abc)
		pred[3*4+3] = int16(bcd)

	case B_VL_PRED:
		// Vertical-Left.
		// Ported from predFunc4VL in golang.org/x/image/vp8/predfunc.go.
		ab := avg2(A, B)
		bc := avg2(B, C)
		cd := avg2(C, D)
		de := avg2(D, E)
		abc := avg3(A, B, C)
		bcd := avg3(B, C, D)
		cde := avg3(C, D, E)
		def := avg3(D, E, F)
		efg := avg3(E, F, G)
		fgh := avg3(F, G, H)
		pred[0*4+0] = int16(ab)
		pred[0*4+1] = int16(bc)
		pred[0*4+2] = int16(cd)
		pred[0*4+3] = int16(de)
		pred[1*4+0] = int16(abc)
		pred[1*4+1] = int16(bcd)
		pred[1*4+2] = int16(cde)
		pred[1*4+3] = int16(def)
		pred[2*4+0] = int16(bc)
		pred[2*4+1] = int16(cd)
		pred[2*4+2] = int16(de)
		pred[2*4+3] = int16(efg)
		pred[3*4+0] = int16(bcd)
		pred[3*4+1] = int16(cde)
		pred[3*4+2] = int16(def)
		pred[3*4+3] = int16(fgh)

	case B_HD_PRED:
		// Horizontal-Down.
		// Ported from predFunc4HD in golang.org/x/image/vp8/predfunc.go.
		pa := avg2(I, X)
		qp := avg2(J, I)
		rq := avg2(K, J)
		sr := avg2(L, K)
		pab := avg3(I, X, A)
		qpa := avg3(J, I, X)
		rqp := avg3(K, J, I)
		srq := avg3(L, K, J)
		abc := avg3(X, A, B)
		bcd := avg3(A, B, C)
		pred[0*4+0] = int16(pa)
		pred[0*4+1] = int16(pab)
		pred[0*4+2] = int16(abc)
		pred[0*4+3] = int16(bcd)
		pred[1*4+0] = int16(qp)
		pred[1*4+1] = int16(qpa)
		pred[1*4+2] = int16(pa)
		pred[1*4+3] = int16(pab)
		pred[2*4+0] = int16(rq)
		pred[2*4+1] = int16(rqp)
		pred[2*4+2] = int16(qp)
		pred[2*4+3] = int16(qpa)
		pred[3*4+0] = int16(sr)
		pred[3*4+1] = int16(srq)
		pred[3*4+2] = int16(rq)
		pred[3*4+3] = int16(rqp)

	case B_HU_PRED:
		// Horizontal-Up.
		// Ported from predFunc4HU in golang.org/x/image/vp8/predfunc.go.
		pq := avg2(I, J)
		qr := avg2(J, K)
		rs := avg2(K, L)
		pqr := avg3(I, J, K)
		qrs := avg3(J, K, L)
		rss := avg3(K, L, L)
		sss := uint8(L)
		pred[0*4+0] = int16(pq)
		pred[0*4+1] = int16(pqr)
		pred[0*4+2] = int16(qr)
		pred[0*4+3] = int16(qrs)
		pred[1*4+0] = int16(qr)
		pred[1*4+1] = int16(qrs)
		pred[1*4+2] = int16(rs)
		pred[1*4+3] = int16(rss)
		pred[2*4+0] = int16(rs)
		pred[2*4+1] = int16(rss)
		pred[2*4+2] = int16(sss)
		pred[2*4+3] = int16(sss)
		pred[3*4+0] = int16(sss)
		pred[3*4+1] = int16(sss)
		pred[3*4+2] = int16(sss)
		pred[3*4+3] = int16(sss)
	}
}

// intra16PredictFromRecon fills pred[256] using reconstructed border pixels from recon.
// This mirrors what the decoder does: it uses previously-decoded pixels as neighbors.
// recon is the full-frame reconstructed buffer with stride reconStride.
// reconW / reconH are the padded dimensions of the recon buffer (multiples of 16),
// used for bounds checking. The true image dimensions are not needed here because
// the YUV planes are already padded with edge-pixel replication.
// For MBs with no top/left neighbors, falls back to 127/129 constants.
func intra16PredictFromRecon(mode int, recon []uint8, reconStride, mbX, mbY, reconW, reconH int, pred []int16) {
	px := mbX * 16
	py := mbY * 16
	hasTop := mbY > 0
	hasLeft := mbX > 0

	getR := func(x, y int) int {
		if x < 0 || y < 0 || x >= reconW || y >= reconH {
			return 128
		}
		return int(recon[y*reconStride+x])
	}

	switch mode {
	case I16_DC_PRED:
		dc := 0
		n := 0
		if hasTop {
			for i := 0; i < 16; i++ {
				dc += getR(px+i, py-1)
			}
			n += 16
		}
		if hasLeft {
			for i := 0; i < 16; i++ {
				dc += getR(px-1, py+i)
			}
			n += 16
		}
		if n == 0 {
			dc = 128
		} else {
			dc = (dc + n/2) / n
		}
		for i := range pred {
			pred[i] = int16(dc)
		}

	case I16_VE_PRED:
		if !hasTop {
			for i := range pred {
				pred[i] = 127
			}
			return
		}
		for y := 0; y < 16; y++ {
			for x := 0; x < 16; x++ {
				pred[y*16+x] = int16(getR(px+x, py-1))
			}
		}

	case I16_HE_PRED:
		if !hasLeft {
			for i := range pred {
				pred[i] = 129
			}
			return
		}
		for y := 0; y < 16; y++ {
			v := int16(getR(px-1, py+y))
			for x := 0; x < 16; x++ {
				pred[y*16+x] = v
			}
		}

	case I16_TM_PRED:
		topLeft := 128
		if hasTop && hasLeft {
			topLeft = getR(px-1, py-1)
		}
		for y := 0; y < 16; y++ {
			leftV := 129
			if hasLeft {
				leftV = getR(px-1, py+y)
			}
			for x := 0; x < 16; x++ {
				topV := 127
				if hasTop {
					topV = getR(px+x, py-1)
				}
				v := topV + leftV - topLeft
				pred[y*16+x] = int16(clip8i(v))
			}
		}
	}
}

// intra16Predict fills pred[256] with the 16x16 prediction for the given mode.
// The yuv plane provides the border pixels.
func intra16Predict(mode int, yuv *yuvImage, mbX, mbY int, pred []int16) {
	px := mbX * 16
	py := mbY * 16
	hasTop := mbY > 0
	hasLeft := mbX > 0

	getY := func(x, y int) int {
		if x < 0 || y < 0 || x >= yuv.width || y >= yuv.height {
			return 128
		}
		return int(yuv.y[y*yuv.yStride+x])
	}

	switch mode {
	case I16_DC_PRED:
		dc := computeDCY(yuv, mbX, mbY)
		for i := range pred {
			pred[i] = int16(dc)
		}

	case I16_VE_PRED:
		// Copy top row down
		if !hasTop {
			for i := range pred {
				pred[i] = 127
			}
			return
		}
		for y := 0; y < 16; y++ {
			for x := 0; x < 16; x++ {
				pred[y*16+x] = int16(getY(px+x, py-1))
			}
		}

	case I16_HE_PRED:
		// Copy left column right
		if !hasLeft {
			for i := range pred {
				pred[i] = 129
			}
			return
		}
		for y := 0; y < 16; y++ {
			v := int16(getY(px-1, py+y))
			for x := 0; x < 16; x++ {
				pred[y*16+x] = v
			}
		}

	case I16_TM_PRED:
		// TrueMotion: pred[x,y] = clip(top[x] + left[y] - topLeft)
		topLeft := 128
		if hasTop && hasLeft {
			topLeft = getY(px-1, py-1)
		}
		for y := 0; y < 16; y++ {
			leftV := 129
			if hasLeft {
				leftV = getY(px-1, py+y)
			}
			for x := 0; x < 16; x++ {
				topV := 127
				if hasTop {
					topV = getY(px+x, py-1)
				}
				v := topV + leftV - topLeft
				pred[y*16+x] = int16(clip8i(v))
			}
		}
	}
}

// ssd4x4 and ssd16x16 are declared per-architecture in ssd_arm64.go /
// ssd_amd64.go (SIMD) or ssd_generic.go (scalar fallback).

// kBModesProba is the 10x10x9 probability table for encoding I4 modes.
// From libwebp/src/enc/tree_enc.c — kBModesProba[NUM_BMODES][NUM_BMODES][NUM_BMODES-1].
// Index: [top_pred][left_pred][bit_index 0..8]
var kBModesProba = [numI4Modes][numI4Modes][numI4Modes - 1]uint8{
	{{231, 120, 48, 89, 115, 113, 120, 152, 112},
		{152, 179, 64, 126, 170, 118, 46, 70, 95},
		{175, 69, 143, 80, 85, 82, 72, 155, 103},
		{56, 58, 10, 171, 218, 189, 17, 13, 152},
		{114, 26, 17, 163, 44, 195, 21, 10, 173},
		{121, 24, 80, 195, 26, 62, 44, 64, 85},
		{144, 71, 10, 38, 171, 213, 144, 34, 26},
		{170, 46, 55, 19, 136, 160, 33, 206, 71},
		{63, 20, 8, 114, 114, 208, 12, 9, 226},
		{81, 40, 11, 96, 182, 84, 29, 16, 36}},
	{{134, 183, 89, 137, 98, 101, 106, 165, 148},
		{72, 187, 100, 130, 157, 111, 32, 75, 80},
		{66, 102, 167, 99, 74, 62, 40, 234, 128},
		{41, 53, 9, 178, 241, 141, 26, 8, 107},
		{74, 43, 26, 146, 73, 166, 49, 23, 157},
		{65, 38, 105, 160, 51, 52, 31, 115, 128},
		{104, 79, 12, 27, 217, 255, 87, 17, 7},
		{87, 68, 71, 44, 114, 51, 15, 186, 23},
		{47, 41, 14, 110, 182, 183, 21, 17, 194},
		{66, 45, 25, 102, 197, 189, 23, 18, 22}},
	{{88, 88, 147, 150, 42, 46, 45, 196, 205},
		{43, 97, 183, 117, 85, 38, 35, 179, 61},
		{39, 53, 200, 87, 26, 21, 43, 232, 171},
		{56, 34, 51, 104, 114, 102, 29, 93, 77},
		{39, 28, 85, 171, 58, 165, 90, 98, 64},
		{34, 22, 116, 206, 23, 34, 43, 166, 73},
		{107, 54, 32, 26, 51, 1, 81, 43, 31},
		{68, 25, 106, 22, 64, 171, 36, 225, 114},
		{34, 19, 21, 102, 132, 188, 16, 76, 124},
		{62, 18, 78, 95, 85, 57, 50, 48, 51}},
	{{193, 101, 35, 159, 215, 111, 89, 46, 111},
		{60, 148, 31, 172, 219, 228, 21, 18, 111},
		{112, 113, 77, 85, 179, 255, 38, 120, 114},
		{40, 42, 1, 196, 245, 209, 10, 25, 109},
		{88, 43, 29, 140, 166, 213, 37, 43, 154},
		{61, 63, 30, 155, 67, 45, 68, 1, 209},
		{100, 80, 8, 43, 154, 1, 51, 26, 71},
		{142, 78, 78, 16, 255, 128, 34, 197, 171},
		{41, 40, 5, 102, 211, 183, 4, 1, 221},
		{51, 50, 17, 168, 209, 192, 23, 25, 82}},
	{{138, 31, 36, 171, 27, 166, 38, 44, 229},
		{67, 87, 58, 169, 82, 115, 26, 59, 179},
		{63, 59, 90, 180, 59, 166, 93, 73, 154},
		{40, 40, 21, 116, 143, 209, 34, 39, 175},
		{47, 15, 16, 183, 34, 223, 49, 45, 183},
		{46, 17, 33, 183, 6, 98, 15, 32, 183},
		{57, 46, 22, 24, 128, 1, 54, 17, 37},
		{65, 32, 73, 115, 28, 128, 23, 128, 205},
		{40, 3, 9, 115, 51, 192, 18, 6, 223},
		{87, 37, 9, 115, 59, 77, 64, 21, 47}},
	{{104, 55, 44, 218, 9, 54, 53, 130, 226},
		{64, 90, 70, 205, 40, 41, 23, 26, 57},
		{54, 57, 112, 184, 5, 41, 38, 166, 213},
		{30, 34, 26, 133, 152, 116, 10, 32, 134},
		{39, 19, 53, 221, 26, 114, 32, 73, 255},
		{31, 9, 65, 234, 2, 15, 1, 118, 73},
		{75, 32, 12, 51, 192, 255, 160, 43, 51},
		{88, 31, 35, 67, 102, 85, 55, 186, 85},
		{56, 21, 23, 111, 59, 205, 45, 37, 192},
		{55, 38, 70, 124, 73, 102, 1, 34, 98}},
	{{125, 98, 42, 88, 104, 85, 117, 175, 82},
		{95, 84, 53, 89, 128, 100, 113, 101, 45},
		{75, 79, 123, 47, 51, 128, 81, 171, 1},
		{57, 17, 5, 71, 102, 57, 53, 41, 49},
		{38, 33, 13, 121, 57, 73, 26, 1, 85},
		{41, 10, 67, 138, 77, 110, 90, 47, 114},
		{115, 21, 2, 10, 102, 255, 166, 23, 6},
		{101, 29, 16, 10, 85, 128, 101, 196, 26},
		{57, 18, 10, 102, 102, 213, 34, 20, 43},
		{117, 20, 15, 36, 163, 128, 68, 1, 26}},
	{{102, 61, 71, 37, 34, 53, 31, 243, 192},
		{69, 60, 71, 38, 73, 119, 28, 222, 37},
		{68, 45, 128, 34, 1, 47, 11, 245, 171},
		{62, 17, 19, 70, 146, 85, 55, 62, 70},
		{37, 43, 37, 154, 100, 163, 85, 160, 1},
		{63, 9, 92, 136, 28, 64, 32, 201, 85},
		{75, 15, 9, 9, 64, 255, 184, 119, 16},
		{86, 6, 28, 5, 64, 255, 25, 248, 1},
		{56, 8, 17, 132, 137, 255, 55, 116, 128},
		{58, 15, 20, 82, 135, 57, 26, 121, 40}},
	{{164, 50, 31, 137, 154, 133, 25, 35, 218},
		{51, 103, 44, 131, 131, 123, 31, 6, 158},
		{86, 40, 64, 135, 148, 224, 45, 183, 128},
		{22, 26, 17, 131, 240, 154, 14, 1, 209},
		{45, 16, 21, 91, 64, 222, 7, 1, 197},
		{56, 21, 39, 155, 60, 138, 23, 102, 213},
		{83, 12, 13, 54, 192, 255, 68, 47, 28},
		{85, 26, 85, 85, 128, 128, 32, 146, 171},
		{18, 11, 7, 63, 144, 171, 4, 4, 246},
		{35, 27, 10, 146, 174, 171, 12, 26, 128}},
	{{190, 80, 35, 99, 180, 80, 126, 54, 45},
		{85, 126, 47, 87, 176, 51, 41, 20, 32},
		{101, 75, 128, 139, 118, 146, 116, 128, 85},
		{56, 41, 15, 176, 236, 85, 37, 9, 62},
		{71, 30, 17, 119, 118, 255, 17, 18, 138},
		{101, 38, 60, 138, 55, 70, 43, 26, 142},
		{146, 36, 19, 30, 171, 255, 97, 27, 20},
		{138, 45, 61, 62, 219, 1, 81, 188, 64},
		{32, 41, 20, 117, 151, 142, 20, 21, 163},
		{112, 19, 12, 61, 195, 128, 48, 4, 24}},
}

// putI4Mode encodes a single 4x4 intra prediction mode into the bool encoder.
// Uses the probability tree from libwebp/src/enc/tree_enc.c — PutI4Mode().
// topPred and leftPred are the neighboring modes (0..9).
func putI4Mode(bw *boolEncoder, mode, topPred, leftPred int) {
	prob := &kBModesProba[topPred][leftPred]

	// Tree (paragraph 11.5 / PutI4Mode in tree_enc.c):
	// bit 0: mode != B_DC_PRED
	if bw.putBitAndReturn(mode != B_DC_PRED, int(prob[0])) {
		// bit 1: mode != B_TM_PRED
		if bw.putBitAndReturn(mode != B_TM_PRED, int(prob[1])) {
			// bit 2: mode != B_VE_PRED
			if bw.putBitAndReturn(mode != B_VE_PRED, int(prob[2])) {
				// bit 3: mode >= B_LD_PRED (i.e. mode is in {LD, VL, HD, HU})
				if !bw.putBitAndReturn(mode >= B_LD_PRED, int(prob[3])) {
					// mode is HE or RD (not HE=3 -> if HE then 1, else RD=0 branch)
					// bit 4: mode != B_HE_PRED
					if bw.putBitAndReturn(mode != B_HE_PRED, int(prob[4])) {
						// bit 5: mode != B_RD_PRED (then VR)
						bw.putBit(boolInt(mode != B_RD_PRED), int(prob[5]))
					}
				} else {
					// mode is LD, VL, HD, or HU
					// bit 6: mode != B_LD_PRED
					if bw.putBitAndReturn(mode != B_LD_PRED, int(prob[6])) {
						// bit 7: mode != B_VL_PRED
						if bw.putBitAndReturn(mode != B_VL_PRED, int(prob[7])) {
							// bit 8: mode != B_HD_PRED (then HU)
							bw.putBit(boolInt(mode != B_HD_PRED), int(prob[8]))
						}
					}
				}
			}
		}
	}
}

// putBitAndReturn encodes a bit and returns whether bit==1.
// Helper to avoid repeating the bit and returning its value.
func (e *boolEncoder) putBitAndReturn(condition bool, prob int) bool {
	v := boolInt(condition)
	e.putBit(v, prob)
	return condition
}

// boolInt converts a bool to 0 or 1.
func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
