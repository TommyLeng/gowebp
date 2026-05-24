// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

//go:build amd64

package gowebp

// intra4Predict fills pred[16] with the 4x4 prediction for the given mode.
// The four simple modes (DC/TM/VE/HE) are handled with tight, compiler-
// friendly Go loops that the amd64 backend can auto-vectorise with SSE2/AVX.
// The six diagonal modes fall through to the scalar reference implementation
// in intra4.go.
func intra4Predict(mode int, ctx pred4Context, pred []int16) {
	switch mode {
	case B_DC_PRED:
		intra4PredDCamd64(ctx, pred)
	case B_TM_PRED:
		intra4PredTMamd64(ctx, pred)
	case B_VE_PRED:
		intra4PredVEamd64(ctx, pred)
	case B_HE_PRED:
		intra4PredHEamd64(ctx, pred)
	default:
		intra4PredScalar(mode, ctx, pred)
	}
}

// intra4PredDCamd64 computes B_DC_PRED for amd64.
// dc = (top[0]+top[1]+top[2]+top[3]+left[0]+left[1]+left[2]+left[3]+4) >> 3
// then broadcasts int16(dc) to all 16 elements of pred.
func intra4PredDCamd64(ctx pred4Context, pred []int16) {
	dc := ctx.top[0] + ctx.top[1] + ctx.top[2] + ctx.top[3] +
		ctx.left[0] + ctx.left[1] + ctx.left[2] + ctx.left[3] + 4
	v := int16(dc >> 3)
	// Unrolled fill — the compiler can emit a single SSE2 broadcast+store.
	pred[0] = v
	pred[1] = v
	pred[2] = v
	pred[3] = v
	pred[4] = v
	pred[5] = v
	pred[6] = v
	pred[7] = v
	pred[8] = v
	pred[9] = v
	pred[10] = v
	pred[11] = v
	pred[12] = v
	pred[13] = v
	pred[14] = v
	pred[15] = v
}

// intra4PredTMamd64 computes B_TM_PRED for amd64.
// pred[y*4+x] = clip8(top[x] + left[y] - topLeft)
func intra4PredTMamd64(ctx pred4Context, pred []int16) {
	X := ctx.topLeft
	// Precompute base[x] = top[x] - topLeft for x=0..3.
	b0 := ctx.top[0] - X
	b1 := ctx.top[1] - X
	b2 := ctx.top[2] - X
	b3 := ctx.top[3] - X
	for y := 0; y < 4; y++ {
		ly := ctx.left[y]
		pred[y*4+0] = int16(clip8i(ly + b0))
		pred[y*4+1] = int16(clip8i(ly + b1))
		pred[y*4+2] = int16(clip8i(ly + b2))
		pred[y*4+3] = int16(clip8i(ly + b3))
	}
}

// intra4PredVEamd64 computes B_VE_PRED for amd64.
// vals[x] = AVG3(topEx[x], topEx[x+1], topEx[x+2])
// where topEx = [topLeft, top[0], top[1], top[2], top[3], top[4]]
// Each row of pred is identical: pred[y*4+x] = vals[x].
func intra4PredVEamd64(ctx pred4Context, pred []int16) {
	v0 := int16(avg3(ctx.topLeft, ctx.top[0], ctx.top[1]))
	v1 := int16(avg3(ctx.top[0], ctx.top[1], ctx.top[2]))
	v2 := int16(avg3(ctx.top[1], ctx.top[2], ctx.top[3]))
	v3 := int16(avg3(ctx.top[2], ctx.top[3], ctx.top[4]))
	// 4 identical rows — the compiler can emit a single MOVQ repeated 4×.
	pred[0] = v0
	pred[1] = v1
	pred[2] = v2
	pred[3] = v3
	pred[4] = v0
	pred[5] = v1
	pred[6] = v2
	pred[7] = v3
	pred[8] = v0
	pred[9] = v1
	pred[10] = v2
	pred[11] = v3
	pred[12] = v0
	pred[13] = v1
	pred[14] = v2
	pred[15] = v3
}

// intra4PredHEamd64 computes B_HE_PRED for amd64.
// vals[y] = AVG3(leftEx[y], leftEx[y+1], leftEx[y+2])
// where leftEx = [topLeft, left[0], left[1], left[2], left[3], left[3]]
// Each row y of pred is filled with vals[y].
func intra4PredHEamd64(ctx pred4Context, pred []int16) {
	r0 := int16(avg3(ctx.topLeft, ctx.left[0], ctx.left[1]))
	r1 := int16(avg3(ctx.left[0], ctx.left[1], ctx.left[2]))
	r2 := int16(avg3(ctx.left[1], ctx.left[2], ctx.left[3]))
	r3 := int16(avg3(ctx.left[2], ctx.left[3], ctx.left[3]))
	pred[0] = r0
	pred[1] = r0
	pred[2] = r0
	pred[3] = r0
	pred[4] = r1
	pred[5] = r1
	pred[6] = r1
	pred[7] = r1
	pred[8] = r2
	pred[9] = r2
	pred[10] = r2
	pred[11] = r2
	pred[12] = r3
	pred[13] = r3
	pred[14] = r3
	pred[15] = r3
}
