// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"image"
	"sync"
)

// yuvImage holds YUV 4:2:0 planar data.
// The Y plane is padded to the next multiple of 16 in both dimensions by
// replicating edge pixels (same as libwebp's WebPPictureAlloc in picture_enc.c).
// yStride / uvStride reflect the padded widths, while width / height store
// the true image dimensions used for VP8 frame header and DC prediction bounds.
type yuvImage struct {
	y        []uint8
	u        []uint8
	v        []uint8
	yStride  int
	uvStride int
	width    int // true image width (encoded in frame header)
	height   int // true image height
	mbW      int // padded width  = (width+15)&^15
	mbH      int // padded height = (height+15)&^15
}

// rgbaToYUV420 converts an image.Image to YUV 4:2:0.
// Uses the same integer arithmetic as libwebp's VP8RGBToY/VP8RGBToU/VP8RGBToV
// in src/dsp/yuv.h (ITU-R BT.601 coefficients).
//
// Y  = (16839*R + 33059*G + 6420*B + YUV_HALF + (16 << 16)) >> 16
// U  = (-9719*R - 19081*G + 28800*B + rounding + (128 << 18)) >> 18
// V  = (28800*R - 24116*G - 4684*B  + rounding + (128 << 18)) >> 18
//
// For chroma, we average 2x2 blocks (sum of 4 pixels, rounding = 1<<(18+2-1) = 1<<19).
//
// The Y plane is allocated at mbW×mbH (next multiples of 16) and the region
// beyond the true image is filled by edge-pixel replication, matching libwebp.
// UV planes are padded to (mbW/2)×(mbH/2).
func rgbaToYUV420(img image.Image) *yuvImage {
	bounds := img.Bounds()
	w := bounds.Max.X - bounds.Min.X
	h := bounds.Max.Y - bounds.Min.Y

	// Pad to next multiple of 16 (macroblock boundary), mirroring libwebp.
	mbW := (w + 15) &^ 15
	mbH := (h + 15) &^ 15

	// UV dimensions: half of padded luma dimensions.
	uvW := mbW / 2
	uvH := mbH / 2

	yuv := &yuvImage{
		y:        make([]uint8, mbW*mbH),
		u:        make([]uint8, uvW*uvH),
		v:        make([]uint8, uvW*uvH),
		yStride:  mbW,
		uvStride: uvW,
		width:    w,
		height:   h,
		mbW:      mbW,
		mbH:      mbH,
	}

	const yuvFix = 16
	const yuvHalf = 1 << (yuvFix - 1) // for Y rounding

	ox := bounds.Min.X
	oy := bounds.Min.Y

	// Fill luma plane for the true image region.
	// Each row writes to non-overlapping indices — safe to parallelise.
	var yuvWG sync.WaitGroup
	for py := 0; py < h; py++ {
		yuvWG.Add(1)
		go func(row int) {
			defer yuvWG.Done()
			for px := 0; px < w; px++ {
				r32, g32, b32, _ := img.At(ox+px, oy+row).RGBA()
				// RGBA() returns 16-bit values; convert to 8-bit
				r := int(r32 >> 8)
				g := int(g32 >> 8)
				b := int(b32 >> 8)

				// Y (full resolution)
				luma := 16839*r + 33059*g + 6420*b
				y := (luma + yuvHalf + (16 << yuvFix)) >> yuvFix
				if y > 235 {
					y = 235
				}
				if y < 16 {
					y = 16
				}
				yuv.y[row*mbW+px] = uint8(y)
			}

			// Pad right edge: replicate last valid column for px = w..mbW-1.
			if w < mbW {
				edge := yuv.y[row*mbW+(w-1)]
				for px := w; px < mbW; px++ {
					yuv.y[row*mbW+px] = edge
				}
			}
		}(py)
	}
	yuvWG.Wait()

	// Pad bottom rows: replicate last valid row for py = h..mbH-1.
	if h < mbH {
		lastRow := yuv.y[(h-1)*mbW : h*mbW]
		for py := h; py < mbH; py++ {
			copy(yuv.y[py*mbW:py*mbW+mbW], lastRow)
		}
	}

	// Chroma: average 2x2 blocks.
	// rounding for UV: (1 << (16+2)) >> 1 = 1 << 17 per pixel, summing 4 = 2 << 18
	// libwebp sums r/g/b over 4 pixels then computes UV on the sum.
	// We replicate that: sum the 4 pixels, scale by 4 to match "4 * r" form.
	// VP8ClipUV: uv = (u + rounding + (128 << 18)) >> 18; clamp [0,255]
	// For a 2x2 block summing 4 pixels each 0..255: sum is 0..1020.
	// We use rounding = 2 << 16 to match libwebp's half-pixel rounding in the 4x sum.
	//
	// UV is computed over the padded grid (uvW×uvH). For positions beyond the
	// true chroma dimensions, edge pixels are replicated (clamped below).
	uvWTrue := (w + 1) / 2
	uvHTrue := (h + 1) / 2

	// Each UV row writes to non-overlapping indices — safe to parallelise.
	for bpy := 0; bpy < uvHTrue; bpy++ {
		yuvWG.Add(1)
		go func(row int) {
			defer yuvWG.Done()
			for bpx := 0; bpx < uvWTrue; bpx++ {
				// Gather 4 pixel block (clamp to image boundary)
				var rSum, gSum, bSum int
				for dy := 0; dy < 2; dy++ {
					for dx := 0; dx < 2; dx++ {
						px := bpx*2 + dx
						py := row*2 + dy
						if px >= w {
							px = w - 1
						}
						if py >= h {
							py = h - 1
						}
						r32, g32, b32, _ := img.At(ox+px, oy+py).RGBA()
						rSum += int(r32 >> 8)
						gSum += int(g32 >> 8)
						bSum += int(b32 >> 8)
					}
				}
				// Compute U/V on the 4-pixel sum.
				// Mirrors libwebp WebPConvertRGBA32ToUV_C → VP8RGBToU/V(r,g,b, YUV_HALF<<2)
				// where r,g,b are the 4-pixel accumulated sums.
				// VP8ClipUV(u, rounding) = (u + rounding + (128 << 18)) >> 18
				// rounding = YUV_HALF << 2 = (1<<15) << 2 = 1 << 17
				uRaw := -9719*rSum - 19081*gSum + 28800*bSum
				vRaw := 28800*rSum - 24116*gSum - 4684*bSum
				rounding := 1 << 17 // = YUV_HALF << 2
				u := (uRaw + rounding + (128 << 18)) >> 18
				v := (vRaw + rounding + (128 << 18)) >> 18
				if u < 0 {
					u = 0
				} else if u > 255 {
					u = 255
				}
				if v < 0 {
					v = 0
				} else if v > 255 {
					v = 255
				}
				yuv.u[row*uvW+bpx] = uint8(u)
				yuv.v[row*uvW+bpx] = uint8(v)
			}

			// Pad right edge of this UV row: replicate last valid column.
			if uvWTrue < uvW {
				uEdge := yuv.u[row*uvW+(uvWTrue-1)]
				vEdge := yuv.v[row*uvW+(uvWTrue-1)]
				for bpx := uvWTrue; bpx < uvW; bpx++ {
					yuv.u[row*uvW+bpx] = uEdge
					yuv.v[row*uvW+bpx] = vEdge
				}
			}
		}(bpy)
	}
	yuvWG.Wait()

	// Pad UV bottom rows: replicate last valid row.
	if uvHTrue < uvH {
		lastRowU := yuv.u[(uvHTrue-1)*uvW : uvHTrue*uvW]
		lastRowV := yuv.v[(uvHTrue-1)*uvW : uvHTrue*uvW]
		for bpy := uvHTrue; bpy < uvH; bpy++ {
			copy(yuv.u[bpy*uvW:bpy*uvW+uvW], lastRowU)
			copy(yuv.v[bpy*uvW:bpy*uvW+uvW], lastRowV)
		}
	}

	return yuv
}

// clip8 clamps v to [0, 255].
func clip8(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}
