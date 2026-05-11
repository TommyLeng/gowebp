// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"encoding/binary"
	"io"
)

// writeLE32 writes a little-endian uint32.
func writeLE32(w io.Writer, v uint32) error {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	_, err := w.Write(b[:])
	return err
}

// writeLE16 writes a little-endian uint16.
func writeLE16(w io.Writer, v uint16) error {
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], v)
	_, err := w.Write(b[:])
	return err
}

// webpHeader writes the RIFF/WEBP container around a VP8 bitstream.
//
// Structure:
//   RIFF <riff_size> WEBP
//   VP8  <vp8_size>
//   <vp8_data>
//
// vp8Data = VP8 frame header (10 bytes) + partition0 + token_partition.
func writeWebPHeader(w io.Writer, vp8Data []byte) error {
	vp8Size := uint32(len(vp8Data))
	// RIFF size = 4 (WEBP) + 8 (VP8 chunk header) + vp8Size
	riffSize := 4 + 8 + vp8Size
	// padding: chunk sizes must be even
	padding := vp8Size & 1

	// RIFF header
	if _, err := w.Write([]byte{'R', 'I', 'F', 'F'}); err != nil {
		return err
	}
	if err := writeLE32(w, riffSize+padding); err != nil {
		return err
	}
	if _, err := w.Write([]byte{'W', 'E', 'B', 'P'}); err != nil {
		return err
	}

	// VP8 chunk header
	if _, err := w.Write([]byte{'V', 'P', '8', ' '}); err != nil {
		return err
	}
	if err := writeLE32(w, vp8Size); err != nil {
		return err
	}

	// VP8 data
	if _, err := w.Write(vp8Data); err != nil {
		return err
	}
	// Padding byte if needed
	if padding != 0 {
		if _, err := w.Write([]byte{0}); err != nil {
			return err
		}
	}
	return nil
}

// buildVP8FrameHeader builds the 10-byte VP8 frame header.
//
// Layout (paragraph 9.1 of VP8 spec):
//   byte[0..2]: frame tag (3 bytes, little-endian)
//     bit 0:     key_frame (0 = key frame)
//     bits 1..3: version (0 = bicubic, no loop filter)
//     bit 4:     show_frame (1)
//     bits 5..23: first_part_size (size of partition 0)
//   byte[3..5]: start code 0x9D 0x01 0x2A
//   byte[6..7]: width  (little-endian uint16, top 2 bits = horiz scale)
//   byte[8..9]: height (little-endian uint16, top 2 bits = vert scale)
//
// profile 0 = bicubic reconstruction filter, no loop filtering.
func buildVP8FrameHeader(width, height int, part0Size int) []byte {
	hdr := make([]byte, 10)
	profile := 0 // version 0
	// bits: 0 (keyframe) | profile<<1 | 1<<4 (show) | part0Size<<5
	bits := uint32(0) | uint32(profile)<<1 | (1 << 4) | (uint32(part0Size) << 5)
	hdr[0] = byte(bits)
	hdr[1] = byte(bits >> 8)
	hdr[2] = byte(bits >> 16)
	// VP8 signature / start code
	hdr[3] = 0x9D
	hdr[4] = 0x01
	hdr[5] = 0x2A
	// Width: low 14 bits = width, top 2 bits = horiz_scale (0)
	hdr[6] = byte(width & 0xff)
	hdr[7] = byte(width >> 8) // scale bits are 0
	// Height: same encoding
	hdr[8] = byte(height & 0xff)
	hdr[9] = byte(height >> 8)
	return hdr
}

// encodePartition0 encodes the VP8 partition 0 (headers + intra modes).
//
// Partition 0 structure (paragraph 9.3):
//   colorspace (1 bit, 0)
//   clamp_type  (1 bit, 0)
//   segmentation_enabled (1 bit, 0 for Phase 1)
//   filter_type (1 bit, 0 = simple)
//   filter_level (6 bits, 0 = disabled)
//   filter_sharpness (3 bits, 0)
//   loop_filter_adj_enable (1 bit, 0)
//   log2_nbr_of_dct_partitions (2 bits, 0 = 1 partition)
//   quant_indices (7 bits base_q + 5*4-bit delta, all 0 deltas)
//   refresh_entropy_probs (1 bit, 0 = use defaults)
//   token_probs (written only if refresh_entropy_probs=1, skip)
//   MB data (intra modes for all macroblocks)
func encodePartition0(bw *boolEncoder, mbW, mbH, baseQ int, mbModes []int) {
	// colorspace = 0, clamp_type = 0
	bw.putBitUniform(0) // colorspace
	bw.putBitUniform(0) // clamp_type

	// Segment header: no segmentation (1 bit = 0)
	bw.putBitUniform(0) // update_segment = 0

	// Filter header
	bw.putBitUniform(0) // filter_type = 0 (simple)
	bw.putBits(0, 6)    // filter_level = 0
	bw.putBits(0, 3)    // filter_sharpness = 0
	bw.putBitUniform(0) // loop_filter_adj_enable = 0

	// Number of DCT partitions: log2(1) = 0
	bw.putBits(0, 2)

	// Quantizer indices: base_q (7 bits), then 5 delta values (each 4-bit signed)
	bw.putBits(uint32(baseQ), 7)
	bw.putSignedBits(0, 4) // y1_dc_delta = 0
	bw.putSignedBits(0, 4) // y2_dc_delta = 0
	bw.putSignedBits(0, 4) // y2_ac_delta = 0
	bw.putSignedBits(0, 4) // uv_dc_delta = 0
	bw.putSignedBits(0, 4) // uv_ac_delta = 0

	// refreshLastFrameBuffer = 0 (key frame: don't refresh golden/altref)
	// This is a single uniform bit in partition 0 before token probs.
	bw.putBitUniform(0)

	// Write token probability update bits.
	// For each [type][band][ctx][prob]: emit 0 (no update) with the update probability.
	// Since we use default tables, all updates are 0.
	// Mirrors VP8WriteProbas() in libwebp/src/enc/tree_enc.c.
	for t := 0; t < numTypes; t++ {
		for b := 0; b < numBands; b++ {
			for c := 0; c < numCtx; c++ {
				for p := 0; p < numProbas; p++ {
					// Emit 0 = "no update" using the update probability
					bw.putBit(0, int(coeffsUpdateProba[t][b][c][p]))
				}
			}
		}
	}

	// use_skip_proba = 0 (don't use per-MB skip probability)
	bw.putBitUniform(0)

	// Encode intra modes for all macroblocks.
	// For each MB: skip_proba not used (no skip_proba in partition 0 for now).
	// Then: mb_type (1 bit: 0=i4x4, 1=i16x16).
	// For i16x16: intra_mode (2-3 bits via tree).
	// For UV: uv_mode (2-3 bits via tree).
	// We use all i16 DC prediction (mode 0 = DC_PRED).
	for i := 0; i < mbW*mbH; i++ {
		mode := mbModes[i] // always 0 (DC_PRED) for Phase 1
		_ = mode

		// mb_type = 1 (i16x16)
		bw.putBit(1, 145)

		// i16 intra mode encoding (PutI16Mode in tree_enc.c):
		// if VP8PutBit(bw, mode==TM_PRED||mode==H_PRED, 156):
		//   VP8PutBit(bw, mode==TM_PRED, 128)  // TM=1 or HE=0
		// else:
		//   VP8PutBit(bw, mode==V_PRED, 163)    // VE=1 or DC=0
		// DC_PRED = 0, TM_PRED = 1, V_PRED = 2, H_PRED = 3
		// For DC_PRED (mode=0): first branch false, second branch false
		bw.putBit(0, 156) // mode != TM_PRED && mode != H_PRED
		bw.putBit(0, 163) // mode != V_PRED (i.e., DC_PRED)

		// UV mode encoding (PutUVMode in tree_enc.c):
		// VP8PutBit(bw, uv_mode != DC_PRED, 142)
		// For DC_PRED (uv_mode=0): false
		bw.putBit(0, 142)
	}
}

// encodePartition0Phase2 encodes partition 0 with full i4/i16 mode support.
// Mirrors VP8CodeIntraModes() in libwebp/src/enc/tree_enc.c.
func encodePartition0Phase2(bw *boolEncoder, mbW, mbH, baseQ int, infos []mbInfo) {
	// colorspace = 0, clamp_type = 0
	bw.putBitUniform(0)
	bw.putBitUniform(0)

	// Segment header: no segmentation
	bw.putBitUniform(0)

	// Filter header
	bw.putBitUniform(0) // filter_type = 0 (simple)
	bw.putBits(0, 6)    // filter_level = 0
	bw.putBits(0, 3)    // filter_sharpness = 0
	bw.putBitUniform(0) // loop_filter_adj_enable = 0

	// Number of DCT partitions: log2(1) = 0
	bw.putBits(0, 2)

	// Quantizer indices
	bw.putBits(uint32(baseQ), 7)
	bw.putSignedBits(0, 4) // y1_dc_delta
	bw.putSignedBits(0, 4) // y2_dc_delta
	bw.putSignedBits(0, 4) // y2_ac_delta
	bw.putSignedBits(0, 4) // uv_dc_delta
	bw.putSignedBits(0, 4) // uv_ac_delta

	// refreshLastFrameBuffer = 0
	bw.putBitUniform(0)

	// Token probability updates: all 0 (use defaults)
	for t := 0; t < numTypes; t++ {
		for b := 0; b < numBands; b++ {
			for c := 0; c < numCtx; c++ {
				for p := 0; p < numProbas; p++ {
					bw.putBit(0, int(coeffsUpdateProba[t][b][c][p]))
				}
			}
		}
	}

	// use_skip_proba = 0
	bw.putBitUniform(0)

	// Per-MB mode encoding.
	// We need to track the top and left i4 mode contexts across MBs.
	// topI4[mbX*4+bx] = bottom-row mode of previously encoded MB above.
	// leftI4[by] = right-column mode of previously encoded MB to the left.
	topI4 := make([]int, mbW*4) // initialized to B_DC_PRED = 0

	for mbY := 0; mbY < mbH; mbY++ {
		leftI4 := [4]int{} // B_DC_PRED = 0 for all

		for mbX := 0; mbX < mbW; mbX++ {
			info := &infos[mbY*mbW+mbX]

			if info.isI4 {
				// mb_type = 0 (i4x4): VP8PutBit(bw, 0, 145)
				bw.putBit(0, 145)

				// Encode 16 sub-block modes in raster order.
				// Each mode uses kBModesProba[top_pred][left_pred].
				// top_pred comes from the block above; left_pred from the block to the left.
				topPred := make([]int, 4)
				copy(topPred, topI4[mbX*4:mbX*4+4])

				for by := 0; by < 4; by++ {
					leftPred := leftI4[by]
					if by == 0 {
						// first row: left of bx=0 comes from leftI4[0]
						leftPred = leftI4[0]
					}
					for bx := 0; bx < 4; bx++ {
						blkIdx := by*4 + bx
						mode := info.i4Modes[blkIdx]

						top := topPred[bx]
						left := leftPred

						putI4Mode(bw, mode, top, left)

						// Update left for next block in this row
						leftPred = mode
						// Update top for next row
						topPred[bx] = mode
					}
					// After row by: leftI4[by] was the rightmost mode in this row
					// (used as left context for bx=0 of the next MB's by row — but
					// that's tracked in leftI4 per row for the next MB).
					// Actually leftI4[by] = mode of (bx=3, by) for next MB's bx=0 at row by.
					leftI4[by] = info.i4Modes[by*4+3]
				}

				// Update topI4 with the bottom row of this MB
				for bx := 0; bx < 4; bx++ {
					topI4[mbX*4+bx] = info.i4Modes[3*4+bx]
				}

			} else {
				// mb_type = 1 (i16x16): VP8PutBit(bw, 1, 145)
				bw.putBit(1, 145)

				// PutI16Mode — maps our constants to libwebp's tree:
				// libwebp: DC=0, TM=1, VE=2, HE=3
				// Our: I16_DC=0, I16_TM=1, I16_VE=2, I16_HE=3
				// Tree (PutI16Mode in tree_enc.c):
				//   VP8PutBit(bw, mode==TM||mode==HE, 156)
				//   if true:  VP8PutBit(bw, mode==TM, 128)
				//   if false: VP8PutBit(bw, mode==VE, 163)
				mode := info.i16Mode
				isTMorHE := mode == I16_TM_PRED || mode == I16_HE_PRED
				bw.putBit(boolInt(isTMorHE), 156)
				if isTMorHE {
					bw.putBit(boolInt(mode == I16_TM_PRED), 128)
				} else {
					bw.putBit(boolInt(mode == I16_VE_PRED), 163)
				}

				// i16 MB: set i4 context to the i16 mode (matches VP8 spec).
				// VP8's parsePredModeY16 sets d.upMB.pred[i] = d.leftMB.pred[j] = i16Mode.
				for bx := 0; bx < 4; bx++ {
					topI4[mbX*4+bx] = info.i16Mode
				}
				for by := 0; by < 4; by++ {
					leftI4[by] = info.i16Mode
				}
			}

			// UV mode encoding (PutUVMode in tree_enc.c):
			// VP8PutBit(bw, uv_mode != DC_PRED, 142)
			// if true:
			//   VP8PutBit(bw, uv_mode != V_PRED, 114)
			//   if true: VP8PutBit(bw, uv_mode != H_PRED, 183)  // else TM
			// We always use DC (uvMode=0).
			bw.putBit(0, 142) // uv_mode == DC_PRED
		}
	}
}

// putCoeffs encodes one set of DCT coefficients using the bool encoder.
// Mirrors PutCoeffs() in libwebp/src/enc/frame_enc.c.
// coeff_type: 0=i16-AC, 1=i16-DC, 2=chroma-AC, 3=i4-AC
// first: 0 for most blocks, 1 for i16 AC (DC is in WHT)
// ctx: context for first coefficient (sum of top_nz + left_nz)
// Returns true if any non-zero coefficient was coded.
func putCoeffs(bw *boolEncoder, ctx int, coeffs []int16, coeffType int, first int, last int) bool {
	n := first
	p := &defaultCoeffProbs[coeffType][n][ctx]

	// Emit: is there any non-zero coeff? (eob_prob)
	if last < 0 {
		// All zeros: emit "no coeff" (EOB at start)
		bw.putBit(0, int(p[0]))
		return false
	}
	bw.putBit(1, int(p[0]))

	for n < 16 {
		c := int(coeffs[n])
		n++
		sign := 0
		v := c
		if c < 0 {
			sign = 1
			v = -c
		}

		// Emit: is this coeff non-zero?
		if v == 0 {
			bw.putBit(0, int(p[1]))
			band := int(vp8EncBands[n])
			p = &defaultCoeffProbs[coeffType][band][0]
			continue
		}
		bw.putBit(1, int(p[1]))

		// Emit: is v > 1?
		if v == 1 {
			bw.putBit(0, int(p[2]))
			band := int(vp8EncBands[n])
			p = &defaultCoeffProbs[coeffType][band][1]
		} else {
			bw.putBit(1, int(p[2]))

			// Emit: is v > 4?
			if v <= 4 {
				bw.putBit(0, int(p[3]))
				// Emit: is v != 2? (i.e., 3 or 4)
				if v == 2 {
					bw.putBit(0, int(p[4]))
				} else {
					bw.putBit(1, int(p[4]))
					// Emit: is v == 4?
					eq4 := 0
					if v == 4 {
						eq4 = 1
					}
					bw.putBit(eq4, int(p[5]))
				}
			} else if v <= 10 {
				bw.putBit(1, int(p[3]))
				bw.putBit(0, int(p[6]))
				// Emit: is v > 6?
				if v <= 6 {
					bw.putBit(0, int(p[7]))
					// VP8PutBit(bw, v==6, 159)
					eq6 := 0
					if v == 6 {
						eq6 = 1
					}
					bw.putBit(eq6, 159)
				} else {
					bw.putBit(1, int(p[7]))
					// VP8PutBit(bw, v>=9, 165)
					ge9 := 0
					if v >= 9 {
						ge9 = 1
					}
					bw.putBit(ge9, 165)
					// VP8PutBit(bw, !(v&1), 145)
					even := 0
					if v&1 == 0 {
						even = 1
					}
					bw.putBit(even, 145)
				}
			} else {
				bw.putBit(1, int(p[3]))
				bw.putBit(1, int(p[6]))
				// Large value: encode using cat tables
				residue := v - 3
				if residue < (8 << 1) { // Cat3: 3 extra bits, v in [3+8, 3+15]
					bw.putBit(0, int(p[8]))
					bw.putBit(0, int(p[9]))
					residue -= 8
					for i := 2; i >= 0; i-- {
						bit := (residue >> uint(i)) & 1
						bw.putBit(bit, int(vp8Cat3[2-i]))
					}
				} else if residue < (8 << 2) { // Cat4
					bw.putBit(0, int(p[8]))
					bw.putBit(1, int(p[9]))
					residue -= 8 << 1
					for i := 3; i >= 0; i-- {
						bit := (residue >> uint(i)) & 1
						bw.putBit(bit, int(vp8Cat4[3-i]))
					}
				} else if residue < (8 << 3) { // Cat5
					bw.putBit(1, int(p[8]))
					bw.putBit(0, int(p[10]))
					residue -= 8 << 2
					for i := 4; i >= 0; i-- {
						bit := (residue >> uint(i)) & 1
						bw.putBit(bit, int(vp8Cat5[4-i]))
					}
				} else { // Cat6
					bw.putBit(1, int(p[8]))
					bw.putBit(1, int(p[10]))
					residue -= 8 << 3
					for i := 10; i >= 0; i-- {
						bit := (residue >> uint(i)) & 1
						bw.putBit(bit, int(vp8Cat6[10-i]))
					}
				}
			}

			band := int(vp8EncBands[n])
			p = &defaultCoeffProbs[coeffType][band][2]
		}

		// Sign bit (uniform)
		bw.putBitUniform(sign)

		// Is there another non-zero coeff? (or EOB)
		if n == 16 {
			return true
		}
		more := 0
		if n <= last {
			more = 1
		}
		bw.putBit(more, int(p[0]))
		if more == 0 {
			return true
		}
	}
	return true
}

// findLast returns the index of the last non-zero coefficient in coeffs[first:16].
// Returns -1 if all are zero.
func findLast(coeffs []int16, first int) int {
	for i := 15; i >= first; i-- {
		if coeffs[i] != 0 {
			return i
		}
	}
	return -1
}
