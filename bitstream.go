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

// putUVMode encodes a UV prediction mode (0=DC, 1=VE, 2=HE, 3=TM) using
// the VP8 probability tree from tree_enc.c PutUVMode().
func putUVMode(bw *boolEncoder, uvMode int) {
	// bit 0: uv_mode != DC_PRED (0)
	if uvMode == 0 {
		bw.putBit(0, 142)
		return
	}
	bw.putBit(1, 142)
	// bit 1: uv_mode != VE (1)
	if uvMode == 1 {
		bw.putBit(0, 114)
		return
	}
	bw.putBit(1, 114)
	// bit 2: uv_mode != HE (2); if true → TM (3)
	bw.putBit(boolInt(uvMode != 2), 183)
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
