// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

// Package lossy implements a VP8 lossy WebP encoder in pure Go.
package gowebp

// boolEncoder is VP8's arithmetic (boolean) coder.
// It operates on MSB-first byte output with carry propagation.
//
// Ported from libwebp src/utils/bit_writer_utils.c — VP8PutBit / Flush.
type boolEncoder struct {
	range_  int32 // range - 1 (starts at 254)
	value   int32
	run     int // pending 0xff bytes (carry propagation)
	nbBits  int // pending bits in value (<= 0 means flush pending)
	buf     []byte
}

// kNorm[i] = 8 - floor(log2(i)) for i in [0..127]
// Used for renormalization shift amount.
var kNorm = [128]uint8{
	7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
}

// kNewRange[i] = ((i + 1) << kNorm[i]) - 1
// The new range after renormalizing range i.
var kNewRange = [128]uint8{
	127, 127, 191, 127, 159, 191, 223, 127, 143, 159, 175, 191, 207, 223, 239,
	127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
	247, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179,
	183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239,
	243, 247, 251, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149,
	151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179,
	181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209,
	211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239,
	241, 243, 245, 247, 249, 251, 253, 127,
}

func newBoolEncoder() *boolEncoder {
	return &boolEncoder{
		range_: 255 - 1, // = 254
		value:  0,
		run:    0,
		nbBits: -8,
		buf:    make([]byte, 0, 4096),
	}
}

func newBoolEncoderSized(cap int) *boolEncoder {
	return &boolEncoder{
		range_: 255 - 1,
		value:  0,
		run:    0,
		nbBits: -8,
		buf:    make([]byte, 0, cap),
	}
}

// flush writes any pending full byte to buf, with carry propagation.
// Mirrors Flush() in bit_writer_utils.c.
func (e *boolEncoder) flush() {
	s := 8 + e.nbBits
	bits := e.value >> s
	e.value -= bits << s
	e.nbBits -= 8
	if bits&0xff != 0xff {
		// Fast path: ~99.6% of calls — no carry, no pending 0xFF run.
		if e.run == 0 && bits&0x100 == 0 {
			e.buf = append(e.buf, byte(bits))
			return
		}
		// Slow path: carry propagation or pending 0xFF bytes.
		pos := len(e.buf)
		e.buf = append(e.buf, make([]byte, e.run+1)...) // bulk extend, zero-filled
		if bits&0x100 != 0 {                            // carry: propagate into previous byte
			if pos > 0 {
				e.buf[pos-1]++
			}
			// pending run bytes become 0x00 — already zero from make, skip fill
		} else if e.run > 0 {
			for i := 0; i < e.run; i++ {
				e.buf[pos+i] = 0xff
			}
		}
		e.buf[pos+e.run] = byte(bits & 0xff)
		e.run = 0
	} else {
		e.run++ // delay writing 0xff bytes pending possible carry
	}
}

// putBit encodes a single bit with given probability (0..255).
// prob is the probability of bit=0; prob=128 is uniform.
// Mirrors VP8PutBit() in bit_writer_utils.c.
func (e *boolEncoder) putBit(bit int, prob int) {
	split := (e.range_ * int32(prob)) >> 8
	if bit != 0 {
		e.value += split + 1
		e.range_ -= split + 1
	} else {
		e.range_ = split
	}
	if e.range_ < 127 {
		shift := kNorm[e.range_]
		e.range_ = int32(kNewRange[e.range_])
		e.value <<= shift
		e.nbBits += int(shift)
		if e.nbBits > 0 {
			e.flush()
		}
	}
}

// putBitUniform encodes a bit with probability 1/2 (range >> 1).
// Mirrors VP8PutBitUniform() in bit_writer_utils.c.
func (e *boolEncoder) putBitUniform(bit int) {
	split := e.range_ >> 1
	if bit != 0 {
		e.value += split + 1
		e.range_ -= split + 1
	} else {
		e.range_ = split
	}
	if e.range_ < 127 {
		e.range_ = int32(kNewRange[e.range_])
		e.value <<= 1
		e.nbBits++
		if e.nbBits > 0 {
			e.flush()
		}
	}
}

// putBits encodes value using nb_bits uniform bits, MSB first.
// Mirrors VP8PutBits() in bit_writer_utils.c.
func (e *boolEncoder) putBits(value uint32, nbBits int) {
	for mask := uint32(1) << uint(nbBits-1); mask != 0; mask >>= 1 {
		b := 0
		if value&mask != 0 {
			b = 1
		}
		e.putBitUniform(b)
	}
}

// putSignedBits encodes a signed value using nbBits.
// Mirrors VP8PutSignedBits() in bit_writer_utils.c.
func (e *boolEncoder) putSignedBits(value int, nbBits int) {
	if value == 0 {
		e.putBitUniform(0)
		return
	}
	e.putBitUniform(1)
	if value < 0 {
		e.putBits(uint32((-value)<<1)|1, nbBits+1)
	} else {
		e.putBits(uint32(value<<1), nbBits+1)
	}
}

// finish flushes remaining bits and returns the encoded bytes.
// Mirrors VP8BitWriterFinish() in bit_writer_utils.c.
func (e *boolEncoder) finish() []byte {
	// Pad with 9-nbBits zeros
	pad := 9 - e.nbBits
	if pad < 0 {
		pad = 0
	}
	e.putBits(0, pad)
	e.nbBits = 0
	e.flush()
	return e.buf
}
