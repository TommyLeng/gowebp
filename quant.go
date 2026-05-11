// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// kDcTable maps quantizer index [0..127] -> DC quantizer step size.
// From libwebp/src/enc/quant_enc.c.
var kDcTable = [128]uint8{
	4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17,
	17, 18, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26,
	27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40,
	41, 42, 43, 44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54,
	55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
	70, 71, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83,
	84, 85, 86, 87, 88, 89, 91, 93, 95, 96, 98, 100, 101, 102, 104,
	106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130, 132, 134, 136,
	138, 140, 143, 145, 148, 151, 154, 157,
}

// kAcTable maps quantizer index [0..127] -> AC quantizer step size.
var kAcTable = [128]uint16{
	4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
	19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
	34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
	49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68,
	70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98,
	100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134,
	137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, 181,
	185, 189, 193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 234, 239, 245,
	249, 254, 259, 264, 269, 274, 279, 284,
}

// kAcTable2 is the AC table for Y2 (WHT plane).
var kAcTable2 = [128]uint16{
	8, 8, 9, 10, 12, 13, 15, 17, 18, 20, 21, 23, 24, 26, 27,
	29, 31, 32, 34, 35, 37, 38, 40, 41, 43, 44, 46, 48, 49, 51,
	52, 54, 55, 57, 58, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74,
	75, 77, 79, 80, 82, 83, 85, 86, 88, 89, 93, 96, 99, 102, 105,
	108, 111, 114, 117, 120, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151,
	155, 158, 161, 164, 167, 170, 173, 176, 179, 184, 189, 193, 198, 203, 207,
	212, 217, 221, 226, 230, 235, 240, 244, 249, 254, 258, 263, 268, 274, 280,
	286, 292, 299, 305, 311, 317, 323, 330, 336, 342, 348, 354, 362, 370, 379,
	385, 393, 401, 409, 416, 424, 432, 440,
}

// qfix is the fixed-point precision for quantizer reciprocals.
const qfix = 17

// biasAC is the AC rounding bias numerator (shifts to QFIX-8=9 bits).
// libwebp uses:
//   type 0 (Y1 AC): bias = 110 << (17-8) = 110 << 9 = 56320
//   type 1 (Y2 DC): bias = 96 << 9 = 49152
//   type 2 (UV):    bias = 115 << 9 = 58880
// We simplify to a single function that takes the bias parameter.

// quantMatrix holds the per-block quantization parameters.
type quantMatrix struct {
	q       [16]uint16 // step sizes (q[0]=DC, q[1..15]=AC)
	iq      [16]uint32 // reciprocals: (1<<QFIX) / q[i]
	bias    [16]uint32 // rounding bias
	zthresh [16]uint32 // zero-threshold: coeff <= zthresh -> quantized to 0
	sharpen [16]int16  // sharpening bias added to coeff before trellis (luma AC only)
}

// kFreqSharpening[j] is the sharpening weight for raster position j.
// Only used for y1 (luma AC, type=0). Zero for y2 and uv.
// From libwebp/src/enc/quant_enc.c kFreqSharpening[].
var kFreqSharpening = [16]int{0, 30, 60, 90, 30, 60, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90}

const sharpenBits = 11 // descaling shift for sharpening bias

// quantMatrices holds Y1 (luma AC), Y2 (WHT/DC), UV quantization matrices.
type quantMatrices struct {
	y1 quantMatrix
	y2 quantMatrix
	uv quantMatrix
}

// clipQ clips q to [lo, hi].
func clipQ(q, lo, hi int) int {
	if q < lo {
		return lo
	}
	if q > hi {
		return hi
	}
	return q
}

// setupMatrix initializes a quantMatrix given its step sizes q[0] (DC) and q[1] (AC).
// biasType: 0=luma-AC, 1=luma-DC(y2), 2=chroma
// From libwebp's ExpandMatrix().
func setupMatrix(m *quantMatrix, biasType int) {
	// bias table: [luma-AC, luma-DC, chroma][dc, ac]
	biasDC := [3]uint32{96, 96, 110}
	biasAC := [3]uint32{110, 108, 115}

	for i := 0; i < 16; i++ {
		isAC := 0
		if i > 0 {
			isAC = 1
		}
		var bias uint32
		if isAC == 0 {
			bias = biasDC[biasType]
		} else {
			bias = biasAC[biasType]
		}
		// iq = (1 << QFIX) / q
		m.iq[i] = (1 << qfix) / uint32(m.q[i])
		// bias in QFIX-8 = 9 extra bits
		m.bias[i] = bias << (qfix - 8)
		// zthresh: largest coeff that rounds to 0
		// zthresh = ((1<<QFIX) - 1 - bias) / iq
		m.zthresh[i] = ((1 << qfix) - 1 - m.bias[i]) / m.iq[i]
		// sharpen: frequency-dependent boost for luma AC (biasType==0 only).
		// Mirrors ExpandMatrix() in libwebp: sharpen[i] = (kFreqSharpening[i] * q[i]) >> SHARPEN_BITS
		if biasType == 0 {
			m.sharpen[i] = int16((kFreqSharpening[i] * int(m.q[i])) >> sharpenBits)
		} else {
			m.sharpen[i] = 0
		}
	}
}

// qualityToLevel maps quality [0..100] to a VP8 quantizer index [0..127].
// Ported from VP8SetSegmentParams() in libwebp/src/enc/quant_enc.c.
// For Phase 1 (no SNS, no multi-segment), using QualityToCompression().
func qualityToLevel(quality int) int {
	q := float64(quality) / 100.0
	// QualityToCompression: linear_c = q*(2/3) if q < 0.75, else 2q-1
	var linearC float64
	if q < 0.75 {
		linearC = q * (2.0 / 3.0)
	} else {
		linearC = 2.0*q - 1.0
	}
	// v = linearC^(1/3)
	v := cubeRoot(linearC)
	// quant = int(127 * (1 - v))
	quant := int(127.0 * (1.0 - v))
	return clipQ(quant, 0, 127)
}

// cubeRoot approximates x^(1/3) via Newton's method.
func cubeRoot(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Initial estimate
	r := x
	for i := 0; i < 50; i++ {
		r = (2.0*r + x/(r*r)) / 3.0
	}
	return r
}

// buildQuantMatrices constructs all three quantization matrices from a quality level.
// Mirrors SetupMatrices() in libwebp/src/enc/quant_enc.c for 1-segment case.
func buildQuantMatrices(quality int) quantMatrices {
	q := qualityToLevel(quality)
	var qm quantMatrices

	// Y1: luma AC blocks
	// q[0] = kDcTable[q] (DC step), q[1] = kAcTable[q] (AC step)
	qDC := int(kDcTable[clipQ(q, 0, 127)])
	qAC := int(kAcTable[clipQ(q, 0, 127)])
	qm.y1.q[0] = uint16(qDC)
	for i := 1; i < 16; i++ {
		qm.y1.q[i] = uint16(qAC)
	}
	setupMatrix(&qm.y1, 0)

	// Y2: DC/WHT plane
	// q[0] = kDcTable[q]*2, q[1] = kAcTable2[q]
	qm.y2.q[0] = uint16(qDC * 2)
	qAC2 := int(kAcTable2[clipQ(q, 0, 127)])
	for i := 1; i < 16; i++ {
		qm.y2.q[i] = uint16(qAC2)
	}
	setupMatrix(&qm.y2, 1)

	// UV: chroma
	// q[0] = kDcTable[clipQ(q,0,117)], q[1] = kAcTable[q]
	qDCuv := int(kDcTable[clipQ(q, 0, 117)])
	qACuv := int(kAcTable[clipQ(q, 0, 127)])
	qm.uv.q[0] = uint16(qDCuv)
	for i := 1; i < 16; i++ {
		qm.uv.q[i] = uint16(qACuv)
	}
	setupMatrix(&qm.uv, 2)

	return qm
}

// quantizeBlock quantizes a 16-element DCT block using the given matrix,
// producing output in zigzag scan order (matching QuantizeBlock_C in libwebp).
//
// in[16] contains DCT coefficients in 4×4 raster order.
// out[n] = quantized level for zigzag scan position n.
// The DC slot is kZigzag[0]=0, so out[0] = DC level.
//
// 'first': 0 = quantize all 16 (i4 or WHT), 1 = skip DC (i16 AC blocks,
//   DC goes to WHT instead). For first=1, out[0] is set to 0.
//
// Ported from QuantizeBlock_C in libwebp/src/dsp/enc.c.
func quantizeBlock(in []int16, out []int16, m *quantMatrix, first int) bool {
	nonZero := false
	for n := 0; n < 16; n++ {
		j := int(kZigzag[n]) // raster position of zigzag slot n
		if n < first {
			// Skip (DC for i16 AC blocks goes to WHT instead)
			out[n] = 0
			continue
		}
		v := int32(in[j])
		sign := 0
		if v < 0 {
			sign = 1
			v = -v
		}
		qv := (uint32(v)*m.iq[j] + m.bias[j]) >> qfix
		if sign != 0 {
			out[n] = -int16(qv)
		} else {
			out[n] = int16(qv)
		}
		if qv != 0 {
			nonZero = true
		}
	}
	return nonZero
}

// quantizeBlockWHT quantizes the 16 WHT coefficients.
// Uses the y2 quantization matrix.
func quantizeBlockWHT(in []int16, out []int16, m *quantMatrix) bool {
	return quantizeBlock(in, out, m, 0)
}

// dequantizeBlock converts quantized zigzag-order levels back to raster-order
// DCT coefficients by multiplying each level by its quantization step size.
// The DC slot (zigzag[0]=raster[0]) is set to dcVal (already in raster space).
// This mirrors the decoder: c = level * quant[DC_or_AC].
func dequantizeBlock(qlevels []int16, raster []int16, m *quantMatrix, dcVal int16) {
	for i := range raster {
		raster[i] = 0
	}
	raster[0] = dcVal // DC comes from inverse WHT
	for n := 1; n < 16; n++ {
		j := int(kZigzag[n]) // raster position
		level := int32(qlevels[n])
		raster[j] = int16(level * int32(m.q[j]))
	}
}
