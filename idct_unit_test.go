package gowebp

import "testing"

// TestIDCTZeroCoeffsConst127: iTransform4x4(zeros, all 127) → all 127.
func TestIDCTZeroCoeffsConst127(t *testing.T) {
	coeffs := make([]int16, 16)
	pred := make([]int16, 16)
	for i := range pred {
		pred[i] = 127
	}
	out := make([]int16, 16)
	iTransform4x4(coeffs, pred, out)
	for i, v := range out {
		if v != 127 {
			t.Errorf("out[%d] = %d, want 127", i, v)
		}
	}
}

// TestIDCTZeroCoeffsConst128: zeros + pred=128 → 128.
func TestIDCTZeroCoeffsConst128(t *testing.T) {
	coeffs := make([]int16, 16)
	pred := make([]int16, 16)
	for i := range pred {
		pred[i] = 128
	}
	out := make([]int16, 16)
	iTransform4x4(coeffs, pred, out)
	for i, v := range out {
		if v != 128 {
			t.Errorf("out[%d] = %d, want 128", i, v)
		}
	}
}

// TestIDCTZeroCoeffsConst129: zeros + pred=129 → 129.
func TestIDCTZeroCoeffsConst129(t *testing.T) {
	coeffs := make([]int16, 16)
	pred := make([]int16, 16)
	for i := range pred {
		pred[i] = 129
	}
	out := make([]int16, 16)
	iTransform4x4(coeffs, pred, out)
	for i, v := range out {
		if v != 129 {
			t.Errorf("out[%d] = %d, want 129", i, v)
		}
	}
}

// TestIDCTDCMinus63Pred43: iTransform4x4([-63,0,0,...], pred=43) → 35.
// Matches the encoder's actual reconstruction for a real face-frame MB.
func TestIDCTDCMinus63Pred43(t *testing.T) {
	coeffs := make([]int16, 16)
	coeffs[0] = -63
	pred := make([]int16, 16)
	for i := range pred {
		pred[i] = 43
	}
	out := make([]int16, 16)
	iTransform4x4(coeffs, pred, out)
	for i, v := range out {
		if v != 35 {
			t.Errorf("out[%d] = %d, want 35", i, v)
		}
	}
}
