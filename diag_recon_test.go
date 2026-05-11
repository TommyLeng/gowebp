package gowebp

import (
	"fmt"
	"testing"
)

func TestReconTracy(t *testing.T) {
	// Test: flat block Y=136, pred=128. Should reconstruct to ~136.
	// (Same as gray 128 → Y=126 test passes, let's verify with Y=136)
	qm := buildQuantMatrices(90)

	// Simulate a single 4x4 block with all pixels = 136, pred = 128
	var src4 [16]int16
	var pred4 [16]int16
	for i := range src4 {
		src4[i] = 136
		pred4[i] = 128
	}

	var dctOut [16]int16
	fTransform(src4[:], pred4[:], dctOut[:])
	fmt.Printf("DCT[0]=%d (DC raw)\n", dctOut[0])

	// Collect DC for WHT (just one block, 16 identical)
	var yDcRaw [16]int16
	for i := range yDcRaw {
		yDcRaw[i] = dctOut[0]
	}
	dctOut[0] = 0

	var acQ [16]int16
	quantizeBlock(dctOut[:], acQ[:], &qm.y1, 1)
	fmt.Printf("AC levels: %v\n", acQ)

	var whtOut [16]int16
	fTransformWHT(yDcRaw[:], whtOut[:])
	fmt.Printf("WHT[0]=%d\n", whtOut[0])

	var dcQ [16]int16
	quantizeBlockWHT(whtOut[:], dcQ[:], &qm.y2)
	fmt.Printf("WHT quant[0]=%d (step=%d)\n", dcQ[0], qm.y2.q[0])

	// Dequantize WHT
	var whtDequant [16]int16
	for n := 0; n < 16; n++ {
		j := int(kZigzag[n])
		whtDequant[n] = int16(int32(dcQ[n]) * int32(qm.y2.q[j]))
	}
	fmt.Printf("WHT dequant[0]=%d\n", whtDequant[0])

	// Inverse WHT → per-block DC coeffs
	var dcBlockCoeffs [16]int16
	inverseWHT16(whtDequant[:], dcBlockCoeffs[:])
	fmt.Printf("dcBlockCoeffs[0]=%d (should be ~%d)\n", dcBlockCoeffs[0], dctOut[0])

	// Now reconstruct block 0
	var rasterCoeffs [16]int16
	dequantizeBlock(acQ[:], rasterCoeffs[:], &qm.y1, dcBlockCoeffs[0])
	fmt.Printf("rasterCoeffs[0]=%d (DC in raster)\n", rasterCoeffs[0])

	var recBlock [16]int16
	iTransform4x4(rasterCoeffs[:], pred4[:], recBlock[:])
	fmt.Printf("Reconstructed pixels[0..3]: %v\n", recBlock[:4])
	fmt.Printf("Expected: ~136 for all\n")
	
	// Also check: what does the decoder compute?
	// Decoder: dequant WHT[0] = dcQ[0]*step = whtDequant[0]
	// inverseWHT16: dcBlockCoeffs[0] = result above
	// inverseDCT4DCOnly: dc = (dcBlockCoeffs[0]+4)>>3, add to pred
	dcDCOnly := (int32(dcBlockCoeffs[0]) + 4) >> 3
	fmt.Printf("DC-only decoded pixel = pred(%d) + %d = %d\n", pred4[0], dcDCOnly, int32(pred4[0])+dcDCOnly)
}
