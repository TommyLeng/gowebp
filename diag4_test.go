package gowebp

import (
	"fmt"
	"testing"
)

func TestQuantLevel(t *testing.T) {
	for _, q := range []int{10, 50, 75, 90, 100} {
		level := qualityToLevel(q)
		qm := buildQuantMatrices(q)
		fmt.Printf("quality=%d -> level=%d, Y1_DC=%d Y1_AC=%d UV_DC=%d UV_AC=%d Y2_DC=%d Y2_AC=%d\n",
			q, level, qm.y1.q[0], qm.y1.q[1], qm.uv.q[0], qm.uv.q[1], qm.y2.q[0], qm.y2.q[1])
	}
}
