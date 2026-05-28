package gowebp

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"strings"
	"testing"

	"golang.org/x/image/webp"
)

// makeTestFrame builds an opaque w×h image with a solid colour. The colour
// varies per frame index so each frame is visibly different — useful for
// verifying that frames are encoded independently.
func makeTestFrame(w, h int, frameIdx int) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	r := uint8((frameIdx * 80) & 0xff)
	g := uint8((frameIdx * 40) & 0xff)
	b := uint8((frameIdx * 20) & 0xff)
	c := color.NRGBA{R: r, G: g, B: b, A: 255}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	return img
}

// parsedChunk is a top-level chunk discovered while walking the WebP body.
type parsedChunk struct {
	tag     string
	size    uint32
	payload []byte
}

// parseWebPChunks parses the RIFF/WEBP wrapper and returns the inner chunks
// in order. Returns an error if the outer wrapper is malformed.
func parseWebPChunks(data []byte) ([]parsedChunk, error) {
	if len(data) < 12 {
		return nil, errEarlyEOF
	}
	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
		return nil, errBadRIFF
	}
	riffSize := binary.LittleEndian.Uint32(data[4:8])
	// riffSize counts everything after the size field. The 4-byte "WEBP"
	// tag is included in riffSize, so the body length is riffSize - 4.
	if int(riffSize)+8 > len(data) {
		return nil, errEarlyEOF
	}
	body := data[12 : 8+riffSize]

	var chunks []parsedChunk
	off := 0
	for off < len(body) {
		if off+8 > len(body) {
			return nil, errEarlyEOF
		}
		tag := string(body[off : off+4])
		sz := binary.LittleEndian.Uint32(body[off+4 : off+8])
		if off+8+int(sz) > len(body) {
			return nil, errEarlyEOF
		}
		chunks = append(chunks, parsedChunk{
			tag:     tag,
			size:    sz,
			payload: body[off+8 : off+8+int(sz)],
		})
		off += 8 + int(sz)
		// pad byte for odd chunk size
		if sz&1 == 1 {
			off++
		}
	}
	return chunks, nil
}

var (
	errEarlyEOF = parseErr("unexpected EOF while parsing WebP container")
	errBadRIFF  = parseErr("not a RIFF/WEBP file")
)

type parseErr string

func (e parseErr) Error() string { return string(e) }

// TestEncodeAllLossy_RoundTrip encodes a 3-frame animation and verifies the
// RIFF structure: VP8X / ANIM / ANMF×3 with the right tags, canvas size,
// loop count, and per-frame parameters.
func TestEncodeAllLossy_RoundTrip(t *testing.T) {
	const W, H = 32, 32
	ani := &Animation{
		Images: []image.Image{
			makeTestFrame(W, H, 1),
			makeTestFrame(W, H, 2),
			makeTestFrame(W, H, 3),
		},
		Durations:       []uint{100, 200, 50},
		Disposals:       []uint{0, 1, 0},
		LoopCount:       7,
		BackgroundColor: 0xFF112233,
	}

	var buf bytes.Buffer
	if err := EncodeAll(&buf, ani, &Options{Quality: 75}); err != nil {
		t.Fatalf("EncodeAll: %v", err)
	}
	out := buf.Bytes()

	chunks, err := parseWebPChunks(out)
	if err != nil {
		t.Fatalf("parseWebPChunks: %v", err)
	}

	// Expected order: VP8X, ANIM, ANMF, ANMF, ANMF.
	if len(chunks) != 5 {
		var tags []string
		for _, c := range chunks {
			tags = append(tags, c.tag)
		}
		t.Fatalf("want 5 chunks, got %d (%s)", len(chunks), strings.Join(tags, ","))
	}

	// --- VP8X ---
	vp8x := chunks[0]
	if vp8x.tag != "VP8X" {
		t.Fatalf("chunk[0].tag = %q, want VP8X", vp8x.tag)
	}
	if vp8x.size != 10 {
		t.Fatalf("VP8X size = %d, want 10", vp8x.size)
	}
	// Flags byte: animation bit (1<<1) must be set; alpha bit (1<<4) must not.
	flags := vp8x.payload[0]
	if flags&(1<<1) == 0 {
		t.Errorf("VP8X animation flag not set: flags=0x%02x", flags)
	}
	if flags&(1<<4) != 0 {
		t.Errorf("VP8X alpha flag should be clear: flags=0x%02x", flags)
	}
	// Canvas width-1 / height-1 (24-bit LE).
	gotW := int(vp8x.payload[4]) | int(vp8x.payload[5])<<8 | int(vp8x.payload[6])<<16
	gotH := int(vp8x.payload[7]) | int(vp8x.payload[8])<<8 | int(vp8x.payload[9])<<16
	if gotW != W-1 || gotH != H-1 {
		t.Errorf("VP8X canvas = (%d+1)x(%d+1), want %dx%d", gotW, gotH, W, H)
	}

	// --- ANIM ---
	anim := chunks[1]
	if anim.tag != "ANIM" {
		t.Fatalf("chunk[1].tag = %q, want ANIM", anim.tag)
	}
	if anim.size != 6 {
		t.Fatalf("ANIM size = %d, want 6", anim.size)
	}
	gotBg := binary.LittleEndian.Uint32(anim.payload[0:4])
	gotLoop := binary.LittleEndian.Uint16(anim.payload[4:6])
	if gotBg != ani.BackgroundColor {
		t.Errorf("ANIM bg = 0x%08x, want 0x%08x", gotBg, ani.BackgroundColor)
	}
	if gotLoop != ani.LoopCount {
		t.Errorf("ANIM loop = %d, want %d", gotLoop, ani.LoopCount)
	}

	// --- ANMF frames ---
	for i := 0; i < 3; i++ {
		anmf := chunks[2+i]
		if anmf.tag != "ANMF" {
			t.Fatalf("chunk[%d].tag = %q, want ANMF", 2+i, anmf.tag)
		}
		if len(anmf.payload) < 16+8 {
			t.Fatalf("ANMF[%d] payload too small: %d bytes", i, len(anmf.payload))
		}
		// 24-bit LE fields: x, y, w-1, h-1, duration.
		read24 := func(off int) uint32 {
			return uint32(anmf.payload[off]) |
				uint32(anmf.payload[off+1])<<8 |
				uint32(anmf.payload[off+2])<<16
		}
		x := read24(0)
		y := read24(3)
		fw1 := read24(6)
		fh1 := read24(9)
		dur := read24(12)
		flagsByte := anmf.payload[15]

		if x != 0 || y != 0 {
			t.Errorf("ANMF[%d] offset = (%d,%d), want (0,0)", i, x, y)
		}
		if int(fw1) != W-1 || int(fh1) != H-1 {
			t.Errorf("ANMF[%d] size = (%d,%d), want (%d,%d)", i, fw1+1, fh1+1, W, H)
		}
		if uint(dur) != ani.Durations[i] {
			t.Errorf("ANMF[%d] duration = %d, want %d", i, dur, ani.Durations[i])
		}
		// disposal = bit 0
		gotDisp := uint(flagsByte & 0x01)
		if gotDisp != ani.Disposals[i] {
			t.Errorf("ANMF[%d] disposal = %d, want %d", i, gotDisp, ani.Disposals[i])
		}

		// Inner chunk must be "VP8 " (with trailing space).
		inner := anmf.payload[16:]
		if string(inner[0:4]) != "VP8 " {
			t.Errorf("ANMF[%d] inner tag = %q, want %q", i, string(inner[0:4]), "VP8 ")
		}
		innerSize := binary.LittleEndian.Uint32(inner[4:8])
		if int(8+innerSize) > len(inner) {
			t.Errorf("ANMF[%d] inner VP8 chunk size %d exceeds payload %d", i, innerSize, len(inner)-8)
		}
		if innerSize == 0 {
			t.Errorf("ANMF[%d] inner VP8 chunk is empty", i)
		}
	}
}

// TestEncodeAllLossy_ErrorCases verifies that input-validation failures
// surface as errors before any bytes are written.
func TestEncodeAllLossy_ErrorCases(t *testing.T) {
	frame := makeTestFrame(16, 16, 1)

	cases := []struct {
		name string
		ani  *Animation
	}{
		{
			name: "nil_animation",
			ani:  nil,
		},
		{
			name: "empty_images",
			ani: &Animation{
				Images:    nil,
				Durations: nil,
				Disposals: nil,
			},
		},
		{
			name: "mismatched_durations",
			ani: &Animation{
				Images:    []image.Image{frame, frame},
				Durations: []uint{100}, // wrong length
				Disposals: []uint{0, 0},
			},
		},
		{
			name: "mismatched_disposals",
			ani: &Animation{
				Images:    []image.Image{frame, frame},
				Durations: []uint{100, 100},
				Disposals: []uint{0}, // wrong length
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := EncodeAll(&buf, tc.ani, nil)
			if err == nil {
				t.Fatalf("EncodeAll(%s): want error, got nil (wrote %d bytes)", tc.name, buf.Len())
			}
		})
	}
}

// TestEncodeAllLossy_ClampDuration confirms that durations > 24-bit max are
// clamped (not truncated) so the encoded 24-bit field still reads as the
// maximum value (2^24-1).
func TestEncodeAllLossy_ClampDuration(t *testing.T) {
	ani := &Animation{
		Images:    []image.Image{makeTestFrame(16, 16, 1)},
		Durations: []uint{1 << 30}, // way over 24-bit
		Disposals: []uint{0},
		LoopCount: 0,
	}
	var buf bytes.Buffer
	if err := EncodeAll(&buf, ani, nil); err != nil {
		t.Fatalf("EncodeAll: %v", err)
	}
	chunks, err := parseWebPChunks(buf.Bytes())
	if err != nil {
		t.Fatalf("parseWebPChunks: %v", err)
	}
	// Find the ANMF and read its duration field.
	var anmf *parsedChunk
	for i := range chunks {
		if chunks[i].tag == "ANMF" {
			anmf = &chunks[i]
			break
		}
	}
	if anmf == nil {
		t.Fatalf("no ANMF chunk in output")
	}
	dur := uint32(anmf.payload[12]) |
		uint32(anmf.payload[13])<<8 |
		uint32(anmf.payload[14])<<16
	if dur != (1<<24)-1 {
		t.Errorf("duration not clamped: got %d, want %d", dur, (1<<24)-1)
	}
}

// TestEncodeAllLossy_FramesDecodable extracts each ANMF's inner VP8 chunk,
// wraps it in a single-frame RIFF/WEBP container, and verifies the standard
// library can decode it. This proves the embedded bitstream is valid VP8 and
// not just byte-structurally well-formed.
//
// The standard x/image/webp decoder doesn't support animation, so we have to
// reconstitute a non-animated WebP per frame.
func TestEncodeAllLossy_FramesDecodable(t *testing.T) {
	const W, H = 48, 48
	ani := &Animation{
		Images: []image.Image{
			makeTestFrame(W, H, 1),
			makeTestFrame(W, H, 2),
		},
		Durations: []uint{100, 200},
		Disposals: []uint{0, 0},
		LoopCount: 0,
	}
	var buf bytes.Buffer
	if err := EncodeAll(&buf, ani, &Options{Quality: 80}); err != nil {
		t.Fatalf("EncodeAll: %v", err)
	}
	chunks, err := parseWebPChunks(buf.Bytes())
	if err != nil {
		t.Fatalf("parseWebPChunks: %v", err)
	}

	frameNum := 0
	for _, c := range chunks {
		if c.tag != "ANMF" {
			continue
		}
		inner := c.payload[16:]
		innerSize := binary.LittleEndian.Uint32(inner[4:8])
		vp8Data := inner[8 : 8+innerSize]

		// Wrap in a standalone RIFF/WEBP container.
		var wrap bytes.Buffer
		wrap.WriteString("RIFF")
		// RIFF size = 4 ("WEBP") + 8 (chunk hdr) + vp8Size.
		riffSize := uint32(4 + 8 + len(vp8Data))
		_ = binary.Write(&wrap, binary.LittleEndian, riffSize)
		wrap.WriteString("WEBP")
		wrap.WriteString("VP8 ")
		_ = binary.Write(&wrap, binary.LittleEndian, uint32(len(vp8Data)))
		wrap.Write(vp8Data)
		if len(vp8Data)&1 == 1 {
			wrap.WriteByte(0)
		}

		img, err := webp.Decode(&wrap)
		if err != nil {
			t.Fatalf("frame %d: webp.Decode failed: %v", frameNum, err)
		}
		b := img.Bounds()
		if b.Dx() != W || b.Dy() != H {
			t.Errorf("frame %d: decoded size = %dx%d, want %dx%d",
				frameNum, b.Dx(), b.Dy(), W, H)
		}
		frameNum++
	}
	if frameNum != 2 {
		t.Errorf("decoded %d frames, want 2", frameNum)
	}
}

// TestEncodeAllLossy_ClampDisposal confirms disposal values > 1 are clamped
// to 1 (matching the lossless reference implementation).
func TestEncodeAllLossy_ClampDisposal(t *testing.T) {
	ani := &Animation{
		Images:    []image.Image{makeTestFrame(16, 16, 1)},
		Durations: []uint{100},
		Disposals: []uint{42}, // out-of-range
		LoopCount: 0,
	}
	var buf bytes.Buffer
	if err := EncodeAll(&buf, ani, nil); err != nil {
		t.Fatalf("EncodeAll: %v", err)
	}
	chunks, err := parseWebPChunks(buf.Bytes())
	if err != nil {
		t.Fatalf("parseWebPChunks: %v", err)
	}
	var anmf *parsedChunk
	for i := range chunks {
		if chunks[i].tag == "ANMF" {
			anmf = &chunks[i]
			break
		}
	}
	if anmf == nil {
		t.Fatalf("no ANMF chunk in output")
	}
	flags := anmf.payload[15]
	if flags&0x01 != 1 {
		t.Errorf("disposal not clamped to 1: flags = 0x%02x", flags)
	}
}
