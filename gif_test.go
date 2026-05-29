// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

import (
	"bytes"
	"image/gif"
	"os"
	"path/filepath"
	"testing"
)

// TestConvertGIF decodes a real GIF from test_data/original/, converts it
// to a lossy WebP animation via ConvertGIF, then sanity-checks the output
// container layout.
func TestConvertGIF(t *testing.T) {
	candidates := []string{
		"test_data/original/baf1a2d038ad43b4bbe8b13799c0987d.gif",
		"test_data/original/ezgif-6beab9280acd98.gif",
		"test_data/original/c8e7b15f1e97f2b4f3854032ca608982.gif",
		"test_data/original/9710eca79d8749e498f95692395114da.gif",
	}
	var gifPath string
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			gifPath = p
			break
		}
	}
	if gifPath == "" {
		t.Skipf("no GIF test data in test_data/original/; skipping")
	}

	f, err := os.Open(gifPath)
	if err != nil {
		t.Fatalf("open %s: %v", gifPath, err)
	}
	defer f.Close()

	g, err := gif.DecodeAll(f)
	if err != nil {
		t.Fatalf("gif.DecodeAll(%s): %v", gifPath, err)
	}
	if len(g.Image) == 0 {
		t.Fatalf("decoded gif has no frames")
	}

	var buf bytes.Buffer
	if err := ConvertGIF(&buf, g, &Options{Quality: 75}); err != nil {
		t.Fatalf("ConvertGIF(%s): %v", filepath.Base(gifPath), err)
	}

	out := buf.Bytes()
	if len(out) == 0 {
		t.Fatalf("ConvertGIF produced 0-byte output")
	}

	// Outer RIFF/WEBP wrapper.
	if len(out) < 12 {
		t.Fatalf("output too short (%d bytes) to contain RIFF/WEBP header", len(out))
	}
	if string(out[0:4]) != "RIFF" {
		t.Fatalf("output does not start with RIFF: got %q", string(out[0:4]))
	}
	if string(out[8:12]) != "WEBP" {
		t.Fatalf("output bytes[8:12] = %q, want %q", string(out[8:12]), "WEBP")
	}

	// Must contain ANIM chunk somewhere.
	if !bytes.Contains(out, []byte("ANIM")) {
		t.Fatalf("output does not contain ANIM chunk; first 64 bytes = %x", out[:min(64, len(out))])
	}
	if !bytes.Contains(out, []byte("ANMF")) {
		t.Fatalf("output does not contain any ANMF chunk")
	}
	if !bytes.Contains(out, []byte("VP8X")) {
		t.Fatalf("output does not contain VP8X chunk")
	}

	// Parse chunks and verify we have one ANMF per GIF frame.
	chunks, err := parseWebPChunks(out)
	if err != nil {
		t.Fatalf("parseWebPChunks: %v", err)
	}
	anmfCount := 0
	for _, c := range chunks {
		if c.tag == "ANMF" {
			anmfCount++
		}
	}
	if anmfCount != len(g.Image) {
		t.Errorf("ANMF chunks = %d, want %d (one per GIF frame)", anmfCount, len(g.Image))
	}

	t.Logf("%s: %d frames → %d bytes WebP (input bytes: see file size)",
		filepath.Base(gifPath), len(g.Image), len(out))
}

// TestConvertGIF_NilGif verifies ConvertGIF rejects a nil GIF.
func TestConvertGIF_NilGif(t *testing.T) {
	var buf bytes.Buffer
	if err := ConvertGIF(&buf, nil, nil); err == nil {
		t.Fatalf("ConvertGIF(nil) returned no error")
	}
}

// TestConvertGIF_EmptyGif verifies ConvertGIF rejects a GIF with no frames.
func TestConvertGIF_EmptyGif(t *testing.T) {
	var buf bytes.Buffer
	if err := ConvertGIF(&buf, &gif.GIF{}, nil); err == nil {
		t.Fatalf("ConvertGIF(empty) returned no error")
	}
}
