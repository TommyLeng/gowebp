// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

package gowebp

// fTransform on amd64 uses the scalar implementation from fTransform_generic.go.
// The SSE2 assembly in dct_amd64.s was removed because it used R14 (the Go
// runtime's goroutine pointer on amd64) as a scratch register without
// saving/restoring it, causing signal-based goroutine preemption to corrupt
// the goroutine pointer and crash.
