// Portions ported from libwebp (https://github.com/webmproject/libwebp)
// Copyright 2011 Google Inc. All Rights Reserved.
// BSD 3-Clause License — see /Users/bayshark/-projects/self/libwebp/COPYING

#include "textflag.h"

// func fTransform(src []int16, ref []int16, out []int16)
//
// SSE2 vectorised horizontal pass + scalar vertical pass.
//
// Go register-based ABI (amd64):
//   AX = src.ptr, BX = src.len, CX = src.cap
//   DI = ref.ptr, SI = ref.len, R8 = ref.cap
//   R9 = out.ptr, R10 = out.len, R11 = out.cap
//
// Horizontal pass: process all 4 rows simultaneously via SSE2.
// src and ref are 16×int16 = 32 bytes each.
//
// Each row has 4 int16 values: [d0,d1,d2,d3] after subtraction.
// Layout: row0 = bytes[0:8], row1 = bytes[8:16], row2 = bytes[16:24], row3 = bytes[24:32]
//
// After loading and subtracting, each X register holds 8 int16 (2 rows):
//   X0 = [d0r0,d1r0,d2r0,d3r0, d0r1,d1r1,d2r1,d3r1]
//   X1 = [d0r2,d1r2,d2r2,d3r2, d0r3,d1r3,d2r3,d3r3]
//
// The horizontal butterfly works row by row.
// For efficiency, we process 2 rows at a time using the full 128-bit register.
//
// Stack: 64 bytes for int32 tmp[16].

TEXT ·fTransform(SB),NOSPLIT,$64-72
	// Load pointers from stack (FP-based calling convention)
	MOVQ    src_base+0(FP), AX
	MOVQ    ref_base+24(FP), DI
	MOVQ    out_base+48(FP), R9

	// Load src and ref
	MOVOU   (AX), X0
	MOVOU   16(AX), X1
	MOVOU   (DI), X2
	MOVOU   16(DI), X3

	// diff = src - ref
	PSUBW   X2, X0    // X0 = diffs rows 0,1
	PSUBW   X3, X1    // X1 = diffs rows 2,3

	// Process horizontal pass row by row using scalar code.
	// Store diffs to stack then process via scalar.
	MOVQ    SP, R10
	MOVOU   X0, (R10)       // rows 0,1 diffs at [SP..SP+15]
	MOVOU   X1, 16(R10)     // rows 2,3 diffs at [SP+16..SP+31]

	// Scalar horizontal pass: tmp[row*4+col] for 4 rows
	// R10 = stack base (diffs), R11 = stack base+32 (tmp output)
	ADDQ    $32, R10         // R10 now points to tmp[] area (at SP+32)
	SUBQ    $32, R10         // actually keep R10 at SP for diff reading
	LEAQ    32(SP), R11      // R11 = &tmp[0]

	MOVQ    $0, CX           // row counter
hloop:
	// Load row CX from diff buffer
	MOVQ    CX, DX
	SHLQ    $3, DX           // DX = CX * 8 (byte offset for row of 4 int16)
	MOVWLSX (SP)(DX*1), R12   // d0 = diff[row*4+0]
	MOVWLSX 2(SP)(DX*1), R13  // d1
	MOVWLSX 4(SP)(DX*1), R14  // d2
	MOVWLSX 6(SP)(DX*1), R15  // d3

	// butterfly
	MOVLQSX R12, R12
	MOVLQSX R13, R13
	MOVLQSX R14, R14
	MOVLQSX R15, R15

	MOVQ    R12, BX
	ADDQ    R15, BX   // a0 = d0+d3
	MOVQ    R13, SI
	ADDQ    R14, SI   // a1 = d1+d2
	MOVQ    R13, R8
	SUBQ    R14, R8   // a2 = d1-d2
	MOVQ    R12, DI
	SUBQ    R15, DI   // a3 = d0-d3

	// tmp[row*4+0] = (a0+a1)*8
	MOVQ    BX, R12
	ADDQ    SI, R12
	SHLQ    $3, R12

	// tmp[row*4+1] = (a2*2217 + a3*5352 + 1812) >> 9
	MOVQ    R8, R13
	IMULQ   $2217, R13
	MOVQ    DI, R14
	IMULQ   $5352, R14
	ADDQ    R14, R13
	ADDQ    $1812, R13
	SARQ    $9, R13

	// tmp[row*4+2] = (a0-a1)*8
	MOVQ    BX, R14
	SUBQ    SI, R14
	SHLQ    $3, R14

	// tmp[row*4+3] = (a3*2217 - a2*5352 + 937) >> 9
	MOVQ    DI, R15
	IMULQ   $2217, R15
	MOVQ    R8, AX
	IMULQ   $5352, AX
	SUBQ    AX, R15
	ADDQ    $937, R15
	SARQ    $9, R15

	// Store to tmp (as int32 in 32-bit slots)
	MOVQ    CX, AX
	SHLQ    $4, AX           // AX = row*16 (byte offset in int32 array, 4 ints × 4 bytes)
	MOVL    R12, (R11)(AX*1)
	MOVL    R13, 4(R11)(AX*1)
	MOVL    R14, 8(R11)(AX*1)
	MOVL    R15, 12(R11)(AX*1)

	INCQ    CX
	CMPQ    CX, $4
	JL      hloop

	// Vertical pass: process 4 columns
	MOVQ    R9, DI           // DI = out.ptr
	MOVQ    $0, CX           // column counter
vloop:
	// Load column CX: tmp[0+i], tmp[4+i], tmp[8+i], tmp[12+i]
	// i=CX, each element is 4 bytes, stride=4 ints*4=16 bytes
	MOVQ    CX, AX
	SHLQ    $2, AX           // AX = i*4 (byte offset within int32 stride)
	MOVLQSX (R11)(AX*1), R12  // tmp[0+i]
	MOVLQSX 16(R11)(AX*1), R13 // tmp[4+i]
	MOVLQSX 32(R11)(AX*1), R14 // tmp[8+i]
	MOVLQSX 48(R11)(AX*1), R15 // tmp[12+i]

	MOVQ    R12, BX
	ADDQ    R15, BX   // a0 = tmp[0+i] + tmp[12+i]
	MOVQ    R13, SI
	ADDQ    R14, SI   // a1 = tmp[4+i] + tmp[8+i]
	MOVQ    R13, R8
	SUBQ    R14, R8   // a2 = tmp[4+i] - tmp[8+i]
	MOVQ    R12, R9
	SUBQ    R15, R9   // a3 = tmp[0+i] - tmp[12+i]

	// out[0+i] = (a0+a1+7)>>4
	MOVQ    BX, AX
	ADDQ    SI, AX
	ADDQ    $7, AX
	SARQ    $4, AX
	MOVQ    CX, DX
	MOVW    AX, (DI)(DX*2)

	// out[8+i] = (a0-a1+7)>>4
	MOVQ    BX, AX
	SUBQ    SI, AX
	ADDQ    $7, AX
	SARQ    $4, AX
	MOVW    AX, 16(DI)(DX*2)

	// out[4+i] = ((a2*2217 + a3*5352 + 12000)>>16) + (a3!=0?1:0)
	MOVQ    R8, AX
	IMULQ   $2217, AX
	MOVQ    R9, R12
	IMULQ   $5352, R12
	ADDQ    R12, AX
	ADDQ    $12000, AX
	SARQ    $16, AX
	CMPQ    R9, $0
	SETNE   R12
	MOVBQZX R12, R12
	ADDQ    R12, AX
	MOVW    AX, 8(DI)(DX*2)

	// out[12+i] = (a3*2217 - a2*5352 + 51000)>>16
	MOVQ    R9, AX
	IMULQ   $2217, AX
	MOVQ    R8, R12
	IMULQ   $5352, R12
	SUBQ    R12, AX
	ADDQ    $51000, AX
	SARQ    $16, AX
	MOVW    AX, 24(DI)(DX*2)

	INCQ    CX
	CMPQ    CX, $4
	JL      vloop

	// Restore R9 (clobbered above as a3)
	// out.ptr already saved in DI, so return is fine
	RET
