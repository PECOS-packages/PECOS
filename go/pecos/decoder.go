// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package pecos

/*
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Mirrors ForeignDecodingResultRaw from pecos-foreign/src/decoder.rs
typedef struct {
    uint8_t* observable_ptr;
    size_t   observable_len;
    double   weight;
    int8_t   converged;    // 0=false, 1=true, -1=unknown
    uint8_t* error_ptr;
    size_t   error_len;
} PecosDecodingResultRaw;

// Mirrors ForeignDecoderVTable from pecos-foreign/src/decoder.rs
typedef struct {
    uint32_t version;
    int32_t (*decode)(void* handle, const uint8_t* input_ptr, size_t input_len, PecosDecodingResultRaw* result_out);
    size_t  (*check_count)(const void* handle);
    size_t  (*bit_count)(const void* handle);
    void    (*free_result)(uint8_t* ptr, size_t len);
    void    (*free_error)(const uint8_t* ptr, size_t len);
    void    (*destroy)(void* handle);
} PecosDecoderVTable;
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// Decoder is the interface that Go decoder authors implement.
//
// This maps directly to the PECOS Decoder trait in Rust. Implement these three
// methods and register your decoder to use it anywhere PECOS expects a decoder.
//
// Example:
//
//	type MyDecoder struct {
//	    checks int
//	    bits   int
//	}
//
//	func (d *MyDecoder) Decode(syndrome []byte) (*DecodingResult, error) {
//	    observable := make([]byte, d.bits)
//	    // ... your decoding logic ...
//	    return &DecodingResult{Observable: observable, Weight: 1.0}, nil
//	}
//
//	func (d *MyDecoder) CheckCount() int { return d.checks }
//	func (d *MyDecoder) BitCount() int   { return d.bits }
type Decoder interface {
	// Decode a syndrome vector and return the decoding result.
	Decode(syndrome []byte) (*DecodingResult, error)

	// CheckCount returns the number of checks (rows in parity check matrix).
	CheckCount() int

	// BitCount returns the number of bits (columns in parity check matrix).
	BitCount() int
}

// DecodingResult holds the output of a decoding operation.
type DecodingResult struct {
	// Observable is the decoded correction/observable vector.
	Observable []byte

	// Weight is the cost of the decoding solution.
	Weight float64

	// Converged indicates whether the decoder converged.
	// nil means unknown.
	Converged *bool
}

// decoderRegistry holds live Go decoder instances keyed by an integer handle.
// This is necessary because Go pointers cannot be stored in C memory directly
// (cgo pointer passing rules), so we store them on the Go side and pass integer
// handles across the FFI boundary.
var (
	decoderRegistry   = make(map[uintptr]Decoder)
	decoderNextHandle uintptr
	decoderMu         sync.Mutex
)

func registerDecoder(d Decoder) uintptr {
	decoderMu.Lock()
	defer decoderMu.Unlock()
	decoderNextHandle++
	h := decoderNextHandle
	decoderRegistry[h] = d
	return h
}

func lookupDecoder(h uintptr) Decoder {
	decoderMu.Lock()
	defer decoderMu.Unlock()
	return decoderRegistry[h]
}

func unregisterDecoder(h uintptr) {
	decoderMu.Lock()
	defer decoderMu.Unlock()
	delete(decoderRegistry, h)
}

// --- C callback implementations ---
// These are exported so their function pointers can populate the vtable.

//export pecos_go_decoder_decode
func pecos_go_decoder_decode(handle unsafe.Pointer, inputPtr *C.uint8_t, inputLen C.size_t, resultOut *C.PecosDecodingResultRaw) C.int32_t {
	h := uintptr(handle)
	d := lookupDecoder(h)
	if d == nil {
		setError(resultOut, "decoder handle not found")
		return -1
	}

	// Build a Go slice backed by the C input (no copy for the input).
	syndrome := C.GoBytes(unsafe.Pointer(inputPtr), C.int(inputLen))

	result, err := d.Decode(syndrome)
	if err != nil {
		setError(resultOut, err.Error())
		return -1
	}

	// Copy observable into C-allocated memory that Rust will own.
	if len(result.Observable) > 0 {
		cObs := C.malloc(C.size_t(len(result.Observable)))
		C.memcpy(cObs, unsafe.Pointer(&result.Observable[0]), C.size_t(len(result.Observable)))
		resultOut.observable_ptr = (*C.uint8_t)(cObs)
		resultOut.observable_len = C.size_t(len(result.Observable))
	}

	resultOut.weight = C.double(result.Weight)

	if result.Converged == nil {
		resultOut.converged = -1
	} else if *result.Converged {
		resultOut.converged = 1
	} else {
		resultOut.converged = 0
	}

	resultOut.error_ptr = nil
	resultOut.error_len = 0
	return 0
}

//export pecos_go_decoder_check_count
func pecos_go_decoder_check_count(handle unsafe.Pointer) C.size_t {
	h := uintptr(handle)
	d := lookupDecoder(h)
	if d == nil {
		return 0
	}
	return C.size_t(d.CheckCount())
}

//export pecos_go_decoder_bit_count
func pecos_go_decoder_bit_count(handle unsafe.Pointer) C.size_t {
	h := uintptr(handle)
	d := lookupDecoder(h)
	if d == nil {
		return 0
	}
	return C.size_t(d.BitCount())
}

//export pecos_go_decoder_free_result
func pecos_go_decoder_free_result(ptr *C.uint8_t, _ C.size_t) {
	if ptr != nil {
		C.free(unsafe.Pointer(ptr))
	}
}

//export pecos_go_decoder_free_error
func pecos_go_decoder_free_error(ptr *C.uint8_t, _ C.size_t) {
	if ptr != nil {
		C.free(unsafe.Pointer(ptr))
	}
}

//export pecos_go_decoder_destroy
func pecos_go_decoder_destroy(handle unsafe.Pointer) {
	if handle != nil {
		unregisterDecoder(uintptr(handle))
	}
}

func setError(resultOut *C.PecosDecodingResultRaw, msg string) {
	cMsg := C.CString(msg)
	resultOut.error_ptr = (*C.uint8_t)(unsafe.Pointer(cMsg))
	resultOut.error_len = C.size_t(len(msg))
}

// DecoderHandle is an opaque reference to a Go decoder registered for use
// by PECOS Rust code. It holds the C handle and vtable needed to pass across FFI.
type DecoderHandle struct {
	handle uintptr
}

// RegisterDecoder registers a Go Decoder implementation and returns a handle
// that can be passed to PECOS Rust code.
//
// The decoder stays alive until the handle is destroyed (by Rust calling the
// vtable's destroy function, or by calling Destroy explicitly).
func RegisterDecoder(d Decoder) *DecoderHandle {
	h := registerDecoder(d)
	return &DecoderHandle{handle: h}
}

// Destroy unregisters the decoder, releasing it for garbage collection.
func (dh *DecoderHandle) Destroy() {
	unregisterDecoder(dh.handle)
}

// Handle returns the raw handle value for passing to Rust FFI.
func (dh *DecoderHandle) Handle() uintptr {
	return dh.handle
}

// String returns a debug representation.
func (dh *DecoderHandle) String() string {
	return fmt.Sprintf("DecoderHandle(%d)", dh.handle)
}
