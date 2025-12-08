// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// Package pecos provides Go bindings for the PECOS quantum error correction simulator.
//
// This package wraps the Rust PECOS library via C FFI, providing a native Go interface
// for quantum computing simulations.
//
// # Building
//
// Before using this package, you need to build the Rust library:
//
//	cd go/pecos-go-ffi
//	cargo build --release
//
// Then set the library path:
//
//	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/PECOS/target/release
//
// # Example
//
//	package main
//
//	import (
//		"fmt"
//		"github.com/PECOS-packages/PECOS/go/pecos"
//	)
//
//	func main() {
//		fmt.Println(pecos.Version())
//
//		q := pecos.NewQubitId(0)
//		fmt.Printf("Created: %s\n", q)
//	}
package pecos

/*
#cgo LDFLAGS: -L${SRCDIR}/../pecos-go-ffi/target/release -lpecos_go
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lpecos_go

#include <stdlib.h>

// FFI function declarations
extern const char* pecos_version();
extern long long create_qubit_id(long long index);
extern const char* qubit_id_to_string(long long index);
extern long long add_two_numbers(long long a, long long b);
extern void free_rust_string(char* s);
*/
import "C"
import "unsafe"

// Version returns the PECOS library version string.
func Version() string {
	cstr := C.pecos_version()
	defer C.free_rust_string((*C.char)(unsafe.Pointer(cstr)))
	return C.GoString(cstr)
}

// QubitId represents a qubit identifier in the PECOS system.
type QubitId struct {
	index int64
}

// NewQubitId creates a new QubitId with the given index.
// Returns nil if the index is negative.
func NewQubitId(index int64) *QubitId {
	result := int64(C.create_qubit_id(C.longlong(index)))
	if result < 0 {
		return nil
	}
	return &QubitId{index: result}
}

// Index returns the qubit's index.
func (q *QubitId) Index() int64 {
	return q.index
}

// String returns a string representation of the QubitId.
func (q *QubitId) String() string {
	cstr := C.qubit_id_to_string(C.longlong(q.index))
	defer C.free_rust_string((*C.char)(unsafe.Pointer(cstr)))
	return C.GoString(cstr)
}

// AddTwoNumbers is a simple test function that adds two numbers.
// This is primarily for testing the FFI connection.
func AddTwoNumbers(a, b int64) int64 {
	return int64(C.add_two_numbers(C.longlong(a), C.longlong(b)))
}
