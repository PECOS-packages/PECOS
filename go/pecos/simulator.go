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

// Mirrors ForeignMeasurementResult from pecos-foreign/src/simulator.rs
typedef struct {
    uint8_t outcome;
    uint8_t is_deterministic;
} PecosMeasurementResult;
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// MeasurementResult holds the outcome of a Z-basis measurement.
type MeasurementResult struct {
	// Outcome is the measurement result: false = |0>, true = |1>.
	Outcome bool
	// IsDeterministic is true if the outcome was deterministic (not random).
	IsDeterministic bool
}

// CliffordSimulator is the interface for Clifford-only simulators.
//
// Implement these 5 methods to create a simulator usable by PECOS.
// All 52 other Clifford gates (X, Y, Z, SX, CZ, SWAP, etc.) are
// automatically decomposed into these primitives by the Rust trait defaults.
//
// Example:
//
//	type MyStabSim struct {
//	    numQubits int
//	    // ... your stabilizer state ...
//	}
//
//	func (s *MyStabSim) SZ(qubits []int)  { /* apply S gate */ }
//	func (s *MyStabSim) H(qubits []int)   { /* apply Hadamard */ }
//	func (s *MyStabSim) CX(pairs [][2]int) { /* apply CNOT */ }
//	func (s *MyStabSim) MZ(qubits []int) []MeasurementResult { /* measure */ }
//	func (s *MyStabSim) Reset() { /* reset to |0...0> */ }
type CliffordSimulator interface {
	// SZ applies the S (sqrt-Z) gate to each qubit.
	SZ(qubits []int)

	// H applies the Hadamard gate to each qubit.
	H(qubits []int)

	// CX applies CNOT to each (control, target) pair.
	CX(pairs [][2]int)

	// MZ measures each qubit in the Z basis.
	MZ(qubits []int) []MeasurementResult

	// Reset resets the simulator to its initial state.
	Reset()
}

// RotationSimulator extends CliffordSimulator with arbitrary rotation gates.
//
// Implement these 3 additional methods for a universal simulator.
// All other rotation gates (RY, T, Tdg, RXX, RYY, U, etc.) are
// automatically decomposed by the Rust trait defaults.
type RotationSimulator interface {
	CliffordSimulator

	// RX applies an X-axis rotation by theta (radians) to each qubit.
	RX(theta float64, qubits []int)

	// RZ applies a Z-axis rotation by theta (radians) to each qubit.
	RZ(theta float64, qubits []int)

	// RZZ applies a ZZ rotation by theta (radians) to each (q0, q1) pair.
	RZZ(theta float64, pairs [][2]int)
}

// --- Simulator registry (same pattern as decoder) ---

var (
	simRegistry   = make(map[uintptr]CliffordSimulator)
	simNextHandle uintptr
	simMu         sync.Mutex
)

func registerSimulator(s CliffordSimulator) uintptr {
	simMu.Lock()
	defer simMu.Unlock()
	simNextHandle++
	h := simNextHandle
	simRegistry[h] = s
	return h
}

func lookupSimulator(h uintptr) CliffordSimulator {
	simMu.Lock()
	defer simMu.Unlock()
	return simRegistry[h]
}

func unregisterSimulator(h uintptr) {
	simMu.Lock()
	defer simMu.Unlock()
	delete(simRegistry, h)
}

// --- Helpers ---

// cToQubits converts a C array of usize to a Go []int slice.
func cToQubits(ptr *C.size_t, n C.size_t) []int {
	count := int(n)
	if count == 0 {
		return nil
	}
	// Build a Go slice viewing the C memory (no copy needed for read-only).
	raw := unsafe.Slice((*C.size_t)(ptr), count)
	qubits := make([]int, count)
	for i, v := range raw {
		qubits[i] = int(v)
	}
	return qubits
}

// cToPairs converts interleaved C pairs [c0, t0, c1, t1, ...] to Go [][2]int.
func cToPairs(ptr *C.size_t, numPairs C.size_t) [][2]int {
	n := int(numPairs)
	if n == 0 {
		return nil
	}
	raw := unsafe.Slice((*C.size_t)(ptr), n*2)
	pairs := make([][2]int, n)
	for i := 0; i < n; i++ {
		pairs[i] = [2]int{int(raw[2*i]), int(raw[2*i+1])}
	}
	return pairs
}

// --- C callback implementations ---

//export pecos_go_sim_sz
func pecos_go_sim_sz(handle unsafe.Pointer, qubits *C.size_t, numQubits C.size_t) {
	s := lookupSimulator(uintptr(handle))
	if s != nil {
		s.SZ(cToQubits(qubits, numQubits))
	}
}

//export pecos_go_sim_h
func pecos_go_sim_h(handle unsafe.Pointer, qubits *C.size_t, numQubits C.size_t) {
	s := lookupSimulator(uintptr(handle))
	if s != nil {
		s.H(cToQubits(qubits, numQubits))
	}
}

//export pecos_go_sim_cx
func pecos_go_sim_cx(handle unsafe.Pointer, pairs *C.size_t, numPairs C.size_t) {
	s := lookupSimulator(uintptr(handle))
	if s != nil {
		s.CX(cToPairs(pairs, numPairs))
	}
}

//export pecos_go_sim_mz
func pecos_go_sim_mz(handle unsafe.Pointer, qubits *C.size_t, numQubits C.size_t, resultsOut *C.PecosMeasurementResult) {
	s := lookupSimulator(uintptr(handle))
	if s == nil {
		return
	}

	goQubits := cToQubits(qubits, numQubits)
	results := s.MZ(goQubits)

	// Write results into the C-allocated buffer.
	out := unsafe.Slice(resultsOut, len(goQubits))
	for i, r := range results {
		if r.Outcome {
			out[i].outcome = 1
		} else {
			out[i].outcome = 0
		}
		if r.IsDeterministic {
			out[i].is_deterministic = 1
		} else {
			out[i].is_deterministic = 0
		}
	}
}

//export pecos_go_sim_rx
func pecos_go_sim_rx(handle unsafe.Pointer, theta C.double, qubits *C.size_t, numQubits C.size_t) {
	s := lookupSimulator(uintptr(handle))
	if rs, ok := s.(RotationSimulator); ok {
		rs.RX(float64(theta), cToQubits(qubits, numQubits))
	}
}

//export pecos_go_sim_rz
func pecos_go_sim_rz(handle unsafe.Pointer, theta C.double, qubits *C.size_t, numQubits C.size_t) {
	s := lookupSimulator(uintptr(handle))
	if rs, ok := s.(RotationSimulator); ok {
		rs.RZ(float64(theta), cToQubits(qubits, numQubits))
	}
}

//export pecos_go_sim_rzz
func pecos_go_sim_rzz(handle unsafe.Pointer, theta C.double, pairs *C.size_t, numPairs C.size_t) {
	s := lookupSimulator(uintptr(handle))
	if rs, ok := s.(RotationSimulator); ok {
		rs.RZZ(float64(theta), cToPairs(pairs, numPairs))
	}
}

//export pecos_go_sim_reset
func pecos_go_sim_reset(handle unsafe.Pointer) {
	s := lookupSimulator(uintptr(handle))
	if s != nil {
		s.Reset()
	}
}

//export pecos_go_sim_destroy
func pecos_go_sim_destroy(handle unsafe.Pointer) {
	if handle != nil {
		unregisterSimulator(uintptr(handle))
	}
}

// SimulatorHandle is an opaque reference to a Go simulator registered for use by PECOS.
type SimulatorHandle struct {
	handle       uintptr
	hasRotations bool
}

// RegisterSimulator registers a Go CliffordSimulator and returns a handle.
// If the simulator also implements RotationSimulator, rotation gates are enabled.
func RegisterSimulator(s CliffordSimulator) *SimulatorHandle {
	h := registerSimulator(s)
	_, hasRot := s.(RotationSimulator)
	return &SimulatorHandle{handle: h, hasRotations: hasRot}
}

// Destroy unregisters the simulator.
func (sh *SimulatorHandle) Destroy() {
	unregisterSimulator(sh.handle)
}

// Handle returns the raw handle for FFI.
func (sh *SimulatorHandle) Handle() uintptr {
	return sh.handle
}

// HasRotations returns true if the simulator supports rotation gates.
func (sh *SimulatorHandle) HasRotations() bool {
	return sh.hasRotations
}

// String returns a debug representation.
func (sh *SimulatorHandle) String() string {
	rot := "Clifford-only"
	if sh.hasRotations {
		rot = "with rotations"
	}
	return fmt.Sprintf("SimulatorHandle(%d, %s)", sh.handle, rot)
}
