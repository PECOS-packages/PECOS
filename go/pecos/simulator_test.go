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

import (
	"testing"
)

// ToyStabSim is a trivial stabilizer simulator for testing the interface.
// It tracks qubit states as simple booleans (0 or 1) -- not a real simulator,
// just enough to verify the Go interface works.
type ToyStabSim struct {
	numQubits int
	state     []bool // true = |1>, false = |0>
}

func NewToyStabSim(n int) *ToyStabSim {
	return &ToyStabSim{
		numQubits: n,
		state:     make([]bool, n),
	}
}

func (s *ToyStabSim) SZ(_ []int) {
	// S gate is a phase gate -- doesn't change computational basis state
}

func (s *ToyStabSim) H(qubits []int) {
	// Toy: H on |0> -> |+>, on |1> -> |->
	// For this toy, we just flip the bit (wrong but tests the interface)
	for _, q := range qubits {
		s.state[q] = !s.state[q]
	}
}

func (s *ToyStabSim) CX(pairs [][2]int) {
	for _, p := range pairs {
		control, target := p[0], p[1]
		if s.state[control] {
			s.state[target] = !s.state[target]
		}
	}
}

func (s *ToyStabSim) MZ(qubits []int) []MeasurementResult {
	results := make([]MeasurementResult, len(qubits))
	for i, q := range qubits {
		results[i] = MeasurementResult{
			Outcome:         s.state[q],
			IsDeterministic: true,
		}
	}
	return results
}

func (s *ToyStabSim) Reset() {
	for i := range s.state {
		s.state[i] = false
	}
}

func TestCliffordSimulatorInterface(t *testing.T) {
	sim := NewToyStabSim(3)

	// Verify it satisfies CliffordSimulator
	var _ CliffordSimulator = sim

	// Apply H to qubit 0 (toy flips it to |1>)
	sim.H([]int{0})

	// CNOT: control=0, target=1
	sim.CX([][2]int{{0, 1}})

	// Measure all
	results := sim.MZ([]int{0, 1, 2})
	if !results[0].Outcome {
		t.Error("qubit 0 should be |1> after H")
	}
	if !results[1].Outcome {
		t.Error("qubit 1 should be |1> after CNOT with control=|1>")
	}
	if results[2].Outcome {
		t.Error("qubit 2 should still be |0>")
	}
}

func TestSimulatorRegistration(t *testing.T) {
	sim := NewToyStabSim(2)
	handle := RegisterSimulator(sim)
	if handle == nil {
		t.Fatal("RegisterSimulator returned nil")
	}
	defer handle.Destroy()

	if handle.Handle() == 0 {
		t.Error("handle should be non-zero")
	}
	if handle.HasRotations() {
		t.Error("ToyStabSim should not have rotations")
	}
}

func TestSimulatorReset(t *testing.T) {
	sim := NewToyStabSim(2)
	sim.H([]int{0, 1})

	// Both should be |1> in our toy
	results := sim.MZ([]int{0, 1})
	if !results[0].Outcome || !results[1].Outcome {
		t.Error("both qubits should be |1> after H")
	}

	sim.Reset()
	results = sim.MZ([]int{0, 1})
	if results[0].Outcome || results[1].Outcome {
		t.Error("both qubits should be |0> after Reset")
	}
}
