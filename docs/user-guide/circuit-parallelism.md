# Circuit parallelism and density

`pecos.analysis.circuit_parallelism` computes static metrics from an existing
`QuantumCircuit`. It does not compile a program, evaluate classical conditions,
assign durations, or produce an executable schedule.

```python
from pecos.analysis import circuit_parallelism
from pecos.circuits import QuantumCircuit

circuit = QuantumCircuit()
circuit.append("CX", {(0, 1)})
circuit.append("CX", {(0, 2)})
circuit.append("CX", {(3, 4)})

analysis = circuit_parallelism(circuit, gate_symbols={"CX"})
assert analysis.all_operations.layer_count == 2
assert analysis.all_operations.operation_count == 3
assert analysis.all_operations.mean_operations == 1.5
assert analysis.all_operations.density(6) == 0.5
```

The selected gates occupy layers with two operations and one operation. Density
is the mean number of occupied qubits per selected layer divided by the supplied
circuit width. Mixed-arity operations contribute their actual number of occupied
qubits. The width must include every selected qubit. Empty results have zero
averages, including zero density for an empty zero-qubit circuit.

## Selection and movement

- `gate_symbols=None` selects all operation symbols. A collection selects exactly
  those symbols; an empty collection selects nothing.
- Selected operations move backward across layers with equal `cond` metadata
  that share none of their qubits. A shared qubit or different condition stops
  movement. Batches with different symbols or parameters can share a layer.
- Unselected operations and operations without qubit locations are barriers by
  default: operations in later ticks cannot move before them. Set
  `ignored_operations_are_barriers=False` only when intentionally measuring a
  projection that ignores their dependencies.
- Operations in one tick are concurrent. Each tick's batches are packed in a
  canonical order of their conditions, so the order in which gates were added
  to a tick never changes the result.

For example, an omitted measurement or classical operation can affect later
execution even when its qubits do not occur in the selected gates. Keeping these
barriers is conservative, but the resulting metrics still do not track classical
reads/writes or prove a valid schedule. Empty ticks do not represent timing here.

The input circuit is not modified. Returned layers contain immutable location
sets and independent copies of their conditions. Because those copies are
mutable, layers and results are unhashable.

## Conditional scenarios

`all_operations` includes all selected layers. `unconditional` drops layers with
truthy `cond` metadata without repacking the remaining layers. These are static
scenarios, not minimum and maximum bounds, probabilities, or expected values.

```python
from pecos.analysis import circuit_parallelism
from pecos.circuits import QuantumCircuit

circuit = QuantumCircuit()
circuit.append("CX", {(0, 1), (2, 3)})
circuit.append("CX", {(4, 5)}, cond={"a": "flag", "op": "==", "b": 1})
analysis = circuit_parallelism(circuit)
assert analysis.all_operations.mean_operations == 1.5
assert analysis.unconditional.mean_operations == 2.0
```

Removing the conditional layer increases the mean in this example. If every
selected layer is conditional, the unconditional scenario is empty and reports
zero. Zero-qubit operations can stop movement but never contribute to averages.

Packing takes time and memory linear in the selected operation supports.
