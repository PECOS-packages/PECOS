<!-- Copyright 2026 The PECOS Developers -->

# Fault-Tolerance Analysis and Distance Certification

This guide covers fault-distance analysis for detector error models and
circuits, together with SAT-backed exact-distance certification. These tools
answer different questions from code-distance searches, so the first step is
choosing the level that matches the object being analyzed.

## What You'll Learn

- Distinguishing code, circuit, and detector-error-model distance
- Finding graphlike and general DEM fault distance
- Diagnosing hook errors and measuring circuit fault distance
- Checking the propagated-fault half of the t-flag condition
- Certifying CSS-code and DEM distance with an independently checked witness
- Exporting SAT and MaxSAT instances for external solvers

```hidden-python
from pecos.qec import (
    CircuitFaultAnalyzer,
    DetectorErrorModel,
    DistanceProblem,
    certified_distance,
)
from pecos.quantum import ParityCheckMatrix, TickCircuit


def repetition_triad_dem():
    circuit = TickCircuit()
    circuit.tick().mz([0, 1, 2])
    circuit.add_detector(records=[-3, -2])
    circuit.add_detector(records=[-3, -1])
    circuit.add_observable(records=[-3])
    return DetectorErrorModel.from_circuit(
        circuit,
        p1=0.0,
        p2=0.0,
        p_meas=0.01,
        p_prep=0.0,
    )


def hook_ladder():
    circuit = TickCircuit()
    circuit.tick().pz([3])
    circuit.tick().cx([(3, 0)])
    circuit.tick().cx([(3, 1)])
    circuit.tick().cx([(3, 2)])
    circuit.tick().mz([3])
    return circuit


def unequal_logical_distance_circuit():
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().pz([1])
    circuit.tick().h([0, 1])
    circuit.tick().cx([(1, 0)])
    circuit.tick().pz([2])
    circuit.tick().h([2])
    return circuit


def weight_four_x_measurement(*, with_flag):
    circuit = TickCircuit()
    circuit.tick().pz([4])
    circuit.tick().h([4])
    if with_flag:
        circuit.tick().pz([5])
    circuit.tick().cx([(4, 0)])
    if with_flag:
        circuit.tick().cx([(4, 5)])
    circuit.tick().cx([(4, 1)])
    circuit.tick().cx([(4, 2)])
    if with_flag:
        circuit.tick().cx([(4, 5)])
    circuit.tick().cx([(4, 3)])
    circuit.tick().h([4])
    circuit.tick().mz([4])
    if with_flag:
        circuit.tick().mz([5])
    return circuit


def steane_distance_problem():
    hamming = ParityCheckMatrix(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ]
    )
    logical = ParityCheckMatrix([[1, 1, 1, 1, 1, 1, 1]])
    return DistanceProblem.from_css_checks(hamming, logical)
```

## Three Levels of Distance

Distance depends on what counts as one fault and what information is retained:

| Level | Minimum counted object | PECOS tool | Question answered |
|-------|------------------------|------------|-------------------|
| Code | Data-qubit Pauli weight | `StabilizerCodeSpec.distance()` or `DistanceProblem.from_css_checks()` | What is the minimum weight of an undetectable logical Pauli? |
| Circuit | Faulty gate locations | `CircuitFaultAnalyzer.fault_distance()` | How many faults in this implementation can cause an undetected logical error? |
| DEM | Error mechanisms | `DetectorErrorModel.graphlike_fault_distance()`, `exhaustive_fault_distance()`, or `DistanceProblem.from_dem()` | How many modeled mechanisms produce no detector events but flip an observable? |

Code distance is not circuit distance. A single ancilla fault can propagate
through later two-qubit gates into a multi-qubit data error called a *hook
error*. Code distance counts the resulting data error's weight; circuit
distance counts the one faulty location that created it. Use code-level tools
to study the stabilizer code, `CircuitFaultAnalyzer` to study a concrete gate
schedule, and DEM tools to study the detector-and-observable abstraction
produced by a noise model.

## Detector-Error-Model Fault Distance

A `DetectorErrorModel` (DEM) describes independent error mechanisms by the
detectors and logical observables they flip. The following three measurement
faults form a repetition-triad model: each pair is distinguishable by
detectors, but all three together cancel the detectors and flip the logical
observable.

```python
dem = repetition_triad_dem()
graphlike = dem.graphlike_fault_distance()
exhaustive = dem.exhaustive_fault_distance(3)

assert graphlike is not None
assert exhaustive is not None
assert graphlike.distance == exhaustive.distance == 3
assert graphlike.mechanism_indices == exhaustive.mechanism_indices
assert len(graphlike.mechanism_indices) == 3
assert graphlike.mechanism_indices == sorted(graphlike.mechanism_indices)
```

`mechanism_indices` is a witness: selecting those DEM mechanisms produces an
undetectable logical failure. `graphlike_fault_distance()` is specialized to
models in which every mechanism touches at most two detectors.
`exhaustive_fault_distance(max_weight)` handles general mechanisms and returns
`None` when it finds no logical failure through the requested weight.

The graphlike method fails fast instead of silently approximating a model with
hyperedges. Here one measurement fault flips three detectors:

```python
circuit = TickCircuit()
circuit.tick().mz([0])
for _ in range(3):
    circuit.add_detector(records=[-1])

hypergraph_dem = DetectorErrorModel.from_circuit(
    circuit,
    p1=0.0,
    p2=0.0,
    p_meas=0.1,
    p_prep=0.0,
)

try:
    hypergraph_dem.graphlike_fault_distance()
except ValueError as error:
    message = str(error)
else:
    raise AssertionError("graphlike distance should reject a hyperedge")
assert "found 1 hyperedge mechanism(s)" in message
```

Use the exhaustive method or the SAT formulation below when hyperedges are
part of the intended model.

## Circuit Fault Analysis

`CircuitFaultAnalyzer` injects Pauli faults at circuit locations and propagates
them through a `TickCircuit`. Its methods accept ancilla and logical supports
as qubit-index lists. A logical is an `(x_support, z_support)` pair.

### Diagnosing Hook Errors

In this three-data-qubit CX ladder, an X fault on the ancilla output of the
first CX propagates through the remaining CX gates. The report identifies the
responsible gate, tick, and qubits:

```python
report = CircuitFaultAnalyzer(hook_ladder()).hook_errors(
    [0, 1, 2],
    [3],
    [],
    [],
    2,
)
hook = next(error for error in report.hook_errors if error.location.tick == 1 and error.fault_paulis == [1, 0])

assert hook.location.gate_type == "CX"
assert hook.location.gate_index == 0
assert hook.location.qubits == [3, 0]
assert hook.data_weight >= 2
print(f"tick={hook.location.tick}, gate={hook.location.gate_type}, qubits={hook.location.qubits}")
```

```text
tick=1, gate=CX, qubits=[3, 0]
```

The integer Pauli labels in `fault_paulis` describe the injected Pauli on each
gate operand. `data_support`, `data_weight`, `detected`, and
`causes_logical_error` describe its propagated effect.

### Overall and Per-Logical Fault Distance

`fault_distance()` returns the first minimum-weight witness across all supplied
logicals. `per_logical_fault_distances()` preserves the distinction between
logical observables. In this small circuit, the two logicals have fault
distances one and two:

```python
analyzer = CircuitFaultAnalyzer(unequal_logical_distance_circuit())
logicals = [([2], []), ([0], [])]

per_logical = analyzer.per_logical_fault_distances(
    [],
    [1],
    logicals,
    2,
    x_only=True,
)
overall = analyzer.fault_distance([], [1], logicals, 2, x_only=True)

assert [result.distance if result is not None else None for result in per_logical] == [1, 2]
assert overall is not None
assert overall.distance == 1
assert overall.logical_index == 0
assert len(overall.witness) == 1
```

A `None` result means no qualifying fault combination was found through
`max_weight`; it is not a proof that no higher-weight combination exists.

### Checking the Flag-Fault Condition

For a weight-four X measurement, placing a flag interaction pair around the
middle data interactions catches the propagated hook that the unflagged
circuit permits:

```python
flagged = CircuitFaultAnalyzer(weight_four_x_measurement(with_flag=True))
flagged_report = flagged.flag_fault_condition(
    [0, 1, 2, 3],
    [5],
    ([0, 1, 2, 3], []),
    1,
)

unflagged = CircuitFaultAnalyzer(weight_four_x_measurement(with_flag=False))
unflagged_report = unflagged.flag_fault_condition(
    [0, 1, 2, 3],
    [],
    ([0, 1, 2, 3], []),
    1,
)

assert flagged_report.fault_condition_satisfied
assert flagged_report.violations == []
assert not unflagged_report.fault_condition_satisfied
assert any(violation.num_faults == 1 and violation.error_weight == 2 for violation in unflagged_report.violations)
```

This method checks only the **propagated-fault half** of the t-flag condition:
dangerous propagated data errors must raise a flag. It does not establish the
separate fault-free behavior. You must verify independently that the flag does
not fire when the circuit has no faults.

## Certified Exact Distance

`DistanceProblem` represents a binary selection problem: choose mechanisms or
qubits that satisfy every detection check and have a nonzero logical effect,
while minimizing the number selected.

### Certifying CSS Distance

For CSS distance, `from_css_checks(hx, lx)` takes the detection-check matrix
and logical-effect matrix. The Steane code uses the Hamming matrix for the
checks and an all-ones logical row:

```python
problem = steane_distance_problem()
result = problem.certified_distance(3)

assert result is not None
assert result.distance == 3
assert result.sat_certified
assert result.unsat_trusted_below == 3
assert problem.verify_witness(result.witness) == 3
assert problem.certified_distance(2) is None
```

The trust split is deliberate. The SAT half is natively verified:
`verify_witness()` independently checks that the returned assignment obeys the
parity constraints, has logical effect, and has the reported weight. The UNSAT
half is solver-trusted: excluding every lower weight, and therefore the claim
of exact minimality, relies on the in-process SAT solver's UNSAT answers.

`verify_witness()` also rejects malformed or invalid assignments:

```python
problem = steane_distance_problem()
result = certified_distance(problem, 3)
assert result is not None

corrupted = result.witness.copy()
corrupted[0] = not corrupted[0]
try:
    problem.verify_witness(corrupted)
except ValueError as error:
    message = str(error)
else:
    raise AssertionError("a corrupted witness should be rejected")
assert "witness violates H row 0" in message
```

The module-level `certified_distance(problem, max_weight)` function and the
method on `DistanceProblem` are equivalent entry points.

### Certifying DEM Distance

`from_dem()` applies the same exact formulation to DEM mechanisms, including
hyperedges:

```python
dem = repetition_triad_dem()
problem = DistanceProblem.from_dem(dem)
result = problem.certified_distance(3)
exhaustive = dem.exhaustive_fault_distance(3)

assert result is not None
assert exhaustive is not None
assert result.distance == exhaustive.distance == 3
assert problem.verify_witness(result.witness) == 3
assert problem.certified_distance(2) is None
```

### Exporting Solver Instances

`to_dimacs(max_weight)` exports a bounded CNF decision problem. `to_wcnf()`
exports the unbounded minimization problem as weighted CNF, with one soft
clause per selected variable:

```python
problem = steane_distance_problem()
dimacs = problem.to_dimacs(3)
wcnf = problem.to_wcnf()

dimacs_header = next(line for line in dimacs.splitlines() if not line.startswith("c "))
wcnf_header = next(line for line in wcnf.splitlines() if not line.startswith("c "))

assert dimacs_header.startswith("p cnf ")
assert wcnf_header.startswith("p wcnf ")
assert sum(line.startswith("1 -") for line in wcnf.splitlines()) == problem.num_vars
```

These strings can be written to files and passed to external SAT or MaxSAT
solvers. Measured in-process capability is strong for medium instances: the
bivariate bicycle `[[72,12,6]]` code certifies in under a second, while
`[[144,12,12]]` takes roughly 15 minutes. Larger instances warrant exporting
WCNF to a branch-and-bound MaxSAT solver.

## Next Steps

- **[Stabilizer-Code Verification](stabilizer-code-verification.md)** - Build codes and search directly for low-weight logical Paulis
- **[Fault Tolerance Analysis](fault-tolerance.md)** - Explore the lower-level Rust checkers and gadget analysis
- **[Fault Catalog Tutorial](fault-catalog.md)** - Construct and inspect circuit fault catalogs
