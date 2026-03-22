# Fault Tolerance Analysis

This document covers fault classification, analysis approaches, and the connection to PECOS's symbolic infrastructure.

## Fault Classification

Individual faults are classified by their effect on syndrome and logical state:

| Class | Syndrome | Logical Error | Meaning |
|-------|----------|---------------|---------|
| **Stabilizer** | None | No | Error equivalent to stabilizer. Harmless. |
| **Detectable** | Non-trivial | No | Triggers syndrome. Decoder corrects. |
| **Detectable + Logical** | Non-trivial | Yes | Detected but may exceed correction capacity. |
| **Undetectable Logical** | None | Yes | Silent failure. Fatal. |

```rust
use pecos_qec::{StabilizerFlipChecker, ErrorClass};

let checker = StabilizerFlipChecker::new(&code);
match checker.classify_error(&error) {
    ErrorClass::Stabilizer => { /* Harmless */ }
    ErrorClass::Detectable { syndrome } => { /* Decoder handles it */ }
    ErrorClass::DetectableLogical { .. } => { /* Detected, may fail */ }
    ErrorClass::UndetectableLogical { .. } => { /* Fatal! */ }
}
```

## Fault Tolerance Verdict

For weight-t fault tolerance, the verdict is binary:

**Fault Tolerant**: All weight-t faults are either:
- Equivalent to stabilizers (harmless), OR
- Detectable with unique correction (decoder succeeds)

**Not Fault Tolerant**: At least one of:
- Undetectable logical errors exist
- Ambiguous syndromes (same syndrome, different logical effects)

```rust
let analysis = checker.analyze_weight(1);

if analysis.is_fault_tolerant() {
    println!("Code is 1-fault tolerant");
} else {
    for failure in analysis.failures() {
        println!("Failure: {}", failure);
    }
}
```

## Four Analysis Approaches

### Approach 1: Stabilizer Flip Analysis (Code-Level)

**Module**: `fault_tolerance::stabilizer_flip_checker`

Works at Level 1 (abstract code). Computes which stabilizers and logicals each error flips using anti-commutation.

```rust
use pecos_qec::StabilizerFlipChecker;

let checker = StabilizerFlipChecker::new(&code);
let analysis = checker.analyze_weight(1);

println!("Undetectable logical errors: {}", analysis.undetectable_logical);
println!("Is FT: {}", analysis.is_fault_tolerant());
```

**Strengths**:
- No circuit needed
- Works for dynamic circuits (stabilizer state is fundamental)
- Fast (direct anti-commutation check)

**Scope**: Analyzes code structure only. No concept of input/output qubits.

**Use when**: Analyzing code properties, designing new codes, dynamic protocols.

### Approach 2: Pauli Propagation Analysis (Circuit-Level)

**Module**: `fault_tolerance::pauli_prop_checker`

Works at Level 3 (circuits). Propagates Pauli errors through gates to determine final syndrome and logical effect.

```rust
use pecos_qec::PauliPropChecker;

let checker = PauliPropChecker::new(&circuit);
let analysis = checker.analyze_decoder_requirements(z_ancillas, x_ancillas, logicals);
```

**Strengths**:
- Circuit-aware (accounts for gate ordering)
- Handles gate-level fault locations
- Can analyze syndrome history for multi-round QEC

**Scope**: Analyzes internal circuit faults. Assumes all qubits start in known states (typically |0⟩). Does not handle input faults on data qubits that enter from a previous gadget.

**Use when**: Verifying self-contained circuits like state preparation.

### Approach 3: Full Simulation (Circuit-Level)

**Module**: `fault_tolerance::circuit_runner`

Runs full stabilizer simulation with fault injection. Most accurate but slowest.

```rust
use pecos_qec::FaultChecker;
use pecos_simulators::SparseStab;

let checker = FaultChecker::new(&circuit);
let result = checker.check(
    |sim: &SparseStab| check_for_failure(sim),  // Failure detection
    || SparseStab::new(n_qubits),                // Fresh simulator
);
```

**Strengths**:
- Exact simulation
- Can handle non-Clifford elements (with appropriate simulator)
- Validates actual circuit behavior
- Can include preparation errors via `with_initial_locations(true)`

**Scope**: Like PauliPropChecker, analyzes internal circuit faults. The `with_initial_locations` option includes preparation errors on ALL qubits, but does not distinguish input data qubits from ancillas.

**Use when**: Final verification, debugging, non-Clifford circuits.

### Approach 4: Gadget-Level Analysis (s + r ≤ t)

**Module**: `fault_tolerance::gadget_checker`

Analyzes QEC gadgets with explicit input/output/ancilla separation. Properly handles the fault tolerance condition s + r ≤ t where:
- **s** = weight of input faults (errors on data qubits entering the gadget)
- **r** = weight of internal faults (errors from gates within the gadget)
- **t** = fault tolerance level

```rust
use pecos_qec::{GadgetChecker, GadgetConfig};

// Syndrome extraction gadget with 7 data qubits
let config = GadgetConfig::builder()
    .input_qubits(vec![0, 1, 2, 3, 4, 5, 6])   // Data enters
    .output_qubits(vec![0, 1, 2, 3, 4, 5, 6])  // Data exits
    .ancilla_qubits(vec![7, 8, 9, 10, 11, 12]) // Prepared fresh
    .z_ancillas(vec![7, 8, 9])
    .x_ancillas(vec![10, 11, 12])
    .build();

let checker = GadgetChecker::new(&circuit, config);
let analysis = checker.analyze_fault_tolerance(1); // t=1
```

**Strengths**:
- Properly handles input/output data flow
- Enumerates all (s, r) combinations where s + r ≤ t
- Separates input faults from internal faults
- Tracks output error weight (must remain correctable)

**Scope**: For gadgets within a larger QEC protocol where data qubits flow through multiple stages.

**Use when**: Analyzing syndrome extraction, logical gates, or any gadget that receives encoded data from a previous stage.

## Choosing the Right Approach

| Scenario | Recommended Tool | Reason |
|----------|------------------|--------|
| Analyzing code distance | StabilizerFlipChecker | No circuit needed |
| State preparation gadget | PauliPropChecker or FaultChecker | All qubits start fresh |
| Syndrome extraction | GadgetChecker | Data qubits have input faults |
| Logical gates on encoded data | GadgetChecker | Must track input errors |
| Final measurement | PauliPropChecker | Often self-contained |
| Full protocol simulation | FaultChecker | Most accurate |

**Key question**: Do qubits enter this gadget already carrying errors from previous stages?
- **No** → Use PauliPropChecker or FaultChecker
- **Yes** → Use GadgetChecker with explicit input_qubits

## Syndrome History Analysis

For multi-round QEC, faults in round k may only be detected in round k+1. Single-shot analysis misses this.

```rust
let result = checker.analyze_with_syndrome_history(logicals);

println!("Measurement rounds: {}", result.rounds.len());
println!("Never-detected logical errors: {}", result.never_detected_logical_errors);
```

Syndrome history tracks:
- Which rounds each fault is detected in
- Whether faults are eventually detected or escape all rounds
- Unique syndrome history patterns for decoder analysis

## CSS Code Analysis

For CSS (Calderbank-Shor-Steane) codes, X and Z errors decouple:

```rust
// X-distance (bit-flip protection)
let x_analysis = checker.analyze_weight_with_types(t, true, false, false);

// Z-distance (phase-flip protection)
let z_analysis = checker.analyze_weight_with_types(t, false, false, true);
```

Example: 3-qubit bit flip code
- X-distance = 3 (corrects 1 X error)
- Z-distance = 1 (no Z protection)

## Connection to Symbolic Infrastructure

### SymbolicSparseStab (pecos-simulators)

Tracks measurement outcomes as functions of prior measurements:
```
m0 = ?           // Random (non-deterministic)
m1 = m0          // Correlated with m0
m2 = m0 ^ m1 ^ 1 // XOR formula
```

Deterministic measurements are implicit **detectors** - they should have fixed values absent errors.

### NoisyMeasurementHistory (pecos-experimental)

Extends symbolic tracking to include faults:
```
m3 = m0 ^ f1 ^ f3  // Depends on measurements AND faults
```

Same Pauli propagation as `PauliPropChecker`, but optimized for sampling rather than enumeration.

### The Detector Perspective

Another way to view fault tolerance:

1. Define **detectors** as measurement parities that should be deterministic
2. A fault **fires** detectors it affects
3. A fault is **detected** if it fires any detector
4. A fault is **undetectable** if it fires no detectors

This is equivalent to syndrome-based analysis but expressed in measurement correlation language. The symbolic infrastructure naturally captures these correlations.

## Complete Example

```rust
use pecos_qec::{StabilizerCode, StabilizerFlipChecker};
use pecos_core::{Xs, Zs};

// Define Steane code
let steane = StabilizerCode::builder(7)
    .check(Xs([0, 2, 4, 6]))
    .check(Xs([1, 2, 5, 6]))
    .check(Xs([3, 4, 5, 6]))
    .check(Zs([0, 2, 4, 6]))
    .check(Zs([1, 2, 5, 6]))
    .check(Zs([3, 4, 5, 6]))
    .logical_x(Xs([0, 1, 2, 3, 4, 5, 6]))
    .logical_z(Zs([0, 1, 2, 3, 4, 5, 6]))
    .build()
    .unwrap();

let checker = StabilizerFlipChecker::new(&steane);

// Verify [[7,1,3]] parameters
assert_eq!(checker.compute_distance(5), Some(3));

// Weight-1 is fault tolerant
let w1 = checker.analyze_weight(1);
assert!(w1.is_fault_tolerant());

// Weight-3 breaks fault tolerance
let w3 = checker.analyze_weight(3);
assert!(!w3.is_fault_tolerant());
```
