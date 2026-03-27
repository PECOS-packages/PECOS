# Fault Tolerance Analysis

This guide covers PECOS's fault tolerance analysis tools: classifying errors, checking fault tolerance at different weights, propagating faults through circuits, and generating detector error models (DEMs).

## What You'll Learn

- Classifying errors as stabilizer, detectable, or undetectable logical
- Using `StabilizerFlipChecker` for code-level fault tolerance analysis
- Using `PauliPropChecker` for circuit-level Pauli propagation
- Generating Stim-compatible detector error models
- Distance calculation and logical operator discovery

## Error Classification

Every Pauli error on a stabilizer code falls into one of four classes:

| Class | Syndrome | Logical Error | Meaning |
|---|---|---|---|
| Stabilizer | None | No | Equivalent to a stabilizer element. Harmless. |
| Detectable | Non-trivial | No | Triggers syndrome. Decoder can correct it. |
| Detectable Logical | Non-trivial | Yes | Triggers syndrome but also causes logical error. Decoder may or may not correct. |
| Undetectable Logical | None | Yes | Logical failure with no syndrome. Fatal -- no decoder can help. |

A code is **fault-tolerant at weight t** if no weight-t error is an undetectable logical and every syndrome pattern maps uniquely to a correction.

## Level 1: Stabilizer Flip Analysis

`StabilizerFlipChecker` analyzes fault tolerance directly from a `StabilizerCodeSpec` using anti-commutation. No circuit needed -- it works purely from the code definition.

**Key insight:** An error E flips stabilizer S if and only if they anti-commute: {E, S} = 0.

### Basic Usage

```rust
use pecos_qec::{StabilizerCodeSpec, StabilizerFlipChecker, ErrorClass};
use pecos_core::{Xs, Zs, PauliString, QuarterPhase};

// Define a 3-qubit bit-flip code
let code = StabilizerCodeSpec::builder(3)
    .check(Zs([0, 1]))
    .check(Zs([1, 2]))
    .logical_z(Zs([0, 1, 2]))
    .logical_x(Xs([0]))
    .build()
    .unwrap();

let checker = StabilizerFlipChecker::new(&code);
```

### Classifying Individual Errors

```rust
use pecos_qec::{StabilizerCodeSpec, StabilizerFlipChecker, ErrorClass};
use pecos_core::{Xs, Zs};
use pecos_core::pauli::constructors::*;

let code = StabilizerCodeSpec::builder(3)
    .check(Zs([0, 1]))
    .check(Zs([1, 2]))
    .logical_z(Zs([0, 1, 2]))
    .logical_x(Xs([0]))
    .build()
    .unwrap();
let checker = StabilizerFlipChecker::new(&code);

// X error on qubit 0 -- detectable (flips first stabilizer, also hits logical X)
let result = checker.classify_error(&X(0));
assert!(result.is_detectable());
assert!(result.causes_logical_error());

// Z error on qubit 0 -- undetectable (commutes with all Z-stabilizers)
// but causes logical error (anticommutes with logical X)
let result = checker.classify_error(&Z(0));
assert!(matches!(result, ErrorClass::UndetectableLogical { .. }));
```

### Computing Flips

For detailed information about which stabilizers and logicals are affected:

```rust
use pecos_qec::{StabilizerCodeSpec, StabilizerFlipChecker};
use pecos_core::{Xs, Zs};
use pecos_core::pauli::constructors::*;

let code = StabilizerCodeSpec::builder(3)
    .check(Zs([0, 1])).check(Zs([1, 2]))
    .logical_z(Zs([0, 1, 2])).logical_x(Xs([0]))
    .build().unwrap();
let checker = StabilizerFlipChecker::new(&code);

let flips = checker.compute_flips(&X(1));
println!("Flipped stabilizers: {:?}", flips.stabilizers);  // {0, 1}
println!("Flipped logical Zs: {:?}", flips.logical_zs);
println!("Flipped logical Xs: {:?}", flips.logical_xs);
println!("Syndrome: {:?}", flips.syndrome(2));  // [true, true]
```

### Analyzing All Errors at a Weight

Enumerate all weight-t Pauli errors and classify each:

```rust
use pecos_qec::{StabilizerCodeSpec, StabilizerFlipChecker};
use pecos_core::{Xs, Zs};

let code = StabilizerCodeSpec::builder(3)
    .check(Zs([0, 1])).check(Zs([1, 2]))
    .logical_z(Zs([0, 1, 2])).logical_x(Xs([0]))
    .build().unwrap();
let checker = StabilizerFlipChecker::new(&code);

// Analyze all weight-1 errors
let analysis = checker.analyze_weight(1);

println!("Total errors: {}", analysis.total_errors);
println!("Stabilizer (harmless): {}", analysis.stabilizer_errors);
println!("Detectable: {}", analysis.detectable_no_logical);
println!("Undetectable logical: {}", analysis.undetectable_logical);
println!("Detectable with logical: {}", analysis.detectable_with_logical);

// Is the code fault-tolerant at weight 1?
println!("Fault-tolerant: {}", analysis.is_fault_tolerant());
```

### Filtering by Pauli Type

For CSS codes, you can analyze X, Y, and Z errors separately:

```rust
use pecos_qec::{StabilizerCodeSpec, StabilizerFlipChecker};
use pecos_core::{Xs, Zs};

let code = StabilizerCodeSpec::builder(3)
    .check(Zs([0, 1])).check(Zs([1, 2]))
    .logical_z(Zs([0, 1, 2])).logical_x(Xs([0]))
    .build().unwrap();
let checker = StabilizerFlipChecker::new(&code);

// Only X errors (bit-flip)
let x_analysis = checker.analyze_weight_with_types(1, true, false, false);

// Only Z errors (phase-flip)
let z_analysis = checker.analyze_weight_with_types(1, false, false, true);

// X and Z but not Y
let xz_analysis = checker.analyze_weight_with_types(1, true, false, true);
```

## Level 2: Pauli Propagation Analysis

`PauliPropChecker` propagates Pauli errors through a specific syndrome extraction circuit. This verifies whether a particular circuit implementation is fault-tolerant.

```rust
use pecos_qec::PauliPropChecker;
use pecos_quantum::TickCircuit;

// Build a syndrome extraction circuit for the 3-qubit bit-flip code
let mut circuit = TickCircuit::new();
circuit.tick().pz(&[0, 1, 2, 3, 4]);    // Initialize all qubits
circuit.tick().cx(&[(0, 3), (1, 4)]);    // CNOT: data -> ancilla
circuit.tick().cx(&[(1, 3), (2, 4)]);    // Second round of CNOTs
circuit.tick().mz(&[3, 4]);              // Measure ancillas

let checker = PauliPropChecker::new(&circuit);

// Define syndrome ancillas and logical operators
let z_ancillas = &[3, 4];
let x_ancillas: &[usize] = &[];
let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])]; // Z logical

// Analyze all single-fault locations
let results = checker.analyze_all_faults(z_ancillas, x_ancillas, logicals);
println!("Total fault locations analyzed: {}", results.len());
```

## Level 3: Gadget-Level Analysis

`GadgetChecker` extends circuit-level analysis with explicit input/output tracking. QEC protocols are composed of gadgets -- circuits where qubits may carry errors from previous stages (inputs) and pass errors to subsequent stages (outputs).

**Key constraint for t-fault tolerance:** `s + r <= t`, where s = input fault weight, r = internal fault weight.

### Gadget Types

| Type | Input | Output | Example |
|------|-------|--------|---------|
| State Preparation | None | Data qubits | Prepare logical zero |
| Syndrome Extraction | Data qubits | Data qubits + syndrome | EC round |
| Measurement | Data qubits | None (all measured) | Final readout |
| Gate | Data qubits | Data qubits | Logical CNOT |
| Self-contained | None | None | Full QEC experiment |

### Basic Usage

```rust
use pecos_qec::fault_tolerance::{GadgetConfig, GadgetChecker};
use pecos_quantum::TickCircuit;

// Build a syndrome extraction gadget
let mut circuit = TickCircuit::new();
// Data qubits 0,1,2 are INPUT (not initialized here)
circuit.tick().pz(&[3, 4]);           // Initialize ancillas only
circuit.tick().cx(&[(0, 3), (1, 4)]); // CNOTs from data to ancilla
circuit.tick().cx(&[(1, 3), (2, 4)]);
circuit.tick().mz(&[3, 4]);           // Measure ancillas
// Data qubits 0,1,2 are OUTPUT (not measured here)

let config = GadgetConfig::syndrome_extraction()
    .with_input_qubits(&[0, 1, 2])   // Data qubits enter with potential errors
    .with_output_qubits(&[0, 1, 2])  // Data qubits leave (may have errors)
    .with_ancilla_qubits(&[3, 4])    // Initialized and measured within gadget
    .with_z_ancillas(&[3, 4])        // For syndrome extraction
    .with_logical_z(&[], &[0, 1, 2]); // Z logical operator

let checker = GadgetChecker::new(&circuit, config);
let analysis = checker.analyze(1); // Check 1-fault tolerance

println!("Is 1-FT: {}", analysis.is_fault_tolerant());
println!("Total tested: {}", analysis.total_tested);
println!("Undetected logical: {}", analysis.undetected_logical);
println!("Excessive output: {}", analysis.excessive_output);
```

### Fault Classification

`GadgetChecker` classifies each fault combination into:

| Class | Description | Failure? |
|---|---|---|
| `Harmless` | No effect (identity or stabilizer) | No |
| `Correctable` | Detectable, no logical error, bounded output weight | No |
| `DetectedLogicalError` | Detectable but causes logical error | No (detected) |
| `UndetectedLogicalError` | No syndrome but causes logical failure | Yes (fatal) |
| `ExcessiveOutputError` | Output error weight exceeds threshold | Yes (fatal) |

### When to Use Which Checker

| Feature | `StabilizerFlipChecker` | `PauliPropChecker` | `GadgetChecker` |
|---|---|---|---|
| Input required | `StabilizerCodeSpec` only | Circuit (`TickCircuit`) | Circuit + `GadgetConfig` |
| What it checks | Code-level properties | Circuit correctness | Gadget fault tolerance |
| Input/output faults | No | No | Yes (s + r <= t) |
| Output error tracking | No | No | Yes (weight bounds) |
| Fault locations | All weight-t Paulis on data qubits | All spacetime locations | Input + spacetime locations |
| Speed | Fast (anti-commutation) | Moderate (propagation) | Moderate (more combinations) |

## Detector Error Models (DEM)

The `DemBuilder` generates Stim-compatible detector error models from fault influence maps. This connects PECOS's fault analysis to external decoders.

```hidden-rust
use pecos_qec::DemBuilder;
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a simple parity check circuit
    let mut dag = DagCircuit::new();
    dag.pz(&[2]);       // prepare ancilla
    dag.cx(&[(0, 2)]);  // parity check
    dag.cx(&[(1, 2)]);
    dag.mz(&[2]);       // measure syndrome

    // Analyze faults to build influence map
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    // Define detectors and observables
    let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;
    let observables_json = r"[]";

    // CODE
    Ok(())
}
```

```rust
use pecos_qec::DemBuilder;

// Build DEM from a fault influence map
let dem = DemBuilder::new(&influence_map)
    .with_noise(0.01, 0.01, 0.01, 0.01)  // p1, p2, p_meas, p_init
    .with_detectors_json(detectors_json)?
    .with_observables_json(observables_json)?
    .build();

println!("DEM has {} detectors, {} contributions",
    dem.num_detectors(), dem.num_contributions());
```

**Decomposition:** MWPM decoders work on graphs, not hypergraphs. When an error mechanism affects 3+ detectors (a hyperedge), it can be decomposed into combinations of graphlike (1-2 detector) errors.

## Distance Calculation

The `distance` module provides configurable distance search:

```rust
use pecos_qec::{StabilizerCodeSpec, calculate_distance, DistanceSearchConfig};
use pecos_core::{Xs, Zs};

let code = StabilizerCodeSpec::builder(7)
    .check(Xs([0, 2, 4, 6]))
    .check(Xs([1, 2, 5, 6]))
    .check(Xs([3, 4, 5, 6]))
    .check(Zs([0, 2, 4, 6]))
    .check(Zs([1, 2, 5, 6]))
    .check(Zs([3, 4, 5, 6]))
    .logical_z(Zs([0, 2, 4, 6]))
    .logical_x(Xs([0, 2, 4, 6]))
    .build()
    .unwrap();

// Basic distance calculation
let result = calculate_distance(&code, &DistanceSearchConfig::default());
if let Some(r) = result {
    println!("Distance: {}", r.distance);
    println!("Min-weight operator: {}", r.min_weight_operator);
}

// CSS-only search (faster for CSS codes)
let result = calculate_distance(&code, &DistanceSearchConfig::css());

// Bounded search (stop at weight 5)
let result = calculate_distance(&code, &DistanceSearchConfig::with_max_weight(5));
```

### Finding All Minimum-Weight Logicals

```rust
use pecos_qec::{StabilizerCodeSpec, find_min_weight_logicals_with_info, DistanceSearchConfig};
use pecos_core::{Xs, Zs};

let code = StabilizerCodeSpec::builder(7)
    .check(Xs([0, 2, 4, 6]))
    .check(Xs([1, 2, 5, 6]))
    .check(Xs([3, 4, 5, 6]))
    .check(Zs([0, 2, 4, 6]))
    .check(Zs([1, 2, 5, 6]))
    .check(Zs([3, 4, 5, 6]))
    .logical_z(Zs([0, 2, 4, 6]))
    .logical_x(Xs([0, 2, 4, 6]))
    .build()
    .unwrap();

let logicals = find_min_weight_logicals_with_info(&code, &DistanceSearchConfig::default());
for op in &logicals {
    println!("Weight {}: {} (equivalent to {})",
        op.weight, op.operator, op.equivalence_string());
}
```

`LogicalOperatorInfo::equivalence_string()` shows which logical operation each operator implements (e.g., "X0", "Z1", "X0*Z1").

## Logical Operator Discovery

When you have stabilizer generators but don't know the logical operators, `discover_logical_operators` finds them automatically using stabilizer simulation:

```rust
use pecos_qec::discover_logical_operators;
use pecos_core::pauli::constructors::Zs;

// Define stabilizers for the 3-qubit bit-flip code
let stabilizers = vec![Zs([0, 1]), Zs([1, 2])];

// Discover logical operators
let result = discover_logical_operators(3, &stabilizers).unwrap();

println!("Logical qubits: {}", result.num_logical_qubits);  // 1
println!("Logical Xs: {:?}", result.logical_xs);
println!("Logical Zs: {:?}", result.logical_zs);
println!("Destabilizers: {:?}", result.destabilizers);
```

The discovery algorithm:
1. Creates a simulator with n + m qubits (n data, m ancilla)
2. Encodes each stabilizer by initializing an ancilla in |+> and applying controlled-Pauli gates
3. Measures ancillas deterministically
4. Extracts logical operators from the remaining tableau generators

The result includes properly paired (X_i, Z_i) logical operators and destabilizers, ready for use with `StabilizerCodeSpec`.

## Putting It Together

A typical fault tolerance analysis workflow:

```rust
use pecos_qec::{
    StabilizerCode, StabilizerCodeSpec, StabilizerFlipChecker,
    calculate_distance, DistanceSearchConfig,
};

// 1. Start from a code definition
let code = StabilizerCode::steane();

// 2. Convert to StabilizerCodeSpec (discovers logicals)
let spec = StabilizerCodeSpec::from_stabilizer_code(&code).unwrap();

// 3. Verify algebraic correctness
spec.verify().unwrap();

// 4. Compute distance
let dist = calculate_distance(&spec, &DistanceSearchConfig::default());
println!("Distance: {:?}", dist.as_ref().map(|r| r.distance));

// 5. Check fault tolerance at weight 1
let checker = StabilizerFlipChecker::new(&spec);
let analysis = checker.analyze_weight(1);
println!("Fault-tolerant at weight 1: {}", analysis.is_fault_tolerant());

// 6. For a distance-3 code, also check weight 2
if dist.as_ref().is_some_and(|r| r.distance >= 3) {
    let analysis_2 = checker.analyze_weight(2);
    println!("Fault-tolerant at weight 2: {}", analysis_2.is_fault_tolerant());
}
```
