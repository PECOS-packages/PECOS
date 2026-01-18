# Pauli Propagation

This document explains how Pauli operators propagate through quantum circuits, and why backward propagation of observables is more efficient than forward simulation of faults.

## Two Layers: Structure and Physics

The propagation system has two distinct layers:

### DagCircuit (Structural Layer)

The `DagCircuit` represents a quantum circuit as a directed acyclic graph:
- **Nodes** are gates (H, CX, S, measurements, preparations, etc.)
- **Edges** are qubit wires connecting gates
- Traversal can start from any node and go forward (following data flow) or backward (against data flow)

This is pure graph structure with no physics. It answers: "What gates are connected to what?"

### Propagator (Physics Layer)

The propagator applies Clifford conjugation rules to track how Pauli operators transform through gates:
- Starts from specific points (measurements, logical operators, or arbitrary locations)
- Applies gate-specific transformation rules
- Tracks which qubits carry X, Z, or Y components

This layer answers: "If I have operator P here, what does it look like there?"

## Forward vs Backward Propagation

Both directions are valid and give equivalent results, but they serve different purposes.

### Forward Propagation: Tracking Errors

Forward propagation asks: "If an error P occurs at location L, what is its effect at the end of the circuit?"

```
Initial:     I ─── I ─── I ─── X ─── I ─── I
                              ↑
                         Error here

After CX:    I ─── I ─── I ─── X ─── X ─── I
                              │     │
                         control  target

After H:     I ─── I ─── I ─── X ─── Z ─── I
                                     │
                              (H swaps X↔Z)
```

Forward propagation is intuitive: you inject a fault and watch it spread. However, to find all faults that affect a specific measurement, you'd need to simulate every possible fault location separately.

### Backward Propagation: Tracking Observables

Backward propagation asks: "What observable is being measured, and where in the circuit would faults affect it?"

```
End:         I ─── I ─── I ─── I ─── I ─── Z   ← Z-measurement
                                           │
                                      observable

Before H:    I ─── I ─── I ─── I ─── X ─── Z
                                     │
                              (H† swaps X↔Z)

Before CX:   I ─── I ─── I ─── X ─── X ─── Z
                              │     │
                         (Z spread to control)
```

The key insight: **a fault P at location L flips the measurement iff P anticommutes with the back-propagated observable at L**.

In the example above, at the leftmost X position:
- An X fault commutes with X (no flip)
- A Z fault anticommutes with X (flips the measurement)
- A Y fault anticommutes with X (flips the measurement)

### Why Backward is More Efficient

| Approach | Work Required |
|----------|---------------|
| Forward (per fault) | Propagate one fault through circuit, check one measurement |
| Forward (all faults) | N fault locations × M measurements = O(N×M) propagations |
| Backward (per measurement) | Propagate one observable, check all N locations in one pass |
| Backward (all measurements) | M measurements × 1 propagation each = O(M) propagations |

For a circuit with N fault locations and M measurements, backward propagation is O(N) times faster.

## Clifford Gate Rules

Pauli operators transform through Clifford gates via conjugation: P → G† P G.

### Hadamard (H)

H is self-adjoint (H† = H), and swaps X ↔ Z:

| Input | Output |
|-------|--------|
| X | Z |
| Z | X |
| Y | -Y (tracked as Y) |

### S Gate (Phase Gate)

S rotates in the XY plane:

| Input | Forward (S† P S) | Backward (S P S†) |
|-------|------------------|-------------------|
| X | Y | -Y (tracked as Y) |
| Y | -X (tracked as X) | X |
| Z | Z | Z |

### CNOT (CX)

CX is self-adjoint. X spreads from control to target; Z spreads from target to control:

| Input | Output |
|-------|--------|
| X_c | X_c X_t |
| X_t | X_t |
| Z_c | Z_c |
| Z_t | Z_c Z_t |
| Y_c = X_c Z_c | X_c X_t Z_c = Y_c X_t |
| Y_t = X_t Z_t | X_t Z_c Z_t = Z_c Y_t |

### CZ

CZ is self-adjoint. X on either qubit adds Z to the other:

| Input | Output |
|-------|--------|
| X_0 | X_0 Z_1 |
| X_1 | Z_0 X_1 |
| Z_0 | Z_0 |
| Z_1 | Z_1 |

### Preparation (Reset)

Preparation resets a qubit to a known state. For backward propagation, this **kills** the observable on that qubit - errors before preparation don't affect measurements after it.

### Measurement

Measurement is a starting point for backward propagation. A Z-measurement starts with Z observable; an X-measurement starts with X observable.

## Anticommutation and Fault Detection

The fundamental rule: **fault P flips detector D iff P anticommutes with the back-propagated observable for D**.

Anticommutation on a single qubit:

| Observable | X fault | Y fault | Z fault |
|------------|---------|---------|---------|
| I | commutes | commutes | commutes |
| X | commutes | anticommutes | anticommutes |
| Y | anticommutes | commutes | anticommutes |
| Z | anticommutes | anticommutes | commutes |

For multi-qubit operators, anticommutation is determined by the product of per-qubit (anti)commutations. An odd number of anticommuting positions means the overall operators anticommute.

## Data Structures

The backward propagator produces a **fault influence map** relating fault locations to detectors/logicals.

### DagFaultInfluenceMap

Cache-optimized format using CSR (Compressed Sparse Row) arrays:

```rust
let map = propagator.build_influence_map();

// Fast classification without allocations
let (has_syndrome, has_logical) = map.classify_fault(loc_idx, pauli);

// Get detector indices for a specific fault
let detectors = map.get_detector_indices(loc_idx, pauli);

// Iterate over locations
for (loc_idx, loc) in map.locations.iter().enumerate() {
    if map.influences.has_detector_flips(loc_idx, Pauli::X) {
        // X fault at this location flips some detector
    }
}
```

Properties:
- Cache-friendly CSR memory layout
- Compact storage with minimal overhead
- Efficient for enumeration workloads

## Example: Syndrome Extraction

Consider a simple syndrome extraction with one data qubit and one ancilla:

```
q0 (data):    ─────────●─────────
                       │
q1 (ancilla): ──|0⟩────X────M────
```

Backward propagation from the measurement:

1. Start with Z on q1 (Z-measurement)
2. Before CX: Z on q1 spreads to Z on q0 (Z_target → Z_control Z_target)
3. Before Prep: Z on q1 is killed (preparation resets state)

Result: The measurement is sensitive to Z errors on q0 (before CX) and various errors on q1 (between prep and measurement).

The influence map captures:
- X fault on q0 before CX: flips measurement (X anticommutes with Z)
- Z fault on q0 before CX: no flip (Z commutes with Z)
- X fault on q1 after prep, before CX: flips measurement
- etc.

## Connection to Fault Tolerance

Backward propagation enables efficient fault tolerance analysis:

1. **Detector construction**: Propagate from each measurement to find which faults flip it
2. **Logical operator tracking**: Propagate from logical operators to find undetectable logical errors
3. **Syndrome ambiguity**: Group faults by syndrome pattern to identify decoder requirements

See [fault-tolerance.md](fault-tolerance.md) for how these maps are used in fault tolerance verification.
