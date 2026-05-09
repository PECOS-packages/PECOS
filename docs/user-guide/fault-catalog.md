# Fault Catalog and Measurement Sampling

## Quick start

If you have a surface code and want to simulate noisy measurements:

<!--skip: requires pecos surface code infrastructure-->
```python
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model
from pecos_rslib_exp import sim_neo, meas_sampling, depolarizing

patch = SurfacePatch.create(distance=3)
tc = _build_surface_tick_circuit_for_native_model(patch, rounds=6, basis="Z")

result = (
    sim_neo(tc)
    .quantum(meas_sampling())
    .noise(depolarizing().p1(0.001).p2(0.01).p_meas(0.005).p_prep(0.005))
    .shots(10000)
    .seed(42)
    .run()
)

# result[shot] gives measurement outcomes for each shot
print(f"{len(result)} shots, {len(result[0])} measurements each")
```

If you want to inspect what faults are possible in that circuit:

<!--skip: requires pecos surface code infrastructure-->
```python
from pecos_rslib_exp import fault_catalog

# Build structural catalog (no noise commitment):
catalog = fault_catalog(tc)
print(f"{len(catalog)} fault locations")

# Parameterize to get probabilities:
catalog.with_noise(p1=0.001, p2=0.01, p_meas=0.005, p_prep=0.005)
```

The rest of this tutorial uses a small hand-built circuit to explain the
concepts. Everything works the same way with surface codes or any other
`TickCircuit`.

## What is the fault catalog?

The fault catalog exposes every possible physical fault event in a circuit:
what Pauli error occurs at each gate, which measurements flip, which
detectors fire, which observables flip, and which tracked operators are
affected.

It serves two purposes:

1. **Research tool** -- explore fault anatomy, build custom decoders, analyze
   noise sensitivity, find undetectable errors, all without committing to a
   noise model.
2. **Sampling infrastructure** -- parameterize with noise, then sample raw
   measurements or enumerate fault configurations for decoding.

The core model is:

- each `FaultLocation` is an independent physical fault mechanism
- each location has one or more `FaultAlternative`s
- exactly one alternative is chosen when the location fires
- measurement, detector, observable, and tracked-operator effects combine by
  XOR parity

## Structural vs Parameterized Catalogs

A fault catalog can be built in two ways:

**Structural (no noise):** includes all fault locations for all gate types.
Probabilities are zero. Use this for topology queries, fault anatomy
exploration, or as a reusable base for parameter sweeps.

**Parameterized (with noise):** fills in probabilities based on a stochastic
noise model. Use this for sampling, decoding, and probability-weighted queries.

<!--setup-->
```python
from pecos.quantum import PauliString, TickCircuit
from pecos_rslib_exp import depolarizing, fault_catalog

circuit = TickCircuit()
circuit.tick().h([0])
circuit.tick().mz([0])

circuit.set_meta("num_measurements", "1")
circuit.set_meta("detectors", '[{"records":[-1]}]')
circuit.set_meta("observables", '[{"records":[-1]}]')

# Structural -- all locations, zero probabilities:
catalog = fault_catalog(circuit)

# Parameterize with noise:
catalog.with_noise(p1=0.03, p2=0.0, p_meas=0.01, p_prep=0.0)

# Or one-shot (structural + parameterize in one call):
catalog = fault_catalog(circuit, p1=0.03, p2=0.0, p_meas=0.01, p_prep=0.0)
```

The circuit must have detector and observable metadata (`num_measurements`,
`detectors`, `observables`). The catalog uses this metadata to map raw
measurement flips into detector and observable flips. Without metadata,
structural fields like `affected_detectors` will be empty, but
`affected_measurements` are still computed from Pauli propagation.

## Re-parameterization

The expensive work (Pauli propagation, detector mapping) is done once during
construction. Changing noise is cheap -- it just updates probability fields:

<!--continuation-->
```python
catalog = fault_catalog(circuit)

# Sweep noise parameters without rebuilding the catalog:
for p in [0.001, 0.005, 0.01, 0.05]:
    catalog.with_noise(p1=p * 0.1, p2=p, p_meas=p * 0.5, p_prep=p * 0.5)
    # ... decode, sample, analyze ...

# Independent copy for parallel comparison:
catalog_a = catalog.parameterized(p1=0.001, p2=0.01, p_meas=0.005, p_prep=0.005)
catalog_b = catalog.parameterized(p1=0.01, p2=0.1, p_meas=0.05, p_prep=0.05)
```

Note: decoders and samplers built from a catalog are snapshots. They read
probabilities at construction time. Re-parameterizing the catalog does NOT
update existing decoders or plans.

The returned object is sequence-like:

<!--continuation-->
```python
print(len(catalog))
print(catalog[0])
print(catalog[-1])

for location in catalog:
    print(location.tick, location.gate_type, location.channel)
```

It also exposes the underlying locations list:

```python
locations = catalog.locations
```

## Location Fields

Each `FaultLocation` represents one physical fault mechanism.

| Field | Meaning |
|---|---|
| `tick` | Tick index in the `TickCircuit` |
| `gate_index` | Gate index within the tick |
| `gate_type` | Gate name, such as `"H"`, `"CX"`, or `"MZ"` |
| `qubits` | Qubits acted on by this gate |
| `channel` | `"p1"`, `"p2"`, `"p_meas"`, or `"p_prep"` |
| `channel_probability` | Total probability the mechanism fires (0 if unparameterized) |
| `no_fault_probability` | `1 - channel_probability` |
| `num_alternatives` | Number of alternatives at this location, `k_i` |
| `faults` | List of `FaultAlternative` objects |

Example:

```python
for loc in catalog:
    print(
        loc.tick,
        loc.gate_type,
        loc.channel,
        loc.channel_probability,
        loc.no_fault_probability,
        loc.num_alternatives,
    )
```

The catalog includes locations with nonzero channel probability even when all
alternatives have empty detector/observable effects. This is necessary for
correct probability accounting.

## Alternative Fields

Each `FaultAlternative` is one possible outcome when its parent location fires.

| Field | Meaning |
|---|---|
| `kind` | `"pauli"`, `"measurement_flip"`, or `"prep_flip"` |
| `pauli` | A PECOS `PauliString` for Pauli faults, or `None` |
| `measurements` | Raw measurement indices flipped |
| `detectors` | Detector indices flipped |
| `observables` | Observable indices flipped |
| `tracked_ops` | Tracked operator indices flipped |
| `conditional_probability` | `1 / k_i` (structural, does not depend on noise) |
| `absolute_probability` | `p_i / k_i` (0 if unparameterized) |
| `channel_probability` | Same `p_i` as the parent location |

The four effect fields (`measurements`, `detectors`, `observables`,
`tracked_ops`) are structural -- they depend on the circuit topology, not the
noise model. They are populated during construction and never change when
noise is re-parameterized.

Example:

```python
for loc in catalog:
    for fault in loc.faults:
        print(f"  {fault.kind}: {fault.pauli}")
        print(f"    measurements: {fault.measurements}")
        print(f"    detectors:    {fault.detectors}")
        print(f"    observables:  {fault.observables}")
        print(f"    tracked_ops:  {fault.tracked_ops}")
```

`fault.absolute_probability` is local to one fault location. It is not the
probability of "this alternative and no other faults in the circuit."

## Probability Semantics

Understanding probabilities matters when you are building decoders, computing
thresholds, or verifying that a noise model produces the expected error rates.

For location `i`:

```text
p_i = location.channel_probability
k_i = location.num_alternatives
P(no fault at i) = 1 - p_i
P(specific alternative at i) = p_i / k_i
```

For a full-circuit configuration:

```text
configuration_probability
  = product selected alternatives (p_i / k_i)
    * product unselected locations (1 - p_i)
```

For a single selected alternative at location `i`, the full event probability is:

```python
selected_location_index = 0
fault = catalog[selected_location_index].faults[0]

event_probability = fault.absolute_probability
for j, loc in enumerate(catalog.locations):
    if j != selected_location_index:
        event_probability *= loc.no_fault_probability
```

## Lazy k-Fault Configurations

Use `catalog.fault_configurations(k)` to enumerate every way exactly `k`
faults can occur simultaneously. This is the foundation for building lookup
decoders, computing truncated ML tables, and analyzing multi-fault error
patterns.

For `k = 0`, the iterator yields exactly one no-fault configuration. Its
probability is the product of every location's `no_fault_probability`:

```python
configs = list(catalog.fault_configurations(0))
assert len(configs) == 1

no_fault = configs[0]
assert no_fault.location_indices == []
assert no_fault.alternative_indices == []
assert no_fault.measurements == []
assert no_fault.detectors == []
assert no_fault.observables == []
assert no_fault.selected_probability == 1.0

expected = 1.0
for loc in catalog.locations:
    expected *= loc.no_fault_probability

assert abs(no_fault.configuration_probability - expected) < 1e-12
```

For `k > 0`, each yielded `FaultConfiguration` has:

| Field | Meaning |
|---|---|
| `location_indices` | Indices into `catalog.locations` |
| `alternative_indices` | Chosen alternative index for each selected location |
| `locations` | The selected `FaultLocation` objects |
| `faults` | The selected `FaultAlternative` objects |
| `measurements` | XOR-combined measurement effects |
| `detectors` | XOR-combined detector effects |
| `observables` | XOR-combined observable effects |
| `selected_probability` | Product of selected `absolute_probability` values |
| `configuration_probability` | Selected probability times no-fault probabilities for unselected locations |

The iterator is lazy:

```python
it = catalog.fault_configurations(1)
first = next(it)

print(first.location_indices)
print(first.alternative_indices)
print(first.locations[0] is catalog.locations[first.location_indices[0]])
print(first.faults[0] is first.locations[0].faults[first.alternative_indices[0]])
```

## XOR Parity

When multiple faults occur simultaneously, their effects combine by XOR parity.
If two faults flip the same detector, that detector cancels. This is fundamental
to QEC -- it's why weight-2 errors can be undetectable even when each individual
fault triggers detectors.

```python
for event in catalog.fault_configurations(2):
    print(event.detectors, event.observables, event.configuration_probability)
```

This is the right behavior for detector syndromes and logical observable flips.
It is also why low-weight decoder tests should apply the selected correction and
check the residual logical by XOR.

## Building a Small Lookup Table

A lookup decoder table groups configuration probability by:

```text
detector syndrome -> logical observable class -> probability
```

For a truncated table:

```python
from collections import defaultdict


def add_weight(table, syndrome, logical, probability):
    table[tuple(syndrome)][tuple(logical)] += probability


table = defaultdict(lambda: defaultdict(float))

for k in range(0, 3):
    for event in catalog.fault_configurations(k):
        add_weight(
            table,
            event.detectors,
            event.observables,
            event.configuration_probability,
        )

decoder = {}
for syndrome, logical_weights in table.items():
    best_logical = max(logical_weights.items(), key=lambda item: item[1])[0]
    decoder[syndrome] = best_logical


# Apply the decoder: XOR the event's logical class with the correction
def xor_sorted(a, b):
    out = set(a)
    for item in b:
        if item in out:
            out.remove(item)
        else:
            out.add(item)
    return tuple(sorted(out))


for event in catalog.fault_configurations(1):
    correction = decoder[tuple(event.detectors)]
    residual = xor_sorted(event.observables, correction)
    print(event.detectors, residual)
```

## Rust API

The Rust API lives in `pecos-qec`:

```rust
use pecos_qec::fault_tolerance::fault_sampler::{
    FaultCatalog, StochasticNoiseParams,
};
use pecos_quantum::{Attribute, TickCircuit};

let mut circuit = TickCircuit::new();
circuit.tick().h(&[0]);
circuit.tick().mz(&[0]);

circuit.set_meta("num_measurements", Attribute::String("1".into()));
circuit.set_meta(
    "detectors",
    Attribute::String(r#"[{"records":[-1]}]"#.into()),
);
circuit.set_meta(
    "observables",
    Attribute::String(r#"[{"records":[-1]}]"#.into()),
);

// Structural catalog (no noise):
let mut catalog = FaultCatalog::from_circuit(&circuit).unwrap();

// Parameterize:
let noise = StochasticNoiseParams {
    p1: 0.03,
    p2: 0.0,
    p_meas: 0.01,
    p_prep: 0.0,
};
catalog.with_noise(&noise);

// Or one-shot convenience:
// let catalog = build_fault_catalog(&circuit, &noise).unwrap();
```

Iterate locations and alternatives:

```rust
for loc in &catalog.locations {
    println!(
        "tick={} gate={:?} channel={:?} p={} k={}",
        loc.tick,
        loc.gate_type,
        loc.channel,
        loc.channel_probability,
        loc.num_alternatives
    );

    for fault in &loc.faults {
        println!(
            "  {:?} dets={:?} obs={:?} tracked={:?} p_alt={}",
            fault.kind,
            fault.affected_detectors,
            fault.affected_observables,
            fault.affected_tracked_ops,
            fault.absolute_probability
        );
    }
}
```

Iterate configurations:

```rust
for event in catalog.fault_configurations(2) {
    println!(
        "locations={:?} alternatives={:?} dets={:?} obs={:?} p={}",
        event.location_indices,
        event.alternative_indices,
        event.affected_detectors,
        event.affected_observables,
        event.configuration_probability
    );
}
```

The Rust iterator borrows the catalog and does not materialize all
configurations up front. On an unparameterized catalog, `fault_configurations(k)`
for k > 0 yields nothing (all probabilities are zero).

## Fault Anatomy Exploration

The structural catalog (no noise needed) lets you explore every fault event:

```python
catalog = fault_catalog(circuit)

# What faults can flip detector D0?
for loc in catalog:
    for alt in loc.faults:
        if 0 in alt.detectors:
            print(f"D0 flipped by {alt.pauli} at {loc.gate_type}({loc.qubits})")

# Find undetectable weight-2 logical errors:
catalog.with_noise(p1=0.01, p2=0.05, p_meas=0.01, p_prep=0.01)
for config in catalog.fault_configurations(2):
    if config.observables and not config.detectors:
        print(f"Undetectable: locations {config.location_indices}")
```

## Tracked Operators

Tracked operators are Pauli operators that the catalog monitors for
anticommutation with fault events. Unlike observables, they have no
measurement records -- they are detected by forward Pauli propagation.

Add tracked operators to a circuit via `pauli_operator`:

```python
tc2 = TickCircuit()
tc2.tick().h([0, 1])
tc2.tick().mz([0, 1])
tc2.set_meta("num_measurements", "2")
tc2.set_meta("detectors", "[]")
tc2.set_meta("observables", "[]")
tc2.pauli_operator(PauliString.from_str("XX"), label="logical_X")

cat2 = fault_catalog(tc2, p1=0.01, p2=0.0, p_meas=0.01, p_prep=0.0)
for loc in cat2:
    for alt in loc.faults:
        if alt.tracked_ops:
            print(f"{alt.pauli} flips tracked ops {alt.tracked_ops}")
```

Each `FaultAlternative` then has a `tracked_ops` field listing which tracked
operators are flipped by that fault. This is useful for studying logical
operator propagation without requiring measurement.

## Raw Measurement Sampling

The `meas_sampling()` backend in `sim_neo` uses the fault catalog internally
to produce raw measurement bitstrings:

```python
from pecos_rslib_exp import sim_neo, meas_sampling, depolarizing

result = (
    sim_neo(circuit)
    .quantum(meas_sampling())
    .noise(depolarizing().p1(0.001).p2(0.01).p_meas(0.005).p_prep(0.005))
    .shots(10000)
    .seed(42)
    .run()
)

# result[shot] gives measurement outcomes for each shot
for shot in range(len(result)):
    measurements = list(result[shot])
```

The sampling architecture:
- **Ideal values** from symbolic stabilizer simulation (respects measurement
  correlations across resets)
- **Physical faults** from geometric skip sampling (O(fired events) per shot)
- Raw measurement = ideal XOR faults

This is fast (millions of shots per second at small distances) and produces
the same measurement format as gate-by-gate stabilizer simulation.

## Common Pitfalls

- `fault.absolute_probability` is `p_i / k_i`, not a full-circuit event
  probability.
- Empty-effect alternatives (no measurements flipped) are real -- they
  represent Pauli errors that commute with subsequent measurements. They
  must stay in the catalog for the correct uniform denominator.
- `catalog.fault_configurations(k)` means exactly `k` distinct physical
  locations fire, not at most `k`. On an unparameterized catalog, k > 0
  yields nothing (zero probabilities).
- Detector and observable metadata must be correct before building the catalog.
  Missing boundary detectors can make a correct decoder appear to fail.
- `with_noise()` mutates the catalog in place. Previously held Python
  references to locations and faults update automatically. Decoders and
  samplers do NOT update -- they are snapshots.
- The structural catalog includes ALL locations even when a channel
  probability is zero. Use `to_mechanisms()` to get only the nonzero
  mechanisms for raw sampling.

## Larger Example

The Rust example `examples/surface/d3_fault_catalog_lookup.rs` builds a
distance-3 surface-code memory experiment, walks `fault_configurations(k)` for
`k = 0..=2`, and aggregates a truncated lookup decoder table.

Run it from the repository root:

```bash
cargo run -p pecos-qec --example surface_d3_fault_catalog_lookup
```
