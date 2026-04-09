# Foreign Language Plugins

PECOS supports bidirectional integration with other programming languages. You can:

1. **Write plugins** (decoders, simulators) in your language that PECOS uses natively
2. **Use PECOS engines** from your language to run quantum circuits

## Quick Start (any language)

If your language can call C functions, you need two files:

1. **Build the shared library:** `cargo build -p pecos-ffi --release`
2. **Link against:** `target/release/libpecos_ffi.so` (Linux) / `libpecos_ffi.dylib` (macOS) / `pecos_ffi.dll` (Windows)
3. **Include:** `crates/pecos-foreign/include/pecos_foreign.h`

No Rust knowledge required. The header is self-contained with all type definitions and
function declarations.

For Python, Go, and Julia there are also language-specific wrappers that provide
idiomatic interfaces (see below), but the C ABI works for any language.

## API Versioning

Every vtable struct has a `version` field as its first member. PECOS checks this on
construction and rejects mismatches with a clear error message. This prevents silent
undefined behavior when a plugin compiled against an older ABI is loaded.

```c
PecosDecoderVTable vtable = {
    .version = PECOS_DECODER_VTABLE_VERSION,  /* always set this */
    .decode = my_decode,
    /* ... */
};
```

When PECOS changes a vtable layout, the version constant is bumped. Old plugins fail
fast with `"ABI version mismatch: plugin has v1, PECOS expects v2"` instead of
calling wrong function pointers.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    PECOS (Rust)                   │
│                                                   │
│  Decoder trait    CliffordGateable    Engine       │
│       ▲                 ▲               │         │
│       │                 │               ▼         │
│  ┌────┴────┐      ┌─────┴────┐   ┌────────────┐  │
│  │ Foreign │      │ Foreign  │   │ ByteMessage │  │
│  │ Decoder │      │Simulator │   │  Protocol   │  │
│  └────┬────┘      └─────┬────┘   └─────┬──────┘  │
│       │                 │               │         │
└───────┼─────────────────┼───────────────┼─────────┘
        │                 │               │
   ┌────┴─────────────────┴───────────────┴────┐
   │          Language-specific wrappers         │
   │                                            │
   │  Python: PyO3 (Py<PyAny>)                  │
   │  Go:     cgo + C ABI vtable                │
   │  Julia:  ccall + C ABI vtable              │
   │  C/C++:  C header (pecos_foreign.h)        │
   └────────────────────────────────────────────┘
```

## Writing a Decoder Plugin

A decoder plugin implements 3 methods:

| Method | Signature | Description |
|--------|-----------|-------------|
| `decode` | syndrome bytes in, observable + weight out | Run the decoding algorithm |
| `check_count` | -> int | Number of checks (parity check matrix rows) |
| `bit_count` | -> int | Number of bits (parity check matrix columns) |

### Python

```python
import pecos_rslib


class MyDecoder:
    def __init__(self, checks, bits):
        self._checks = checks
        self._bits = bits

    def decode(self, syndrome: bytes) -> dict:
        observable = bytearray(self._bits)
        # ... your decoding logic ...
        return {"observable": bytes(observable), "weight": 1.0, "converged": True}

    def check_count(self) -> int:
        return self._checks

    def bit_count(self) -> int:
        return self._bits


# Wrap and use
decoder = pecos_rslib.PyForeignDecoder(MyDecoder(100, 50))
result = decoder.decode(syndrome_bytes)
```

The `decode` method receives syndrome bytes and returns a dict with:
- `"observable"`: bytes or list of ints -- the decoded correction
- `"weight"`: float -- cost of the solution
- `"converged"` (optional): bool -- whether the decoder converged

### Go

```go
package main

import "github.com/PECOS-packages/PECOS/go/pecos"

type MyDecoder struct {
    checks int
    bits   int
}

func (d *MyDecoder) Decode(syndrome []byte) (*pecos.DecodingResult, error) {
    observable := make([]byte, d.bits)
    // ... your decoding logic ...
    converged := true
    return &pecos.DecodingResult{
        Observable: observable,
        Weight:     1.0,
        Converged:  &converged,
    }, nil
}

func (d *MyDecoder) CheckCount() int { return d.checks }
func (d *MyDecoder) BitCount() int   { return d.bits }

func main() {
    handle := pecos.RegisterDecoder(&MyDecoder{checks: 100, bits: 50})
    defer handle.Destroy()
}
```

### C/C++ (or any language via C ABI)

Include `pecos_foreign.h` and link against `libpecos_ffi.so`:

```c
#include "pecos_foreign.h"

int32_t my_decode(void *handle, const uint8_t *input, size_t len,
                  PecosDecodingResultRaw *result) {
    // ... your decoding logic ...
    result->observable_ptr = malloc(bit_count);
    result->observable_len = bit_count;
    result->weight = 1.0;
    result->converged = 1;
    result->error_ptr = NULL;
    return 0;
}

// Fill the vtable (version field is required)
PecosDecoderVTable vtable = {
    .version = PECOS_DECODER_VTABLE_VERSION,
    .decode = my_decode,
    .check_count = my_check_count,
    .bit_count = my_bit_count,
    .free_result = my_free_result,
    .free_error = my_free_error,
    .destroy = my_destroy,
};

// Register with PECOS
PecosDecoder *dec = pecos_foreign_decoder_create(my_state, &vtable);

// Use it
PecosDecodingResultRaw result = {0};
pecos_foreign_decoder_decode(dec, syndrome, syndrome_len, &result);

// Cleanup
pecos_foreign_decoder_free_observable(result.observable_ptr, result.observable_len);
pecos_foreign_decoder_free(dec);
```

## Writing a Simulator Plugin

A Clifford simulator needs 5 methods. All 52 other Clifford gates (X, Y, Z, SX, CZ, SWAP, etc.)
are decomposed into these 4 primitives automatically:

| Method | Signature | Description |
|--------|-----------|-------------|
| `sz` | qubits -> () | S (phase) gate |
| `h` | qubits -> () | Hadamard gate |
| `cx` | pairs -> () | CNOT gate |
| `mz` | qubits -> measurements | Z-basis measurement |
| `reset` | () -> () | Reset to initial state |

For universal (non-Clifford) simulators, add 3 rotation methods. All other rotations
(RY, T, Tdg, RXX, RYY, U, etc.) are decomposed automatically:

| Method | Signature | Description |
|--------|-----------|-------------|
| `rx` | (theta, qubits) -> () | X rotation (radians) |
| `rz` | (theta, qubits) -> () | Z rotation (radians) |
| `rzz` | (theta, pairs) -> () | ZZ rotation (radians) |

### Python

```python
import pecos_rslib


class MyStabilizerSim:
    def __init__(self, n):
        self.n = n
        # ... initialize your state ...

    def sz(self, qubits: list[int]):
        for q in qubits:
            pass  # apply S gate to qubit q

    def h(self, qubits: list[int]):
        for q in qubits:
            pass  # apply H gate to qubit q

    def cx(self, pairs: list[tuple[int, int]]):
        for control, target in pairs:
            pass  # apply CNOT

    def mz(self, qubits: list[int]) -> list[tuple[bool, bool]]:
        results = []
        for q in qubits:
            outcome = False  # measurement outcome
            deterministic = True  # whether outcome was deterministic
            results.append((outcome, deterministic))
        return results

    def reset(self):
        pass  # reset to |0...0>


sim = pecos_rslib.PyForeignSimulator(MyStabilizerSim(10))
```

### Go

```go
type MyStabSim struct {
    numQubits int
}

func (s *MyStabSim) SZ(qubits []int)                     { /* ... */ }
func (s *MyStabSim) H(qubits []int)                      { /* ... */ }
func (s *MyStabSim) CX(pairs [][2]int)                   { /* ... */ }
func (s *MyStabSim) MZ(qubits []int) []pecos.MeasurementResult { /* ... */ }
func (s *MyStabSim) Reset()                               { /* ... */ }

handle := pecos.RegisterSimulator(&MyStabSim{numQubits: 10})
```

## Using PECOS Engines from Foreign Languages

Foreign code can create and run PECOS quantum engines via the C ABI:

### Workflow

1. Create an engine (`pecos_engine_create`)
2. Build a circuit (`pecos_circuit_new`, `pecos_circuit_h`, etc.)
3. Serialize the circuit (`pecos_circuit_build`)
4. Run it (`pecos_engine_process`)
5. Parse results (`pecos_parse_outcomes`)
6. Reset for next shot (`pecos_engine_reset`)
7. Clean up (`pecos_engine_free`, `pecos_circuit_free`)

### Available Engine Types

| Engine | Type string | Description |
|--------|------------|-------------|
| State vector | `"state_vec"` | Full state vector simulation |
| Sparse stabilizer | `"sparse_stab"` | Clifford-only, sparse tableau |
| Stabilizer | `"stabilizer"` | Clifford-only, standard tableau |
| Clifford+RZ | `"clifford_rz"` | Sum-over-Cliffords for T/RZ gates |
| Density matrix | `"density_matrix"` | Mixed state simulation |
| Coin toss | `"coin_toss"` | Random outcomes (testing) |

### Available Circuit Gates

| Function | Gate | Parameters |
|----------|------|------------|
| `pecos_circuit_h` | Hadamard | qubits |
| `pecos_circuit_x` | Pauli X | qubits |
| `pecos_circuit_z` | Pauli Z | qubits |
| `pecos_circuit_sz` | S (phase) | qubits |
| `pecos_circuit_cx` | CNOT | pairs |
| `pecos_circuit_rx` | RX rotation | theta (radians), qubits |
| `pecos_circuit_rz` | RZ rotation | theta (radians), qubits |
| `pecos_circuit_rzz` | RZZ rotation | theta (radians), pairs |
| `pecos_circuit_mz` | Z measurement | qubits |

### C Example

```c
#include "pecos_foreign.h"

// Create a 2-qubit state vector engine with seed 42
PecosEngine *engine = pecos_engine_create("state_vec", 2, 42);

// Build a Bell state circuit
PecosCircuitBuilder *circuit = pecos_circuit_new();
size_t q0 = 0, q1 = 1;
size_t pair[] = {0, 1};
pecos_circuit_h(circuit, &q0, 1);
pecos_circuit_cx(circuit, pair, 1);
pecos_circuit_mz(circuit, (size_t[]){0, 1}, 2);

// Serialize and run
uint8_t *circuit_bytes;
size_t circuit_len;
pecos_circuit_build(circuit, &circuit_bytes, &circuit_len);

uint8_t *output_bytes;
size_t output_len;
pecos_engine_process(engine, circuit_bytes, circuit_len, &output_bytes, &output_len);

// Parse measurement results
uint32_t *outcomes;
size_t num_outcomes;
pecos_parse_outcomes(output_bytes, output_len, &outcomes, &num_outcomes);
// outcomes[0] and outcomes[1] will be correlated (Bell state)

// Cleanup
pecos_free_outcomes(outcomes, num_outcomes);
pecos_free_bytes(output_bytes, output_len);
pecos_free_bytes(circuit_bytes, circuit_len);
pecos_circuit_free(circuit);
pecos_engine_reset(engine);
pecos_engine_free(engine);
```

## Compatibility with pecos-neo

Foreign simulators work with both the current `pecos-engines` stack and the experimental
`pecos-neo` stack. Both systems use `CliffordGateable` as the simulator interface:

- **pecos-engines**: uses `ByteMessage` serialization and `Box<dyn QuantumEngine>`
- **pecos-neo**: uses typed commands and `CircuitRunner<S: CliffordGateable>` (generic, no vtable)

Since `ForeignSimulator` and `PyForeignSimulator` implement `CliffordGateable`, they plug
into either stack without modification:

```rust
// Works with pecos-neo's generic runner
let runner = CircuitRunner::<ForeignSimulator>::new();
```

The same applies to decoders -- the `Decoder` trait is shared between both stacks.

pecos-neo also has plugin systems for gate decompositions, noise channels, and orchestration
(Bevy-style `Tool` plugins). Foreign noise channels and classical control sources are
possible future extensions but require exposing additional Rust-specific types over FFI.

## Crate Structure

| Crate | Type | Purpose |
|-------|------|---------|
| `pecos-foreign` | rlib | Core: vtable types, `ForeignDecoder`/`ForeignSimulator`, engine API, FFI bridge functions |
| `pecos-ffi` | cdylib | Universal shared library (`libpecos_ffi.so`) -- link from any language |
| `pecos-foreign/include/pecos_foreign.h` | header | Self-contained C header with all types and function declarations |
| `pecos-rslib` | cdylib | Python: `PyForeignDecoder`, `PyForeignSimulator` via PyO3 |
| `pecos-go-ffi` | cdylib | Go: convenience wrapper (links `pecos-foreign` + Go-specific scaffolding) |
| `go/pecos/` | Go | `Decoder`/`CliffordSimulator` interfaces + cgo glue |
| `pecos-julia-ffi` | cdylib | Julia: convenience wrapper (links `pecos-foreign` + Julia scaffolding) |
| `julia/PECOS.jl/` | Julia | `AbstractDecoder`/`AbstractCliffordSimulator` types |

For new languages, only `pecos-ffi` + the C header are needed. The Go/Julia crates are
optional ergonomic wrappers -- they are not required for FFI access.

## Conformance Testing

Foreign simulators can run a built-in conformance test suite that verifies correctness.
The suite tests deterministic gates, Bell state correlation, reset behavior, and
derived gate decomposition. Any correct Clifford simulator must pass.

### From C

```c
PecosConformanceReport report;
int ok = pecos_run_conformance_tests(sim_handle, &vtable, num_qubits, &report);
printf("Passed %u/%u tests\n", report.tests_passed, report.tests_run);
```

### From Rust

```rust
let report = pecos_foreign::conformance::run_conformance_tests(&mut sim);
assert!(report.all_passed());
```

## Plugin Discovery

PECOS can discover and load foreign plugins at runtime from shared libraries.

### Plugin directory

Default: `~/.pecos/plugins/`

Place `.so` (Linux), `.dylib` (macOS), or `.dll` (Windows) files in this directory.
PECOS scans it and loads each plugin automatically.

### Plugin contract

A discoverable plugin exports one C function:

```c
int pecos_plugin_init(PecosPluginDescriptor *desc) {
    desc->name = "my-decoder";
    desc->plugin_api_version = PECOS_PLUGIN_API_VERSION;
    desc->decoder_handle = my_state;
    desc->decoder_vtable = &my_vtable;
    return 0;
}
```

The descriptor can provide a decoder, simulator, or both. PECOS validates the
API version and vtable versions on load.

### From Rust

```rust
use pecos_foreign::discovery::discover_plugins;
let plugins = discover_plugins(); // scans ~/.pecos/plugins/
for plugin in &plugins {
    println!("Loaded: {} (decoder: {}, simulator: {})",
        plugin.name, plugin.decoder.is_some(), plugin.simulator.is_some());
}
```

## Native Gate Set Configuration (pecos-neo)

When using foreign simulators with pecos-neo, the `gate_support` module (behind the
`neo` feature flag) automatically configures the `CircuitRunner` decomposition based
on what the foreign simulator supports:

```rust
use pecos_foreign::gate_support::configure_runner_for_foreign;

let sim = ForeignSimulator::new(handle, vtable);
let mut runner = configure_runner_for_foreign(&sim);
// If sim supports rotations: runner uses RX, RZ, RZZ natively
// Otherwise: Clifford-only, everything decomposes into {SZ, H, CX}
let outcomes = runner.apply_circuit(&mut sim, &commands)?;
```
