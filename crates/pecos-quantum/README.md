# pecos-quantum

Quantum circuit representation data structures.

## Purpose

Provides quantum circuit representation data structures for PECOS, including DAG-based and tick-based circuit representations.

## Key Types

- `DagCircuit` - Quantum circuit as a directed acyclic graph
- `TickCircuit` - Quantum circuit as sequences of parallel time slices
- `Circuit`, `CircuitMut` - Circuit traits
- `Gate`, `GateType` - Gate representations

## Usage

```rust
use pecos_quantum::{DagCircuit, Gate, QubitId};

let mut circuit = DagCircuit::new();
let h = circuit.add_gate(Gate::h(&[0]));
let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));
circuit.connect(h, cx, QubitId::from(0)).unwrap();
```

### Gate validation

`Gate` constructors check native angle arity, and DAG insertion and mutation
validate gate payloads. `Gate` fields remain public: these checks do not make
malformed states unrepresentable. Full encapsulation of both the gate type and
angles is separate work.

### Atomic tick insertion and lookup

`Tick::try_add_gates` checks the complete request, including overlaps between its
batches, before inserting any gate. It preserves command boundaries and returns
indices for attaching per-command metadata. Single-batch typed broadcasts use
`try_add_gate`, which likewise validates the entire batch before mutation.

`Tick::gate_on_qubit` uses a maintained per-qubit index to locate an occupying
batch in O(log n) time. Insertions, replacements, splitting, coalescing and
removals maintain that index; callers can read the returned batch and its
attributes without cloning the tick. These operations support Python's prepared
requests: only free batches are committed after native fields and decoded views
have established that every overlap is an identical application.

Single removal deletes the removed qubits and scans the index to decrement shifted
batch indices when removing before the last batch. Removing the last batch does
not scan the remaining index, including when a merge has already reassigned its
qubits to a surviving batch. Bulk removal (including discard and optimization
passes) compacts commands and attributes together and remaps the qubit index once.

Custom gate signatures are checked without mutation before insertion and are
registered only when the gate is accepted, in both typed and Python builders.

## Circuit diagrams

Both circuit types provide `to_svg()`, `to_tikz()`, `to_dot()`, `to_ascii()`, and
`to_unicode()`. They return strings and require no Python, notebook, plotting
library, or external renderer. Save SVG directly with `std::fs::write`; TikZ
and DOT exports are source for use in external tools.

Use `DiagramStyle` and `render_with` for styling:

```rust
use pecos_quantum::{AngleUnit, DiagramStyle, TickCircuit};

let mut circuit = TickCircuit::new();
circuit.tick().h(&[0]);
circuit.tick().cx(&[(0, 1)]);
let style = DiagramStyle::builder()
    .color(false)
    .angle_unit(AngleUnit::Turns)
    .build();
let svg = circuit.render_with(&style).svg();
let tikz = circuit.render_with(&style).tikz();
```

For source-aware inspection, `circuit.inspect(&style)` returns an owned SVG and
records for its stored gate batches or DAG nodes. `inspection.to_html()` exports
a standalone interactive inspector with batch highlighting, keyboard selection,
and operation details. It works offline, with all assets embedded. The same
Rust viewer supports Python notebook embedding. IDs are scoped to the unchanged
source circuit; a stored batch can contain several gate instances.

Run the standalone export example with an output directory:

```sh
cargo run -p pecos-quantum --example circuit_exports -- /tmp/pecos-diagrams
```

Its `circuit-inspector.html` demonstrates interactive operation inspection.

The [inspector regression suite](tests/inspector/README.md) tests freshly exported
HTML for selection, search, trace linking, and viewport navigation. Its DOM and
modeled geometry checks run in CI; browser visual checks remain separate.

For annotated examples covering controls, interactions, measurement identities,
leakage-aware readout, ranges, and angle formatting, generate the visual gallery:

```sh
cargo run -p pecos-quantum --example circuit_gallery -- /tmp/pecos-gallery
```

Open its `index.html` locally to inspect SVGs and follow links to TikZ, DOT, and
text exports. The combined `gallery.tex` can also be compiled with LaTeX and the
TikZ, `geometry`, and `graphicx` packages. Both examples replace their named
outputs when rerun.

`TickCircuit` diagrams use explicit ticks; `DagCircuit` diagrams use dependency
layers. Neither axis represents physical time. DOT output renders the wire
diagram rather than exporting the original DAG's complete edge structure.
