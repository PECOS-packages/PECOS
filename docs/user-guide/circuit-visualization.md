# Circuit Visualization

Render `TickCircuit`, `DagCircuit`, or a saved QIS operation trace in a notebook,
terminal, or file. These diagrams use PECOS's built-in Rust renderer; no plotting
package, Graphviz installation, or TeX compiler is required to generate them.
SVG and TikZ generation work independently of notebooks in both Rust and Python.

## Draw a Circuit

```python
from pecos.quantum import TickCircuit
from pecos.visualization import circuit_diagram

circuit = TickCircuit()
circuit.tick().pz([0, 1])
circuit.tick().h([0])
circuit.tick().cx([(0, 1)])
circuit.tick().mz_with_ids([0, 1], [0, 1])

diagram = circuit_diagram(circuit)
diagram  # SVG display when this is the last expression in a notebook cell
```

The returned `CircuitDiagram` is a snapshot: later edits to the circuit do not
change it. `str(diagram)` returns the rendered content. To save the SVG:

<!--continuation-->
```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    diagram.save(Path(directory) / "bell.svg")
```

`save` writes UTF-8 and replaces an existing file. The filename does not change
the format; select the format when calling `circuit_diagram`.

## Choose an Output Format

| Format | Output | Typical use |
|--------|--------|-------------|
| `svg` | SVG image, with automatic notebook display | Inspect or share a circuit |
| `ascii` | Text wire diagram | Terminal output and logs |
| `unicode` | Text with box-drawing characters | Readable terminal output |
| `tikz` | A TikZ `tikzpicture` source fragment | Include in a LaTeX document |
| `dot` | Graphviz DOT source for the wire diagram | External graph rendering |

<!--continuation-->
```python
print(circuit_diagram(circuit, format="unicode"))
monochrome = circuit_diagram(circuit, color=False)
turns = circuit_diagram(circuit, angle_unit="turns")
```

`color` affects graphical formats only; text output has no ANSI escape codes.
The default angle unit is `radians`, using compact pi notation where possible.
`turns` expresses a full rotation as one turn. Unsupported formats and angle
units raise `ValueError`.

Both circuit classes also expose direct string-returning methods:

<!--continuation-->
```python
svg = circuit.to_svg()
text = circuit.to_ascii()
unicode_text = circuit.to_unicode()
tikz = circuit.to_tikz()
dot = circuit.to_dot()
styled_svg = circuit.render("svg", color=False, angle_unit="turns")
```

Circuit objects themselves support notebook SVG display. Their ordinary Python
`repr` remains the compact circuit summary.

## Read Gate Symbols

A filled dot connected to an `X`, `Y`, or `H` box is a control; the box is the
target. Connected dots without a label denote CZ. Linked lowercase `x` boxes
represent SWAP. For symmetric interactions such as RZZ, the name appears on the
connector between wires; only the marked endpoints participate.

Custom gates retain their names. Multi-qubit custom gates, general `U2q` gates,
and correlated channels use `[0]`, `[1]`, etc. to identify operand order. These
indices refer to positions in the operation's qubit list, not qubit IDs.

`Ch` marks a channel's support, but does not show its payload or probability.
`XT-excl` marks exclusions from global crosstalk; `XT-local` marks local
crosstalk metadata. `TP` marks tracked-Pauli metadata without displaying its
contents. These metadata markers are not applied quantum gates. Consult the
circuit's channel and metadata payloads for quantitative noise analysis.

Angles are displayed modulo a full turn. Ordinary decimal labels are rounded;
very small nonzero angles use scientific notation so they remain distinct from
zero. A small negative value can denote the equivalent residual below a full
turn.

## Show Measurement Identities

Set `show_measurement_ids=True` to show stored identities beside measurement
symbols. It works with every output format, named panels, and tick/layer ranges:

<!--continuation-->
```python
readout = circuit_diagram(circuit, start=3, stop=4, show_measurement_ids=True)
assert "MZ[m0]" in str(readout)
assert "MZ[m1]" in str(readout)
```

For example, `MZ[m7]` means that measurement has identity `m7`. It is neither
an outcome nor the seventh entry in the result stream: IDs can be sparse and
out of order. Batched measurements retain the association between each qubit
and its own ID. Selecting a range never renumbers them.

`m?` means the gate has no usable stored ID. Rendering does not allocate IDs or
repair duplicate IDs. For a malformed measurement batch with a mismatched
number of IDs, every operand is marked `m?` instead of guessing an association.
Default diagrams omit IDs to keep the layout compact.

Leakage-aware Pauli measurements use the convention `M<Pauli>_L`, replacing
`<Pauli>` with the measured Pauli operator: for example, `MZ_L`, `MX_L`, or
`MZZ_L`. The `_L` suffix means a distinct leakage outcome is reported; it is
not an additional Pauli factor. This convention does not add new gate types.

`MZ_L` denotes the existing `MeasureLeaked` operation, a leakage-aware Z-basis
measurement that can return 0, 1, or 2 (`L`). The quantum engine produces
measurement results, and the general noise model reports 2 for a qubit that
was leaked at measurement. `MZ_L[m7]` identifies the measurement; it does not
say that its outcome was leaked. The
current circuit allocator does not automatically assign IDs to every `MeasureLeaked` gate,
so `MZ_L[m?]` is possible even though the operation produces a runtime result.

Trace visualization preserves `MeasureLeaked` as a native gate and renders it as
`MZ_L`, retaining each runtime result ID. This is distinct from
`qis_operation_trace_to_tick_circuit`, whose analysis replay projects it onto
Boolean `MZ`. Pass the original saved trace to the visualization helper to retain
the leakage-aware operation. Neither view displays individual outcome values.

## Inspect Operations and Trace Origins

Use `inspect_circuit` to obtain an SVG together with records connecting its gate
symbols and connectors to the source circuit. It accepts the same circuits and
saved QIS traces as `circuit_diagram`, including range and styling options:

<!--continuation-->
```python
from pecos.visualization import inspect_circuit

inspection = inspect_circuit(circuit, show_measurement_ids=True)
operation = inspection.operations[0]
print(operation["id"], operation["source"], operation["gate_type"])
inspection  # ordinary SVG display in a notebook
```

`inspection.save(path)` writes a JSON snapshot containing the SVG and operation
records; `inspection.save_svg(path)` writes the standalone SVG. Both replace an
existing file. `inspection.to_json()` returns the JSON string. Reading
`inspection.operations` returns an independent copy, and later circuit edits
do not change the snapshot. The circuit's native `.inspect(...)` method returns
the underlying dictionary directly.

Each operation record describes **one stored gate batch or DAG node**. A batched
CX can therefore have several control symbols, target symbols, and connectors
sharing its ID. Those SVG components carry a `data-pecos-operation` attribute
matching the record's `id`. Wires, tick backgrounds, and row labels are not
operations. An operation with no displayed support may have a record without
a visible symbol.

### Interactive Inspector

The same snapshot can produce a standalone interactive HTML document. Select a
gate to highlight all components of its stored batch and show source locations,
ordered qubits, stored measurement IDs, angles, parameters, and metadata. Hover
highlights the matching components; the selected operation remains pinned until
another is selected or cleared. The operation selector and Previous/Next buttons
also reach records without visible symbols. Focus a gate and press Enter or Space
to select it from the keyboard.

The operation selector and Previous/Next buttons scroll the diagram pane to
bring the selected batch into view. **Show selection** returns to it after
manual scrolling. If a batch is larger than the viewport, its upper-left
extent is shown first; the rest remains reachable by scrolling. Clicking a
visible gate keeps the current viewing position.

Use **+** and **−** to zoom, **100%** to restore the original scale, or
**Fit width** for an overview of the current diagram pane. Zoom ranges from 1%
to 400%; very large diagrams may still require scrolling at the minimum scale.
Zoom preserves the pane's viewing center where scroll bounds permit. These
controls affect only the interactive view, not saved SVG or snapshot data.

Search is case-insensitive and treats whitespace-separated terms as literal
substrings that must all match an operation. It searches the gate name and
displayed symbol (including `MZ_L`), source location, qubit IDs such as `q3`,
measurement IDs such as `m7`, parameters, and metadata. For example, `CX q3`
finds CX batches involving qubit 3 (and IDs beginning with `q3`, such as `q30`).
Search narrows the operation selector and Previous/Next navigation; it dims
nonmatching SVG components without changing wires or layout. A selection that
no longer matches is cleared. Reset search restores all operations.

With **Link trace stages** enabled, selecting a replayed operation also marks
other stored operations with the same complete recorded trace origin using a
dashed teal outline. The selected batch stays orange. This includes related
stages outside the search matches, shown for context; the selection message
reports their count. Linking uses all four origin coordinates and preserves
large integer IDs exactly. Missing, malformed, or unrecognized origins are not
linked. Counts include only operations present in the selected tick/layer range,
so they do not claim to describe every stage in the original trace.

**Previous trace stage** and **Next trace stage** move between operations with
that same origin, in their stored order within this view. They work independently
of the highlighting checkbox. If the next stage is outside the current search,
the button says **(resets search)**; using it clears the search, selects that
stage, and brings it into view. Stage positions and counts are local to the
selected range, not positions in the complete original trace.

<!--continuation-->
```python
viewer = inspection.interactive()
viewer  # sandboxed interactive notebook display
```

Use `inspection.save_html(path)` or `viewer.save(path)` to write a standalone
HTML file. Open it in a browser; no server, network connection, Python runtime,
or notebook is needed. `inspection.to_html()` returns the document as a string.
The native circuit method `circuit.inspect(include_html=True)` includes it in
the returned dictionary under `html`; the default dictionary omits that field.
The inspection's JSON export remains SVG plus operation records, without HTML.

Notebook interactivity is explicit: displaying `inspection` still shows SVG.
Displaying `viewer` uses an iframe with `sandbox="allow-scripts"`, isolating its
styles and scripts from the notebook. Notebook clients may require trusted
output or disallow scripted iframes; in that case use the SVG display or save
the HTML and open it separately. Without JavaScript, standalone HTML retains
the diagram and a readable list of all operation details.

HTML details are formatted in Rust, so large integer identities stay exact in
the viewer. Metadata is displayed as text. Channel payload presence is shown,
but its contents are not displayed. The inspector uses structural source
locations and does not report execution duration or measurement outcomes.

For a `TickCircuit`, `source` contains the original tick and batch indices. For
a `DagCircuit`, it contains the original node index. IDs remain consistent
between ranges of the same unchanged circuit. They are scoped to that source:
do not use them to match independently edited, converted, or compacted circuits,
or confuse them with measurement IDs.

For QIS traces, `source` describes the replayed, compacted circuit. The separate
`metadata["_pecos_trace_origin"]` record identifies the original
`engine_trace_id`, `shot_index`, `chunk_index`, and zero-based
`lowered_op_index` in the chunk's `lowered_quantum_ops` array. Decomposed stages
share that origin, while distinct operations retain distinct origins even when
compaction puts them in one tick. Original operation metadata accompanies each
stage. The origin identifies a captured lowered operation; it does not infer a
compiler node or an unexecuted branch.

The JSON format is `pecos_circuit_inspection_v1`. Records include gate type,
ordered qubits, measurement IDs, angles, parameters, and metadata.
`angle_turn_fractions` preserves each exact `Angle64` numerator as a decimal
string, representing that numerator divided by 2**64 turns; `angles_radians`
provides floating-point approximations. IDs elsewhere remain JSON integers,
so consumers handling values above 2**53 - 1 need lossless integer parsing.
Non-finite floating-point values are rejected instead of silently replaced.
Channel payloads are not serialized: `has_channel_payload` indicates their
presence. Rust inspection records retain the full owned `Gate`.

## Rust Export and Styling

The rendering and layout implementation lives in Rust. Both circuit types
return standalone SVG and TikZ strings directly:

```rust
use pecos::quantum::{DiagramStyle, TickCircuit};

let mut circuit = TickCircuit::new();
circuit.tick().pz(&[0, 1]);
circuit.tick().h(&[0]);
circuit.tick().cx(&[(0, 1)]);
circuit.tick().mz(&[0, 1]);

let svg = circuit.to_svg();
let tikz = circuit.to_tikz();
let style = DiagramStyle::builder()
    .color(false)
    .show_measurement_ids(true)
    .build();
let monochrome_svg = circuit.render_with(&style).svg();

assert!(svg.contains("<svg"));
assert!(tikz.contains("tikzpicture"));
```

Applications can write these strings using `std::fs::write`. `DiagramStyle`
also controls angle units, palettes, and dash patterns. Direct users of the
`pecos-quantum` crate import these types from `pecos_quantum`.

Rust users can obtain the same SVG associations and owned operation records
without Python or a notebook. Both circuit types expose `inspect` and
`inspect_range`:

<!--continuation-->
```rust
let inspection = circuit.inspect(&style);
let annotated_svg = inspection.svg();
let inspector_html = inspection.to_html();
for operation in inspection.operations() {
    println!("{}: {:?}", operation.id, operation.source);
    // operation.gate and operation.metadata retain the source data.
}
assert!(annotated_svg.contains("data-pecos-operation"));
assert!(inspector_html.starts_with("<!doctype html>"));
```

The runnable example exports SVG, TikZ, DOT, Unicode text, a monochrome variant,
an annotated inspection SVG, and a standalone interactive HTML inspector, with
batch locations printed in the terminal.
It replaces its named output files if they already exist:

```sh
cargo run -p pecos-quantum --example circuit_exports -- /tmp/pecos-diagrams
```

For a broader visual inspection, generate the Rust gallery and open its
`index.html` in a browser:

```sh
cargo run -p pecos-quantum --example circuit_gallery -- /tmp/pecos-gallery
```

The gallery includes explicit ticks versus dependency layers, nested controls,
nonadjacent interactions, sparse custom operands, correlated channel support,
leakage-aware readout, small angles, empty ticks, and selected ranges. Each
example explains what to look for and links to standalone SVG, TikZ, DOT, and
text files. Wide diagrams scroll horizontally rather than shrinking their
labels. Everything is generated locally without a notebook or web server.

`gallery.tex` combines the TikZ fragments into a document with one diagram per
page. Compile it from the output directory with a LaTeX engine that provides
TikZ, `geometry`, and `graphicx`. Generating the gallery does not require LaTeX.
The example replaces its named output files when rerun. This is a manual
inspection aid, not an automated pixel-comparison test.

## Understand the Horizontal Axis

| Input | Diagram positions mean |
|-------|------------------------|
| `TickCircuit` | Explicit circuit ticks |
| `DagCircuit` | Dependency layers, grouping gates that can run in parallel |
| QIS operation trace | Ticks produced by replay and ASAP compaction |

These are **structural circuit diagrams**, not physical execution timelines.
Idle gates display their duration in abstract time units, for example `Idle(20)`,
but that duration does not scale the horizontal spacing. Independent
gates in different explicit ticks can appear in the same layer after conversion
to a DAG. Gates with crossing connectors may occupy separate visual columns
within a single tick or layer to keep the diagram readable. Group labels use
`tN` for ticks and `lN` for dependency layers. Empty ticks on populated circuits
are retained and labeled, including trailing empty ticks.

## Select Ticks and Name Panels

Use a half-open range: `start` is included and `stop` is excluded. The selected
view retains the source's qubit rows and original tick or dependency-layer
numbers. Omit either bound to use the beginning or end. Negative, reversed,
and out-of-bounds ranges raise `ValueError`; an empty interval shows only its
header. The source circuit is unchanged.

<!--continuation-->
```python
from pecos.visualization import circuit_diagram_panels

interaction = circuit_diagram(circuit, start=1, stop=3, title="Entanglement")
panels = circuit_diagram_panels(circuit, [
    ("Preparation", 0, 1),
    ("Entanglement", 1, 3),
    ("Readout", 3, 4),
])
# Display each panel in a notebook, or call panel.save(path).
```

The same options work on `circuit.render(...)` and with every output format.
Titles appear alongside the source type and selected range. Named panels may
overlap and preserve the supplied order. For a trace, ranges refer to replayed,
ASAP-compacted ticks; the complete trace is validated and replayed once before
rendering the panels.

Rust provides the same checked selection through `render_range`:

```rust
use pecos::quantum::{DiagramStyle, TickCircuit};
let mut circuit = TickCircuit::new();
circuit.tick().h(&[0]);
circuit.tick().cx(&[(0, 1)]);
let style = DiagramStyle::default();
let panel = circuit.render_range(1..2, &style).unwrap().with_title("Entanglement");
assert!(panel.svg().contains("ticks 1..2 of 2"));
```

`DiagramStyle` now includes `show_measurement_ids`. Rust callers using exhaustive
struct literals must supply that field; using `DiagramStyle::builder()` or
`..DiagramStyle::default()` allows the default (`false`).

Invalid Rust ranges return `DiagramRangeError`. Round boundaries must be
supplied explicitly or derived from metadata specific to the circuit builder;
they are not inferred from gate patterns. The runnable surface-code example
uses the builder's phase and syndrome-round metadata to separate preparation,
initialization, repeated rounds, and final data readout:

```sh
python examples/python_examples/circuit_round_panels.py /tmp/pecos-rounds
```

It exports SVG and TikZ files with measurement IDs and replaces its named outputs.
Detector/observable overlays and collapsed-round summaries are not yet shown.
These diagrams support inspection of local gate structure, rather than a
complete QEC debugging view.

DOT export describes the same gates and qubit wires, but Graphviz controls the
final layout and may route connectors differently from SVG and TikZ. It does
not export every edge of the original `DagCircuit`; use the DAG's graph
inspection methods when examining its actual dependencies.

## Render a Saved QIS Trace

Load the JSON-compatible trace returned by `capture_qis_operation_trace` and
pass its chunk list directly to `circuit_diagram`:

<!--skip: requires a saved runtime trace-->
```python
import json
from pathlib import Path
from pecos.visualization import circuit_diagram

trace = json.loads(Path("runtime-qis-trace.json").read_text())
diagram = circuit_diagram(trace)
diagram.save("runtime-circuit.svg")
```

Rendering does not execute the original program. It uses
the same audited replay validation as `qis_operation_trace_to_tick_circuit`
for framing, completeness, and measurement provenance, while preserving
leakage-aware measurement kinds for visualization. Incomplete,
mixed-shot, or unaudited traces are rejected by the same validation as ordinary
trace replay. A filename or program object is not accepted as a diagram input.

A trace describes one execution path. Its circuit diagram does not show
unexecuted branches, physical timestamps, measurement outcomes, detector
overlays, or injected noise events. Keep the original trace for inspecting
result values and provenance.

The convenience capture helper uses a coin-toss engine. To inspect a path
selected by a particular quantum simulator, capture through a configured
`sim(...).quantum(...).capture_operation_trace()` builder and select one
completed shot before rendering.

See [Runtime QIS Tracing](runtime-qis-tracing.md) for capture, validation, and
replay semantics, and [Circuit Representation](circuit-representation.md) for
the circuit data structures.
