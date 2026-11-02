# Circuit inspector regression tests

From the workspace root, install the locked test dependencies and run:

```sh
npm ci --prefix crates/pecos-quantum/tests/inspector --ignore-scripts
npm test --prefix crates/pecos-quantum/tests/inspector
```

This requires Cargo and Node.js 22.13+ on the 22.x line (CI uses Node 22).
Node 20.19+ and 24+ are also supported by the test dependencies. Node and npm
are development dependencies only; exporting or viewing diagrams needs neither.
After installing dependencies, `just test-inspector` runs the same suite.

Each run builds the `inspector_fixtures` Rust example and exports fresh HTML into
a unique temporary directory. The Node test runner executes the embedded viewer
script in jsdom, then removes the directory even if a command fails. Generation
errors fail the run; old HTML files cannot substitute for a failed export.
Nothing is fetched by the viewer. The fixtures are trusted, locally generated
test inputs; do not point these script-executing tests at arbitrary HTML.

The suite covers TickCircuit and DagCircuit selection, multi-component batches,
literal search, measurement IDs, escaped metadata, no-script fallback, exact
64-bit trace origins, filtered sibling stages, stage navigation, empty and
invisible operations, zoom limits, and scrolling. The trace examples use
synthetic origin metadata on native gates. Actual trace validation and replay
remain covered by the Python inspection/visualization integration tests.

jsdom has no browser layout engine. Geometry tests supply explicit bounding
boxes and viewport dimensions, including classic scrollbar width changes.
These check scrolling calculations, not rendered placement, clipping, font
metrics, browser CSP enforcement, or notebook iframe behavior. A browser visual
pass remains necessary for layout changes; this suite does not replace it.

The `Circuit inspector` GitHub Actions workflow runs this suite on relevant
Rust, dependency, test, and workflow changes. It also runs Rust rendering,
inspection, and leakage measurement ordering tests. There is no browser download or Python extension build in this job.
