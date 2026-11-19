# Public Guppy WASM execution

Implement from the public `tket.wasm` extension specification and public Selene
interfaces. No proprietary source, APIs, packages, or build steps are required.

The implementation supports mixed scalar i64/f64 calls with zero or one result.
A Python HUGR lowering pass replaces WASM types and operations with explicit
external function calls, then uses the released public HUGR/QIS compiler.
Generated LLVM adapters call a versioned public WASM ABI shared by PECOS and a
Selene Utility plugin. Module bytes are supplied explicitly and embedded in the
compiled program; HUGR filenames do not authorize filesystem reads.

External calls and their enclosing executable regions receive explicit order
edges from a stable topological schedule that respects all existing data/order
dependencies. This makes context disposal/reacquisition deterministic even for
older Guppy versions that omit context effects. Mutually exclusive cases and
function definitions are never sequenced against each other.

Context state belongs to a shot and worker. Public tests must exercise repeated
shots, multiple contexts, traps, signed values, and quantum feed-forward.

The initial adapter rejects loop-containing programs because released public
compilers 0.4.2 and 0.5.2 can silently constant-fold loop-carried external-call
results to their initial values. This reproduces with ordinary Guppy external
declarations independently of WASM and at LLVM opt_level=0. Keep a rejection
regression until a released compiler includes the
[external-call analysis fix](https://github.com/Quantinuum/tket2/pull/2177) and
permits a real loop-execution assertion.

Typed signatures use an additive `pecos_wasm_v2_lookup` export with byte type
tags (0=i64, 1=f64). Integer-only lowering retains `pecos_wasm_v1_lookup`.
Both use the existing call/read buffers: each slot holds 64 raw bits. LLVM
bitcasts and Wasmtime's `Val::F64` preserve float representations without numeric
integer conversions. Function handles include the full typed signature, and
execution validates it against the actual caller-supplied module.

The legacy `ForeignObject::exec` integer/clamping contract remains unchanged.
Mixed float signatures use a separate strict scalar method, which shares trap
and timeout reporting with the legacy path. Integer-only signatures retain the
existing validated integer execution path.
