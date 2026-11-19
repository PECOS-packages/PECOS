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
older Guppy versions that omit context effects. For unordered acquisitions,
the success check's same-region predecessors also precede the acquisition,
provided this does not contradict an existing dependency path. Mutually exclusive cases and
function definitions are never sequenced against each other.

Context state belongs to a shot and worker. Public tests must exercise repeated
shots, multiple contexts, traps, signed values, and quantum feed-forward.

Released public compilers can silently constant-fold loop-carried external-call
results to their initial values, including at LLVM opt_level=0. PECOS protects
value-returning external declarations by using
`LoadFunction -> prelude.Barrier -> CallIndirect`.
The public Barrier has no constant-fold rule and emits as an identity, keeping
the indirect target unknown to HUGR analysis without changing the function that
executes. Existing loads of value-returning external declarations receive the
same protection. Void calls remain direct. This uses released public dependencies
without a compiler fork or patch.

Effect scheduling runs before this rewrite: appended load/barrier nodes must
not change the original topological tie-breaking between external calls and
local helpers. Existing call order edges remain on the indirect call. Each
region's final effect is also ordered before its Output: otherwise dead-code elimination can discard
indirect calls with unused results, including their enclosing regions. Regression
tests check real loop results, result-dependent branches, nested loops, helper
calls and float feedback across both backends. The Selene matrix covers both
platforms and optimization levels 0 through 3; keep these tests when upgrading
the compiler because the workaround depends on its conservative Barrier analysis.

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
