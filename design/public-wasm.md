# Public Guppy WASM execution

Implement from the public `tket.wasm` extension specification and public Selene
interfaces. No proprietary source, APIs, packages, or build steps are required.

The first implementation supports scalar i64 calls with zero or one result.
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
regression until an upstream fix permits a real loop-execution assertion.
