# Live decoding with supplied DEM windows

## Implemented boundary

`sim(Qis(...)).classical(selene_engine(runtime).runtime_decoder(factory))` now
supports synchronous embedded decoding. The caller supplies structured DEM
windows, possibly assembled with `DemSliceRoundSchedule` and cached slices.
`RuntimeDecoder` accepts either a collection indexed by window ID or a factory
of shot-local providers. It uses `DetectorErrorModel.build_decoder`, including
its existing structured windowed-decoder path and unsupported-model checks.
It does not compile a DEM from runtime operations.

The program calls this C ABI (LLVM declaration shown):

```llvm
declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)
; Arguments: window_id, instance_id, observed detector bytes, byte count.
; Return: an unsigned 64-bit program frame update.
```

The buffer contains dense, unpacked bytes, each exactly zero or one. The program
must first read the measurements it needs through the selected runtime and
compute detector parities, including reference signs. The call copies the bytes
before suspension. It rejects null nonempty buffers, oversized lengths (more
than 1,048,576 bytes), nonbinary data and collection-only execution. As with the
other native imports, a caller must supply a valid readable pointer. Failed
calls fail the shot; zero is a valid successful answer, never an error fallback.
The QIS FFI data ABI is now version 5; rebuild host and FFI artifacts together.

The returned mask is an XOR update to program-defined frame bits. It is derived
from standard DEM observable predictions; it is not an arbitrary physical
recovery operator and is never compared with hidden logical truth. The program
can update a classical frame, branch, or issue physical correction gates through
ordinary QIS. The host does not apply a simulator-only correction. This first
ABI supports any admitted detector count and at most 64 destination frame bits;
wide model outputs can still be explicitly routed or projected into those bits.

See [the runnable example](../../examples/runtime_decoding.py). Language frontends
can emit the intrinsic; this PR does not add a Guppy library primitive or modify
the HUGR compiler.

## Execution and lifecycle

The existing route remains `sim → MonteCarlo → HybridEngine → QuantumSystem`.
Monte Carlo clones classical-engine templates for workers, resets each shot and
assigns its program index. HybridEngine alternates QisEngine command production
with QuantumSystem noise/simulation and feedback. QisEngine intercepts the
host-only Decode request before runtime lowering, retaining it while preceding
ready batches and their measurement feedback execute. It then invokes the
session exactly once and responds on the existing synchronous response lane,
which is separate from measurement IDs and native future handles.

An empty lowering poll processes work already ready. Decoding inserts no barrier,
forces no unread future, changes no native timestamp and does not advance the
runtime clock by its wall-clock duration. Modelled classical delay must be an
explicit program/runtime request. Pending physical work whose result was not
read may remain pending. Only the runtime determines when that work executes.

The zero-argument decoder factory runs lazily on the controlling worker's first
call in each shot. Clones share factory configuration only. Reset and the next
shot discard the session, pending request and instance counter. Callback or
backend errors latch the shot; neither later calls nor result retrieval can
turn it into a successful shot. No simulator object, noise sample, RNG seed,
measurement store or logical truth is passed to a decoder/provider.

`RuntimeDecoder` creates a fresh provider session for each shot. Its provider
receives only `RuntimeDecodeRequest(window_id, instance_id)`. A collection shares
immutable definitions; a provider factory must return fresh mutable state.
Each native decoder is built, used and released inside its decode call on that
worker, because Python structured decoders are thread-bound. Prepared model and
decoder caching is an optimization for a later change. The lower-level Rust
`RuntimeDecoderSession` and Python session protocol are trusted extension points:
the implementer must enforce its model contract and must not share mutable shot
state between factory invocations.

## Checked supplied-window contract

`RuntimeDecodeWindow` deliberately supports **independently closed regions**.
A hard boundary by itself is insufficient: the existing slice composer may
project outside detectors at its backward boundary. Callers must certify that
no physical source or persistent state crosses these independently decoded
instances; `independent=True` is an explicit assertion of that obligation.
PECOS checks the declaration, not a proof of the physical experiment.

| Contract | Check and meaning |
| --- | --- |
| Identity | Unsigned 64-bit window ID agrees with the collection key or requested provider ID. Instance IDs are contiguous from zero per shot; a reused definition needs a new instance. |
| Detectors | Unique named identities cover every dense model `D` column. `detector_order[i]` selects the program byte for model detector `i`; it is a complete permutation, with exact width and binary values checked at each call. |
| Outputs | Unique names cover every model `L` column. Every output has a GF(2) routing row into frame bits 0–63. Empty rows explicitly project; repeated destinations cancel. Runtime windows explicitly reject tracked-Pauli outputs, which standard decoder conversion would otherwise omit. |
| Commit/buffer | The host commit is the entire detector tuple in model order. Host buffers and partial commits are rejected. Windowed decoding inside one complete model still uses the existing internal buffer/carry algorithm. |
| Boundary state | Empty only in this milestone. Nonempty boundary/carry/leakage state is rejected. An output mask alone cannot transport full detector residuals or source correlations. |
| Fault ownership | Nonempty, unique source-group names declare all sources owned by a region, including undetectable sources. Absolute ownership is `(instance_id, fault_id)`. The caller certifies complete coverage and disjoint physical ownership; name uniqueness cannot prove those facts. Correlated sources may not be split into independently owned groups. |
| Failure/reuse | Invalid identities, shapes, mappings, unsupported boundaries, provider errors and decoder errors abort the shot. Factories and counters restart for the next shot; failed sessions cannot be reused. |

Names and mappings must describe the actual program data. Supplying the correct
number of arbitrary Boolean values does not establish a valid model. The live
integration tests use independent reset/measurement regions with a known
one-bit response. They establish host feedback behavior, not adaptive fault
distance or a production streaming decoder.

## Follow-up: providers driven by emitted physical batches

Keep full JIT compilation separate from this supplied-model milestone. Extend
the provider with a versioned immutable context containing *observed execution
provenance*, not simulator fault samples. Use the existing runtime batch,
source-metadata and slice-composition seams, rather than a second circuit or
window implementation.

A future compiler/runtime provenance record must include:

- Program and compiler revisions and stable QIS instruction / HUGR node IDs.
- Branch-site ID, selected outcome and occurrence; nested loop-site IDs and
  iteration counts. Outcome sequences alone cannot distinguish different sites.
- Relevant typed classical inputs, including their dependency on observed reads
  or prior decoder results. Preserve ordering and measurement/future provenance.
- Runtime batch identity, emitted operation ordering, timing units, durations,
  explicit idles, runtime/configuration revision and completion/availability.
- Detector and output schema versions and instance relabelings, incoming boundary
  state and outgoing port obligations, with a declared correlation horizon.

Current traces capture concrete QIS operations, named results and physical
lowering provenance. They do **not** supply this complete branch/loop dependency
record. Do not infer unrecorded branch sites from the trace's Boolean values.
The new Decode operation records request IDs and observed bytes in operation
traces; it is not a control-flow certificate.

Cache bounded *physical fragments* with keys sufficient to determine their
physics: normalized emitted gates, relevant timing and scheduling context,
geometry, noise parameters and support topology, schema, temporal horizon,
initialization/readout family and boundary-state identity. Keep detector/output
relabelings, placement translations and supported Clifford frame transforms
separate using `DemSliceInstance`. Use bounded provider caches; do not use the
entire Boolean history as the sole key. If a future-dependent output cannot yet
be propagated, retain a supported symbolic interface or defer the decision;
do not guess the future Clifford suffix.

Before allowing overlapping commits, introduce a typed boundary handoff carrying
committed correction incidence, unresolved detector ports, source/correlation
identity and any supported persistent-state representation. Reuse existing
`DemWindowSpec` horizon/ownership checks and decoder-core's column-aware commit
and carry logic. An observable-only prediction cannot distinguish errors with
the same logical action but different outgoing boundary effects. Keep that
unsupported until the decoder can supply the required handoff.

Four cases require explicit admission rules:

1. **Fault-dependent branches.** A fault can change the branch itself, changing
   operations, timing, detector definitions and the conditional noise law. A DEM
   from the one realized path omits counterfactual effects. Admit only a proved
   branch-invariant/closed gadget contract or a supported joint conditional
   model; otherwise reject. Branch coverage by testing alone is not such a proof.
2. **Cross-boundary correlations.** One source must have one owner; retain all
   decomposition components and outgoing effects. Existing partial-ownership,
   unresolved hard-port and undersized-buffer failures remain failures. A new
   instance number never makes a shared physical source independent.
3. **Persistent leakage.** A leaked physical state can survive through several
   batches and alter later interactions. A reset is a closing boundary only if
   the actual reset/noise contract establishes that fact. A Pauli DEM and a zero
   boundary token cannot silently erase leakage; defer or reject until there is
   a supported stateful model/handoff. Observed leakage flags are distinct from
   hidden simulator leakage truth.
4. **Unavailable future operations.** Decoding cannot force program progress
   past its own dependent decision. Require enough already available evidence
   for the commit, a justified boundary model, or an explicit latency protocol.
   Preserve `BufferTooSmall` and static adaptive-DEM rejection; never pad with
   invented operations or regard a soft boundary as permission to truncate a
   correlation touching the commit.

## Literature and limits

[Fowler et al., Autotune, Appendix B.5](https://arxiv.org/html/1202.6111v2)
provides the reusable local recipe approach already embodied in the slice layer.
[Dennis et al., section VI.B](https://arxiv.org/html/quant-ph/0110143) motivates
overlapping recovery rather than independent truncation of continuous history.
[Bombín et al., sections IV–V](https://arxiv.org/html/2303.04846v1) requires a
buffering condition and locally sound decoding; an arbitrary number of buffered
rounds is not a universal proof for a different correlation model.
[Kliuchnikov, Lietz and Pastawski, sections 4 and 7](https://arxiv.org/html/2609.13087v1)
formalize compositional interfaces and committed boundary information for
stabilizer circuits with stochastic Pauli faults. That scope does not certify
arbitrary leakage or faulty classical control.
[Ikari et al., section 4.2](https://arxiv.org/html/2608.11719v1) describes temporal
ports and JIT stitching for its transversal-gate schedules; its results do not
justify an arbitrary adaptive trace.
[Skoric et al.](https://arxiv.org/html/2209.08552v2) makes feed-forward latency and
backlog part of the real-time decoding problem. This integration adds synchronous
functional feedback, with no real-time throughput or latency claim.
