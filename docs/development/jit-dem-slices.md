# Just-in-time DEM slices

PECOS represents reusable detector-error-model fragments as `DemSlice` values.
A slice owns the independent fault contributions introduced by one bounded
physical circuit block. Its detector targets use `(local detector, relative
round)` addresses instead of final `D<n>` indices.

This representation supports two separate phases:

1. An offline compiler analyzes constant-depth physical templates such as
   initialization, idle syndrome extraction, measurement, and transversal
   Clifford operations.
2. `DemStitcher` instantiates the cached templates for one decoding window,
   resolves their temporal dependencies, and returns a structured
   `DetectorErrorModel`.

The split follows the DEM Stitch design in
[Ikari et al.](https://arxiv.org/abs/2608.11719). PECOS keeps the more
specialized static-Tanner-graph optimization from
[Ye, Maksymov, and Delfosse](https://arxiv.org/abs/2608.25027) separate: a
decoder may update only its prior vector when a window has identical incidence
and logical-action matrices, but physical logical gates can also change those
matrices and require general slice assembly.

## Ownership and temporal ports

Every physical fault contribution must be owned by exactly one slice. Detector
support outside the owning round is represented by a signed round offset. A
positive offset is a forward temporal port; a negative offset is a dependency
on a preceding slice.

`DemTemporalHorizon` turns the assumption of bounded temporal correlation into
a checked integer contract. Slice construction fails when any target exceeds
the declared past or future reach.

`DemSliceDetector::new(id)` declares a detector emitted at every instance's
owning round. `DemSliceDetector::port(id)` registers an identity that the slice
may target without emitting it. The latter is needed at initialization and
other asymmetric boundaries where the adjacent slice owns the detector
declaration.

## Relabeling

A cached slice contains only local detector and output identities.
`DemSliceInstance` supplies the algorithm-wide mappings when the slice is
scheduled. Logical labels, absolute round numbers, patch placement, and
zero-time automorphisms therefore do not multiply the cache entries for the
same physical operation.

The cache deliberately accepts a caller-defined ordered key. A physical
template compiler should include circuit identity, code geometry, detector
schema, temporal horizon, and noise-support topology in that key. It should not
include instance-only relabeling state.

## Existing DEM integration

`DemSlice::from_detector_error_model` adapts PECOS's structured
`DetectorErrorModel` directly. `DemSliceModelMap` explicitly maps every source
detector declaration to a local detector identity and signed round offset, and
maps standard `L<n>` and PECOS `TP<n>` outputs into their local identity spaces.
Missing declarations or output mappings fail instead of being dropped.

The adapter retains contributions individually. Y-specific decomposition and
arbitrary source-frame component lists are both preserved, including
multi-component sources used by native two-qubit Clifford and replacement
branches. Component XOR is checked against the source contribution's complete
effect.

For a physical template containing halo operations,
`DemSlice::from_detector_error_model_for_locations` accepts the owned
`DagFaultInfluenceMap` location IDs. A contribution is included only if all of
its source locations are owned by the slice and omitted if none are. Partial or
unattributed ownership fails loudly so one correlated source cannot be split
or counted twice.

## Bounded physical-template compiler

`DemSliceTemplateCompiler` extracts selected owner rounds from a bounded,
source-tracked physical model. It validates the annotated circuit's source
ownership and detector-stream layout once, then emits absolute-round-independent
`DemSlice` values suitable for `DemSliceCache`. The bounded model needs only
enough neighboring rounds to expose the operation's complete temporal horizon;
it is not an algorithm-length model.

For a surface-code memory experiment, a three-SEC-round fixture is enough to
compile four relevant families: initialization, stationary bulk SEC, the SEC
round immediately before destructive readout, and the terminal measurement.
The pre-terminal SEC family is intentionally distinct from bulk because its
faults terminate on data-readout detectors rather than the next full syndrome
round. Tests instantiate one cached bulk slice at multiple absolute rounds and
reconstruct a separately compiled five-round physical DEM exactly.

At the Rust layer, callers compose cached `DemSliceInstance` values with
`DemSliceRoundSchedule::from_instances`. Python exposes the same narrow path as
opaque `DemSliceTemplate` values returned by `schedule.template(...)` and
`DemSliceRoundSchedule.from_templates(...)`. The Python constructor currently
uses identity detector/output mappings plus checked global or per-stream
spatial translations. Per-stream translation lets independently placed code
blocks reuse one physical template; general stream-identity routing and logical
relabeling remain in the Rust instance API.

The production `LogicalCircuitBuilder` uses a bounded three-round compile for
eligible single-patch memory operations of two or more rounds. Its cache key
contains the circuit family, complete patch geometry, measurement basis, and
noise parameters, but deliberately excludes the requested memory length, patch
label, qubit offset, and spatial placement. Detector coordinates are translated
on `DemSliceInstance` construction. Consequently, a later memory experiment of
any supported length or placement reuses the same initialization, stationary
bulk, pre-terminal, and terminal objects.
`build_dem`, `build_sampler_and_decoder`, and `build_algorithm_descriptor` all
share this provider. One-round memories, multiple patches, and circuits with
unsupported logical gates conservatively retain the full structured fallback
until their bounded families and instance mappings are implemented.

A second bounded provider covers a single transversal H between two memory
segments of at least two rounds each. A six-round canonical fixture yields
initialization, ordinary pre-H bulk, the pre-H boundary round, H plus the first
post-H SEC round, ordinary post-H bulk, pre-terminal, and terminal templates.
The rounds adjacent to H are distinct physical families: faults immediately
before the gate can propagate through it, while H itself shares ownership with
the first post-gate detector round. The cache key adds the initial and final
measurement bases but still excludes both requested segment lengths, patch
label, qubit offset, and spatial placement. Exact tests cover different depths,
orientations, bases, labels, offsets, and noise keys. Shallow boundary cases,
multiple H gates, and other Clifford operations retain the full-model fallback.

A third bounded provider covers a transversal CX between two matching patch
shapes. Its seven families use the same temporal positions as H but contain two
patches and the correlated boundary sources introduced by the physical CX.
Canonical stream IDs are partitioned by patch, then independently translated
to the control and target placements at instantiation. The cache key includes
both geometries, orientations, four boundary bases, and noise parameters, while
excluding both memory depths, labels, qubit offsets, and patch coordinates.
Patches whose shapes differ, reversed/noncanonical operation ordering, or a
memory side shorter than two rounds retain the full structured fallback.

The current standalone SZ/SZdg emitter is deliberately not cached. Its full DEM
contains mechanisms whose detector span grows with the entire preceding memory
segment (observed spans 3, 5, and 9 for corresponding pre-gate depths), violating
the bounded-correlation requirement for JIT templates. A sound provider requires
the documented mid-cycle fold-transversal S-SE construction, rather than
caching the current between-round physical phase layer with a depth-dependent
key.

Stable detector-stream IDs are seeded from the earliest round with the maximum
number of declarations. This uses stationary SEC record order instead of a
partial initialization boundary, ensuring composed `D<n>` ordering matches the
physical syndrome stream exactly. Surface-memory tests therefore compare the
entire rendered model byte for byte, not only up to detector relabeling.

## Native round schedule

`DemSliceRoundSchedule::from_annotated_circuit` derives the repetitive layout
from metadata PECOS already carries. Detector coordinates are interpreted as
`[x, y, round]`: equal spatial pairs form a stable detector stream, while the
time coordinate supplies the emitted round. Physical DAG gates carry the
integer `dem_slice_round` attribute, and every `DagFaultInfluenceMap` location
inherits its owner from its gate node.

The surface `LogicalCircuitBuilder` writes this attribute on every generated
gate. Initialization is owned by round zero, syndrome-extraction operations by
their current round, transversal gates by the following round, and terminal
data measurements by the terminal boundary round. Missing or non-integral
metadata fails instead of guessing. A multi-location correlated source whose
locations disagree on the owner round is also rejected.

The schedule derives relative detector maps, temporal horizons, standard
output mappings, and tracked-Pauli mappings, then exposes the resulting slice
instances to `DemStitcher`. This removes the hand-authored ownership and mapping
tables from the equivalence path. Full-circuit source-tracked DEMs remain the
independent equivalence oracle for bounded template composition.

Python callers can exercise the same structured path through
`DetectorErrorModel.stitched_round_window(...)`. It accepts the originating
influence map and annotated DAG and returns another structured model; rendered
DEM text appears only at an explicit final serialization boundary.
`required_buffer_rounds(...)` computes the exact minimum look-ahead for a
commit region from source ownership and detector targets. Omitting
`buffer_rounds` from `stitched_round_window(...)` applies that safe value;
supplying an undersized value still fails loudly.

For multiple windows, `DetectorErrorModel.round_schedule(...)` returns a
reusable `DemSliceRoundSchedule`. Its `stitch(...)` and
`required_buffer_rounds(...)` methods reuse the already-derived ownership,
stream layout, relative mappings, and slice contributions instead of compiling
the schedule again for every window. The one-shot model methods remain
convenience wrappers over the same provider.

`LogicalCircuitBuilder.build_algorithm_descriptor(...)` uses this path for its
per-segment models. Segment DEMs may contain look-ahead detectors needed to
preserve a cross-boundary source, while segment metadata counts only the
non-overlapping detector partition consumed by the streaming decoder. An
omitted `buffer` derives the safe forward overlap. An explicit `buffer` also
adds that many look-behind rounds and is rejected if it is smaller than the
derived forward requirement.

## Window boundaries

`DemWindowSpec` describes a half-open commit region followed by a half-open
buffer region. The backward boundary is hard. Callers may supply instances
before the window as a bounded source halo; their outside detectors are
projected away while effects reaching into the window are retained.

The forward boundary is explicit:

- `Soft` permits unresolved forward ports to project to the decoder boundary.
- `Hard` requires all forward ports to resolve inside the window, as expected
  after terminal destructive measurement.

A soft boundary is not permission to truncate a correlation that touches the
commit region. If a contribution reaches from a commit detector through the
entire buffer, stitching reports `BufferTooSmall`. Increasing the buffer or
rejecting the physical model is required.

Projection never graphifies a mechanism. Hyperedges remain hyperedges, and
independent contributions that become identical after projection are combined
by the existing XOR probability rule when the structured model is rendered or
converted to mechanism columns.

## Current scope

This layer provides the stable slice, cache, structured-DEM adapter, automatic
round layout, ownership, bounded template extraction, mapping, and stitching
API. Initialization, stationary bulk SEC, pre-terminal SEC, and terminal
surface-memory families have exact composition coverage and a production
single-patch memory provider. A single transversal-H boundary and its adjacent
memory families also have exact composition coverage and a production provider,
as does a two-patch transversal CX between matching patch shapes. It does not
yet implement the bounded mid-cycle SZ/SZdg circuit, general multi-patch or
lattice-surgery families, repeated logical gates, decoder prior mutation, the
anti-snake logical-subgraph window decoder, or adaptive syndrome-extraction
templates. Full-circuit DEM construction remains the equivalence oracle and the
conservative fallback outside supported families.
