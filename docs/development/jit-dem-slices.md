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
tables from the equivalence path; it still uses a full source-tracked DEM as the
oracle until the operation-template compiler is introduced.

Python callers can exercise the same structured path through
`DetectorErrorModel.stitched_round_window(...)`. It accepts the originating
influence map and annotated DAG and returns another structured model; rendered
DEM text appears only at an explicit final serialization boundary.
`required_buffer_rounds(...)` computes the exact minimum look-ahead for a
commit region from source ownership and detector targets. Omitting
`buffer_rounds` from `stitched_round_window(...)` applies that safe value;
supplying an undersized value still fails loudly.

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
round layout, ownership, mapping, and stitching API. It does not yet compile
physical template/halo circuits independently, mutate a decoder's prior
vector, enable the anti-snake logical-subgraph window decoder, or support
adaptive syndrome-extraction templates. The full-circuit DEM builder remains
the equivalence oracle and the source of round-schedule slices until the
operation-template compiler is introduced.
