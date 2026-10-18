# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.

"""Typed surface for the dynamically registered ``pecos_rslib.qec`` module."""

from __future__ import annotations

import os
from typing import Any, ClassVar, Mapping, Sequence, TypedDict

from pecos_rslib import decoders
from pecos_rslib.decoders import ObservableFlips

from pecos_rslib import DagCircuit, ParityCheckMatrix, PauliString, Qis, StabilizerCodeSpec, TickCircuit

DEM_SLICE_ROUND_ATTRIBUTE: str

class DiscoveredDetectors:
    """Record parities with integer noiseless references (0 or 1)."""

    @property
    def detectors(self) -> list[tuple[list[int], int]]: ...
    @property
    def observables(self) -> list[tuple[list[int], int]]: ...

def discover_detectors(tc: TickCircuit, observables: Sequence[Sequence[int]] = ()) -> DiscoveredDetectors:
    """Discover detectors; undeclared deterministic logical parities become detectors."""
    ...

class BivariateBicycleCode:
    """A validated bivariate-bicycle CSS code."""

    def __init__(
        self,
        l: int,
        m: int,
        a_terms: list[tuple[int, int]],
        b_terms: list[tuple[int, int]],
    ) -> None: ...
    @property
    def l_order(self) -> int: ...
    @property
    def m_order(self) -> int: ...
    @property
    def hx(self) -> ParityCheckMatrix: ...
    @property
    def hz(self) -> ParityCheckMatrix: ...
    @property
    def logical_x(self) -> ParityCheckMatrix: ...
    @property
    def logical_z(self) -> ParityCheckMatrix: ...
    def num_qubits(self) -> int: ...
    def num_logical_qubits(self) -> int: ...

def bb_memory_circuit(
    l: int,
    m: int,
    a_terms: list[tuple[int, int]],
    b_terms: list[tuple[int, int]],
    rounds: int,
    basis: str,
) -> TickCircuit: ...
def coloration_memory_circuit(
    hx: ParityCheckMatrix,
    hz: ParityCheckMatrix,
    rounds: int,
    basis: str,
) -> TickCircuit: ...
def transform_two_patch_pauli(pauli: int, gate: str) -> int: ...

class FaultDistanceResult:
    """A unit-weight mechanism distance and one witnessing index set."""

    @property
    def distance(self) -> int: ...
    @property
    def mechanism_indices(self) -> list[int]: ...
    def __repr__(self) -> str: ...

class FaultDistanceUpperBoundConfig:
    """Fully explicit randomized fault-distance upper-bound configuration."""

    def __init__(
        self,
        samples: int,
        seed: int,
        observable_subset_strategy: str,
        error_rate: float,
        max_iterations: int,
        bp_method: str,
        bp_schedule: str,
        min_sum_scaling_factor: float,
        osd_method: str,
        osd_order: int,
        omp_threads: int,
    ) -> None: ...
    @property
    def samples(self) -> int: ...
    @property
    def seed(self) -> int: ...
    def __repr__(self) -> str: ...

class FaultDistanceUpperBoundResult:
    """A natively verified upper bound, never an exact fault-distance claim."""

    @property
    def weight(self) -> int: ...
    @property
    def mechanism_indices(self) -> list[int]: ...
    @property
    def samples_run(self) -> int: ...
    @property
    def bound_kind(self) -> str: ...
    def __repr__(self) -> str: ...

class CachedDemSlice:
    """Reusable absolute-round-independent DEM slice."""

    @property
    def name(self) -> str: ...
    @property
    def temporal_horizon(self) -> tuple[int, int]: ...
    @property
    def num_contributions(self) -> int: ...
    @property
    def dem_outputs(self) -> list[int]: ...
    @property
    def tracked_paulis(self) -> list[int]: ...
    def __repr__(self) -> str: ...

class DemSliceRoundSchedule:
    """Reusable structured-DEM schedule compiled from annotated round metadata."""

    @staticmethod
    def from_cached_slices(
        output_model: DetectorErrorModel,
        cached_slices: Sequence[tuple[CachedDemSlice, int]],
        expected_dem_outputs: Sequence[int],
        expected_tracked_paulis: Sequence[int],
        coordinate_offset: tuple[float, float] | None = ...,
        detector_coordinate_offsets: Mapping[int, tuple[float, float]] | None = ...,
        dem_output_routings: Mapping[int, Mapping[int, Sequence[int]]] | None = ...,
        tracked_pauli_routings: Mapping[int, Mapping[int, Sequence[int]]] | None = ...,
        detector_order_routings: Mapping[int, Mapping[int, int]] | None = ...,
    ) -> DemSliceRoundSchedule: ...
    @property
    def num_instances(self) -> int: ...
    def cached_slice(self, owner_round: int) -> CachedDemSlice: ...
    def rounds(self) -> list[int]: ...
    def required_buffer_rounds(self, start_round: int, commit_rounds: int) -> int: ...
    def compose(
        self,
        start_round: int,
        commit_rounds: int,
        buffer_rounds: int | None = ...,
        forward_boundary: str = ...,
    ) -> DetectorErrorModel: ...
    def __repr__(self) -> str: ...

class DemInferenceSettings:
    """Immutable fluent settings for empirical DEM annotation inference."""

    def __init__(self) -> None: ...
    def with_raw_tag(self, tag: str) -> DemInferenceSettings: ...
    def with_detector_tag(self, tag: str) -> DemInferenceSettings: ...
    def with_observable_tags(self, tags: Sequence[str]) -> DemInferenceSettings: ...
    def with_probe_shots(self, shots: int) -> DemInferenceSettings: ...
    def with_provenance_shots(self, shots: int) -> DemInferenceSettings: ...
    def with_validation_rows(self, rows: int) -> DemInferenceSettings: ...
    def with_require_raw_provenance(self, required: bool) -> DemInferenceSettings: ...
    def validate(self) -> None: ...
    def resolved_probe_shots(self, raw_width: int) -> int: ...
    def infer(
        self,
        raw_measurement_ids: Sequence[int],
        raw_rows: Sequence[Sequence[int]],
        detector_rows: Sequence[Sequence[int]],
        observable_rows: Sequence[Sequence[int]],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Infer detector and observable supports from bound bit-valued probe rows."""
        ...

    @property
    def raw_tag(self) -> str: ...
    @property
    def detector_tag(self) -> str: ...
    @property
    def observable_tags(self) -> list[str]: ...
    @property
    def probe_shots(self) -> int | None: ...
    @property
    def provenance_shots(self) -> int: ...
    @property
    def validation_rows(self) -> int: ...
    @property
    def require_raw_provenance(self) -> bool: ...

class DetectorErrorModel:
    """Rust-backed detector error model."""

    def detector_coordinates(self) -> list[tuple[int, list[float] | None]]: ...
    def detector_fault_sectors(self) -> list[tuple[int, str | None]]: ...
    def detector_bases(self) -> list[str]: ...
    def build_decoder(self, decoder: decoders.DecoderSpec | str) -> StructuredDemDecoder: ...
    def round_schedule(self, influence_map: Any, circuit: DagCircuit) -> DemSliceRoundSchedule: ...
    def required_buffer_rounds(
        self,
        influence_map: Any,
        circuit: DagCircuit,
        start_round: int,
        commit_rounds: int,
    ) -> int: ...
    def composed_round_window(
        self,
        influence_map: Any,
        circuit: DagCircuit,
        start_round: int,
        commit_rounds: int,
        buffer_rounds: int | None = ...,
        forward_boundary: str = ...,
    ) -> DetectorErrorModel: ...
    def graphlike_fault_distance(self) -> FaultDistanceResult | None: ...
    def connected_cluster_fault_distance(self, max_weight: int) -> FaultDistanceResult | None: ...
    def per_observable_fault_distances(self, max_weight: int) -> list[FaultDistanceResult | None]: ...
    def exhaustive_fault_distance(self, max_weight: int) -> FaultDistanceResult | None: ...
    def randomized_fault_distance_upper_bound(
        self, config: FaultDistanceUpperBoundConfig
    ) -> FaultDistanceUpperBoundResult | None: ...
    def __getattr__(self, name: str) -> Any: ...

class StructuredDemDecoder:
    """Decoder built from standard DEM text plus the object's fault_sector labels."""

    @property
    def dem(self) -> str: ...
    @property
    def projection(self) -> ProjectionReport | None:
        """Shared report, or None if unneeded or projected within members without a shared policy."""

    @property
    def num_detectors(self) -> int | None: ...
    def decode_syndrome(self, syndrome: list[int]) -> decoders.ObservableFlips: ...

class CircuitFaultLocation:
    """A Pauli-fault location in a tick circuit."""

    @property
    def tick(self) -> int: ...
    @property
    def gate_type(self) -> str: ...
    @property
    def qubits(self) -> list[int]: ...
    @property
    def gate_index(self) -> int: ...
    @property
    def before(self) -> bool: ...
    def __repr__(self) -> str: ...

class HookError:
    """A single-location fault that amplifies across the data block."""

    @property
    def location(self) -> CircuitFaultLocation: ...
    @property
    def fault_paulis(self) -> list[int]: ...
    @property
    def data_support(self) -> list[int]: ...
    @property
    def data_weight(self) -> int: ...
    @property
    def detected(self) -> bool: ...
    @property
    def causes_logical_error(self) -> bool: ...
    def __repr__(self) -> str: ...

class HookErrorReport:
    """Summary of hook-error diagnosis over a selected Pauli fault set."""

    @property
    def hook_errors(self) -> list[HookError]: ...
    @property
    def total_faults_examined(self) -> int: ...
    @property
    def max_data_weight(self) -> int: ...
    def __repr__(self) -> str: ...

class FlagViolation:
    """A counterexample to the propagated-fault condition for a flag circuit."""

    @property
    def faults(self) -> list[tuple[CircuitFaultLocation, list[int]]]: ...
    @property
    def num_faults(self) -> int: ...
    @property
    def error_weight(self) -> int: ...
    def __repr__(self) -> str: ...

class FlagFaultToleranceReport:
    """Result of checking the propagated-fault condition through fault weight ``t``."""

    @property
    def fault_condition_satisfied(self) -> bool: ...
    @property
    def t(self) -> int: ...
    @property
    def violations(self) -> list[FlagViolation]: ...
    @property
    def total_configurations_tested(self) -> int: ...
    def __repr__(self) -> str: ...

class CircuitDistanceResult:
    """A circuit fault distance and its first iterator-ordered witness."""

    @property
    def distance(self) -> int: ...
    @property
    def witness(self) -> list[tuple[CircuitFaultLocation, list[int]]]: ...
    @property
    def logical_index(self) -> int: ...
    def __repr__(self) -> str: ...

class CircuitFaultAnalyzer:
    """Fault-tolerance diagnostics and distance searches for a tick circuit."""

    def __init__(self, circuit: TickCircuit) -> None: ...
    def hook_errors(
        self,
        data_qubits: list[int],
        z_ancillas: list[int],
        x_ancillas: list[int],
        logicals: list[tuple[list[int], list[int]]],
        min_data_weight: int,
        *,
        x_only: bool = False,
        y_only: bool = False,
        z_only: bool = False,
    ) -> HookErrorReport: ...
    def flag_fault_condition(
        self,
        data_qubits: list[int],
        flag_qubits: list[int],
        measured_stabilizer: tuple[list[int], list[int]],
        t: int,
        *,
        x_only: bool = False,
        y_only: bool = False,
        z_only: bool = False,
    ) -> FlagFaultToleranceReport: ...
    def fault_distance(
        self,
        z_ancillas: list[int],
        x_ancillas: list[int],
        logicals: list[tuple[list[int], list[int]]],
        max_weight: int,
        *,
        x_only: bool = False,
        y_only: bool = False,
        z_only: bool = False,
    ) -> CircuitDistanceResult | None: ...
    def per_logical_fault_distances(
        self,
        z_ancillas: list[int],
        x_ancillas: list[int],
        logicals: list[tuple[list[int], list[int]]],
        max_weight: int,
        *,
        x_only: bool = False,
        y_only: bool = False,
        z_only: bool = False,
    ) -> list[CircuitDistanceResult | None]: ...
    def __repr__(self) -> str: ...

class CertifiedDistance:
    """A natively checked SAT witness and solver-trusted UNSAT prefix."""

    @property
    def distance(self) -> int: ...
    @property
    def witness(self) -> list[bool]: ...
    @property
    def sat_certified(self) -> bool: ...
    @property
    def unsat_trusted_below(self) -> int: ...
    def __repr__(self) -> str: ...

class StabilizerDistanceSearchResult:
    """Certified stabilizer distance or a proven lower bound after budget exhaustion."""

    @property
    def certified(self) -> bool: ...
    @property
    def distance(self) -> int | None: ...
    @property
    def min_weight_operator(self) -> Any | None: ...
    @property
    def lower_bound(self) -> int: ...
    @property
    def max_weight(self) -> int | None: ...
    def __repr__(self) -> str: ...

class ClassicalDistanceSearchResult:
    """Certified classical distance, empty kernel, or a budget-exhausted lower bound."""

    @property
    def certified(self) -> bool: ...
    @property
    def distance(self) -> int | None: ...
    @property
    def witness(self) -> list[bool] | None: ...
    @property
    def lower_bound(self) -> int | None: ...
    @property
    def max_weight(self) -> int | None: ...
    @property
    def no_nonzero_codeword(self) -> bool: ...
    def __repr__(self) -> str: ...

class DistanceProblem:
    """A binary undetectable-error problem with nonzero logical effect."""

    @classmethod
    def from_css_checks(cls, hx: ParityCheckMatrix, lx: ParityCheckMatrix) -> DistanceProblem: ...
    @classmethod
    def from_css_code_x_distance(cls, spec: StabilizerCodeSpec) -> DistanceProblem: ...
    @classmethod
    def from_css_code_z_distance(cls, spec: StabilizerCodeSpec) -> DistanceProblem: ...
    @classmethod
    def from_dem(cls, dem: DetectorErrorModel) -> DistanceProblem: ...
    @property
    def num_vars(self) -> int: ...
    def to_dimacs(self, max_weight: int) -> str: ...
    def to_wcnf(self) -> str: ...
    def verify_witness(self, witness: list[bool]) -> int: ...
    def certified_distance(self, max_weight: int) -> CertifiedDistance | None: ...
    def __repr__(self) -> str: ...

def certified_distance(problem: DistanceProblem, max_weight: int) -> CertifiedDistance | None: ...
def connected_cluster_code_distance(
    h: ParityCheckMatrix,
    l: ParityCheckMatrix,
    max_weight: int,
) -> FaultDistanceResult | None: ...
def randomized_code_distance_upper_bound(
    h: ParityCheckMatrix,
    l: ParityCheckMatrix,
    config: FaultDistanceUpperBoundConfig,
) -> FaultDistanceUpperBoundResult | None: ...
def x_distance(spec: StabilizerCodeSpec, max_weight: int) -> FaultDistanceResult | None: ...
def z_distance(spec: StabilizerCodeSpec, max_weight: int) -> FaultDistanceResult | None: ...
def stabilizer_code_distance(spec: StabilizerCodeSpec, max_weight: int) -> StabilizerDistanceSearchResult: ...

class DistanceResult:
    @property
    def distance(self) -> int: ...
    @property
    def min_weight_operator(self) -> PauliString: ...

def subsystem_dressed_distance(
    num_qubits: int,
    stabilizers: list[PauliString],
    gauge_generators: list[PauliString],
    logical_zs: list[PauliString],
    logical_xs: list[PauliString],
    max_weight: int,
) -> DistanceResult | None: ...

# The native QEC module predates this focused stub. Preserve the untyped behavior of its other
# classes and functions until that complete API is migrated rather than falsely narrowing them.
def __getattr__(name: str) -> Any: ...
def certified_coset_weight(
    group: ParityCheckMatrix, representative: list[int], max_weight: int
) -> CertifiedDistance | None: ...
def certified_stabilizer_coset_weight(
    code: StabilizerCodeSpec, operator: Any, max_weight: int
) -> CertifiedDistance | None: ...
def logical_generator_coset_weights(code: StabilizerCodeSpec, max_weight: int) -> list[CertifiedDistance | None]: ...
def certified_classical_distance(h: ParityCheckMatrix, max_weight: int) -> ClassicalDistanceSearchResult: ...

class HypergraphProductCode:
    """A hypergraph-product CSS code built from two classical parity-check matrices."""

    def __init__(self, h1: ParityCheckMatrix, h2: ParityCheckMatrix) -> None: ...
    @property
    def hx(self) -> ParityCheckMatrix: ...
    @property
    def hz(self) -> ParityCheckMatrix: ...
    @property
    def logical_x(self) -> ParityCheckMatrix: ...
    @property
    def logical_z(self) -> ParityCheckMatrix: ...
    def num_qubits(self) -> int: ...
    def num_logical_qubits(self) -> int: ...

class ProjectionReport:
    """Immutable projection diagnostics. Pickling is not supported."""

    __hash__: ClassVar[None]

    @property
    def input_hyperedges(self) -> int: ...
    @property
    def projected(self) -> int: ...
    @property
    def zero_probability_omitted(self) -> list[int]: ...
    @property
    def dropped_indices(self) -> list[int]: ...
    @property
    def cross_sector_components(self) -> int: ...
    @property
    def label_conflicts(self) -> int: ...
    @property
    def unlabelled_detectors(self) -> int: ...
    @property
    def distinct_sectors(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

class DecoderComparisonResult:
    @property
    def dut_projection(self) -> ProjectionReport | None:
        """Shared report, or None if unneeded or projected within members without a shared policy."""

    @property
    def reference_projection(self) -> ProjectionReport | None:
        """Shared report, or None if unneeded or projected within members without a shared policy."""

    @property
    def counts(self) -> list[list[int]]: ...
    @property
    def total_shots(self) -> int: ...
    @property
    def alpha(self) -> float: ...
    @property
    def dut_correct_reference_correct(self) -> int: ...
    @property
    def dut_correct_reference_mismatch(self) -> int: ...
    @property
    def dut_correct_reference_error(self) -> int: ...
    @property
    def dut_mismatch_reference_correct(self) -> int: ...
    @property
    def dut_mismatch_reference_mismatch(self) -> int: ...
    @property
    def dut_mismatch_reference_error(self) -> int: ...
    @property
    def dut_error_reference_correct(self) -> int: ...
    @property
    def dut_error_reference_mismatch(self) -> int: ...
    @property
    def dut_error_reference_error(self) -> int: ...
    @property
    def dut_only_failures(self) -> int: ...
    @property
    def dut_only_failure_proportion(self) -> float: ...
    @property
    def dut_only_failure_interval(self) -> tuple[float, float]: ...
    @property
    def both_failed(self) -> int: ...
    @property
    def both_failed_proportion(self) -> float: ...
    @property
    def both_failed_interval(self) -> tuple[float, float]: ...

class SampleBatch:
    def __init__(
        self,
        detection_events: Sequence[Sequence[int | bool]],
        observable_masks: Sequence[int],
        *,
        num_observables: int | None = ...,
    ) -> None: ...
    @property
    def num_shots(self) -> int: ...
    @property
    def num_observables(self) -> int: ...
    @property
    def seed(self) -> int | None: ...
    @property
    def dem(self) -> str | None: ...
    @property
    def dem_fault_sectors(self) -> list[str | None] | None: ...
    @property
    def metadata_json(self) -> str | None: ...
    @property
    def generator(self) -> str | None: ...
    @property
    def format_version(self) -> int | None: ...
    def get_syndrome(self, i: int) -> list[int]: ...
    def get_observable_flips(self, i: int) -> ObservableFlips: ...
    def detector_events(self) -> list[list[bool]]: ...
    def observable_flips(self) -> list[list[bool]]: ...
    def save(
        self,
        path: str | os.PathLike[str],
        *,
        dem: str | DetectorErrorModel,
        metadata_json: str | None = ...,
        clear_metadata: bool = ...,
        allow_dem_mismatch: bool = ...,
    ) -> None:
        """Write v2 (older PECOS rejects it); readers accept v1 and v2.

        Objects supply labels; equal embedded text preserves stored labels;
        other text writes all-null labels. Identity compares only text.
        Saving does not attach a model or mutate the source batch.
        """
        ...

    @staticmethod
    def load(path: str | os.PathLike[str]) -> SampleBatch:
        """Read v1 (unlabelled) or v2; format_version reports the disk version.

        Every save writes v2, which older PECOS installations reject.
        """
        ...

    def decode(
        self,
        dem: str | DetectorErrorModel | None = ...,
        decoder: decoders.DecoderSpec | str = ...,
        *,
        workers: int | None = ...,
        predictions: bool = ...,
        timing: bool = ...,
        allow_dem_mismatch: bool = ...,
    ) -> DecodeResult:
        """Decode all shots with planned native, sequential, or parallel execution.

        ``decoder`` is a typed decoder spec or legacy string. ``dem=None``
        requires a loaded corpus and uses its text plus stored fault-sector labels.
        Labelled embedded models use object semantics and refuse providers;
        all-null labels use text semantics. Explicit text is unlabelled.
        An object decodes its standard text plus its fault_sector labels. Explicit
        ``workers`` requests that exact count; automatic execution respects
        stateful and wall-clock-limited decoder contracts. Predictions are
        wide Python integers, and timing forces per-shot dispatch.
        """
        ...

    def compare_decoders(
        self,
        dem: str | DetectorErrorModel,
        dut_decoder_type: str,
        reference_decoder_type: str,
        alpha: float = ...,
        *,
        allow_dem_mismatch: bool = ...,
    ) -> DecoderComparisonResult: ...

class DecodeStats:
    num_shots: int
    num_errors: int
    logical_error_rate: float
    total_seconds: float
    wall_elapsed: float
    summed_decode_elapsed: float
    num_timing_samples: int
    per_shot_mean: float
    per_shot_median: float
    per_shot_p99: float
    per_shot_min: float
    per_shot_max: float
    quantiles: list[float]

class DecodeRecovery:
    shot_index: int
    undecoded: bool
    logical_error: bool | None
    fallback_used: bool
    syndrome_consistent: bool | None
    deltas_tried: list[float]

class DecodeResult:
    @property
    def projection(self) -> ProjectionReport | None:
        """Shared report, or None if unneeded or projected within members without a shared policy."""
    num_shots: int
    num_errors: int
    num_scored: int
    num_undecoded: int
    num_fallback: int
    recovery: list[DecodeRecovery]
    logical_error_rate: float
    execution_path: str
    workers_used: int
    reproducibility_warnings: list[str]
    sampling_seed_used: int | None
    predictions: list[int | None] | None
    stats: DecodeStats | None
    def interval(self, alpha: float = ...) -> tuple[float, float]: ...

class DemSampler:
    @staticmethod
    def from_circuit(
        circuit: DagCircuit | TickCircuit,
        p1: float = ...,
        p2: float = ...,
        p_meas: float = ...,
        p_prep: float = ...,
        **noise_options: Any,
    ) -> DemSampler: ...
    @staticmethod
    def from_dem_string(dem_string: str) -> DemSampler: ...
    @staticmethod
    def raw_uniform(influence_map: DagFaultInfluenceMap, p_error: float) -> DemSampler: ...
    @staticmethod
    def raw(
        influence_map: DagFaultInfluenceMap,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> DemSampler: ...
    @staticmethod
    def with_detectors(
        influence_map: DagFaultInfluenceMap,
        detectors: Sequence[Sequence[int]],
        observables: Sequence[Sequence[int]],
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
        **noise_options: Any,
    ) -> DemSampler: ...
    @staticmethod
    def from_influence_map(
        influence_map: DagFaultInfluenceMap,
        p_error: float,
    ) -> DemSampler: ...
    @staticmethod
    def from_influence_map_circuit_noise(
        influence_map: DagFaultInfluenceMap,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> DemSampler: ...
    def to_detector_error_model(self) -> DetectorErrorModel: ...
    @property
    def num_mechanisms(self) -> int: ...
    @property
    def num_outputs(self) -> int: ...
    @property
    def num_detectors(self) -> int: ...
    @property
    def num_observables(self) -> int: ...
    @property
    def num_dem_outputs(self) -> int: ...
    @property
    def num_tracked_paulis(self) -> int: ...
    def sample(self, seed: int | None = ...) -> tuple[list[bool], list[bool]]: ...
    def sample_batch(self, num_shots: int, seed: int | None = ...) -> SampleBatch: ...
    def sample_batch_with_pauli_masks(
        self,
        num_shots: int,
        lookup: PauliFrameLookup,
        pauli_masks: Any,
        seed: int | None = ...,
    ) -> SampleBatch: ...
    def sample_tracked_paulis(self, seed: int | None = ...) -> list[bool]: ...
    def sample_tracked_pauli_batch(self, num_shots: int, seed: int | None = ...) -> list[list[bool]]: ...
    def sample_statistics(self, num_shots: int, seed: int | None = ...) -> dict[str, Any]: ...
    def decode(
        self,
        dem: str | DetectorErrorModel,
        num_shots: int,
        decoder: decoders.DecoderSpec | str,
        *,
        seed: int | None = ...,
        workers: int | None = ...,
        predictions: bool = ...,
        timing: bool = ...,
    ) -> DecodeResult:
        """Sample and decode under sampling ABI v1.

        A fixed seed produces the same shot stream for all worker counts and
        execution paths. Predictions and counts match too, except when
        ``reproducibility_warnings`` is non-empty -- a wall-clock-limited
        decoder run in parallel can decode differently under CPU contention.
        The resolved replay seed is returned in ``sampling_seed_used``;
        timing is always outside the reproducibility guarantee.
        """
        ...

    def labels(self) -> dict[str, Any]: ...

class FaultLocation:
    @property
    def node(self) -> int: ...
    @property
    def qubits(self) -> list[int]: ...
    @property
    def before(self) -> bool: ...
    @property
    def gate_type(self) -> str: ...

class DagFaultInfluenceMap:
    @property
    def num_locations(self) -> int: ...
    @property
    def num_detectors(self) -> int: ...
    @property
    def num_dem_outputs(self) -> int: ...
    @property
    def num_observables(self) -> int: ...
    @property
    def num_tracked_paulis(self) -> int: ...
    def get_locations(self) -> list[FaultLocation]: ...
    def get_location(self, loc_idx: int) -> FaultLocation | None: ...
    def classify_fault(self, loc_idx: int, pauli: int) -> tuple[bool, bool]: ...
    def get_detector_indices(self, loc_idx: int, pauli: int) -> list[int]: ...
    def get_dem_output_indices(self, loc_idx: int, pauli: int) -> list[int]: ...
    def get_internal_dem_output_indices(self, loc_idx: int, pauli: int) -> list[int]: ...
    def get_tracked_pauli_indices(self, loc_idx: int, pauli: int) -> list[int]: ...
    def get_observable_indices(self, loc_idx: int, pauli: int) -> list[int]: ...
    def has_detector_flips(self, loc_idx: int, pauli: int) -> bool: ...
    def has_dem_output_flips(self, loc_idx: int, pauli: int) -> bool: ...
    def has_observable_flips(self, loc_idx: int, pauli: int) -> bool: ...
    def has_tracked_pauli_flips(self, loc_idx: int, pauli: int) -> bool: ...
    def merge_dem_outputs_from(self, other: DagFaultInfluenceMap) -> None: ...
    def memory_stats(self) -> dict[str, Any]: ...
    def export_csr(self) -> dict[str, Any]: ...
    def measurements(self) -> list[tuple[int, int, int]]: ...
    def __len__(self) -> int: ...

class PauliFrameLookup:
    @staticmethod
    def from_circuit(
        dag: DagCircuit,
        detectors: Sequence[Sequence[int]],
        observables: Sequence[Sequence[int]],
    ) -> PauliFrameLookup: ...
    @property
    def num_pauli_sites(self) -> int: ...
    @property
    def num_tracked_paulis(self) -> int: ...
    @property
    def num_detectors(self) -> int: ...
    @property
    def num_observables(self) -> int: ...
    def row(self, tracked_idx: int) -> tuple[list[int], list[int]]: ...
    def mask_firings(self, pauli_masks: Any) -> list[list[bool]]: ...
    def compute_mask_xor(self, pauli_masks: Any) -> tuple[list[list[bool]], list[list[bool]]]: ...

class _SeleneRuntimeSpec:
    def __init__(self) -> None: ...
    def selene_runtime(
        self,
        runtime_name: str | None = None,
        *,
        custom_event_policy: str = "reject_unhandled",
        native_gates: list[str] | None = None,
    ) -> _SeleneRuntimeSpec: ...
    def selene_runtime_plugin(
        self,
        library_file: str,
        init_args: list[str] | None = None,
        library_search_dirs: list[str] | None = None,
        *,
        custom_event_policy: str = "reject_unhandled",
        native_gates: list[str] | None = None,
    ) -> _SeleneRuntimeSpec: ...

class _DemEvidence(TypedDict):
    circuit: TickCircuit
    raw_measurement_ids: list[int]
    raw_rows: list[list[int]]
    detector_rows: list[list[int]]
    observable_rows: list[list[int]]
    binding: str
    detector_outputs: list[tuple[str, int]]
    observable_outputs: list[tuple[str, int]]
    probe_shots: int
    correlation_shots: int
    schedule_shots: int
    named_result_traces: list[dict[str, Any]]

def _capture_dem_evidence(
    program: Qis,
    num_qubits: int,
    seed: int,
    runtime: _SeleneRuntimeSpec,
    settings: DemInferenceSettings,
    detector_outputs: Sequence[tuple[str, int]] | None = None,
    observable_outputs: Sequence[tuple[str, int]] | None = None,
) -> _DemEvidence: ...
