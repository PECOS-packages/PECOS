"""Python boundary for Rust-backed Guppy detector error model construction.

This module attaches ``from_guppy`` to the Rust detector error model and
exposes the typed builder. Python selects program carriers, checks HUGR static
schedules when required, and converts runtime, noise, and definition objects.
Rust prepares the trace, resolves result tags and references, applies idle
passes, and builds the DEM from a reusable schema.

Guppy preparation lowers Clifford rotations first, so gate-rate keys name
lowered gates such as ``SZZ``. Direct ``from_circuit`` paths retain scheduled
rotation names unless the caller lowers the circuit first. Result-tag
resolution belongs to Guppy preparation; direct circuit builds consume
``records`` and ``meas_ids`` metadata.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

from pecos_rslib import Qis
from pecos_rslib.qec import DemInferenceSettings, _prepare_guppy_dem, _SeleneRuntimeSpec
from pecos_rslib.qec import DetectorErrorModel as _RustDetectorErrorModel

from pecos._engine_builders import _configure_selene_runtime
from pecos.programs import Guppy
from pecos.qec._idle_noise import _translate_structured_idle_noise
from pecos.qec.dem_spec import GuppyDemBuild, ResultRef, _definition_record
from pecos.qec.guppy_output_dem import InferredGuppyDemAnnotations, _guppy_dem_program

if TYPE_CHECKING:
    from typing import Self

    from hugr import Hugr

    from pecos.qec.dem_spec import Detector, Observable
    from pecos.qec.surface.decode import NoiseParameters

P1Weights = Mapping[str, float]
P2Weights = Mapping[str, float]

_GENERATOR_LAYOUT_ATTR = "__pecos_named_measurement_layout_v2__"

_GUPPY_NOISE_KEYWORDS = (
    "p1",
    "p1_weights",
    "p2",
    "p2_weights",
    "p2_replacement_approximation",
    "p_meas",
    "p_prep",
    "p_idle_linear",
    "p_idle_linear_model",
    "p_idle_sin_squared",
    "p_idle_sin_squared_model",
    "p_idle_coherent",
    "p_idle_coherent_model",
    "t1",
    "t2",
    "p_idle_linear_rate",
    "p_idle_quadratic_rate",
    "p_idle_x_linear_rate",
    "p_idle_y_linear_rate",
    "p_idle_z_linear_rate",
    "p_idle_x_quadratic_rate",
    "p_idle_y_quadratic_rate",
    "p_idle_z_quadratic_rate",
    "p_idle_quadratic_sine_rate",
    "p_idle_x_quadratic_sine_rate",
    "p_idle_y_quadratic_sine_rate",
    "p_idle_z_quadratic_sine_rate",
)
_NOISE_PARAMETER_INTERNAL_IDLE_FIELDS = frozenset(
    {
        "p_idle_linear_rate",
        "p_idle_quadratic_rate",
        "p_idle_x_linear_rate",
        "p_idle_y_linear_rate",
        "p_idle_z_linear_rate",
        "p_idle_x_quadratic_rate",
        "p_idle_y_quadratic_rate",
        "p_idle_z_quadratic_rate",
        "p_idle_quadratic_sine_rate",
        "p_idle_x_quadratic_sine_rate",
        "p_idle_y_quadratic_sine_rate",
        "p_idle_z_quadratic_sine_rate",
    },
)


class _NoiseKeywordDefault:
    """Track whether a flat noise keyword was explicitly supplied."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)


_NOISE_DEFAULT_NONE = _NoiseKeywordDefault(None)
_NOISE_DEFAULT_P1 = _NoiseKeywordDefault(0.001)
_NOISE_DEFAULT_P2 = _NoiseKeywordDefault(0.01)


def _resolve_guppy_noise(noise: NoiseParameters | None, call_arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one grouped or flat Guppy DEM noise configuration."""
    explicitly_flat = [
        name for name in _GUPPY_NOISE_KEYWORDS if not isinstance(call_arguments[name], _NoiseKeywordDefault)
    ]
    if noise is None:
        return {
            name: (
                call_arguments[name].value
                if isinstance(call_arguments[name], _NoiseKeywordDefault)
                else call_arguments[name]
            )
            for name in _GUPPY_NOISE_KEYWORDS
        }

    if explicitly_flat:
        conflicts = ", ".join(explicitly_flat)
        msg = f"noise cannot be combined with flat noise keyword(s): {conflicts}"
        raise ValueError(msg)

    # Import locally so dem.py remains below surface.decode in the package's
    # initialization graph instead of introducing a module-level back edge.
    from pecos.qec.surface.decode import NoiseParameters

    if not isinstance(noise, NoiseParameters):
        msg = f"noise must be a NoiseParameters instance or None, got {type(noise).__name__}"
        raise TypeError(msg)

    unsupported = {
        "p_idle": "use p_idle_linear instead",
        "p2_szz": "use the shared p2 rate instead",
        "p2_szzdg": "use the shared p2 rate instead",
    }
    for field, guidance in unsupported.items():
        if getattr(noise, field) is not None:
            msg = f"NoiseParameters.{field} is not supported by the Guppy DEM entry points; {guidance}"
            raise ValueError(msg)

    expanded = {
        name: getattr(noise, f"_{name}" if name in _NOISE_PARAMETER_INTERNAL_IDLE_FIELDS else name)
        for name in _GUPPY_NOISE_KEYWORDS
    }
    for weights_name in ("p1_weights", "p2_weights"):
        if expanded[weights_name] is not None:
            expanded[weights_name] = dict(expanded[weights_name])
    return expanded


def _certifiable_hugr_bytes(guppy: Any) -> bytes | None:
    """Return the HUGR bytes that certify this program's static schedule.

    Accepts ``@guppy`` definitions (compiled through the shared cache),
    ``pecos.Guppy`` wrappers (their ``hugr_bytes``, from the same cache), and ``pecos.Hugr``
    wrappers or raw HUGR envelope bytes (used directly, so the audit inspects
    the exact bytes that would execute). Returns ``None`` when the input
    shape is not HUGR-certifiable; audited callers fail closed on ``None``
    rather than tracing one sampled execution of an uninspectable program.
    """
    if isinstance(guppy, (bytes, bytearray)):
        return bytes(guppy)
    wrapped_bytes = getattr(guppy, "hugr_bytes", None)
    if isinstance(wrapped_bytes, (bytes, bytearray)):
        return bytes(wrapped_bytes)
    from pecos._compilation import guppy_to_hugr

    target = getattr(guppy, "wrapped_function", guppy)
    try:
        return guppy_to_hugr(target)
    except ValueError:
        # Not a Guppy definition at all; a genuine compile failure of a real
        # definition raises RuntimeError and propagates to the caller.
        return None


def _certificate_carrier(guppy: Any) -> Any | None:
    """Return the object whose generator certificate may be honored, if any.

    Certificates are stamped by built-in generators on Guppy *definition*
    objects only. ``pecos.Guppy`` is honored through the definition it wraps,
    even though it also exposes the compiled envelope as ``hugr_bytes``: those
    bytes are compiled from that same definition.

    Byte carriers (``pecos.Hugr``, raw HUGR bytes, duck-typed ``hugr_bytes``
    holders) never carry an honorable certificate, whatever ``wrapped_function``
    they claim: they are opaque data, and honoring an attribute there would let
    any bytes suppress the control-flow guard by stapling a self-consistent
    digest to themselves.
    """
    if isinstance(guppy, (bytes, bytearray)):
        return None
    if isinstance(guppy, Guppy):
        return guppy.wrapped_function
    if isinstance(getattr(guppy, "hugr_bytes", None), (bytes, bytearray)):
        return None
    return getattr(guppy, "wrapped_function", guppy)


def _generator_certified_layout(guppy: Any, hugr_bytes: bytes | None = None) -> tuple[str, Sequence[Any]] | None:
    """Validate and return a built-in generator's program-bound layout."""
    certificate = getattr(guppy, _GENERATOR_LAYOUT_ATTR, None)
    if certificate is None:
        return None
    if (
        not isinstance(certificate, Sequence)
        or isinstance(certificate, (str, bytes))
        or len(certificate) != 2
        or not isinstance(certificate[0], str)
    ):
        msg = "invalid Guppy generator measurement-layout certificate"
        raise ValueError(msg)
    digest, layout = certificate
    if hugr_bytes is None:
        from pecos._compilation import guppy_to_hugr

        hugr_bytes = guppy_to_hugr(guppy)
    from pecos_rslib import generator_certificate_digest

    generator_certificate_digest(hugr_bytes, layout, expected=digest)
    return digest, layout


class _DetectorErrorModelMixin:
    """Namespace for the Python Guppy/QIS-trace convenience constructor."""

    __slots__ = ()

    @classmethod
    def builder(cls) -> GuppyDemBuilder:
        """Create a chained builder for an audited Guppy detector error model."""
        return GuppyDemBuilder()

    @classmethod
    def from_guppy(
        cls,
        guppy: Any,
        *,
        num_qubits: int,
        detectors_json: str,
        observables_json: str = "[]",
        num_measurements: int | None = None,
        noise: NoiseParameters | None = None,
        p1: float = _NOISE_DEFAULT_P1,
        p1_weights: P1Weights | None = _NOISE_DEFAULT_NONE,
        p2: float = _NOISE_DEFAULT_P2,
        p2_weights: P2Weights | None = _NOISE_DEFAULT_NONE,
        p2_replacement_approximation: str | None = _NOISE_DEFAULT_NONE,
        p_meas: float = _NOISE_DEFAULT_P1,
        p_prep: float = _NOISE_DEFAULT_P1,
        p_idle_linear: float | None = _NOISE_DEFAULT_NONE,
        p_idle_linear_model: Mapping[str, float] | None = _NOISE_DEFAULT_NONE,
        p_idle_sin_squared: float | None = _NOISE_DEFAULT_NONE,
        p_idle_sin_squared_model: Mapping[str, float] | None = _NOISE_DEFAULT_NONE,
        p_idle_coherent: float | None = _NOISE_DEFAULT_NONE,
        p_idle_coherent_model: Mapping[str, float] | None = _NOISE_DEFAULT_NONE,
        t1: float | None = _NOISE_DEFAULT_NONE,
        t2: float | None = _NOISE_DEFAULT_NONE,
        p_idle_linear_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_quadratic_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_x_linear_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_y_linear_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_z_linear_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_x_quadratic_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_y_quadratic_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_z_quadratic_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_quadratic_sine_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_x_quadratic_sine_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_y_quadratic_sine_rate: float | None = _NOISE_DEFAULT_NONE,
        p_idle_z_quadratic_sine_rate: float | None = _NOISE_DEFAULT_NONE,
        strip_traced_idles: bool | None = None,
        idle_after_2q_duration: float | None = None,
        runtime: object | None = None,
        seed: int = 0,
        require_hosted_operation_order: bool = False,
        max_hosted_tick_separation: int | None = None,
    ) -> _RustDetectorErrorModel:
        """Build a circuit-level DEM from a Guppy program by tracing it.

        Runs ``guppy`` under the Selene QIS engine with operation tracing,
        replays the captured gate stream into a ``TickCircuit``, attaches the
        caller-supplied detector/observable definitions, and builds the DEM via
        native PECOS fault propagation. All metadata validation happens in the
        Rust DEM builder (single source of truth). Preparation computes exact
        noiseless detector and observable parities and writes ``reference`` into
        their metadata. Random parities and disagreeing declared references
        are rejected. Explicit builds require a HUGR static-schedule preflight
        or a trusted generator layout on an eligible definition carrier.

        The three structured idle models contain relative-rate multipliers:
        each axis rate is ``family_rate * axis_multiplier``. The linear law is
        additive, so its multipliers must sum to 1.0 and coincide with the
        engines relative-probability distribution. The nonlinear sine-squared
        and coherent laws are not additive, so their finite, non-negative
        multipliers have no sum constraint.

        Args:
            guppy: A HUGR-certifiable program: a ``@guppy``-decorated function,
                a compiled Guppy program (e.g. the object returned by
                ``pecos.guppy_gen.make_surface_code``), a ``pecos.Guppy`` or
                ``pecos.Hugr`` wrapper, or raw HUGR envelope bytes. Inputs whose
                HUGR cannot be obtained (e.g. already-lowered QIS/QIR) are
                rejected: the audited build must be able to certify the static
                schedule before tracing, and the exact certified bytes are what
                gets executed.
            num_qubits: Number of qubits to allocate for the trace. QIS/HUGR
                programs require an explicit qubit count.
            detectors_json: Detector definitions as a JSON list, e.g.
                ``[{"id": 0, "records": [-1, -5]}, ...]``. ``id`` may be a bare
                integer or, for convenience, the DEM-label form ``"D0"``
                (observables likewise accept ``"L0"``); both normalize to the
                same integer.

                Each entry references measurements in one of three ways
                (provide exactly one form; co-presence is allowed only if the
                forms reference the same measurements):

                - ``records``: negative measurement offsets (Stim convention),
                  positional in the traced measurement record.
                - ``meas_ids``: stable stamped ``MeasId``s -- resolved in Rust
                  against the circuit's actual ids, so robust to any
                  measurement reordering Guppy/Selene compilation may
                  introduce.
                - ``result_tags``: Guppy ``result(tag, ...)`` tag strings
                  (e.g. ``[{"id": 0, "result_tags": ["syn_a"]}]``). The
                  reorder-immune ``tag -> measurement`` binding is recovered
                  from the executed lowered QIS and resolved to stable runtime
                  ``MeasId`` values by Rust preparation. Supported only for
                  **straight-line, canonical** programs:
                  ``result(tag, measure(q))`` of a raw scalar measurement.
                  Computed (``result(tag, m0 == m1)``), constant
                  (``result(tag, True)``), and array-valued
                  (``result(tag, measure_array(qs))``) forms are not
                  resolvable and an unknown tag is a hard error. The HUGR
                  control-flow check rejects branches and runtime loops before
                  execution. Generator-certified programs bypass that check;
                  their ``result_tags`` still require lowered-program certification.
                  ``result_tags`` accepts the HUGR-certifiable input forms
                  listed above. Use ``records`` or ``meas_ids`` when scalar
                  tag provenance cannot be certified.
            observables_json: Observable definitions as a JSON list, e.g.
                ``[{"id": 0, "records": [-1]}]`` (same id/records rules as
                detectors).

                Tracked Paulis: **hand-authored JSON tracked Paulis are NOT
                supported** by this path. Tracked Paulis are only produced from
                circuit *annotations* (e.g. the surface builder), not from
                ``observables_json``; a ``{"kind": "tracked_pauli", ...}``
                entry here is rejected by the builder.
            num_measurements: Total measurement count, used to resolve negative
                ``records`` offsets. If omitted, it is inferred from the traced
                circuit; if given, it must match the traced count.
            noise: Complete grouped noise configuration. When supplied, its
                values replace all flat noise keywords, including this entry
                point's defaults. In particular, ``NoiseParameters`` defaults such
                as ``p1=0.0`` apply instead of this function's ``p1=0.001``.
                Mixing ``noise`` with any flat noise keyword is rejected.
            p1: Single-qubit gate Pauli error rate. The categorical Pauli
                channel is converted after equal propagated signatures merge.
            p1_weights: Optional relative probabilities over single-qubit
                Pauli error labels ``"X"``, ``"Y"``, and ``"Z"``. Values must
                sum to 1.0; ``p1`` remains the total single-qubit error rate.
            p2: Two-qubit gate depolarizing rate. Its 15 categorical branches
                are converted together after propagation.
            p2_weights: Optional relative probabilities over two-qubit Pauli
                error labels. Plain labels such as ``"XX"`` are post-gate
                Pauli branches; labels such as ``"~XX"`` or
                ``":replace:XX"`` are replacement branches that omit the ideal
                two-qubit gate before applying the Pauli. ``"~II"`` (or
                ``":replace:II"``) omits the gate without applying a Pauli.
                In a label ``"AB"``, ``A`` acts on the gate's first operand and
                ``B`` on its second, whatever their qubit indices.
                The two replacement spellings are equivalent and cannot both
                name the same entry. Values must sum to 1.0; ``p2`` remains the
                total two-qubit error rate.
            p2_replacement_approximation: Approximation used for
                replacement labels. ``"pauli_twirl_omitted_gate"`` convolves
                with the omitted two-qubit gate's Pauli twirl;
                ``"branch_impact"`` evaluates replacement entries as replacement
                branch impacts; ``"exact_branch_replay"`` is reserved for a
                future circuit-aware exact replay provider and currently fails
                loudly for replacement entries; ``"ignore_gate_removal"``
                treats replacement entries like plain post-gate Pauli entries.
            p_meas: Measurement flip rate.
            p_prep: Preparation (reset) error rate.
            p_idle_linear: Optional total stochastic idle-noise rate linear in
                duration. Uses the engines ``GeneralNoiseModel`` convention.
            p_idle_linear_model: Optional relative weights over ``"X"``, ``"Y"``,
                ``"Z"``, and ``"L"`` for ``p_idle_linear``. Weights must be
                finite, non-negative, and sum to 1.0 within ``1e-5``, including
                any explicit ``"L"`` weight. Defaults to the engines' uniform
                Pauli model. DEM fault propagation is Pauli-only, so ``"L"``
                must be zero here; the engines simulators support nonzero
                leakage weights.
            p_idle_sin_squared: Optional stochastic sine-law idle rate. An
                axis multiplier ``m`` gives probability
                ``sin((p_idle_sin_squared * m) * t)^2`` for an idle of duration
                ``t``. By default X, Y, and Z each use the full family rate.
            p_idle_sin_squared_model: Optional relative-rate multipliers over
                ``"X"``, ``"Y"``, ``"Z"``, and ``"L"`` for
                ``p_idle_sin_squared``. Values must be finite and non-negative,
                with no sum constraint. Defaults to
                ``{"X": 1.0, "Y": 1.0, "Z": 1.0}``. DEM fault propagation
                is Pauli-only, so ``"L"`` must be zero here; the engines
                simulators support nonzero leakage weights.
            p_idle_coherent: Optional coherent-rotation rate. The standard DEM
                builder cannot represent coherent idle noise and rejects every
                nonzero rate rather than silently storing a Pauli twirl that
                discards the requested coherence. Use the EEG coherent route
                in ``exp/pecos-eeg`` for coherent idle noise; it supports only
                an RZ generator. The honest stochastic equivalent—the exact
                Pauli twirl of ``RZ(rate * t)``—is
                ``p_idle_sin_squared=rate/2`` with
                ``p_idle_sin_squared_model={"Z": 1.0}``. Zero has no effect.
            p_idle_coherent_model: Optional relative-rate multipliers over
                ``"RX"``, ``"RY"``, and ``"RZ"`` for ``p_idle_coherent``.
                Values must be finite and non-negative, with no sum constraint;
                the default is ``{"RX": 1.0, "RY": 1.0, "RZ": 1.0}``. The
                keys are validation-only on this route because any nonzero
                coherent family rate is rejected. ``"L"`` is not a
                coherent-model key, and ``"U"`` is reserved for future
                Hamiltonian-level support and rejected.
            t1: Optional T1 relaxation time for explicit idle gates; requires ``t2``.
            t2: Optional T2 dephasing time for explicit idle gates; requires ``t1``.
            p_idle_linear_rate: Deprecated bare Z-only alias for a stochastic
                rate linear in idle duration. Use ``p_idle_linear`` with a
                Z-only model, or ``p_idle_z_linear_rate`` for literal behavior.
            p_idle_quadratic_rate: Deprecated bare Z-only coefficient-style
                rate quadratic in idle duration. Use ``p_idle_sin_squared`` for
                the structured engines-style dephasing interface, or
                ``p_idle_z_quadratic_rate`` for literal behavior.
            p_idle_x_linear_rate: Optional stochastic X-memory rate linear in idle duration.
            p_idle_y_linear_rate: Optional stochastic Y-memory rate linear in idle duration.
            p_idle_z_linear_rate: Optional stochastic Z-memory rate linear in idle duration.
            p_idle_x_quadratic_rate: Optional stochastic X-memory rate quadratic in idle duration.
            p_idle_y_quadratic_rate: Optional stochastic Y-memory rate quadratic in idle duration.
            p_idle_z_quadratic_rate: Optional stochastic Z-memory rate quadratic in idle duration.
            p_idle_quadratic_sine_rate: Deprecated bare Z-only alias for a
                stochastic rate with probability ``sin(rate * duration)^2``.
                Use ``p_idle_sin_squared`` or ``p_idle_z_quadratic_sine_rate``.
            p_idle_x_quadratic_sine_rate: Optional stochastic X-memory sine-law rate.
            p_idle_y_quadratic_sine_rate: Optional stochastic Y-memory sine-law rate.
            p_idle_z_quadratic_sine_rate: Optional stochastic Z-memory sine-law rate.
            strip_traced_idles: If true, remove ``Idle`` gates from the
                normalized traced circuit while preserving ``I`` and
                zero-angle rotations as gate-noise locations. This pass runs
                before idle insertion when both idle-pass options are set.
                Defaults to ``None``, which strips exactly when
                ``idle_after_2q_duration`` is set:
                inserting a uniform idle convention on top of runtime-emitted
                idles would double-count idle noise. Pass ``False`` explicitly
                to keep runtime-emitted idles alongside inserted ones.
            idle_after_2q_duration: If set, insert an ``Idle`` gate of this
                duration on both qubits after every two-qubit gate in the
                normalized traced circuit. Insertion runs after
                ``strip_traced_idles`` and before result-tag resolution and
                detector/observable metadata attachment.
            runtime: Optional Selene runtime selector/plugin. ``None`` selects
                the default Selene runtime. Runtime plugin objects configure
                the Rust Selene adapter using the same plugin protocol as
                ``pecos.selene_engine(runtime)``.
            seed: Seed for the ideal trace run.
            require_hosted_operation_order: If true, validate generic
                hosted-operation metadata after trace replay. A gate with
                ``local_role`` metadata must bind to a later same-``host_id``
                host gate sharing a qubit.
            max_hosted_tick_separation: Optional maximum absolute signed tick
                separation accepted by the hosted-operation validator.

        Returns:
            A ``DetectorErrorModel`` built from the traced circuit.

        Raises:
            ValueError: If ``num_measurements`` disagrees with the traced
                measurement count, if a detector/observable is malformed or
                references an out-of-range ``record`` or an absent
                ``meas_id``, if ``idle_after_2q_duration`` is not a finite
                positive number, if ``p_idle_coherent`` is nonzero, if active
                idle noise is requested but the final traced circuit has no
                ``Idle`` gates, if only one of ``t1`` and ``t2`` is supplied, or if the traced operation stream
                cannot be replayed. To provide targets for idle noise, pass
                ``idle_after_2q_duration`` or use a Selene runtime that emits
                scheduled idles. Explicit zero rates need no idle gates.

        Note:
            Runtime-lowered idles are replayed as nanosecond PECOS
            ``TimeUnits``. If idle parameters come from a per-second
            simulator/runtime model, use
            ``noise.for_runtime_idle_time_units()`` and pass the converted
            model through this constructor's ``noise`` keyword.

            **Measurement-dependent (dynamic) control flow is unsupported.**
            ``from_guppy`` traces one ideal execution; a Guppy program whose
            quantum operations depend on a measurement *outcome* (e.g.
            ``if measure(q): x(other)``) is rejected before tracing. Generic
            branching and looping HUGR control flow is conservatively rejected
            because one sampled branch cannot certify a static circuit. Built-in
            generators may carry a trusted static measurement-layout certificate.

            Every measurement is anchored to a stable MeasId automatically:
            ``measure()`` itself allocates the result slot in the trace (a
            ``result(...)`` call is not required for MeasId assignment).

            Source-anchored tag-referenced detectors are exposed via the
            ``result_tags`` field on detectors/observables (see the
            ``detectors_json`` argument). The supported scope is canonical
            scalar ``result(tag, measure(q))`` in straight-line programs; the
            runtime-loop case (per-occurrence binding) remains deferred.
        """
        noise_keywords = {name: value for name, value in locals().items() if name in _GUPPY_NOISE_KEYWORDS}
        builder = (
            cls.builder()
            .with_program(guppy)
            .with_qubits(num_qubits)
            .with_detectors_json(detectors_json)
            .with_observables_json(observables_json)
            .with_strip_traced_idles(strip_traced_idles)
            .with_idle_after_2q(idle_after_2q_duration)
            .with_runtime(runtime)
            .with_seed(seed)
            .with_require_hosted_operation_order(require_hosted_operation_order)
            .with_max_hosted_tick_separation(max_hosted_tick_separation)
        )
        if num_measurements is not None:
            builder.with_num_measurements(num_measurements)
        # Same-module private seam: the flat keyword surface stays on this
        # function while noise() remains strictly a NoiseParameters-instance setter.
        return builder._legacy_noise(noise, noise_keywords).build().dem  # noqa: SLF001


def _result_tags_present(detectors_json: str, observables_json: str) -> bool:
    """Cheap gate: does any entry use ``result_tags``? (substring check).

    Only decides whether JSON metadata needs static evidence. Tag extraction,
    resolution, and the measurement-count guard belong to Rust preparation.
    """
    return '"result_tags"' in (detectors_json or "") or '"result_tags"' in (observables_json or "")


class _GuppyProgramAnalysis:
    """Compile and analyze a Guppy program on demand, sharing results within one DEM build."""

    def __init__(self, hugr_bytes: bytes) -> None:
        self.hugr_bytes = hugr_bytes

    @cached_property
    def hugr(self) -> Hugr:
        """Return the validated module, loading it only on first use."""
        from pecos.qec._hugr_preflight import load_hugr_from_bytes

        return load_hugr_from_bytes(self.hugr_bytes)

    @cached_property
    def qis(self) -> str:
        """Lower once with the same defaults as Hugr._to_program."""
        from pecos_rslib.hugr_lowering import compile_hugr_to_qis

        return compile_hugr_to_qis(self.hugr_bytes)


def _preflight_guppy_static_schedule(
    guppy: Any,
    *,
    required_tags: Sequence[str],
    json_result_tags: bool = False,
) -> tuple[Sequence[Any] | None, _GuppyProgramAnalysis, str | None]:
    """Validate program-level trust before any runtime trace is captured.

    Returns ``(generator_layout, analysis, digest)``. Runs the generator-certificate
    digest check and the branching/looping HUGR rejection *before* execution,
    so an unsupported program cannot run (or hang) ahead of its rejection.
    Fails closed on any input whose HUGR cannot be obtained: one sampled
    execution of an uninspectable program is not a certifiable static circuit.
    The analysis holds the bytes to lower and caches the exact QIS text to execute. Certified
    generators bypass loading; other programs load for the control-flow check.
    """
    hugr_bytes = _certifiable_hugr_bytes(guppy)
    if hugr_bytes is None:
        if required_tags:
            msg = (
                "result_ref(...) requires a HUGR-compilable Guppy program; use "
                "DetectorErrorModel.from_circuit(...) for circuit inputs"
            )
            raise ValueError(msg)
        if json_result_tags:
            msg = (
                "result_tags requires a @guppy-decorated function (or a GuppyFunctionDefinition) so the program can "
                "be compiled to a HUGR; use DetectorErrorModel.from_circuit(...) for circuit inputs"
            )
            raise ValueError(msg)
        msg = (
            "GuppyDemBuilder.program() requires a HUGR-certifiable program (a @guppy "
            "function, pecos.Guppy, pecos.Hugr, or HUGR envelope bytes); a "
            f"{type(guppy).__name__!r} input cannot be certified as statically "
            "scheduled, so an audited DEM cannot be built from it; use "
            "DetectorErrorModel.from_circuit(...) for circuit inputs"
        )
        raise ValueError(msg)

    certificate_carrier = _certificate_carrier(guppy)
    generator_layout = (
        _generator_certified_layout(certificate_carrier, hugr_bytes) if certificate_carrier is not None else None
    )
    analysis = _GuppyProgramAnalysis(hugr_bytes)
    if generator_layout is not None:
        digest, layout = generator_layout
        return layout, analysis, digest

    from pecos.qec._hugr_preflight import has_nontrivial_control_flow

    if has_nontrivial_control_flow(analysis.hugr):
        msg = (
            "GuppyDemBuilder requires a statically straight-line Guppy program unless it carries "
            "a trusted generator-owned measurement layout; branching or looping control flow cannot be "
            "certified from one runtime trace"
        )
        raise ValueError(msg)
    return None, analysis, None


def _guppy_dem_evidence(
    program: Any,
    specs: Sequence[Any],
    settings: Any,
    definitions: Mapping[str, Any],
    *,
    analysis: _GuppyProgramAnalysis | None = None,
) -> tuple[Qis, dict[str, Any]]:
    """Select static evidence and the exact QIS consumed by Rust preparation."""
    needs_inference = any(spec.inferred_output is not None for spec in specs) or (
        settings is not None
        and any(role not in definitions and f"{role}_json" not in definitions for role in ("detectors", "observables"))
    )
    tags = sorted({ref.tag for spec in specs for ref in spec.refs if isinstance(ref, ResultRef)})
    json_tags = _result_tags_present(definitions.get("detectors_json", "[]"), definitions.get("observables_json", "[]"))
    evidence = {}
    if not needs_inference or tags or json_tags:
        layout, checked, digest = _preflight_guppy_static_schedule(
            program,
            required_tags=tags,
            json_result_tags=json_tags,
        )
        if analysis is None:
            analysis = checked
        if layout is None:
            evidence["frontend_static_schedule"] = "python_hugr_preflight"
        else:
            evidence["trusted_generator_layout"] = (analysis.hugr_bytes, list(layout), digest)
    return (Qis(analysis.qis) if analysis is not None else _guppy_dem_program(program)), evidence


_UNSET = object()
_LEGACY_NOISE = object()


def _builder_noise_defaults() -> dict[str, Any]:
    """Return the legacy Guppy entry-point defaults with explicitness intact."""
    defaults = dict.fromkeys(_GUPPY_NOISE_KEYWORDS, _NOISE_DEFAULT_NONE)
    defaults.update(
        p1=_NOISE_DEFAULT_P1,
        p2=_NOISE_DEFAULT_P2,
        p_meas=_NOISE_DEFAULT_P1,
        p_prep=_NOISE_DEFAULT_P1,
    )
    return defaults


class GuppyDemBuilder:
    """Configure and build an audited detector error model from a Guppy program."""

    __slots__ = (
        "_detectors_kind",
        "_detectors_value",
        "_idle_after_2q",
        "_inference",
        "_max_hosted_tick_separation",
        "_noise",
        "_num_measurements",
        "_observables_kind",
        "_observables_value",
        "_program",
        "_qubits",
        "_require_hosted_operation_order",
        "_residual_warning_threshold",
        "_runtime",
        "_seed",
        "_strip_traced_idles",
    )

    def __init__(self) -> None:
        """Create an empty builder whose required inputs are not yet set."""
        self._inference: Any = _UNSET
        self._program: Any = _UNSET
        self._qubits: Any = _UNSET
        self._detectors_kind: str | object = _UNSET
        self._detectors_value: Any = _UNSET
        self._observables_kind: str | object = _UNSET
        self._observables_value: Any = _UNSET
        self._num_measurements: Any = _UNSET
        self._noise: Any = _UNSET
        self._idle_after_2q: Any = _UNSET
        self._strip_traced_idles: Any = _UNSET
        self._runtime: Any = _UNSET
        self._seed: Any = _UNSET
        self._residual_warning_threshold: Any = _UNSET
        self._require_hosted_operation_order: Any = _UNSET
        self._max_hosted_tick_separation: Any = _UNSET

    def _set_once(self, attribute: str, value: Any, setter: str) -> None:
        if getattr(self, attribute) is not _UNSET:
            msg = f"{setter}() may only be called once"
            raise ValueError(msg)
        setattr(self, attribute, value)

    def _set_specs(self, role: str, kind: str, value: Any) -> None:
        kind_attribute = f"_{role}_kind"
        value_attribute = f"_{role}_value"
        current_kind = getattr(self, kind_attribute)
        setter = f"with_{role}" if kind == "typed" else f"with_{role}_json"
        if current_kind is not _UNSET:
            previous = f"with_{role}" if current_kind == "typed" else f"with_{role}_json"
            if current_kind == kind:
                msg = f"{setter}() may only be called once"
                raise ValueError(msg)
            msg = f"{setter}() cannot be combined with {previous}()"
            raise ValueError(msg)
        if kind == "typed" and self._num_measurements is not _UNSET:
            msg = f"{setter}() cannot be combined with with_num_measurements()"
            raise ValueError(msg)
        setattr(self, kind_attribute, kind)
        setattr(self, value_attribute, value)

    def with_program(self, program: Any) -> Self:
        """Set the Guppy or HUGR program to trace."""
        self._set_once("_program", program, "with_program")
        return self

    def with_qubits(self, num_qubits: int) -> Self:
        """Set the number of qubits allocated to the trace."""
        self._set_once("_qubits", num_qubits, "with_qubits")
        return self

    def with_detectors(self, specs: Sequence[Detector]) -> Self:
        """Set typed detector specifications."""
        self._set_specs("detectors", "typed", tuple(specs))
        return self

    def with_observables(self, specs: Sequence[Observable]) -> Self:
        """Set typed logical-observable specifications."""
        self._set_specs("observables", "typed", tuple(specs))
        return self

    def with_detectors_json(self, text: str) -> Self:
        """Set raw JSON detector specifications."""
        self._set_specs("detectors", "json", text)
        return self

    def with_observables_json(self, text: str) -> Self:
        """Set raw JSON logical-observable specifications."""
        self._set_specs("observables", "json", text)
        return self

    def with_inference(self, settings: DemInferenceSettings) -> Self:
        """Configure inference and infer roles whose definitions are unspecified.

        Settings are immutable values shared with the Rust API. Inference is
        empirical; the returned build records its evidence separately from
        compiler-certified explicit bindings.
        """
        if not isinstance(settings, DemInferenceSettings):
            msg = "with_inference() requires a DemInferenceSettings instance"
            raise TypeError(msg)
        settings.validate()
        if self._num_measurements is not _UNSET:
            msg = "with_inference() cannot be combined with with_num_measurements()"
            raise ValueError(msg)
        self._set_once("_inference", settings, "with_inference")
        return self

    def with_num_measurements(self, count: int) -> Self:
        """Set the measurement count used by raw JSON record references."""
        if self._inference is not _UNSET:
            msg = "with_num_measurements() cannot be combined with with_inference()"
            raise ValueError(msg)
        if self._detectors_kind == "typed" or self._observables_kind == "typed":
            msg = "with_num_measurements() cannot be combined with typed with_detectors() or with_observables()"
            raise ValueError(msg)
        self._set_once("_num_measurements", count, "with_num_measurements")
        return self

    def with_noise(self, noise_model: NoiseParameters) -> Self:
        """Set the complete grouped noise configuration."""
        from pecos.qec.surface.decode import NoiseParameters

        if not isinstance(noise_model, NoiseParameters):
            msg = f"noise() requires a NoiseParameters instance, got {type(noise_model).__name__}"
            raise TypeError(msg)
        self._set_once("_noise", noise_model, "with_noise")
        return self

    def _legacy_noise(self, noise_model: NoiseParameters | None, flat_keywords: Mapping[str, Any]) -> Self:
        """Carry the legacy entry points' flat noise keywords through the builder.

        Private: the flat keyword surface stays on ``from_guppy`` and
        ``build_dem_from_guppy``; ``noise()`` accepts only a ``NoiseParameters``.
        """
        self._set_once("_noise", (_LEGACY_NOISE, noise_model, dict(flat_keywords)), "with_noise")
        return self

    def with_idle_after_2q(self, duration: float | None) -> Self:
        """Set the idle duration inserted after every two-qubit gate."""
        self._set_once("_idle_after_2q", duration, "with_idle_after_2q")
        return self

    def with_strip_traced_idles(self, flag: bool | None) -> Self:
        """Choose whether runtime-emitted ``Idle`` gates are stripped."""
        self._set_once("_strip_traced_idles", flag, "with_strip_traced_idles")
        return self

    def with_runtime(self, runtime: object | None) -> Self:
        """Set the Selene runtime that lowers the program into the traced QIS stream.

        The runtime produces the trace rather than annotating it: the Guppy
        program is lowered and unrolled through it, and the ``TickCircuit`` this
        DEM is built from comes out the far side. A runtime that models timing
        therefore emits its own ``Idle`` gates, which is why
        :meth:`with_idle_after_2q` strips traced idles first by default rather
        than stacking a second convention on top.

        Accepts four forms:

        - ``None`` selects the default runtime, preferring a freshly built
          artifact and falling back to the installed plugin package.
        - A name without path separators is treated as a built runtime library.
        - A path-like value is loaded as a shared library.
        - A runtime plugin object is duck-typed. It must expose
          ``library_file``; ``get_init_args()`` and ``library_search_dirs`` are
          used when present and default to empty otherwise. An object without
          ``library_file`` raises :class:`TypeError` at configuration time.

        See ``pecos._engine_builders._configure_selene_runtime`` for the
        dispatch.
        """
        self._set_once("_runtime", runtime, "with_runtime")
        return self

    def with_seed(self, seed: int) -> Self:
        """Set the ideal trace seed."""
        self._set_once("_seed", seed, "with_seed")
        return self

    def with_residual_warning_threshold(self, fraction: float) -> Self:
        """Accept channel-conversion residuals up to a relative physics tolerance.

        ``fraction`` is a fraction of each requested channel's total error
        weight, not an absolute probability. At or below this tolerance the
        channel remains an accepted inexact conversion, with the exact figures
        retained in ``dem.idle_noise_residuals`` and the build audit. The default
        is ``0.0``, so every nonzero residual warns.

        Use :func:`warnings.filterwarnings` when the intent is to silence a
        warning category wholesale; this setter records a physics tolerance.
        """
        try:
            finite = math.isfinite(fraction)
        except (TypeError, ValueError):
            finite = False
        if not finite or not isinstance(fraction, (int, float)) or fraction < 0.0:
            msg = (
                "with_residual_warning_threshold() requires a finite fraction of "
                "the channel's total error weight in [0.0, 1.0]; "
                f"got {fraction!r}"
            )
            raise ValueError(msg)
        if fraction > 1.0:
            msg = (
                "with_residual_warning_threshold() is a fraction of the channel's "
                "total error weight in [0.0, 1.0], not an absolute probability; "
                f"got {fraction!r}"
            )
            raise ValueError(msg)
        self._set_once("_residual_warning_threshold", float(fraction), "with_residual_warning_threshold")
        return self

    def with_require_hosted_operation_order(self, flag: bool) -> Self:
        """Choose whether hosted-operation ordering is validated."""
        self._set_once("_require_hosted_operation_order", flag, "with_require_hosted_operation_order")
        return self

    def with_max_hosted_tick_separation(self, count: int | None) -> Self:
        """Set the maximum hosted-operation tick separation."""
        self._set_once("_max_hosted_tick_separation", count, "with_max_hosted_tick_separation")
        return self

    def _noise_parameters(self) -> dict[str, Any]:
        if isinstance(self._noise, tuple) and len(self._noise) == 3 and self._noise[0] is _LEGACY_NOISE:
            _, noise, call_arguments = self._noise
            return _resolve_guppy_noise(noise, call_arguments)
        noise = None if self._noise is _UNSET else self._noise
        return _resolve_guppy_noise(noise, _builder_noise_defaults())

    def _required(self, attribute: str, setter: str) -> Any:
        value = getattr(self, attribute)
        if value is _UNSET:
            msg = f"build() requires {setter}()"
            raise ValueError(msg)
        return value

    def build(self) -> GuppyDemBuild:
        """Prepare explicit or inferred annotations and return an audited DEM build."""
        from pecos.qec.dem_spec import Detector, Observable
        from pecos.quantum.hosted import _hosted_tick_separation_limit

        program = self._required("_program", "with_program")
        num_qubits = self._required("_qubits", "with_qubits")
        inferred_observables = self._observables_kind == "typed" and any(
            isinstance(spec, Observable) and spec.inferred_output is not None for spec in self._observables_value
        )
        if self._detectors_kind is _UNSET and self._inference is _UNSET and not inferred_observables:
            msg = "build() requires with_detectors() or with_detectors_json() or with_inference()"
            raise ValueError(msg)

        noise = self._noise_parameters()
        structured = {
            name: noise.pop(name)
            for name in (
                "p_idle_linear",
                "p_idle_linear_model",
                "p_idle_sin_squared",
                "p_idle_sin_squared_model",
                "p_idle_coherent",
                "p_idle_coherent_model",
            )
        }
        translated_names = (
            "p_idle_x_linear_rate",
            "p_idle_y_linear_rate",
            "p_idle_z_linear_rate",
            "p_idle_x_quadratic_sine_rate",
            "p_idle_y_quadratic_sine_rate",
            "p_idle_z_quadratic_sine_rate",
        )
        axes = _translate_structured_idle_noise(
            **structured,
            **{
                name: noise[name]
                for name in (
                    "p_idle_linear_rate",
                    "p_idle_quadratic_rate",
                    "p_idle_x_linear_rate",
                    "p_idle_y_linear_rate",
                    "p_idle_z_linear_rate",
                    "p_idle_quadratic_sine_rate",
                    "p_idle_x_quadratic_sine_rate",
                    "p_idle_y_quadratic_sine_rate",
                    "p_idle_z_quadratic_sine_rate",
                )
            },
        )
        noise.update(zip(translated_names, axes, strict=True))
        definitions = {}
        specs = []
        for role, expected in (("detectors", Detector), ("observables", Observable)):
            kind, values = getattr(self, f"_{role}_kind"), getattr(self, f"_{role}_value")
            if kind == "typed":
                definitions[role] = [_definition_record(spec, expected, role) for spec in values]
                specs.extend(values)
            elif kind == "json":
                definitions[f"{role}_json"] = values
        settings = None if self._inference is _UNSET else self._inference
        qis, evidence = _guppy_dem_evidence(program, specs, settings, definitions)
        runtime = _configure_selene_runtime(_SeleneRuntimeSpec(), self._value("_runtime"))
        prepared = _prepare_guppy_dem(
            qis,
            num_qubits,
            self._value("_seed", 0),
            runtime,
            settings=settings,
            num_measurements=self._value("_num_measurements"),
            require_hosted_operation_order=self._value("_require_hosted_operation_order", default=False),
            max_hosted_tick_separation=_hosted_tick_separation_limit(
                self._value("_max_hosted_tick_separation"),
                context="GuppyDemBuilder.build",
            ),
            idle_after_2q=self._value("_idle_after_2q"),
            strip_traced_idles=self._value("_strip_traced_idles"),
            **definitions,
            **evidence,
        )
        built = prepared.build_dem(**noise)
        _warn_on_noise_channel_residuals(built, self._value("_residual_warning_threshold", 0.0))
        inference = None if prepared.inference is None else InferredGuppyDemAnnotations(prepared)
        return GuppyDemBuild(built, inference)

    def _value(self, attribute: str, default: Any = None) -> Any:
        value = getattr(self, attribute)
        return default if value is _UNSET else value


def _warn_on_noise_channel_residuals(dem: Any, relative_threshold: float = 0.0) -> None:
    """Warn about channel residuals above the accepted relative tolerance."""
    summary = dem.residual_summary(relative_threshold)
    if not summary:
        return
    count = sum(group["count"] for group in summary)
    kinds = ", ".join(
        f"{group['count']} {group['channel_kind']} (largest relative {group['max_relative_magnitude']:.3e}; "
        f"largest TV {group['max_tv_magnitude']:.3e})"
        for group in summary
    )
    warnings.warn(
        f"{count} categorical noise channel(s) were approximated: {kinds}. "
        "A non-negative boundary fit was emitted; relative magnitudes are fractions "
        "of each requested channel's total error weight, and TV magnitudes are "
        "total-variation distances. See dem.idle_noise_residuals for details.",
        UserWarning,
        stacklevel=3,
    )


def build_dem_from_guppy(
    guppy: Any,
    *,
    num_qubits: int,
    detectors: Sequence[Detector],
    observables: Sequence[Observable] = (),
    noise: NoiseParameters | None = None,
    p1: float = _NOISE_DEFAULT_P1,
    p1_weights: P1Weights | None = _NOISE_DEFAULT_NONE,
    p2: float = _NOISE_DEFAULT_P2,
    p2_weights: P2Weights | None = _NOISE_DEFAULT_NONE,
    p2_replacement_approximation: str | None = _NOISE_DEFAULT_NONE,
    p_meas: float = _NOISE_DEFAULT_P1,
    p_prep: float = _NOISE_DEFAULT_P1,
    p_idle_linear: float | None = _NOISE_DEFAULT_NONE,
    p_idle_linear_model: Mapping[str, float] | None = _NOISE_DEFAULT_NONE,
    p_idle_sin_squared: float | None = _NOISE_DEFAULT_NONE,
    p_idle_sin_squared_model: Mapping[str, float] | None = _NOISE_DEFAULT_NONE,
    p_idle_coherent: float | None = _NOISE_DEFAULT_NONE,
    p_idle_coherent_model: Mapping[str, float] | None = _NOISE_DEFAULT_NONE,
    t1: float | None = _NOISE_DEFAULT_NONE,
    t2: float | None = _NOISE_DEFAULT_NONE,
    p_idle_linear_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_quadratic_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_x_linear_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_y_linear_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_z_linear_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_x_quadratic_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_y_quadratic_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_z_quadratic_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_quadratic_sine_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_x_quadratic_sine_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_y_quadratic_sine_rate: float | None = _NOISE_DEFAULT_NONE,
    p_idle_z_quadratic_sine_rate: float | None = _NOISE_DEFAULT_NONE,
    strip_traced_idles: bool | None = None,
    idle_after_2q_duration: float | None = None,
    runtime: object | None = None,
    seed: int = 0,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> GuppyDemBuild:
    """Prepare and build an audited DEM from typed definitions in Rust.

    ``rec[-k]`` references the canonical dense Guppy result-id stream, not the
    runtime's scheduled measurement-gate order. ``result_ref(...)`` resolves
    through the executed lowered program's dataflow and is checked against the traced
    measurement count. Both forms lower to stable ``MeasId`` metadata before
    DEM construction.

    Measurement-dependent quantum control remains unsupported because one
    captured execution is not a static circuit model. Explicit and named mixed
    builds require static schedule evidence. Pure inference and positional-only
    mixed builds check schedules empirically across probes. Rust computes exact
    noiseless references, rejects random parities or mismatched declarations,
    and stores ``reference`` in returned metadata. Evaluators report events and
    flips relative to those references; the ``*_parities`` methods return raw
    parities.

    The three structured idle models contain relative-rate multipliers: each
    axis rate is ``family_rate * axis_multiplier``. The linear law is additive,
    so its multipliers must sum to 1.0 and coincide with the engines
    relative-probability distribution. The nonlinear sine-squared and coherent
    laws are not additive, so their finite, non-negative multipliers have no
    sum constraint.

    Args:
        guppy: A Guppy program executed under the Selene QIS engine. Explicit
            builds require static schedule evidence; inference uses probes.
        num_qubits: Number of qubits to allocate for the trace.
        detectors: Typed detector definitions using ``rec[...]`` or
            ``result_ref(...)`` measurement references.
        observables: Typed logical-observable definitions using the same
            measurement-reference forms as ``detectors``.
        noise: Complete grouped noise configuration. When supplied, its values
            replace all flat noise keywords, including this entry point's
            defaults. In particular, ``NoiseParameters`` defaults such as
            ``p1=0.0`` apply instead of this function's ``p1=0.001``. Mixing
            ``noise`` with any flat noise keyword is rejected.
        p1: Single-qubit gate Pauli error rate. The categorical Pauli channel
            is converted after equal propagated signatures merge.
        p1_weights: Optional relative probabilities over single-qubit Pauli
            error labels ``"X"``, ``"Y"``, and ``"Z"``.
        p2: Two-qubit gate depolarizing rate. Its 15 categorical branches are
            converted together after propagation.
        p2_weights: Optional relative probabilities over two-qubit Pauli error
            labels, including replacement branches written as ``"~XX"`` or
            ``":replace:XX"``.
        p2_replacement_approximation: Approximation used for
            replacement labels in ``p2_weights``.
        p_meas: Measurement flip rate.
        p_prep: Preparation (reset) error rate.
        p_idle_linear: Optional total stochastic idle-noise rate linear in
            duration. Uses the engines ``GeneralNoiseModel`` categorical-Pauli
            convention. The DEM groups non-empty propagated flip signatures
            before converting distinct signatures to independent mechanisms.
            If exact conversion would require a negative mechanism, the build
            uses a non-negative boundary fit and reports its quantified
            both-fire residual through ``dem.idle_noise_residuals`` and the
            audited build's ``audit["idle_noise_residuals"]`` entry.
        p_idle_linear_model: Optional relative weights over ``"X"``, ``"Y"``,
            ``"Z"``, and ``"L"`` for ``p_idle_linear``. Weights must be finite,
            non-negative, and sum to 1.0 within ``1e-5``, including any
            explicit ``"L"`` weight. Defaults to the engines' uniform Pauli
            model. DEM fault propagation is Pauli-only, so ``"L"`` must be zero
            here; the engines simulators support nonzero leakage weights.
        p_idle_sin_squared: Optional stochastic sine-law idle rate. An
            axis multiplier ``m`` gives probability
            ``sin((p_idle_sin_squared * m) * t)^2``. By default X, Y, and Z
            each use the full family rate. These axis mechanisms remain
            separate from the linear family.
        p_idle_sin_squared_model: Optional finite, non-negative relative-rate
            multipliers over ``"X"``, ``"Y"``, ``"Z"``, and ``"L"``. There is
            no sum constraint; the default is
            ``{"X": 1.0, "Y": 1.0, "Z": 1.0}``. DEM fault propagation is
            Pauli-only, so ``"L"`` must be zero here; the engines simulators
            support nonzero leakage weights.
        p_idle_coherent: Optional coherent-rotation rate. The standard DEM
            builder cannot represent coherent idle noise and rejects every
            nonzero rate rather than silently storing a Pauli twirl that
            discards the requested coherence. Use the EEG coherent route in
            ``exp/pecos-eeg`` for coherent idle noise; it supports only an RZ
            generator. The honest stochastic equivalent—the exact Pauli twirl
            of ``RZ(rate * t)``—is ``p_idle_sin_squared=rate/2`` with
            ``p_idle_sin_squared_model={"Z": 1.0}``. Zero has no effect.
        p_idle_coherent_model: Optional finite, non-negative relative-rate
            multipliers over ``"RX"``, ``"RY"``, and ``"RZ"``, with no sum
            constraint. Defaults to ``{"RX": 1.0, "RY": 1.0, "RZ": 1.0}``.
            The keys are validation-only on this route because any nonzero
            coherent family rate is rejected. ``"L"`` is not a coherent-model
            key; ``"U"`` is reserved for future Hamiltonian-level support and
            rejected.
        t1: Optional T1 relaxation time for explicit idle gates; requires ``t2``.
        t2: Optional T2 dephasing time for explicit idle gates; requires ``t1``.
        p_idle_linear_rate: Deprecated bare Z-only alias for a stochastic rate
            linear in idle duration. Use ``p_idle_linear`` with a Z-only model,
            or ``p_idle_z_linear_rate`` for literal behavior.
        p_idle_quadratic_rate: Deprecated bare Z-only coefficient-style rate
            quadratic in idle duration. Use ``p_idle_sin_squared`` for the
            structured engines-style dephasing interface, or
            ``p_idle_z_quadratic_rate`` for literal behavior.
        p_idle_x_linear_rate: Optional stochastic X-memory rate linear in idle duration.
        p_idle_y_linear_rate: Optional stochastic Y-memory rate linear in idle duration.
        p_idle_z_linear_rate: Optional stochastic Z-memory rate linear in idle duration.
        p_idle_x_quadratic_rate: Optional stochastic X-memory rate quadratic in idle duration.
        p_idle_y_quadratic_rate: Optional stochastic Y-memory rate quadratic in idle duration.
        p_idle_z_quadratic_rate: Optional stochastic Z-memory rate quadratic in idle duration.
        p_idle_quadratic_sine_rate: Deprecated bare Z-only alias for a
            stochastic rate with probability ``sin(rate * duration)^2``. Use
            ``p_idle_sin_squared`` or ``p_idle_z_quadratic_sine_rate``.
        p_idle_x_quadratic_sine_rate: Optional stochastic X-memory sine-law rate.
        p_idle_y_quadratic_sine_rate: Optional stochastic Y-memory sine-law rate.
        p_idle_z_quadratic_sine_rate: Optional stochastic Z-memory sine-law rate.
        strip_traced_idles: If true, remove ``Idle`` gates from the normalized
            trace while preserving ``I`` and zero-angle rotations as
            gate-noise locations. This pass runs before idle insertion when
            both idle-pass options are set. Defaults to ``None``, which strips
            exactly when ``idle_after_2q_duration`` is set; pass ``False``
            explicitly to keep runtime-emitted idles alongside inserted
            ones.
        idle_after_2q_duration: If set, insert an ``Idle`` gate of this
            duration on both qubits after every two-qubit gate. Insertion runs
            after ``strip_traced_idles`` within Rust preparation, preserving
            the resolved measurement identities and definitions.
        runtime: Optional Selene runtime selector/plugin. ``None`` selects the
            default Selene runtime.
        seed: Seed for the ideal trace run.
        require_hosted_operation_order: If true, validate generic
            hosted-operation metadata after trace replay.
        max_hosted_tick_separation: Optional maximum absolute signed tick
            separation accepted by the hosted-operation validator.

    Raises:
        ValueError: If ``idle_after_2q_duration`` is not a finite positive
            number, if ``p_idle_coherent`` is nonzero, or if any representable
            idle-noise rate is nonzero but the final traced circuit has no
            ``Idle`` gates. Pass ``idle_after_2q_duration`` or use a Selene
            runtime that emits scheduled idles to provide targets for idle
            noise. Explicit zero rates require no idle targets; ``t1`` and
            ``t2`` must be supplied together.
    """
    noise_keywords = {name: value for name, value in locals().items() if name in _GUPPY_NOISE_KEYWORDS}
    builder = (
        GuppyDemBuilder()
        .with_program(guppy)
        .with_qubits(num_qubits)
        .with_detectors(detectors)
        .with_observables(observables)
        .with_strip_traced_idles(strip_traced_idles)
        .with_idle_after_2q(idle_after_2q_duration)
        .with_runtime(runtime)
        .with_seed(seed)
        .with_require_hosted_operation_order(require_hosted_operation_order)
        .with_max_hosted_tick_separation(max_hosted_tick_separation)
    )
    # Same-module private seam: see the note in DetectorErrorModel.from_guppy.
    return builder._legacy_noise(noise, noise_keywords).build()  # noqa: SLF001


DetectorErrorModel = _RustDetectorErrorModel
DetectorErrorModel.builder = classmethod(_DetectorErrorModelMixin.__dict__["builder"].__func__)
DetectorErrorModel.from_guppy = classmethod(_DetectorErrorModelMixin.__dict__["from_guppy"].__func__)
