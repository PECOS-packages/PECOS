"""Python-level ``DetectorErrorModel`` with a Guppy convenience constructor.

The core ``DetectorErrorModel`` is implemented in Rust
(``pecos_rslib.qec.DetectorErrorModel``). The Guppy -> Selene -> QIS-trace
pipeline, however, lives entirely in Python (``pecos.sim``, ``pecos.guppy``,
``pecos.qec.surface.decode``). To keep the convenient
``DetectorErrorModel.from_guppy(...)`` call site without making the low-level
Rust extension import the high-level Python package (a dependency cycle), this
module attaches a Python :meth:`from_guppy` classmethod to the Rust-backed
``pecos_rslib.qec.DetectorErrorModel`` and re-exports that class as the public
``pecos.qec.DetectorErrorModel``.

This wrapper is intentionally thin: it traces the Guppy program into a
``TickCircuit``, compiles Guppy inputs to a HUGR to reject unverified control
flow and, when requested, recover the sound tag -> measurement binding via
``pecos_hugr_qis::extract_result_tag_measurements``, and hands
the caller's detector/observable JSON to the Rust DEM builder. The metadata
validation that applies to **every** ingest path (``from_guppy``,
``from_circuit``, ``DemSampler.from_circuit``, public ``DemBuilder``) lives
solely in the Rust DEM builder
(``pecos_qec::fault_tolerance::dem_builder``): JSON shape, ``D0``/``L0`` id
forms, tracked-Pauli rejection, ``num_measurements`` consistency,
out-of-range records, ``meas_id`` resolution against the circuit's stable
stamped ``MeasId``s, and the ``records``-vs-``meas_ids`` redundancy rule.

The ``result_tags`` -> record-offset resolution (loop guard included) is
applied **only** through ``from_guppy``: the rewriter
(``pecos_qec::resolve_result_tags``, invoked via the pyo3
``resolve_result_tags_for_guppy`` binding) runs from this wrapper before
``from_circuit`` is called, so the downstream DEM builder only ever sees
already-resolved ``records``. ``result_tags`` in circuit metadata fed
directly to ``from_circuit`` / ``DemSampler.from_circuit`` /
``DemBuilder.build`` is **not** resolved -- those paths build from
``records``/``meas_ids`` as usual.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pecos_rslib.qec import DetectorErrorModel as _RustDetectorErrorModel

from pecos._traced_circuit import (
    measurement_ids_in_execution_order,
    normalize_traced_tick_circuit,
)
from pecos.qec.dem_spec import GuppyDemBuild, ResultRef, _resolve_dem_specs

if TYPE_CHECKING:
    from pecos.qec.dem_spec import Detector, Observable

P1Weights = Mapping[str, float]
P2Weights = Mapping[str, float]

_GENERATOR_LAYOUT_ATTR = "__pecos_named_measurement_layout_v2__"
_IDLE_MODEL_NORMALIZATION_TOLERANCE = 1.0e-5
_IDLE_MODEL_FLOAT_EPSILON = 1.0e-10


def _translate_structured_idle_noise(
    *,
    p_idle: float | None,
    p_idle_linear: float | None,
    p_idle_linear_model: Mapping[str, float] | None,
    p_idle_quadratic: float | None,
    p_idle_coherent: bool,
    p_idle_linear_rate: float | None,
    p_idle_quadratic_rate: float | None,
    p_idle_x_linear_rate: float | None,
    p_idle_y_linear_rate: float | None,
    p_idle_z_linear_rate: float | None,
    p_idle_x_quadratic_rate: float | None,
    p_idle_y_quadratic_rate: float | None,
    p_idle_z_quadratic_rate: float | None,
    p_idle_quadratic_sine_rate: float | None,
    p_idle_x_quadratic_sine_rate: float | None,
    p_idle_y_quadratic_sine_rate: float | None,
    p_idle_z_quadratic_sine_rate: float | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Validate and translate engines-style idle noise to DEM primitives."""
    linear_primitives = {
        "p_idle_linear_rate": p_idle_linear_rate,
        "p_idle_x_linear_rate": p_idle_x_linear_rate,
        "p_idle_y_linear_rate": p_idle_y_linear_rate,
        "p_idle_z_linear_rate": p_idle_z_linear_rate,
    }
    if (p_idle_linear is not None or p_idle_linear_model is not None) and any(
        value is not None for value in linear_primitives.values()
    ):
        conflicts = ", ".join(name for name, value in linear_primitives.items() if value is not None)
        msg = f"p_idle_linear/p_idle_linear_model cannot be combined with low-level idle rate(s): {conflicts}"
        raise ValueError(msg)
    if p_idle is not None and p_idle_linear is not None:
        msg = "p_idle and p_idle_linear cannot be combined; p_idle is the uniform-model shorthand"
        raise ValueError(msg)

    quadratic_primitives = {
        "p_idle_quadratic_rate": p_idle_quadratic_rate,
        "p_idle_x_quadratic_rate": p_idle_x_quadratic_rate,
        "p_idle_y_quadratic_rate": p_idle_y_quadratic_rate,
        "p_idle_z_quadratic_rate": p_idle_z_quadratic_rate,
        "p_idle_quadratic_sine_rate": p_idle_quadratic_sine_rate,
        "p_idle_x_quadratic_sine_rate": p_idle_x_quadratic_sine_rate,
        "p_idle_y_quadratic_sine_rate": p_idle_y_quadratic_sine_rate,
        "p_idle_z_quadratic_sine_rate": p_idle_z_quadratic_sine_rate,
    }
    if (p_idle_quadratic is not None or p_idle_coherent) and any(
        value is not None for value in quadratic_primitives.values()
    ):
        conflicts = ", ".join(name for name, value in quadratic_primitives.items() if value is not None)
        msg = f"p_idle_quadratic/p_idle_coherent cannot be combined with low-level idle rate(s): {conflicts}"
        raise ValueError(msg)

    if p_idle_linear_model is not None and p_idle_linear is None:
        msg = "p_idle_linear_model requires p_idle_linear; otherwise the model is inert"
        raise ValueError(msg)
    if p_idle_coherent and p_idle_quadratic is None:
        msg = "p_idle_coherent=True requires p_idle_quadratic; otherwise it is inert"
        raise ValueError(msg)

    legacy_replacements = {
        "p_idle_linear_rate": (
            p_idle_linear_rate,
            (
                "p_idle_linear with p_idle_linear_model={'Z': 1.0} for the engines-consistent interface, "
                "or p_idle_z_linear_rate for literal Z-only behavior"
            ),
        ),
        "p_idle_quadratic_rate": (
            p_idle_quadratic_rate,
            (
                "p_idle_quadratic for the engines-consistent quadratic interface, "
                "or p_idle_z_quadratic_rate for literal coefficient-style Z-only behavior"
            ),
        ),
        "p_idle_quadratic_sine_rate": (
            p_idle_quadratic_sine_rate,
            (
                "p_idle_quadratic for the engines-consistent quadratic interface, "
                "or p_idle_z_quadratic_sine_rate for literal Z-only behavior"
            ),
        ),
    }
    for name, (value, replacement) in legacy_replacements.items():
        if value is not None:
            warnings.warn(
                f"{name} is deprecated; use {replacement}",
                DeprecationWarning,
                stacklevel=3,
            )

    if p_idle_linear is not None:
        if p_idle_linear_model is not None and not isinstance(p_idle_linear_model, Mapping):
            msg = "p_idle_linear_model must be a mapping from 'X', 'Y', and 'Z' to weights"
            raise ValueError(msg)
        model = (
            p_idle_linear_model if p_idle_linear_model is not None else {"X": 1.0 / 3.0, "Y": 1.0 / 3.0, "Z": 1.0 / 3.0}
        )
        normalized_model: dict[str, float] = {}
        for key, weight in model.items():
            if key == "L":
                msg = (
                    "p_idle_linear_model key 'L' denotes leakage, which is supported by the engines simulators "
                    "but not by DEM construction"
                )
                raise ValueError(msg)
            if key not in {"X", "Y", "Z"}:
                msg = f"invalid p_idle_linear_model key {key!r}; expected 'X', 'Y', or 'Z'"
                raise ValueError(msg)
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError) as exc:
                msg = f"p_idle_linear_model weight for {key!r} must be a finite, non-negative float"
                raise ValueError(msg) from exc
            if not math.isfinite(numeric_weight) or numeric_weight < 0.0:
                msg = f"p_idle_linear_model weight for {key!r} must be a finite, non-negative float"
                raise ValueError(msg)
            normalized_model[key] = numeric_weight

        total_weight = sum(normalized_model.values())
        if total_weight <= 0.0 or abs(total_weight - 1.0) > _IDLE_MODEL_NORMALIZATION_TOLERANCE:
            msg = (
                "p_idle_linear_model weights must sum to 1.0 within tolerance "
                f"{_IDLE_MODEL_NORMALIZATION_TOLERANCE:g}; got {total_weight}"
            )
            raise ValueError(msg)
        if abs(total_weight - 1.0) > _IDLE_MODEL_FLOAT_EPSILON:
            normalized_model = {key: weight / total_weight for key, weight in normalized_model.items()}

        p_idle_x_linear_rate = p_idle_linear * normalized_model.get("X", 0.0)
        p_idle_y_linear_rate = p_idle_linear * normalized_model.get("Y", 0.0)
        p_idle_z_linear_rate = p_idle_linear * normalized_model.get("Z", 0.0)

    idle_rz = None
    if p_idle_quadratic is not None:
        if p_idle_coherent:
            idle_rz = p_idle_quadratic
        else:
            p_idle_z_quadratic_sine_rate = p_idle_quadratic

    return (
        p_idle_x_linear_rate,
        p_idle_y_linear_rate,
        p_idle_z_linear_rate,
        p_idle_z_quadratic_sine_rate,
        idle_rz,
    )


def _certifiable_hugr_bytes(guppy: Any) -> bytes | None:
    """Return the HUGR bytes that certify this program's static schedule.

    Accepts ``@guppy`` definitions (compiled through the shared cache),
    ``pecos.Guppy`` wrappers (unwrapped and compiled), and ``pecos.Hugr``
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
    objects only. Byte carriers (``pecos.Hugr``, raw HUGR bytes, duck-typed
    ``hugr_bytes`` holders) never carry an honorable certificate: they are
    opaque data, and honoring an attribute there would let any bytes suppress
    the control-flow guard by stapling a self-consistent digest to themselves.
    """
    if isinstance(guppy, (bytes, bytearray)):
        return None
    if isinstance(getattr(guppy, "hugr_bytes", None), (bytes, bytearray)):
        return None
    return getattr(guppy, "wrapped_function", guppy)


def _generator_certified_layout(guppy: Any, hugr_bytes: bytes | None = None) -> Sequence[Any] | None:
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
    layout_json = json.dumps(layout, separators=(",", ":"))
    expected = hashlib.sha256(hugr_bytes + b"\0" + layout_json.encode()).hexdigest()
    if digest != expected:
        msg = "Guppy generator measurement-layout certificate does not match the program and layout"
        raise ValueError(msg)
    return layout


def _from_circuit_with_noise(
    tc: Any,
    *,
    p1: float,
    p1_weights: P1Weights | None,
    p2: float,
    p2_weights: P2Weights | None,
    p2_replacement_approximation: str | None,
    p_meas: float,
    p_prep: float,
    p_idle: float | None,
    t1: float | None,
    t2: float | None,
    idle_rz: float | None,
    p_idle_linear_rate: float | None,
    p_idle_quadratic_rate: float | None,
    p_idle_x_linear_rate: float | None,
    p_idle_y_linear_rate: float | None,
    p_idle_z_linear_rate: float | None,
    p_idle_x_quadratic_rate: float | None,
    p_idle_y_quadratic_rate: float | None,
    p_idle_z_quadratic_rate: float | None,
    p_idle_quadratic_sine_rate: float | None,
    p_idle_x_quadratic_sine_rate: float | None,
    p_idle_y_quadratic_sine_rate: float | None,
    p_idle_z_quadratic_sine_rate: float | None,
) -> _RustDetectorErrorModel:
    return _RustDetectorErrorModel.from_circuit(
        tc,
        p1=p1,
        p1_weights=p1_weights,
        p2=p2,
        p2_weights=p2_weights,
        p2_replacement_approximation=p2_replacement_approximation,
        p_meas=p_meas,
        p_prep=p_prep,
        p_idle=p_idle,
        t1=t1,
        t2=t2,
        idle_rz=idle_rz,
        p_idle_linear_rate=p_idle_linear_rate,
        p_idle_quadratic_rate=p_idle_quadratic_rate,
        p_idle_x_linear_rate=p_idle_x_linear_rate,
        p_idle_y_linear_rate=p_idle_y_linear_rate,
        p_idle_z_linear_rate=p_idle_z_linear_rate,
        p_idle_x_quadratic_rate=p_idle_x_quadratic_rate,
        p_idle_y_quadratic_rate=p_idle_y_quadratic_rate,
        p_idle_z_quadratic_rate=p_idle_z_quadratic_rate,
        p_idle_quadratic_sine_rate=p_idle_quadratic_sine_rate,
        p_idle_x_quadratic_sine_rate=p_idle_x_quadratic_sine_rate,
        p_idle_y_quadratic_sine_rate=p_idle_y_quadratic_sine_rate,
        p_idle_z_quadratic_sine_rate=p_idle_z_quadratic_sine_rate,
    )


def _apply_traced_idle_passes(
    circuit: Any,
    *,
    strip_traced_idles: bool | None,
    idle_after_2q_duration: float | None,
    idle_noise_parameters: Sequence[float | None],
) -> None:
    """Apply requested idle passes and reject idle noise with no target gates."""
    if strip_traced_idles is None:
        # Inserting a uniform idle convention on top of runtime-emitted idles
        # would double-count idle noise, so insertion implies stripping first.
        strip_traced_idles = idle_after_2q_duration is not None
    if strip_traced_idles:
        circuit.remove_identity()
    if idle_after_2q_duration is not None:
        if not math.isfinite(idle_after_2q_duration) or idle_after_2q_duration <= 0.0:
            msg = (
                "idle_after_2q_duration must be a finite, positive duration; "
                f"got {idle_after_2q_duration!r} (a non-positive duration would insert idle "
                "gates that contribute zero idle noise)"
            )
            raise ValueError(msg)
        circuit.insert_idle_after_two_qubit_gates(idle_after_2q_duration)

    if any(value is not None for value in idle_noise_parameters) and circuit.gate_counts_by_type().get("Idle", 0) == 0:
        msg = (
            "idle-noise parameters have no idle gates to attach to; either pass "
            "idle_after_2q_duration=..., or use a Selene runtime that emits scheduled idles"
        )
        raise ValueError(msg)


class _DetectorErrorModelMixin:
    """Namespace for the Python Guppy/QIS-trace convenience constructor."""

    __slots__ = ()

    @classmethod
    def from_guppy(
        cls,
        guppy: Any,
        *,
        num_qubits: int,
        detectors_json: str,
        observables_json: str = "[]",
        num_measurements: int | None = None,
        p1: float = 0.001,
        p1_weights: P1Weights | None = None,
        p2: float = 0.01,
        p2_weights: P2Weights | None = None,
        p2_replacement_approximation: str | None = None,
        p_meas: float = 0.001,
        p_prep: float = 0.001,
        p_idle: float | None = None,
        p_idle_linear: float | None = None,
        p_idle_linear_model: Mapping[str, float] | None = None,
        p_idle_quadratic: float | None = None,
        p_idle_coherent: bool = False,
        t1: float | None = None,
        t2: float | None = None,
        p_idle_linear_rate: float | None = None,
        p_idle_quadratic_rate: float | None = None,
        p_idle_x_linear_rate: float | None = None,
        p_idle_y_linear_rate: float | None = None,
        p_idle_z_linear_rate: float | None = None,
        p_idle_x_quadratic_rate: float | None = None,
        p_idle_y_quadratic_rate: float | None = None,
        p_idle_z_quadratic_rate: float | None = None,
        p_idle_quadratic_sine_rate: float | None = None,
        p_idle_x_quadratic_sine_rate: float | None = None,
        p_idle_y_quadratic_sine_rate: float | None = None,
        p_idle_z_quadratic_sine_rate: float | None = None,
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
        Rust DEM builder (single source of truth).

        Args:
            guppy: A HUGR-certifiable program: a ``@guppy``-decorated function,
                a compiled Guppy program (e.g. the object returned by
                ``pecos.guppy.make_surface_code``), a ``pecos.Guppy`` or
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
                  from the compiled HUGR by
                  ``pecos_hugr_qis::extract_result_tag_measurements`` and
                  resolved to stable runtime ``MeasId`` values in Rust. Supported only for
                  **straight-line, canonical** programs:
                  ``result(tag, measure(q))`` of a raw scalar measurement.
                  Computed (``result(tag, m0 == m1)``), constant
                  (``result(tag, True)``), and array-valued
                  (``result(tag, measure_array(qs))``) forms are not
                  resolvable and an unknown tag is a hard error. Runtime
                  ``for _ in range(comptime(n))`` loops (e.g. the surface
                  code's round structure) have one static measure op per
                  loop body in the HUGR, not per occurrence -- ``result_tags``
                  is rejected fail-loud for such programs. ``result_tags``
                  also requires ``guppy`` to be a ``@guppy``-decorated
                  function / ``GuppyFunctionDefinition`` (not an arbitrary
                  ``pecos.sim``-acceptable wrapper); use ``records`` for the
                  surface-code path.
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
            p1: Single-qubit gate Pauli error rate.
            p1_weights: Optional relative probabilities over single-qubit
                Pauli error labels ``"X"``, ``"Y"``, and ``"Z"``. Values must
                sum to 1.0; ``p1`` remains the total single-qubit error rate.
            p2: Two-qubit gate depolarizing rate.
            p2_weights: Optional relative probabilities over two-qubit Pauli
                error labels. Plain labels such as ``"XX"`` are post-gate
                Pauli branches; labels prefixed by ``"*"`` such as ``"*XX"``
                are replacement branches that omit the ideal two-qubit gate
                before applying the Pauli. Values must sum to 1.0; ``p2``
                remains the total two-qubit error rate.
            p2_replacement_approximation: Approximation used for starred
                replacement labels. ``"pauli_twirl_omitted_gate"`` convolves
                with the omitted two-qubit gate's Pauli twirl;
                ``"branch_impact"`` evaluates starred entries as replacement
                branch impacts; ``"exact_branch_replay"`` is reserved for a
                future circuit-aware exact replay provider and currently fails
                loudly for starred entries; ``"ignore_gate_removal"`` treats starred
                entries like plain post-gate Pauli entries.
            p_meas: Measurement flip rate.
            p_prep: Preparation (reset) error rate.
            p_idle: Optional shorthand for ``p_idle_linear`` with the uniform
                ``{"X": 1/3, "Y": 1/3, "Z": 1/3}`` model.
            p_idle_linear: Optional total stochastic idle-noise rate linear in
                duration. Uses the engines ``GeneralNoiseModel`` convention.
            p_idle_linear_model: Optional relative weights over ``"X"``, ``"Y"``,
                and ``"Z"`` for ``p_idle_linear``. Weights must be finite,
                non-negative, and sum to 1.0 within ``1e-5``. Defaults to the
                engines' uniform model. The engines leakage key ``"L"`` is
                reserved but unsupported by DEM construction.
            p_idle_quadratic: Optional quadratic dephasing rate. With
                ``p_idle_coherent=False``, an idle of duration ``t`` produces a
                stochastic Z fault with probability ``sin(rate * t)^2``.
                With ``p_idle_coherent=True``, the same rate is forwarded as a
                coherent ``RZ(rate * t)`` angle; the DEM converts an isolated
                rotation to ``sin(rate * t / 2)^2`` and coherently accumulates
                angles for matching detector sets.
            p_idle_coherent: Select coherent RZ rather than stochastic Z
                interpretation for ``p_idle_quadratic``. Defaults to ``False``.
            t1: Optional T1 relaxation time for explicit idle gates.
            t2: Optional T2 dephasing time for explicit idle gates.
            p_idle_linear_rate: Deprecated bare Z-only alias for a stochastic
                rate linear in idle duration. Use ``p_idle_linear`` with a
                Z-only model, or ``p_idle_z_linear_rate`` for literal behavior.
            p_idle_quadratic_rate: Deprecated bare Z-only coefficient-style
                rate quadratic in idle duration. Use ``p_idle_quadratic`` for
                engines semantics, or ``p_idle_z_quadratic_rate`` for literal
                behavior.
            p_idle_x_linear_rate: Optional stochastic X-memory rate linear in idle duration.
            p_idle_y_linear_rate: Optional stochastic Y-memory rate linear in idle duration.
            p_idle_z_linear_rate: Optional stochastic Z-memory rate linear in idle duration.
            p_idle_x_quadratic_rate: Optional stochastic X-memory rate quadratic in idle duration.
            p_idle_y_quadratic_rate: Optional stochastic Y-memory rate quadratic in idle duration.
            p_idle_z_quadratic_rate: Optional stochastic Z-memory rate quadratic in idle duration.
            p_idle_quadratic_sine_rate: Deprecated bare Z-only alias for a
                stochastic rate with probability ``sin(rate * duration)^2``.
                Use ``p_idle_quadratic`` or ``p_idle_z_quadratic_sine_rate``.
            p_idle_x_quadratic_sine_rate: Optional stochastic X-memory sine-law rate.
            p_idle_y_quadratic_sine_rate: Optional stochastic Y-memory sine-law rate.
            p_idle_z_quadratic_sine_rate: Optional stochastic Z-memory sine-law rate.
            strip_traced_idles: If true, remove identity-like gates from the
                normalized traced circuit, including ``I``, ``Idle``, and
                zero-angle rotations. This pass runs before idle insertion
                when both idle-pass options are set. Defaults to ``None``,
                which strips exactly when ``idle_after_2q_duration`` is set:
                inserting a uniform idle convention on top of runtime-emitted
                idles would double-count idle noise. Pass ``False`` explicitly
                to keep runtime-emitted idles alongside inserted ones.
            idle_after_2q_duration: If set, insert an ``Idle`` gate of this
                duration on both qubits after every two-qubit gate in the
                normalized traced circuit. Insertion runs after
                ``strip_traced_idles`` and before result-tag resolution and
                detector/observable metadata attachment.
            runtime: Optional Selene runtime selector/plugin. ``None`` selects
                the default Selene runtime. Runtime plugin objects are passed
                through to ``pecos.selene_engine(runtime)``.
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
                positive number, if any idle-noise parameter is set but the
                final traced circuit has no ``Idle`` gates, or if the traced
                operation stream cannot be replayed. To provide targets for
                idle noise, pass ``idle_after_2q_duration`` or use a Selene
                runtime that emits scheduled idles.

        Note:
            Runtime-lowered idles are replayed as nanosecond PECOS
            ``TimeUnits``. If idle parameters come from a per-second
            simulator/runtime model, use
            ``noise.for_runtime_idle_time_units()`` and pass the converted
            scalar idle-rate fields to this constructor.

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
        from pecos.tracing import trace_program_to_tick_circuit

        (
            p_idle_x_linear_rate,
            p_idle_y_linear_rate,
            p_idle_z_linear_rate,
            p_idle_z_quadratic_sine_rate,
            idle_rz,
        ) = _translate_structured_idle_noise(
            p_idle=p_idle,
            p_idle_linear=p_idle_linear,
            p_idle_linear_model=p_idle_linear_model,
            p_idle_quadratic=p_idle_quadratic,
            p_idle_coherent=p_idle_coherent,
            p_idle_linear_rate=p_idle_linear_rate,
            p_idle_quadratic_rate=p_idle_quadratic_rate,
            p_idle_x_linear_rate=p_idle_x_linear_rate,
            p_idle_y_linear_rate=p_idle_y_linear_rate,
            p_idle_z_linear_rate=p_idle_z_linear_rate,
            p_idle_x_quadratic_rate=p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate=p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate=p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate=p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate=p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate=p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate=p_idle_z_quadratic_sine_rate,
        )

        # Tag-referenced detectors require the compiled HUGR (to recover the
        # sound, reorder-immune Guppy `result(tag, ...)` -> measurement
        # binding). `guppy_to_hugr` accepts @guppy-decorated functions and
        # `GuppyFunctionDefinition`s (e.g. `make_surface_code(...)`), but
        # not arbitrary callables / non-Guppy `pecos.sim`-acceptable inputs.
        # Compile upfront so a wrong input fails loud here, before tracing,
        # with a clear @guppy-mentioning message instead of crashing later
        # inside the HUGR step.
        needs_tags = _result_tags_present(detectors_json, observables_json)
        hugr_bytes = _certifiable_hugr_bytes(guppy)
        if hugr_bytes is None:
            if needs_tags:
                msg = (
                    "result_tags requires a @guppy-decorated function (or a "
                    "GuppyFunctionDefinition, e.g. the object "
                    "make_surface_code(...) returns) so the program can be "
                    "compiled to a HUGR. Pass such an input directly, or use "
                    "positional 'records' / 'meas_ids' instead."
                )
                raise ValueError(msg)
            msg = (
                "DetectorErrorModel.from_guppy requires a HUGR-certifiable program "
                "(a @guppy function, pecos.Guppy, pecos.Hugr, or HUGR envelope "
                f"bytes); a {type(guppy).__name__!r} input cannot be certified as "
                "statically scheduled, so an audited DEM cannot be built from it"
            )
            raise ValueError(msg)
        certificate_carrier = _certificate_carrier(guppy)
        generator_layout = (
            _generator_certified_layout(certificate_carrier, hugr_bytes) if certificate_carrier is not None else None
        )
        if generator_layout is None:
            from pecos_rslib import guppy_hugr_has_nontrivial_control_flow

            if guppy_hugr_has_nontrivial_control_flow(hugr_bytes):
                msg = (
                    "DetectorErrorModel.from_guppy requires a statically straight-line Guppy program; "
                    "branching or looping control flow cannot be certified from one runtime trace"
                )
                raise ValueError(msg)

        # Trace the EXACT bytes that were certified above: re-compiling the
        # original object for execution would let the audit and the execution
        # diverge (and pays a second compile for nothing).
        from pecos.programs import Hugr as _HugrProgram

        tc = trace_program_to_tick_circuit(
            _HugrProgram(hugr_bytes),
            num_qubits,
            seed=seed,
            runtime=runtime,
            require_hosted_operation_order=require_hosted_operation_order,
            max_hosted_tick_separation=max_hosted_tick_separation,
        )

        # Compilation passes required for traced QIS circuits before fault
        # analysis: normalize parameterized Clifford rotations to named gates,
        # stamp stable MeasIds onto measurement gates, and fail loudly if raw
        # traced-QIS rotations survived normalization.
        normalize_traced_tick_circuit(tc, context="DetectorErrorModel.from_guppy")
        _apply_traced_idle_passes(
            tc,
            strip_traced_idles=strip_traced_idles,
            idle_after_2q_duration=idle_after_2q_duration,
            idle_noise_parameters=(
                p_idle,
                p_idle_linear,
                p_idle_quadratic,
                t1,
                t2,
                p_idle_linear_rate,
                p_idle_quadratic_rate,
                p_idle_x_linear_rate,
                p_idle_y_linear_rate,
                p_idle_z_linear_rate,
                p_idle_x_quadratic_rate,
                p_idle_y_quadratic_rate,
                p_idle_z_quadratic_rate,
                p_idle_quadratic_sine_rate,
                p_idle_x_quadratic_sine_rate,
                p_idle_y_quadratic_sine_rate,
                p_idle_z_quadratic_sine_rate,
            ),
        )

        # Resolve `result_tags` -> record offsets via Rust (sound HUGR
        # extraction + runtime-loop guard via static-vs-traced measurement
        # count). After this, `detectors_json` / `observables_json` no longer
        # contain `result_tags`; the downstream Rust DEM builder is unchanged.
        if needs_tags:
            from pecos_rslib import resolve_result_tags_for_guppy

            source_ids_json = tc.get_meta("qis_source_measurement_ids") or tc.get_meta("guppy_source_measurement_ids")
            source_measurement_ids = json.loads(source_ids_json) if source_ids_json else []

            detectors_json, observables_json = resolve_result_tags_for_guppy(
                detectors_json,
                observables_json,
                hugr_bytes,
                source_measurement_ids,
                measurement_ids_in_execution_order(tc),
            )

        # Hand the caller's metadata to the Rust builder verbatim; it owns all
        # schema/ref validation (including D0/L0 id forms, tracked-Pauli
        # rejection, num_measurements consistency, and stamped-MeasId
        # resolution).
        tc.set_meta("detectors", detectors_json)
        tc.set_meta("observables", observables_json)
        if num_measurements is not None:
            tc.set_meta("num_measurements", str(num_measurements))

        return _from_circuit_with_noise(
            tc,
            p1=p1,
            p1_weights=p1_weights,
            p2=p2,
            p2_weights=p2_weights,
            p2_replacement_approximation=p2_replacement_approximation,
            p_meas=p_meas,
            p_prep=p_prep,
            p_idle=p_idle,
            t1=t1,
            t2=t2,
            idle_rz=idle_rz,
            p_idle_linear_rate=p_idle_linear_rate,
            p_idle_quadratic_rate=p_idle_quadratic_rate,
            p_idle_x_linear_rate=p_idle_x_linear_rate,
            p_idle_y_linear_rate=p_idle_y_linear_rate,
            p_idle_z_linear_rate=p_idle_z_linear_rate,
            p_idle_x_quadratic_rate=p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate=p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate=p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate=p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate=p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate=p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate=p_idle_z_quadratic_sine_rate,
        )


def _result_tags_present(detectors_json: str, observables_json: str) -> bool:
    """Cheap gate: does any entry use ``result_tags``? (substring check).

    Only decides whether to compile the Guppy program to HUGR; the actual
    extraction, loop-guard, resolution, and validation are all done in Rust.
    """
    return '"result_tags"' in (detectors_json or "") or '"result_tags"' in (observables_json or "")


def _validated_source_measurement_ids(circuit: Any) -> list[int]:
    """Return source IDs after proving they match the lowered runtime identities."""
    source_ids_json = circuit.get_meta("qis_source_measurement_ids") or circuit.get_meta(
        "guppy_source_measurement_ids",
    )
    source_measurement_ids = json.loads(source_ids_json) if source_ids_json else list(range(circuit.num_measurements()))
    if not isinstance(source_measurement_ids, list) or any(
        isinstance(meas_id, bool) or not isinstance(meas_id, int) or meas_id < 0 for meas_id in source_measurement_ids
    ):
        msg = "source measurement identities must be a list of non-negative integers"
        raise ValueError(msg)

    runtime_measurement_ids = measurement_ids_in_execution_order(circuit)
    if (
        len(source_measurement_ids) != len(set(source_measurement_ids))
        or len(runtime_measurement_ids) != len(set(runtime_measurement_ids))
        or set(source_measurement_ids) != set(runtime_measurement_ids)
    ):
        msg = "source and runtime measurement identities must be unique and describe the same set"
        raise ValueError(msg)
    return source_measurement_ids


def _preflight_guppy_static_schedule(
    guppy: Any,
    *,
    required_tags: Sequence[str],
) -> tuple[Sequence[Any] | None, bytes]:
    """Validate program-level trust before any runtime trace is captured.

    Returns ``(generator_layout, hugr_bytes)``. Runs the generator-certificate
    digest check and the branching/looping HUGR rejection *before* execution,
    so an unsupported program cannot run (or hang) ahead of its rejection.
    Fails closed on any input whose HUGR cannot be obtained: one sampled
    execution of an uninspectable program is not a certifiable static circuit.
    The returned bytes are the exact bytes the caller must execute.
    """
    hugr_bytes = _certifiable_hugr_bytes(guppy)
    if hugr_bytes is None:
        if required_tags:
            msg = "result_ref(...) requires a HUGR-compilable Guppy program"
            raise ValueError(msg)
        msg = (
            "build_dem_from_guppy requires a HUGR-certifiable program (a @guppy "
            "function, pecos.Guppy, pecos.Hugr, or HUGR envelope bytes); a "
            f"{type(guppy).__name__!r} input cannot be certified as statically "
            "scheduled, so an audited DEM cannot be built from it"
        )
        raise ValueError(msg)

    certificate_carrier = _certificate_carrier(guppy)
    generator_layout = (
        _generator_certified_layout(certificate_carrier, hugr_bytes) if certificate_carrier is not None else None
    )
    if generator_layout is not None:
        return generator_layout, hugr_bytes

    from pecos_rslib import guppy_hugr_has_nontrivial_control_flow

    if guppy_hugr_has_nontrivial_control_flow(hugr_bytes):
        msg = (
            "build_dem_from_guppy requires a statically straight-line Guppy program unless it carries "
            "a trusted generator-owned measurement layout; branching or looping control flow cannot be "
            "certified from one runtime trace"
        )
        raise ValueError(msg)
    return None, hugr_bytes


def _compiler_certified_result_traces(
    generator_layout: Sequence[Any] | None,
    hugr_bytes: bytes,
    circuit: Any,
    runtime_result_traces: Sequence[Mapping[str, Any]],
    *,
    required_tags: Sequence[str],
) -> list[dict[str, Any]]:
    """Resolve direct scalar result tags without trusting runtime read timing."""
    if generator_layout is not None:
        return _generator_certified_result_traces(
            generator_layout,
            circuit,
            runtime_result_traces,
            required_tags=required_tags,
        )

    candidate_tags = sorted(
        {
            *required_tags,
            *(trace.get("name") for trace in runtime_result_traces if isinstance(trace.get("name"), str)),
        },
    )
    if not candidate_tags:
        return []

    from pecos_rslib import extract_result_tag_measurements_for_guppy

    tag_occurrences, static_measurement_count = extract_result_tag_measurements_for_guppy(hugr_bytes)
    source_measurement_ids = _validated_source_measurement_ids(circuit)
    if static_measurement_count != len(source_measurement_ids):
        if required_tags:
            msg = (
                "result_ref(...) is not supported for Guppy programs with runtime loops: "
                f"the HUGR has {static_measurement_count} static measurement op(s) but "
                f"the traced program emits {len(source_measurement_ids)} measurement(s)"
            )
            raise ValueError(msg)
        return []

    required = set(required_tags)
    runtime_arities: dict[tuple[str, int], int] = {}
    runtime_occurrences: dict[str, int] = {}
    for trace in runtime_result_traces:
        tag = trace.get("name")
        if not isinstance(tag, str):
            continue
        occurrence = runtime_occurrences.get(tag, 0)
        runtime_occurrences[tag] = occurrence + 1
        values = trace.get("values")
        if isinstance(values, list):
            runtime_arities[(tag, occurrence)] = max(len(values), 1)
    certified: list[dict[str, Any]] = []
    for tag in candidate_tags:
        occurrences = tag_occurrences.get(tag)
        if occurrences is None:
            if tag in required:
                msg = f"result_ref {tag!r} is absent from the compiled Guppy program"
                raise ValueError(msg)
            continue
        for occurrence, ordinal in enumerate(occurrences):
            if ordinal is None:
                arity = runtime_arities.get((tag, occurrence), 1)
                certified.append(
                    {
                        "name": tag,
                        "occurrence": occurrence,
                        "values": [False] * arity,
                        "result_ids": [],
                    },
                )
                continue
            if ordinal >= len(source_measurement_ids):
                msg = f"compiler measurement ordinal {ordinal} is outside the traced source measurement stream"
                raise ValueError(msg)
            certified.append(
                {
                    "name": tag,
                    "occurrence": occurrence,
                    "values": [False],
                    "result_ids": [int(source_measurement_ids[ordinal])],
                },
            )
    return certified


def _generator_certified_result_traces(
    layout: Any,
    circuit: Any,
    runtime_result_traces: Sequence[Mapping[str, Any]],
    *,
    required_tags: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate a generator-supplied named-output to source-measurement layout."""
    if not isinstance(layout, Sequence) or isinstance(layout, (str, bytes)):
        msg = "Guppy named measurement layout certificate must be a sequence"
        raise TypeError(msg)
    entries: list[tuple[str, int]] = []
    for entry in layout:
        if (
            not isinstance(entry, Sequence)
            or isinstance(entry, (str, bytes))
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or isinstance(entry[1], bool)
            or not isinstance(entry[1], int)
            or entry[1] < 0
        ):
            msg = f"invalid Guppy named measurement layout entry: {entry!r}"
            raise ValueError(msg)
        entries.append((entry[0], entry[1]))

    source_measurement_ids = _validated_source_measurement_ids(circuit)
    if len(entries) != len(source_measurement_ids):
        msg = (
            "Guppy named measurement layout has "
            f"{len(entries)} entries but the source trace has {len(source_measurement_ids)} measurements"
        )
        raise ValueError(msg)

    runtime_value_count: dict[str, int] = {}
    for trace in runtime_result_traces:
        tag = trace.get("name")
        values = trace.get("values")
        if isinstance(tag, str) and isinstance(values, list):
            runtime_value_count[tag] = runtime_value_count.get(tag, 0) + len(values)
    layout_value_count: dict[str, int] = {}
    seen_slots: set[tuple[str, int]] = set()
    for slot in entries:
        if slot in seen_slots:
            msg = f"Guppy named measurement layout repeats output slot {slot!r}"
            raise ValueError(msg)
        seen_slots.add(slot)
        layout_value_count[slot[0]] = max(layout_value_count.get(slot[0], 0), slot[1] + 1)
    for tag, count in layout_value_count.items():
        if runtime_value_count.get(tag) != count:
            msg = (
                f"Guppy named measurement layout expects {count} value(s) for {tag!r}, "
                f"but the runtime trace emits {runtime_value_count.get(tag, 0)}"
            )
            raise ValueError(msg)
    missing_required = sorted(set(required_tags).difference(layout_value_count))
    if missing_required:
        msg = f"result_ref tag(s) are absent from the generator-certified layout: {missing_required}"
        raise ValueError(msg)

    # Defense-in-depth: the layout binds by abstract-circuit position, which
    # relies on the invariant that abstract measurement order equals source
    # `result()` emission order. Every generator-certified scalar slot must
    # be backed by the runtime trace's own scalar result id AND agree with
    # the positional binding: a missing id (a provenance regression) or a
    # disagreement (order drift in a future generator variant) fails loud
    # instead of silently misbinding detectors.
    runtime_scalar_ids: dict[tuple[str, int], int] = {}
    runtime_occurrences: dict[str, int] = {}
    for trace in runtime_result_traces:
        tag = trace.get("name")
        if not isinstance(tag, str):
            continue
        occurrence = runtime_occurrences.get(tag, 0)
        runtime_occurrences[tag] = occurrence + 1
        result_ids = trace.get("result_ids")
        if (
            isinstance(result_ids, list)
            and len(result_ids) == 1
            and isinstance(result_ids[0], int)
            and not isinstance(result_ids[0], bool)
            and result_ids[0] >= 0
        ):
            runtime_scalar_ids[(tag, occurrence)] = result_ids[0]
    for source_index, (tag, value_index) in enumerate(entries):
        expected = int(source_measurement_ids[source_index])
        actual = runtime_scalar_ids.get((tag, value_index))
        if actual is None:
            msg = (
                f"runtime trace does not expose a scalar result id for generator "
                f"slot ({tag!r}, occurrence {value_index}); the certified layout "
                "cannot be cross-checked against runtime provenance"
            )
            raise ValueError(msg)
        if actual != expected:
            msg = (
                f"generator layout binds {tag!r} occurrence {value_index} to source "
                f"measurement {expected}, but the runtime trace reports result id "
                f"{actual}; abstract and source measurement order have diverged"
            )
            raise ValueError(msg)

    return [
        {
            "name": tag,
            "occurrence": value_index,
            "values": [False],
            "result_ids": [int(source_measurement_ids[source_index])],
        }
        for source_index, (tag, value_index) in enumerate(entries)
    ]


def build_dem_from_guppy(
    guppy: Any,
    *,
    num_qubits: int,
    detectors: Sequence[Detector],
    observables: Sequence[Observable] = (),
    p1: float = 0.001,
    p1_weights: P1Weights | None = None,
    p2: float = 0.01,
    p2_weights: P2Weights | None = None,
    p2_replacement_approximation: str | None = None,
    p_meas: float = 0.001,
    p_prep: float = 0.001,
    p_idle: float | None = None,
    p_idle_linear: float | None = None,
    p_idle_linear_model: Mapping[str, float] | None = None,
    p_idle_quadratic: float | None = None,
    p_idle_coherent: bool = False,
    t1: float | None = None,
    t2: float | None = None,
    p_idle_linear_rate: float | None = None,
    p_idle_quadratic_rate: float | None = None,
    p_idle_x_linear_rate: float | None = None,
    p_idle_y_linear_rate: float | None = None,
    p_idle_z_linear_rate: float | None = None,
    p_idle_x_quadratic_rate: float | None = None,
    p_idle_y_quadratic_rate: float | None = None,
    p_idle_z_quadratic_rate: float | None = None,
    p_idle_quadratic_sine_rate: float | None = None,
    p_idle_x_quadratic_sine_rate: float | None = None,
    p_idle_y_quadratic_sine_rate: float | None = None,
    p_idle_z_quadratic_sine_rate: float | None = None,
    strip_traced_idles: bool | None = None,
    idle_after_2q_duration: float | None = None,
    runtime: object | None = None,
    seed: int = 0,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> GuppyDemBuild:
    """Trace once and build an audited DEM from typed measurement references.

    ``rec[-k]`` references the canonical dense Guppy result-id stream, not the
    runtime's scheduled measurement-gate order. ``result_ref(...)`` resolves
    through the compiled HUGR's dataflow and is checked against the traced
    measurement count. Both forms lower to stable ``MeasId`` metadata before
    DEM construction.

    Measurement-dependent quantum control remains unsupported because one
    captured execution is not a static circuit model.

    Args:
        guppy: A HUGR-certifiable Guppy program to trace once under the Selene
            QIS engine.
        num_qubits: Number of qubits to allocate for the trace.
        detectors: Typed detector definitions using ``rec[...]`` or
            ``result_ref(...)`` measurement references.
        observables: Typed logical-observable definitions using the same
            measurement-reference forms as ``detectors``.
        p1: Single-qubit gate Pauli error rate.
        p1_weights: Optional relative probabilities over single-qubit Pauli
            error labels ``"X"``, ``"Y"``, and ``"Z"``.
        p2: Two-qubit gate depolarizing rate.
        p2_weights: Optional relative probabilities over two-qubit Pauli error
            labels, including starred replacement branches.
        p2_replacement_approximation: Approximation used for starred
            replacement labels in ``p2_weights``.
        p_meas: Measurement flip rate.
        p_prep: Preparation (reset) error rate.
        p_idle: Optional shorthand for ``p_idle_linear`` with the uniform
            ``{"X": 1/3, "Y": 1/3, "Z": 1/3}`` model.
        p_idle_linear: Optional total stochastic idle-noise rate linear in
            duration. Uses the engines ``GeneralNoiseModel`` convention.
        p_idle_linear_model: Optional relative weights over ``"X"``, ``"Y"``,
            and ``"Z"`` for ``p_idle_linear``. Weights must be finite,
            non-negative, and sum to 1.0 within ``1e-5``. Defaults to the
            engines' uniform model. The engines leakage key ``"L"`` is
            reserved but unsupported by DEM construction.
        p_idle_quadratic: Optional quadratic dephasing rate. The default
            stochastic interpretation gives probability ``sin(rate * t)^2``;
            the coherent interpretation forwards ``RZ(rate * t)`` and the DEM
            uses its ``sin(rate * t / 2)^2`` half-angle convention.
        p_idle_coherent: Select coherent RZ rather than stochastic Z
            interpretation for ``p_idle_quadratic``. Defaults to ``False``.
        t1: Optional T1 relaxation time for explicit idle gates.
        t2: Optional T2 dephasing time for explicit idle gates.
        p_idle_linear_rate: Deprecated bare Z-only alias for a stochastic rate
            linear in idle duration. Use ``p_idle_linear`` with a Z-only model,
            or ``p_idle_z_linear_rate`` for literal behavior.
        p_idle_quadratic_rate: Deprecated bare Z-only coefficient-style rate
            quadratic in idle duration. Use ``p_idle_quadratic`` for engines
            semantics, or ``p_idle_z_quadratic_rate`` for literal behavior.
        p_idle_x_linear_rate: Optional stochastic X-memory rate linear in idle duration.
        p_idle_y_linear_rate: Optional stochastic Y-memory rate linear in idle duration.
        p_idle_z_linear_rate: Optional stochastic Z-memory rate linear in idle duration.
        p_idle_x_quadratic_rate: Optional stochastic X-memory rate quadratic in idle duration.
        p_idle_y_quadratic_rate: Optional stochastic Y-memory rate quadratic in idle duration.
        p_idle_z_quadratic_rate: Optional stochastic Z-memory rate quadratic in idle duration.
        p_idle_quadratic_sine_rate: Deprecated bare Z-only alias for a
            stochastic rate with probability ``sin(rate * duration)^2``. Use
            ``p_idle_quadratic`` or ``p_idle_z_quadratic_sine_rate``.
        p_idle_x_quadratic_sine_rate: Optional stochastic X-memory sine-law rate.
        p_idle_y_quadratic_sine_rate: Optional stochastic Y-memory sine-law rate.
        p_idle_z_quadratic_sine_rate: Optional stochastic Z-memory sine-law rate.
        strip_traced_idles: If true, remove identity-like gates from the
            normalized trace, including ``I``, ``Idle``, and zero-angle
            rotations. This pass runs before idle insertion when both
            idle-pass options are set. Defaults to ``None``, which strips
            exactly when ``idle_after_2q_duration`` is set; pass ``False``
            explicitly to keep runtime-emitted idles alongside inserted
            ones.
        idle_after_2q_duration: If set, insert an ``Idle`` gate of this
            duration on both qubits after every two-qubit gate. Insertion runs
            after ``strip_traced_idles`` and before typed result-reference
            resolution and detector/observable metadata attachment.
        runtime: Optional Selene runtime selector/plugin. ``None`` selects the
            default Selene runtime.
        seed: Seed for the ideal trace run.
        require_hosted_operation_order: If true, validate generic
            hosted-operation metadata after trace replay.
        max_hosted_tick_separation: Optional maximum absolute signed tick
            separation accepted by the hosted-operation validator.

    Raises:
        ValueError: If ``idle_after_2q_duration`` is not a finite positive
            number, or if any idle-noise parameter is set but the final traced
            circuit has no ``Idle`` gates. Pass ``idle_after_2q_duration`` or
            use a Selene runtime that emits scheduled idles to provide targets
            for idle noise.
    """
    from pecos.tracing import _trace_program_to_tick_circuit_with_result_traces

    (
        p_idle_x_linear_rate,
        p_idle_y_linear_rate,
        p_idle_z_linear_rate,
        p_idle_z_quadratic_sine_rate,
        idle_rz,
    ) = _translate_structured_idle_noise(
        p_idle=p_idle,
        p_idle_linear=p_idle_linear,
        p_idle_linear_model=p_idle_linear_model,
        p_idle_quadratic=p_idle_quadratic,
        p_idle_coherent=p_idle_coherent,
        p_idle_linear_rate=p_idle_linear_rate,
        p_idle_quadratic_rate=p_idle_quadratic_rate,
        p_idle_x_linear_rate=p_idle_x_linear_rate,
        p_idle_y_linear_rate=p_idle_y_linear_rate,
        p_idle_z_linear_rate=p_idle_z_linear_rate,
        p_idle_x_quadratic_rate=p_idle_x_quadratic_rate,
        p_idle_y_quadratic_rate=p_idle_y_quadratic_rate,
        p_idle_z_quadratic_rate=p_idle_z_quadratic_rate,
        p_idle_quadratic_sine_rate=p_idle_quadratic_sine_rate,
        p_idle_x_quadratic_sine_rate=p_idle_x_quadratic_sine_rate,
        p_idle_y_quadratic_sine_rate=p_idle_y_quadratic_sine_rate,
        p_idle_z_quadratic_sine_rate=p_idle_z_quadratic_sine_rate,
    )

    referenced_tags = sorted(
        {ref.tag for item in (*detectors, *observables) for ref in item.refs if isinstance(ref, ResultRef)},
    )
    generator_layout, hugr_bytes = _preflight_guppy_static_schedule(
        guppy,
        required_tags=referenced_tags,
    )
    has_generator_layout = generator_layout is not None
    # Trace the EXACT bytes that were certified: re-compiling the original
    # object for execution would let the audit and the execution diverge (and
    # pays a second compile for nothing).
    from pecos.programs import Hugr as _HugrProgram

    circuit, result_traces = _trace_program_to_tick_circuit_with_result_traces(
        _HugrProgram(hugr_bytes),
        num_qubits,
        seed=seed,
        runtime=runtime,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        allow_raw_measurement_id_fallback=False,
    )
    normalize_traced_tick_circuit(circuit, context="build_dem_from_guppy")
    _apply_traced_idle_passes(
        circuit,
        strip_traced_idles=strip_traced_idles,
        idle_after_2q_duration=idle_after_2q_duration,
        idle_noise_parameters=(
            p_idle,
            p_idle_linear,
            p_idle_quadratic,
            t1,
            t2,
            p_idle_linear_rate,
            p_idle_quadratic_rate,
            p_idle_x_linear_rate,
            p_idle_y_linear_rate,
            p_idle_z_linear_rate,
            p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate,
        ),
    )
    result_traces = _compiler_certified_result_traces(
        generator_layout,
        hugr_bytes,
        circuit,
        result_traces,
        required_tags=referenced_tags,
    )
    schema = _resolve_dem_specs(
        detectors,
        observables,
        circuit=circuit,
        result_traces=result_traces,
    )
    if has_generator_layout:
        named_result_binding = "generator_layout_v2_program_bound"
    else:
        result_ids = [result_id for _, ids in schema.result_ids_by_tag for result_id in ids]
        if not result_ids or all(result_id is None for result_id in result_ids):
            named_result_binding = "none"
        elif any(result_id is None for result_id in result_ids):
            named_result_binding = "compiler_direct_scalar_partial"
        else:
            named_result_binding = "compiler_direct_scalar_complete"
    circuit.set_meta("detectors", schema.detectors_json)
    circuit.set_meta("observables", schema.observables_json)
    circuit.set_meta("num_measurements", str(circuit.num_measurements()))
    circuit.set_meta("dem_schema_fingerprint", schema.schema_fingerprint)

    dem = _from_circuit_with_noise(
        circuit,
        p1=p1,
        p1_weights=p1_weights,
        p2=p2,
        p2_weights=p2_weights,
        p2_replacement_approximation=p2_replacement_approximation,
        p_meas=p_meas,
        p_prep=p_prep,
        p_idle=p_idle,
        t1=t1,
        t2=t2,
        idle_rz=idle_rz,
        p_idle_linear_rate=p_idle_linear_rate,
        p_idle_quadratic_rate=p_idle_quadratic_rate,
        p_idle_x_linear_rate=p_idle_x_linear_rate,
        p_idle_y_linear_rate=p_idle_y_linear_rate,
        p_idle_z_linear_rate=p_idle_z_linear_rate,
        p_idle_x_quadratic_rate=p_idle_x_quadratic_rate,
        p_idle_y_quadratic_rate=p_idle_y_quadratic_rate,
        p_idle_z_quadratic_rate=p_idle_z_quadratic_rate,
        p_idle_quadratic_sine_rate=p_idle_quadratic_sine_rate,
        p_idle_x_quadratic_sine_rate=p_idle_x_quadratic_sine_rate,
        p_idle_y_quadratic_sine_rate=p_idle_y_quadratic_sine_rate,
        p_idle_z_quadratic_sine_rate=p_idle_z_quadratic_sine_rate,
    )
    return GuppyDemBuild(
        dem=dem,
        circuit=circuit,
        detectors_json=schema.detectors_json,
        observables_json=schema.observables_json,
        measurement_ledger=schema.ledger,
        schema_fingerprint=schema.schema_fingerprint,
        named_result_binding=named_result_binding,
        _detector_meas_ids=schema.detector_meas_ids,
        _observable_meas_ids=schema.observable_meas_ids,
        _result_ids_by_tag=schema.result_ids_by_tag,
    )


DetectorErrorModel = _RustDetectorErrorModel
DetectorErrorModel.from_guppy = classmethod(_DetectorErrorModelMixin.__dict__["from_guppy"].__func__)
