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
``TickCircuit``, optionally compiles the program to a HUGR (only when
``result_tags`` is requested -- to recover the sound tag -> measurement
binding via ``pecos_hugr_qis::extract_result_tag_measurements``), and hands
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

from collections.abc import Mapping
from typing import Any

from pecos_rslib.qec import DetectorErrorModel as _RustDetectorErrorModel

P1Weights = Mapping[str, float]
P2Weights = Mapping[str, float]


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
            guppy: Anything ``pecos.sim`` accepts -- a ``@guppy``-decorated
                function, a compiled Guppy program (e.g. the object returned by
                ``pecos.guppy.make_surface_code``), or a program wrapper. There
                is no Guppy *source-string* form in PECOS; pass a program/
                function, not source text.
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
                  resolved to record offsets in Rust. Supported only for
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
            p_idle: Optional uniform depolarizing idle-noise rate per idle duration.
            t1: Optional T1 relaxation time for explicit idle gates.
            t2: Optional T2 dephasing time for explicit idle gates.
            p_idle_linear_rate: Optional legacy alias for stochastic Z-memory rate
                linear in idle duration.
            p_idle_quadratic_rate: Optional legacy alias for stochastic Z-memory rate
                quadratic in idle duration.
            p_idle_x_linear_rate: Optional stochastic X-memory rate linear in idle duration.
            p_idle_y_linear_rate: Optional stochastic Y-memory rate linear in idle duration.
            p_idle_z_linear_rate: Optional stochastic Z-memory rate linear in idle duration.
            p_idle_x_quadratic_rate: Optional stochastic X-memory rate quadratic in idle duration.
            p_idle_y_quadratic_rate: Optional stochastic Y-memory rate quadratic in idle duration.
            p_idle_z_quadratic_rate: Optional stochastic Z-memory rate quadratic in idle duration.
            p_idle_quadratic_sine_rate: Optional legacy alias for stochastic Z-memory
                rate with probability ``sin(rate * duration)^2``.
            p_idle_x_quadratic_sine_rate: Optional stochastic X-memory sine-law rate.
            p_idle_y_quadratic_sine_rate: Optional stochastic Y-memory sine-law rate.
            p_idle_z_quadratic_sine_rate: Optional stochastic Z-memory sine-law rate.
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
                ``meas_id``, or if the traced operation stream cannot be
                replayed.

        Note:
            **Measurement-dependent (dynamic) control flow is unsupported.**
            ``from_guppy`` traces one ideal execution; a Guppy program whose
            quantum operations depend on a measurement *outcome* (e.g.
            ``if measure(q): x(other)``) would yield a DEM built from a single
            sampled branch, silently wrong and seed-dependent. No reliable
            runtime-trace heuristic distinguishes that from the
            statically-scheduled post-measurement gates a normal QEC circuit
            has (the surface code has these every round), so no guard is
            attempted -- pass straight-line programs only. Sound detection
            would require HUGR conditional-on-measurement analysis (deferred).

            Every measurement is anchored to a stable MeasId automatically:
            ``measure()`` itself allocates the result slot in the trace (a
            ``result(...)`` call is not required for MeasId assignment).

            Source-anchored tag-referenced detectors are exposed via the
            ``result_tags`` field on detectors/observables (see the
            ``detectors_json`` argument). The supported scope is canonical
            scalar ``result(tag, measure(q))`` in straight-line programs; the
            runtime-loop case (per-occurrence binding) remains deferred.
        """
        from pecos.qec.surface.circuit_builder import normalize_traced_qis_tick_circuit
        from pecos.qec.surface.decode import trace_guppy_into_tick_circuit

        # Tag-referenced detectors require the compiled HUGR (to recover the
        # sound, reorder-immune Guppy `result(tag, ...)` -> measurement
        # binding). `guppy_to_hugr` accepts @guppy-decorated functions and
        # `GuppyFunctionDefinition`s (e.g. `make_surface_code(...)`), but
        # not arbitrary callables / non-Guppy `pecos.sim`-acceptable inputs.
        # Compile upfront so a wrong input fails loud here, before tracing,
        # with a clear @guppy-mentioning message instead of crashing later
        # inside the HUGR step.
        needs_tags = _result_tags_present(detectors_json, observables_json)
        hugr_bytes: bytes | None = None
        if needs_tags:
            from pecos._compilation import guppy_to_hugr

            try:
                hugr_bytes = guppy_to_hugr(guppy)
            except ValueError as exc:
                msg = (
                    "result_tags requires a @guppy-decorated function (or a "
                    "GuppyFunctionDefinition, e.g. the object "
                    "make_surface_code(...) returns) so the program can be "
                    "compiled to a HUGR. Pass such an input directly, or use "
                    "positional 'records' / 'meas_ids' instead."
                )
                raise ValueError(msg) from exc

        tc = trace_guppy_into_tick_circuit(
            guppy,
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
        normalize_traced_qis_tick_circuit(tc, context="DetectorErrorModel.from_guppy")

        # Resolve `result_tags` -> record offsets via Rust (sound HUGR
        # extraction + runtime-loop guard via static-vs-traced measurement
        # count). After this, `detectors_json` / `observables_json` no longer
        # contain `result_tags`; the downstream Rust DEM builder is unchanged.
        if needs_tags:
            from pecos_rslib import resolve_result_tags_for_guppy

            detectors_json, observables_json = resolve_result_tags_for_guppy(
                detectors_json,
                observables_json,
                hugr_bytes,
                tc.num_measurements(),
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


DetectorErrorModel = _RustDetectorErrorModel
DetectorErrorModel.from_guppy = classmethod(_DetectorErrorModelMixin.__dict__["from_guppy"].__func__)
