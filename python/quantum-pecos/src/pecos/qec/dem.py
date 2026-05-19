"""Python-level ``DetectorErrorModel`` with a Guppy convenience constructor.

The core ``DetectorErrorModel`` is implemented in Rust
(``pecos_rslib.qec.DetectorErrorModel``). The Guppy -> Selene -> QIS-trace
pipeline, however, lives entirely in Python (``pecos.sim``, ``pecos.guppy``,
``pecos.qec.surface.decode``). To keep the convenient
``DetectorErrorModel.from_guppy(...)`` call site without making the low-level
Rust extension import the high-level Python package (a dependency cycle), this
module defines a thin Python subclass that adds :meth:`from_guppy` and is
re-exported as the public ``pecos.qec.DetectorErrorModel``.

The subclass is behaviorally identical to the Rust class for every other
operation; all existing methods (``from_circuit``, ``from_pecos_metadata_json``,
``to_string``, ``to_sampler``, ...) are inherited unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from pecos_rslib.qec import DetectorErrorModel as _RustDetectorErrorModel


def _collect_measurement_info(tc: Any) -> tuple[int, set[int]]:
    """Return (measurement count, set of MeasIds) for the traced circuit.

    Counts measured qubits across all MZ gates and gathers the stable MeasIds
    stamped on them.
    """
    dag = tc.to_dag_circuit()
    count = 0
    meas_ids: set[int] = set()
    for node_id in dag.nodes():
        gate = dag.gate(node_id)
        if gate is None or gate.gate_type.name != "MZ":
            continue
        qubits = list(gate.qubits)
        ids = list(gate.meas_ids)
        count += len(qubits)
        if len(ids) != len(qubits):
            msg = (
                "Traced Guppy circuit has an MZ gate without a stable MeasId "
                f"(qubits={qubits}, meas_ids={ids}) after replay and "
                "assign_missing_meas_ids(); this indicates an internal "
                "inconsistency in the traced-circuit pipeline, not a problem "
                "with the caller's inputs."
            )
            raise ValueError(msg)
        meas_ids.update(int(i) for i in ids)
    return count, meas_ids


def _validate_measurement_contract(
    tc: Any,
    *,
    detectors_json: str,
    observables_json: str,
    num_measurements: int | None,
) -> None:
    """Fail loudly if the caller's detector/observable JSON is inconsistent.

    Catches the common ``from_guppy`` misuse where detector ``records``/
    ``meas_ids`` do not line up with the measurements the traced program
    actually emits, instead of silently building a wrong DEM.
    """
    measured, present_ids = _collect_measurement_info(tc)

    if num_measurements is not None and num_measurements != measured:
        msg = (
            f"num_measurements={num_measurements} does not match the "
            f"{measured} measurement(s) the traced Guppy program emits. The "
            "detector/observable record offsets are defined against the "
            "traced measurement order; a mismatch means the DEM would be "
            "silently wrong."
        )
        raise ValueError(msg)
    effective = num_measurements if num_measurements is not None else measured

    def _check(kind: str, entries: list[dict[str, Any]]) -> None:
        alt_id = "detector_id" if kind == "Detector" else "observable_id"
        # NB: malformed input raises ValueError (not TypeError) to keep one
        # consistent failure type across from_guppy's documented contract and
        # the sibling record/meas_id checks below -- hence the TRY004 noqas.
        for entry in entries:
            if not isinstance(entry, dict):
                msg = f"{kind} entry is not a JSON object: {entry!r}"
                raise ValueError(msg)  # noqa: TRY004
            # Tracked Paulis reference qubits via "pauli", not measurements.
            if entry.get("kind") == "tracked_pauli":
                continue
            # Schema: the Rust DEM-builder JSON parser requires an integer id
            # and records; on a parse failure it silently builds an empty DEM.
            # Validate here so malformed input fails loud instead.
            raw_id = entry.get("id", entry.get(alt_id))
            if not isinstance(raw_id, int) or isinstance(raw_id, bool):
                msg = (
                    f"{kind} entry {entry!r} is missing a valid integer "
                    f"'id' (or '{alt_id}'); the DEM builder would silently "
                    "drop it and produce an empty DEM."
                )
                raise ValueError(msg)  # noqa: TRY004
            if not (entry.get("records") or entry.get("meas_ids")):
                msg = (
                    f"{kind} {raw_id} has no 'records' or 'meas_ids'; it "
                    "would contribute nothing and silently weaken the DEM."
                )
                raise ValueError(msg)
            for rec in entry.get("records", []) or []:
                idx = effective + int(rec)
                if not 0 <= idx < effective:
                    msg = (
                        f"{kind} {entry.get('id', entry)} references record "
                        f"{rec}, which is out of range for a circuit with "
                        f"{effective} measurement(s)."
                    )
                    raise ValueError(msg)
            for mid in entry.get("meas_ids", []) or []:
                if int(mid) not in present_ids:
                    msg = (
                        f"{kind} {entry.get('id', entry)} references "
                        f"meas_id {mid}, which is not present in the traced "
                        "circuit. meas_ids must match the stable MeasIds the "
                        "traced program assigns (one per measured qubit, in "
                        "trace order)."
                    )
                    raise ValueError(msg)

    try:
        detectors = json.loads(detectors_json) if detectors_json else []
        observables = json.loads(observables_json) if observables_json else []
    except json.JSONDecodeError as exc:
        msg = f"detectors_json/observables_json is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    _check("Detector", detectors)
    _check("Observable", observables)


def _normalize_entry_ids(blob: str, prefix: str) -> str:
    """Normalize ``"id": "D0"``/``"L0"`` to the integer the pipeline expects.

    ``prefix`` is ``"D"`` for detectors, ``"L"`` for observables. Integer ids
    and entries without ``"id"`` (e.g. those using ``detector_id`` /
    ``observable_id``) pass through unchanged. A string id with the wrong
    prefix or a non-numeric body is a hard error -- silently reinterpreting it
    would risk a mislabeled DEM.
    """
    if not blob:
        return blob
    try:
        entries = json.loads(blob)
    except json.JSONDecodeError:
        return blob  # validation downstream reports the parse error
    if not isinstance(entries, list):
        return blob

    changed = False
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        raw = entry["id"].strip()
        body = raw[len(prefix):] if raw.startswith(prefix) else None
        if body is None or not body.isdigit():
            msg = (
                f"id {entry['id']!r} is not a valid identifier for this list; "
                f"expected an integer or {prefix!r}-prefixed form like "
                f"{prefix}0 (detectors use 'D', observables use 'L')."
            )
            raise ValueError(msg)
        entry["id"] = int(body)
        changed = True

    return json.dumps(entries, separators=(",", ":")) if changed else blob


class DetectorErrorModel(_RustDetectorErrorModel):
    """Detector error model with a Guppy/QIS-trace convenience constructor.

    Identical to :class:`pecos_rslib.qec.DetectorErrorModel` except for the
    added :meth:`from_guppy` classmethod.
    """

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
        p2: float = 0.01,
        p_meas: float = 0.001,
        p_prep: float = 0.001,
        seed: int = 0,
    ) -> _RustDetectorErrorModel:
        """Build a circuit-level DEM from a Guppy program by tracing it.

        Runs ``guppy`` under the Selene QIS engine with operation tracing,
        replays the captured gate stream into a ``TickCircuit``, attaches the
        caller-supplied detector/observable definitions, and builds the DEM via
        native PECOS fault propagation.

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
                same integer. ``records`` are
                negative measurement offsets (Stim convention); ``meas_ids``
                may be used instead. Defined against the *traced* program's own
                measurement order.
            observables_json: Observable definitions as a JSON list, e.g.
                ``[{"id": 0, "records": [-1]}]`` (same id/records rules as
                detectors).

                Tracked Paulis: **hand-authored JSON tracked Paulis are NOT
                supported** by this path. The DEM builder's JSON observable
                parser reads only ``id``/``records``; it ignores ``kind`` /
                ``label`` / ``pauli``. Tracked Paulis are only produced from
                circuit *annotations* (e.g. the surface builder), not from
                ``observables_json``. A ``{"kind": "tracked_pauli", ...}``
                entry here is silently treated as a (malformed) observable --
                do not use it.
            num_measurements: Total measurement count, used to resolve negative
                ``records`` offsets. If omitted, it is inferred from the traced
                circuit.
            p1: Single-qubit gate depolarizing rate.
            p2: Two-qubit gate depolarizing rate.
            p_meas: Measurement flip rate.
            p_prep: Preparation (reset) error rate.
            seed: Seed for the ideal trace run.

        Returns:
            A ``DetectorErrorModel`` built from the traced circuit.

        Raises:
            ValueError: If ``num_measurements`` disagrees with the traced
                measurement count, if a detector/observable is malformed or
                references an out-of-range ``record`` or an absent
                ``meas_id``, or if the traced operation stream cannot be
                replayed.

        Note:
            Every measurement is anchored to a stable MeasId automatically:
            ``measure()`` itself allocates the result slot in the trace (a
            ``result(...)`` call is not required for MeasId assignment).

            Detector/observable ``records``/``meas_ids`` reference measurements
            by *traced (post-compilation)* order and are therefore sensitive to
            any measurement reordering introduced by Guppy/Selene compilation.
            Source-anchored tag-referenced detectors are **not exposed here**:
            the sound HUGR-based binding
            (``pecos_hugr_qis::extract_result_tag_measurements``) only covers
            the canonical scalar ``result(tag, measure(q))`` pattern and is not
            yet wired into ``from_guppy``; runtime-loop programs remain
            unsupported. See
            ``docs/proposals/001-from-guppy-tag-referenced-detectors.md``.
        """
        from pecos.qec.surface.decode import trace_guppy_into_tick_circuit

        # Convenience: allow "id": "D0" / "L0" (matching DEM labels) in
        # addition to bare integers. Normalized to ints here so the schema,
        # Rust parser, and surface path are untouched.
        detectors_json = _normalize_entry_ids(detectors_json, "D")
        observables_json = _normalize_entry_ids(observables_json, "L")

        tc = trace_guppy_into_tick_circuit(guppy, num_qubits, seed=seed)

        # Compilation passes required for traced QIS circuits before fault
        # analysis: normalize parameterized Clifford rotations to named gates
        # and stamp stable MeasIds onto measurement gates.
        tc.lower_clifford_rotations()
        tc.assign_missing_meas_ids()

        _validate_measurement_contract(
            tc,
            detectors_json=detectors_json,
            observables_json=observables_json,
            num_measurements=num_measurements,
        )

        tc.set_meta("detectors", detectors_json)
        tc.set_meta("observables", observables_json)
        if num_measurements is not None:
            tc.set_meta("num_measurements", str(num_measurements))

        return _RustDetectorErrorModel.from_circuit(
            tc,
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
