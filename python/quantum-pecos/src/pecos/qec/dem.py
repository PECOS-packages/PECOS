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

    meas_tags_meta = tc.get_meta("meas_tags")
    meas_tags: dict[str, list[int]] = json.loads(meas_tags_meta) if meas_tags_meta else {}

    def _check(kind: str, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            # Tracked Paulis reference qubits via "pauli", not measurements.
            if entry.get("kind") == "tracked_pauli":
                continue
            for tag in entry.get("result_tags", []) or []:
                if tag not in meas_tags:
                    known = ", ".join(sorted(meas_tags)[:8]) or "<none>"
                    msg = (
                        f"{kind} {entry.get('id', entry)} references "
                        f"result_tag {tag!r}, which the traced program never "
                        f"recorded via result(...). Known tags: {known}"
                        f"{' ...' if len(meas_tags) > 8 else ''}."
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


def _uses_result_tags(detectors_json: str, observables_json: str) -> bool:
    """True if any detector/observable references measurements by result tag."""
    for blob in (detectors_json, observables_json):
        if not blob:
            continue
        try:
            entries = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if any(e.get("result_tags") for e in entries if isinstance(e, dict)):
            return True
    return False


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
            observables_json: Observable / tracked-Pauli definitions as a JSON
                list. Plain observables look like ``[{"id": 0, "records":
                [-1]}]``. Tracked Paulis are entries in this same list carrying
                ``"kind": "tracked_pauli"`` (plus ``"label"`` and ``"pauli"``,
                e.g. ``"+X0 Z2"``); the DEM builder splits them out
                automatically. There is no separate tracked-Pauli argument --
                this matches the underlying circuit-metadata contract exactly.

                Limitation: a tracked Pauli references **qubits** (via its
                ``pauli`` string), not measurements, so the ``result_tags``
                anchor does not apply to it. Its qubit indices are interpreted
                in the *traced (post-compilation)* qubit numbering and are
                therefore **not** source-stable the way tag-referenced
                detectors/observables are -- Guppy exposes no ``result()``-style
                identity for a qubit. For a hand-authored general Guppy program
                the caller must supply tracked-Pauli qubit indices in the
                traced numbering; geometry-derived paths (e.g. the surface
                builder) avoid this by construction.
                Reorder-robust alternative: instead of positional ``records``/
                ``meas_ids``, an entry may carry ``"result_tags": ["sx0:meas:0",
                ...]`` to reference measurements by the stable Guppy
                ``result(tag, ...)`` tag they were recorded under. Tags are
                fixed in the Guppy source, so they survive any measurement
                reordering introduced by Guppy/Selene compilation. The DEM
                builder resolves tags via the trace's ``meas_tags`` linkage;
                ``result_tags`` and ``records`` may be combined on one entry.
            num_measurements: Total measurement count, used to resolve negative
                ``records`` offsets. If omitted, it is inferred from the traced
                circuit (and is always set automatically when ``result_tags``
                are used).
            p1: Single-qubit gate depolarizing rate.
            p2: Two-qubit gate depolarizing rate.
            p_meas: Measurement flip rate.
            p_prep: Preparation (reset) error rate.
            seed: Seed for the ideal trace run.

        Returns:
            A ``DetectorErrorModel`` built from the traced circuit.

        Raises:
            ValueError: If ``num_measurements`` disagrees with the traced
                measurement count, if a detector/observable references an
                out-of-range ``record``, an absent ``meas_id``, or a
                ``result_tag`` the traced program never recorded, or if the
                traced operation stream is malformed (the strict
                ``AllocateResult``/``Measure`` pairing in the replay fails).

        Note:
            Every measurement is anchored to a stable MeasId automatically:
            ``measure()`` itself allocates the result slot in the trace. A
            ``result(...)`` call is not required for MeasId assignment, but it
            *is* what enables reorder-robust ``result_tags`` references: the
            trace records, per tag, exactly which MeasIds it captured
            (``meas_tags`` metadata), an identity fixed in the Guppy source.

            Positional ``records``/``meas_ids`` reference measurements by
            *traced (post-compilation)* order and are therefore sensitive to
            measurement reordering by Guppy/Selene compilation; ``result_tags``
            are not. See
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
        if num_measurements is None and _uses_result_tags(detectors_json, observables_json):
            # The DEM builder resolves result_tags -> record offsets as
            # meas_id - num_measurements, so num_measurements must be present.
            num_measurements, _ = _collect_measurement_info(tc)
        if num_measurements is not None:
            tc.set_meta("num_measurements", str(num_measurements))

        return _RustDetectorErrorModel.from_circuit(
            tc,
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
