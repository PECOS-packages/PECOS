# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Local detector processes and exact detector-conditioned quantum instruments.

Sector names describe persistent quantum subspaces, not measurements. Only
declared measurement outcomes become records. The exact dense compiler is a
small-system reference with explicit dimension and Kraus-path limits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from pecos.experimental.detector_models._numeric import (
    array,
    close,
    decode_array,
    effect,
    encode_array,
    fail,
    identity,
    matrices,
    positive_semidefinite,
    trace_preserving,
    zeros,
)
from pecos.experimental.detector_models._numeric import (
    operators as _kraus_stack,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pecos as pc

SCHEMA_VERSION = 1
TENSOR_CONVENTION = "most-significant system first; explicit local target order"
BranchKey = tuple[int | str | None, ...]


def _json_digest(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _save(path: str | Path, manifest: dict[str, object], arrays: dict[str, object]) -> dict[str, object]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.update(
        complex_encoding="shape and flattened real/imaginary pairs",
        arrays=arrays,
        arrays_sha256=_json_digest(arrays),
    )
    text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return manifest


def _load(path: str | Path, kind: str) -> dict[str, object]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != kind:
        fail("unsupported experimental detector-model schema")
    if manifest.get("complex_encoding") != "shape and flattened real/imaginary pairs":
        fail("unsupported complex encoding")
    if _json_digest(manifest["arrays"]) != manifest["arrays_sha256"]:
        fail("serialized array hash mismatch")
    return manifest


@dataclass(frozen=True)
class SystemSpec:
    """Declare a persistent system and a partition of its basis into subspaces."""

    id: str
    basis: tuple[str, ...]
    sectors: Mapping[str, tuple[int, ...]]
    role: str = "physical"

    @property
    def dimension(self) -> int:
        """Return the dimension, including leakage and vacancy sectors."""
        return len(self.basis)

    def validate(self) -> None:
        """Validate distinct basis labels and an exhaustive sector partition."""
        if (
            not isinstance(self.id, str)
            or not self.id
            or not self.basis
            or any(not isinstance(label, str) or not label for label in self.basis)
            or len(set(self.basis)) != len(self.basis)
        ):
            fail("system IDs and distinct basis labels are required")
        partition = [index for indices in self.sectors.values() for index in indices]
        if not self.sectors or any(
            not isinstance(name, str) or not name or not indices for name, indices in self.sectors.items()
        ):
            fail("sectors must be named nonempty subspaces")
        if any(type(index) is not int for index in partition) or sorted(partition) != list(range(self.dimension)):
            fail("sectors must partition the declared basis exactly once")
        if self.role not in ("physical", "memory"):
            fail("system role must be physical or explicit memory")


@dataclass(frozen=True)
class ChannelFactor:
    """Declare unobserved local CP/TP evolution; Kraus indices are not records."""

    id: str
    targets: tuple[str, ...]
    kraus: object
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class InstrumentFactor:
    """Declare an observed outcome and its local, possibly multi-Kraus CP map."""

    id: str
    targets: tuple[str, ...]
    outcomes: Mapping[str, object]
    record_id: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorDef:
    """Interpret a parity of actual observed records, including unavailable values."""

    id: str
    records: tuple[str, ...]
    bit_values: Mapping[str, int | None] = field(default_factory=lambda: {"0": 0, "1": 1, "no_readout": None})
    expected_parity: int | None = 0

    def evaluate(self, record: Mapping[str, str]) -> int | None:
        """Evaluate relative to the declared reference, propagating unavailable data."""
        parity = 0
        for name in self.records:
            label = record[name]
            if label not in self.bit_values:
                fail(f"detector {self.id} has no interpretation of outcome {label!r}")
            value = self.bit_values[label]
            if value is None:
                return None
            parity ^= value
        return parity if self.expected_parity is None else parity ^ self.expected_parity


@dataclass(frozen=True)
class DetectorProcess:
    """Store local factors without expanding all physical states or histories.

    ``input_isometry`` optionally maps a logical input into the initial physical
    state, including any declared memory. Outputs stay physical: recovery and
    logical reduction are not implicit. Retained records default to all actual
    observations; an explicit empty tuple requests detector-only coarse graining.
    """

    systems: tuple[SystemSpec, ...]
    factors: tuple[ChannelFactor | InstrumentFactor, ...]
    detectors: tuple[DetectorDef, ...] = ()
    retained_records: tuple[str, ...] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    input_isometry: object | None = None

    @property
    def dimension(self) -> int:
        """Return the complete persistent physical Hilbert-space dimension."""
        return prod(system.dimension for system in self.systems)

    def validate(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Validate physical maps, detector definitions and retained-record names."""
        if not self.systems:
            fail("at least one persistent system is required")
        for system in self.systems:
            system.validate()
        lookup = {system.id: system for system in self.systems}
        if len(lookup) != len(self.systems):
            fail("duplicate system IDs")
        ids, records = set(), {}
        for factor in self.factors:
            if not isinstance(factor, (ChannelFactor, InstrumentFactor)):
                fail("unsupported factor type")
            if not isinstance(factor.id, str) or not factor.id or factor.id in ids:
                fail("factor IDs must be distinct")
            ids.add(factor.id)
            if (
                not factor.targets
                or len(set(factor.targets)) != len(factor.targets)
                or any(name not in lookup for name in factor.targets)
            ):
                fail("factor targets must be distinct declared systems")
            dimension = prod(lookup[name].dimension for name in factor.targets)
            json.dumps(dict(factor.metadata), allow_nan=False)
            if isinstance(factor, ChannelFactor):
                combined = _kraus_stack(factor.kraus)
            else:
                if (
                    not isinstance(factor.record_id, str)
                    or not factor.record_id
                    or (factor.record_id in records or not factor.outcomes)
                ):
                    fail("instruments require distinct record IDs and complete outcomes")
                if any(not isinstance(label, str) or not label for label in factor.outcomes):
                    fail("instrument outcome labels must be nonempty strings")
                groups = [_kraus_stack(values) for values in factor.outcomes.values()]
                if any(group.shape[1:] != (dimension, dimension) for group in groups):
                    fail("instrument operator dimensions disagree with targets")
                combined = _kraus_stack([matrix for group in groups for matrix in group.tolist()])
                records[factor.record_id] = tuple(factor.outcomes)
            if combined.shape[1:] != (dimension, dimension):
                fail("channel operator dimensions disagree with targets")
            trace_preserving(combined)
        detector_ids = set()
        for detector in self.detectors:
            if not isinstance(detector.id, str) or not detector.id or detector.id in detector_ids:
                fail("detector IDs must be distinct")
            detector_ids.add(detector.id)
            if detector.expected_parity is not None and (
                type(detector.expected_parity) is not int or detector.expected_parity not in (0, 1)
            ):
                fail("ideal reference parity must be 0, 1, or explicitly unspecified")
            if any(
                value is not None and (type(value) is not int or value not in (0, 1))
                for value in detector.bit_values.values()
            ):
                fail("detector outcome interpretations must be binary or unavailable")
            for name in detector.records:
                if name not in records:
                    fail("detector refers to an undefined observed record")
                if any(label not in detector.bit_values for label in records[name]):
                    fail("detector must interpret every possible referenced outcome")
        retained = tuple(records) if self.retained_records is None else self.retained_records
        if len(set(retained)) != len(retained) or any(name not in records for name in retained):
            fail("retained records must be distinct observed records")
        json.dumps(dict(self.metadata), allow_nan=False)
        if self.input_isometry is not None:
            _check_isometry(self.input_isometry, self.dimension)
        return tuple(records), tuple(retained)

    def save(self, path: str | Path) -> dict[str, object]:
        """Save portable JSON with complex pairs, without expanding global maps."""
        self.validate()
        arrays, factors = {}, []
        if self.input_isometry is not None:
            arrays["input_isometry"] = encode_array(self.input_isometry)
        for index, factor in enumerate(self.factors):
            row = {"id": factor.id, "targets": list(factor.targets), "metadata": dict(factor.metadata)}
            if isinstance(factor, ChannelFactor):
                name = f"factor_{index}"
                arrays[name] = encode_array(_kraus_stack(factor.kraus))
                row.update(type="channel", array=name)
            else:
                outcomes = {}
                for branch, (label, values) in enumerate(factor.outcomes.items()):
                    name = f"factor_{index}_outcome_{branch}"
                    arrays[name] = encode_array(_kraus_stack(values))
                    outcomes[label] = name
                row.update(type="instrument", record_id=factor.record_id, outcomes=outcomes)
            factors.append(row)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "local_instrument_detector_process",
            "tensor_convention": TENSOR_CONVENTION,
            "systems": _system_rows(self.systems),
            "factors": factors,
            "detectors": [
                {
                    "id": detector.id,
                    "records": list(detector.records),
                    "bit_values": dict(detector.bit_values),
                    "expected_parity": detector.expected_parity,
                }
                for detector in self.detectors
            ],
            "retained_records": self.retained_records,
            "input_isometry": None if self.input_isometry is None else "input_isometry",
            "metadata": dict(self.metadata),
        }
        return _save(path, manifest, arrays)

    @classmethod
    def load(cls, path: str | Path) -> DetectorProcess:
        """Load the experimental JSON format and validate all physical declarations."""
        manifest = _load(path, "local_instrument_detector_process")
        if manifest.get("tensor_convention") != TENSOR_CONVENTION:
            fail("unsupported serialized tensor ordering convention")
        arrays = manifest["arrays"]
        systems = tuple(
            SystemSpec(
                row["id"],
                tuple(row["basis"]),
                {name: tuple(indices) for name, indices in row["sectors"].items()},
                row["role"],
            )
            for row in manifest["systems"]
        )
        factors = []
        for row in manifest["factors"]:
            if row["type"] == "channel":
                factor = ChannelFactor(
                    row["id"],
                    tuple(row["targets"]),
                    decode_array(arrays[row["array"]]),
                    row.get("metadata", {}),
                )
            elif row["type"] == "instrument":
                factor = InstrumentFactor(
                    row["id"],
                    tuple(row["targets"]),
                    {label: decode_array(arrays[name]) for label, name in row["outcomes"].items()},
                    row["record_id"],
                    row.get("metadata", {}),
                )
            else:
                fail("unsupported serialized factor type")
            factors.append(factor)
        detectors = tuple(
            DetectorDef(row["id"], tuple(row["records"]), row["bit_values"], row["expected_parity"])
            for row in manifest["detectors"]
        )
        retained, isometry_key = manifest["retained_records"], manifest["input_isometry"]
        result = cls(
            systems,
            tuple(factors),
            detectors,
            None if retained is None else tuple(retained),
            manifest["metadata"],
            None if isometry_key is None else decode_array(arrays[isometry_key]),
        )
        result.validate()
        return result


@dataclass(frozen=True, init=False)
class KrausMap:
    """Store an unnormalized, completely positive trace-nonincreasing map."""

    _values: pc.Array

    def __init__(self, operators: object) -> None:
        """Snapshot finite operators and validate the trace bound."""
        values = _kraus_stack(operators)
        if not positive_semidefinite(identity(values.shape[2]) - effect(values)):
            fail("a detector-conditioned map must be trace nonincreasing")
        object.__setattr__(self, "_values", values)

    @property
    def operators(self) -> pc.Array:
        """Return a defensive copy of the validated Kraus stack."""
        return array(self._values)

    def apply(self, rho: object) -> pc.Array:
        """Apply without normalization; matrix-basis inputs are also permitted."""
        density = array(rho)
        dimension = self._values.shape[2]
        if density.shape != (dimension, dimension):
            fail("input density matrix has the wrong dimension")
        return sum(
            (operator * density * operator.conj().T for operator in matrices(self._values)),
            zeros(self._values.shape[1], self._values.shape[1]),
        )


class DetectorInstrument:
    """Store detector/record-conditioned CP maps with full physical quantum output.

    Branch keys contain detector values followed by retained observed records.
    Hidden Kraus indices and leakage/vacancy sector labels never become keys.
    """

    def __init__(self, branches: Mapping[BranchKey, KrausMap], metadata: Mapping[str, object] | None = None) -> None:
        """Validate branch dimensions and completeness of the overall instrument."""
        if not branches:
            fail("an instrument must have at least one branch")
        self.branches = MappingProxyType(dict(branches))
        shapes = {value.operators.shape[1:] for value in self.branches.values()}
        if len(shapes) != 1:
            fail("branch map input/output dimensions must agree")
        self.output_dimension, self.input_dimension = shapes.pop()
        trace_preserving(
            _kraus_stack([matrix for value in self.branches.values() for matrix in value.operators.tolist()]),
        )
        self.metadata = {} if metadata is None else dict(metadata)
        json.dumps(self.metadata, allow_nan=False)
        for key in self.branches:
            if not isinstance(key, tuple) or any(value is not None and type(value) not in (int, str) for value in key):
                fail("branch keys must be tuples of detector values/record labels")

    def apply(self, rho: object) -> dict[BranchKey, pc.Array]:
        """Return the unnormalized quantum output for every classical branch."""
        return {key: value.apply(rho) for key, value in self.branches.items()}

    def total(self, rho: object) -> pc.Array:
        """Sum the branch maps, discarding the classical outcome."""
        return sum(self.apply(rho).values(), zeros(self.output_dimension, self.output_dimension))

    def save(self, path: str | Path) -> dict[str, object]:
        """Persist all branch operators as portable non-executable JSON data."""
        arrays, branch_rows = {}, []
        for index, (key, channel) in enumerate(self.branches.items()):
            name = f"branch_{index}"
            arrays[name] = encode_array(channel.operators)
            branch_rows.append({"key": list(key), "array": name})
        return _save(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "detector_conditioned_quantum_instrument",
                "semantics": "sum_a K[d,a] rho K[d,a]^dagger; outputs remain unnormalized",
                "input_dimension": self.input_dimension,
                "output_dimension": self.output_dimension,
                "branches": branch_rows,
                "metadata": self.metadata,
            },
            arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> DetectorInstrument:
        """Load and validate dimension, trace and serialized-array invariants."""
        manifest = _load(path, "detector_conditioned_quantum_instrument")
        branches = {}
        for row in manifest["branches"]:
            key = tuple(row["key"])
            if key in branches:
                fail("duplicate serialized branch keys")
            branches[key] = KrausMap(decode_array(manifest["arrays"][row["array"]]))
        result = cls(branches, manifest["metadata"])
        if (result.input_dimension, result.output_dimension) != (
            manifest["input_dimension"],
            manifest["output_dimension"],
        ):
            fail("serialized instrument dimensions disagree")
        return result


def _system_rows(systems: tuple[SystemSpec, ...]) -> list[dict[str, object]]:
    return [
        {
            "id": system.id,
            "basis": list(system.basis),
            "role": system.role,
            "sectors": {name: list(indices) for name, indices in system.sectors.items()},
        }
        for system in systems
    ]


def _check_isometry(value: object, dimension: int) -> pc.Array:
    initial = array(value)
    if initial.ndim != 2 or initial.shape[0] != dimension or not initial.shape[1]:
        fail("encoding must be a finite physical-by-input isometry")
    if not close(initial.conj().T * initial, identity(initial.shape[1])):
        fail("encoding must preserve the input trace")
    return initial


def _unravel(index: int, dimensions: tuple[int, ...]) -> list[int]:
    digits = []
    for dimension in reversed(dimensions):
        index, digit = divmod(index, dimension)
        digits.append(digit)
    return list(reversed(digits))


def _ravel(digits: list[int], dimensions: tuple[int, ...]) -> int:
    result = 0
    for digit, dimension in zip(digits, dimensions, strict=True):
        result = result * dimension + digit
    return result


def _embed(operator: pc.Array, targets: tuple[int, ...], dimensions: tuple[int, ...]) -> pc.Array:
    dimension = prod(dimensions)
    local_dimensions = tuple(dimensions[index] for index in targets)
    result = [[0j] * dimension for _ in range(dimension)]
    data = operator.tolist()
    for column in range(dimension):
        source = _unravel(column, dimensions)
        local_column = _ravel([source[index] for index in targets], local_dimensions)
        for local_row in range(prod(local_dimensions)):
            coefficient = data[local_row][local_column]
            if coefficient == 0:
                continue
            destination = source.copy()
            for target, value in zip(targets, _unravel(local_row, local_dimensions), strict=True):
                destination[target] = value
            result[_ravel(destination, dimensions)][column] = coefficient
    return array(result)


def compile_instrument(
    process: DetectorProcess,
    encoding: object | None = None,
    *,
    max_paths: int = 8192,
    max_dimension: int = 64,
) -> DetectorInstrument:
    """Materialize a small exact detector instrument with explicit resource limits.

    Coherent alternatives stay inside each Kraus matrix. Unobserved environment
    alternatives and discarded observed histories are concatenated as CP maps,
    never summed as amplitudes. The optional encoding declares the initial
    logical embedding and memory state; output retains all physical sectors.
    """
    records, retained = process.validate()
    if type(max_paths) is not int or max_paths < 1 or type(max_dimension) is not int or max_dimension < 1:
        fail("positive integer reference compiler limits are required")
    dimension = process.dimension
    if dimension > max_dimension:
        fail("exact reference dimension limit exceeded; use a structured backend")
    supplied_encoding = process.input_isometry if encoding is None else encoding
    initial = identity(dimension) if supplied_encoding is None else _check_isometry(supplied_encoding, dimension)
    paths = [({}, initial)]
    dimensions = tuple(system.dimension for system in process.systems)
    positions = {system.id: index for index, system in enumerate(process.systems)}
    for factor in process.factors:
        groups = {None: factor.kraus} if isinstance(factor, ChannelFactor) else factor.outcomes
        number = sum(_kraus_stack(group).shape[0] for group in groups.values())
        if len(paths) * number > max_paths:
            fail("exact reference path limit exceeded; retain the local factor model")
        targets = tuple(positions[name] for name in factor.targets)
        embedded = [
            (label, _embed(operator, targets, dimensions))
            for label, values in groups.items()
            for operator in matrices(_kraus_stack(values))
        ]
        expanded = []
        for record, previous in paths:
            for label, operator in embedded:
                updated = record.copy()
                if isinstance(factor, InstrumentFactor):
                    updated[factor.record_id] = label
                expanded.append((updated, operator * previous))
        paths = expanded
    grouped = {}
    for record, operator in paths:
        key = tuple(detector.evaluate(record) for detector in process.detectors) + tuple(
            record[name] for name in retained
        )
        grouped.setdefault(key, []).append(operator.tolist())
    metadata = {
        "compiler": "exact dense reference; no Pauli twirling or factorization",
        "factor_order": [factor.id for factor in process.factors],
        "factor_provenance": [
            {"id": factor.id, "targets": list(factor.targets), "metadata": dict(factor.metadata)}
            for factor in process.factors
        ],
        "system_order": [system.id for system in process.systems],
        "tensor_convention": TENSOR_CONVENTION,
        "systems": _system_rows(process.systems),
        "detector_order": [detector.id for detector in process.detectors],
        "retained_record_order": list(retained),
        "discarded_record_ids": [name for name in records if name not in retained],
        "record_source": "declared measurement instruments only",
        "output_convention": "retained full physical quantum state; no implicit code projection",
        "encoding_isometry_supplied": supplied_encoding is not None,
        "approximations": [],
        "process_metadata": dict(process.metadata),
    }
    return DetectorInstrument({key: KrausMap(values) for key, values in grouped.items()}, metadata)
