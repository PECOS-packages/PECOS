"""Immutable compiler-owned LLVM control snapshots, separate from declared context.

A snapshot certifies control execution in the admitted compiler subset, not an
adaptive DEM or attribution of individual deferred physical operations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pecos.qec.runtime_decode_context import _array, _object, _reject_constant, _uint, _unique_object

if TYPE_CHECKING:
    from pecos.qec.runtime_decoder_batches import RuntimeDecodeBatchPrefix


@dataclass(frozen=True, kw_only=True)
class RuntimeControlSite:
    """Compiler-generated LLVM site; dependencies are static possible input sites."""

    site_id: int
    block: int
    kind: str
    scalar_type: str
    input_dependencies: tuple[int, ...]

    def __post_init__(self) -> None:
        """Freeze caller-supplied sequences."""
        object.__setattr__(self, "input_dependencies", tuple(self.input_dependencies))


@dataclass(frozen=True, kw_only=True)
class RuntimeControlManifest:
    """Identity and CFG of the admitted module; contains no caller source labels."""

    module_sha256: str
    instrumenter: str
    sites: tuple[RuntimeControlSite, ...]
    successors: tuple[tuple[int, ...], ...]
    loop_members: tuple[tuple[int, ...], ...]
    loop_headers: tuple[int, ...]

    def __post_init__(self) -> None:
        """Freeze caller-supplied sequences."""
        for name in ("sites", "loop_headers"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("successors", "loop_members"):
            object.__setattr__(self, name, tuple(tuple(row) for row in getattr(self, name)))


@dataclass(frozen=True, kw_only=True)
class RuntimeControlEvent:
    """An executed compiler site and its scalar value, with shot-local occurrence."""

    site_id: int
    occurrence: int
    value: int


@dataclass(frozen=True, kw_only=True)
class RuntimeControlLoop:
    """Active natural loop: entry occurrence and completed-backedge count."""

    header: int
    occurrence: int
    iteration: int


@dataclass(frozen=True, kw_only=True)
class RuntimeControlSnapshot:
    """Control interval joined to the decoder's emitted-batch watermarks."""

    manifest: RuntimeControlManifest
    decode_site: int
    events: tuple[RuntimeControlEvent, ...]
    active_loops: tuple[RuntimeControlLoop, ...]
    runtime_shot_id: int
    first_batch_index: int
    next_batch_index: int

    def __post_init__(self) -> None:
        """Freeze caller-supplied sequences."""
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "active_loops", tuple(self.active_loops))

    @classmethod
    def from_json(cls, data: bytes, prefix: RuntimeDecodeBatchPrefix) -> RuntimeControlSnapshot:
        """Validate and freeze the native bridge's compiler snapshot."""
        if type(data) is not bytes or not 0 < len(data) <= 4 * 1024 * 1024:
            msg = "control snapshot must contain 1..4194304 bytes"
            raise ValueError(msg)
        try:
            raw = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            return _snapshot(raw, prefix)
        except (TypeError, ValueError, KeyError, RecursionError) as error:
            msg = f"invalid compiler control snapshot: {error}"
            raise ValueError(msg) from error


def _fields(raw: object, names: str) -> dict:
    value = _object(raw)
    if set(value) != set(names.split()):
        msg = "unexpected fields"
        raise ValueError(msg)
    return value


def _index(value: int, length: int) -> int:
    _uint(value, "index")
    if value >= length:
        msg = "index outside compiler table"
        raise ValueError(msg)
    return value


def _indices(raw: object, length: int) -> tuple[int, ...]:
    values = tuple(_index(i, length) for i in _array(raw))
    if len(set(values)) != len(values):
        msg = "duplicate compiler identity"
        raise ValueError(msg)
    return values


def _snapshot(raw: object, prefix: RuntimeDecodeBatchPrefix) -> RuntimeControlSnapshot:
    raw = _fields(
        raw,
        "version manifest decode_site events active_loops runtime_shot_id first_batch_index next_batch_index",
    )
    version = raw.pop("version")
    if type(version) is not int or version != 1:
        msg = "invalid control version"
        raise ValueError(msg)
    # Version is checked before dropping it; bool never aliases version one.
    for key in ("runtime_shot_id", "first_batch_index", "next_batch_index"):
        _uint(raw[key], key)
        if raw[key] != getattr(prefix, key):
            msg = "control/batch interval mismatch"
            raise ValueError(msg)
    manifest = _fields(raw["manifest"], "module_sha256 instrumenter sites successors loop_members loop_headers")
    if (
        type(manifest["module_sha256"]) is not str
        or re.fullmatch("[0-9a-f]{64}", manifest["module_sha256"]) is None
        or manifest["instrumenter"] != "pecos-qis-control-v1"
    ):
        msg = "unsupported compiler identity"
        raise ValueError(msg)
    edges = _array(manifest["successors"])
    sites = _array(manifest["sites"])
    if not 0 < len(edges) <= 256 or not 0 < len(sites) <= 4096:
        msg = "compiler table limit exceeded"
        raise ValueError(msg)
    successors = []
    for raw_row in edges:
        row = _array(raw_row)
        if len(row) > 2:
            msg = "unsupported successor arity"
            raise ValueError(msg)
        successors.append(tuple(_index(i, len(edges)) for i in row))
    parsed_sites = []
    kinds = {
        "block": "u64",
        "shot": "u64",
        "branch": "bool",
        "read_bool": "bool",
        "decode": "u64",
        "decoder_frame": "u64",
    }
    for index, raw_item in enumerate(sites):
        item = _fields(raw_item, "site_id block kind scalar_type input_dependencies")
        if _index(item["site_id"], len(sites)) != index or kinds.get(item["kind"]) != item["scalar_type"]:
            msg = "invalid compiler site"
            raise ValueError(msg)
        _index(item["block"], len(edges))
        item["input_dependencies"] = _indices(item["input_dependencies"], len(sites))
        parsed_sites.append(RuntimeControlSite(**item))
    for item in parsed_sites:
        if any(parsed_sites[i].kind not in ("shot", "read_bool", "decoder_frame") for i in item.input_dependencies):
            msg = "dependency is not a compiler input"
            raise ValueError(msg)
    headers = _indices(manifest["loop_headers"], len(edges))
    members = tuple(_indices(row, len(edges)) for row in _array(manifest["loop_members"]))
    if len(headers) != len(members) or any(h not in m for h, m in zip(headers, members, strict=True)):
        msg = "invalid compiler loops"
        raise ValueError(msg)
    manifest.update(sites=tuple(parsed_sites), successors=tuple(successors), loop_headers=headers, loop_members=members)
    raw["manifest"] = RuntimeControlManifest(**manifest)
    decode_site = _index(raw["decode_site"], len(sites))
    if parsed_sites[decode_site].kind != "decode":
        msg = "expected compiler decode site"
        raise ValueError(msg)
    events = _array(raw["events"])
    if not events or len(events) > 4096:
        msg = "invalid control event interval"
        raise ValueError(msg)
    parsed = []
    for raw_event in events:
        event = _fields(raw_event, "site_id occurrence value")
        site = parsed_sites[_index(event["site_id"], len(sites))]
        _uint(event["occurrence"], "occurrence")
        _uint(event["value"], "value")
        if (site.scalar_type == "bool" and event["value"] > 1) or (
            site.kind in ("block", "decode") and event["value"] != 0
        ):
            msg = "invalid control value"
            raise ValueError(msg)
        parsed.append(RuntimeControlEvent(**event))
    if parsed[-1].site_id != decode_site:
        msg = "control interval must end at its decode site"
        raise ValueError(msg)
    raw["events"] = tuple(parsed)
    loops = []
    for raw_loop in _array(raw["active_loops"]):
        loop = _fields(raw_loop, "header occurrence iteration")
        if _index(loop["header"], len(edges)) not in headers:
            msg = "unknown loop header"
            raise ValueError(msg)
        _uint(loop["occurrence"], "loop occurrence")
        _uint(loop["iteration"], "loop iteration")
        loops.append(RuntimeControlLoop(**loop))
    if len({loop.header for loop in loops}) != len(loops):
        msg = "duplicate active loop"
        raise ValueError(msg)
    raw["active_loops"] = tuple(loops)
    return RuntimeControlSnapshot(**raw)
