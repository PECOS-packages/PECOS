# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Certify generated surface programs against their abstract measurement order."""

import hashlib
import json
from typing import TYPE_CHECKING

from pecos._compilation import guppy_to_hugr
from pecos.qec.surface.decode import _surface_abstract_measurement_result_refs

if TYPE_CHECKING:
    from pecos_rslib.quantum import TickCircuit


def certify_surface_measurement_layout(program: object, abstract_tc: "TickCircuit") -> None:
    """Bind a generated surface program to its abstract measurement order."""
    occurrence_by_tag: dict[str, int] = {}
    layout: list[tuple[str, int]] = []
    for ref in _surface_abstract_measurement_result_refs(abstract_tc):
        if ref[0] == "scalar":
            _, tag = ref
            occurrence = occurrence_by_tag.get(tag, 0)
            occurrence_by_tag[tag] = occurrence + 1
            layout.append((tag, occurrence))
        else:
            _, tag, element = ref
            layout.append((f"{tag}:meas:{element}", 0))
    certified_layout = tuple(layout)
    layout_json = json.dumps(certified_layout, separators=(",", ":"))
    digest = hashlib.sha256(guppy_to_hugr(program) + b"\0" + layout_json.encode()).hexdigest()
    object.__setattr__(
        program,
        "__pecos_named_measurement_layout_v2__",
        (digest, certified_layout),
    )
