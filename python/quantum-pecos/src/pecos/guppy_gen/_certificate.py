# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Certify generated programs against their abstract measurement order."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pecos.qec.measurement_layout import abstract_result_refs

if TYPE_CHECKING:
    from collections.abc import Callable

    from pecos_rslib.quantum import TickCircuit

    from pecos.qec.measurement_layout import ResultRef


def certify_measurement_layout(
    program: object,
    abstract_tc: TickCircuit,
    new_classifier: Callable[[], Callable[[str, Callable[[str], object]], ResultRef]],
    *,
    family: str,
) -> None:
    """Bind a generated program to its abstract measurement order."""
    # Imported here so that `import pecos` does not load guppylang.
    from pecos._compilation import guppy_to_hugr

    occurrence_by_tag: dict[str, int] = {}
    layout: list[tuple[str, int]] = []
    for ref in abstract_result_refs(abstract_tc, new_classifier, family=family):
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
