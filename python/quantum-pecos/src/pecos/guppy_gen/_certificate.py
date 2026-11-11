# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Certify generated programs and their named output-slot inventories."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pecos.qec.measurement_layout import abstract_result_refs, named_result_slots

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
    """Bind a generated program to its output slots, in informational abstract order.

    The v2 digest covers HUGR and layout. The certificate admits the generator's
    static schedule and supplies counts and required tags; measurement identity
    is resolved from runtime provenance or compiled scalar analysis. Per-tag
    occurrences assume result() calls retain their abstract source order.
    """
    # Imported here so that `import pecos` does not load guppylang.
    from pecos._compilation import guppy_to_hugr

    certified_layout = named_result_slots(abstract_result_refs(abstract_tc, new_classifier, family=family))
    layout_json = json.dumps(certified_layout, separators=(",", ":"))
    digest = hashlib.sha256(guppy_to_hugr(program) + b"\0" + layout_json.encode()).hexdigest()
    object.__setattr__(
        program,
        "__pecos_named_measurement_layout_v2__",
        (digest, certified_layout),
    )
