# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""QEC protocol geometry and abstractions.

This module provides geometry structures for quantum error correction
protocols like magic state distillation.
"""

from pecos.qec.protocols.msd import (
    InnerCodeGeometry,
    MSDProtocol,
    OuterCodeGeometry,
    create_msd_protocol,
)

__all__ = [
    "InnerCodeGeometry",
    "MSDProtocol",
    "OuterCodeGeometry",
    "create_msd_protocol",
]
