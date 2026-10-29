# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Shared C4-Helix fixtures."""

import pytest
from pecos.qec.helix import c4_helix_code, c4_helix_gates, min_cost_words, symplectic_action


@pytest.fixture(scope="module")
def compiled():
    matrices = {name: symplectic_action(c4_helix_code(), gate) for name, gate in c4_helix_gates().items()}
    costs = {name: int(name == "S0S1") for name in matrices}
    return matrices, min_cost_words(matrices, costs)
