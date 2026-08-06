# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0

"""VerifyStabilizers emits a DeprecationWarning pointing at the replacement."""

import pytest


def test_verify_stabilizers_warns_and_still_works() -> None:
    from pecos.analysis import VerifyStabilizers

    with pytest.warns(DeprecationWarning, match="StabilizerCodeSpec"):
        qecc = VerifyStabilizers()

    # The deprecated workflow must keep functioning during the cycle.
    qecc.check("Z", (0, 1))
    qecc.check("Z", (1, 2))
    qecc.compile()
