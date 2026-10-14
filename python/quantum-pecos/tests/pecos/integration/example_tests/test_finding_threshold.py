# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Integration tests for quantum error correction threshold finding."""

from math import isfinite

import pecos as pc
import pytest
from pecos.analysis.threshold_curve import func


@pytest.mark.slow
def test_finding_threshold() -> None:
    """Test threshold finding for quantum error correction codes."""
    depolar = pc.noise.DepolarModel(
        model_level="code_capacity",
        perp_errors=True,
    )
    ps = [0.19, 0.17, 0.15, 0.13, 0.11]
    ds = [5, 7, 9]
    plist = pc.array(ps * len(ds))

    dlist = [d for d in ds for _ in ps]
    dlist = pc.array(dlist)

    plog = []
    for d in ds:
        surface = pc.qeccs.Surface4444(distance=d)
        mwpm2d = pc.decoders.MWPM2D(surface)
        for i, p in enumerate(ps):
            plog.append(
                pc.analysis.codecapacity_logical_rate(
                    250,
                    surface,
                    d,
                    depolar,
                    error_params={"p": p},
                    decoder=mwpm2d,
                    seed=101 + 100 * d + i,
                    verbose=False,
                )[0],
            )

    plog = pc.array(plog)

    p0 = (0.155, 1.5, 0.15, 1, 1)
    opt, std = pc.analysis.threshold_fit(plist, dlist, plog, func, p0, maxfev=10000)

    # Conventional matching has a depolarizing code-capacity threshold of about 15.5%:
    # https://arxiv.org/html/2212.11632v3#S2 (nonrotated planar code).
    # With d=5,7,9 and 250 shots the fitted value varies with the seed: twelve seed offsets
    # gave 0.10 to 0.18 (mean 0.144, sd 0.021), so the band is about three sd wide each way.
    # A decoder that does not decode leaves no crossing, and the fit then fails to converge.
    assert all(isfinite(value) for value in opt)
    assert all(isfinite(value) for value in std)
    assert 0.08 < opt[0] < 0.22
