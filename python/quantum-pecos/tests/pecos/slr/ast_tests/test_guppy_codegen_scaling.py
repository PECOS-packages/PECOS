# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Large Guppy programs must not copy the slot-name set for each qubit."""

from time import process_time
from timeit import Timer

from pecos.slr import CReg, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit as qb


def test_large_register_codegen_scales_linearly() -> None:
    # Warm up lazy imports before timing. CPU time excludes CI scheduling delays;
    # timeit disables cyclic GC during each sample. A 4x size step gives about 4x
    # for linear emission and well over 8x for the quadratic set-copy regression.
    SlrConverter(Main()).guppy()
    timings = []
    for size in (16_000, 64_000):
        q, c = QReg("q", size), CReg("c", size)
        program = Main(q, c, qb.Measure(q) > c)
        timer = Timer(lambda program=program: SlrConverter(program).guppy(), timer=process_time)
        timings.append(timer.timeit(number=1))
    ratio = timings[1] / timings[0]
    assert ratio < 8.0, f"Quadrupling register width took {ratio:.2f}x CPU time: {timings}"
