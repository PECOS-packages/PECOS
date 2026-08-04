# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""`DemAwareDecoder` must not wrap observable bits at 64 (issue #430)."""

from __future__ import annotations

import pytest
from pecos.decoders import DemAwareDecoder

# Observable 70 is the probe: a u64 mask would fold it onto bit 70 % 64 == 6.
_WIDE_OBSERVABLE = 70
_WRAPPED_BIT = _WIDE_OBSERVABLE % 64


def _wide_dem(num_observables: int = _WIDE_OBSERVABLE + 1) -> str:
    lines = ["error(0.1) D0 L0", f"error(0.1) D1 L{_WIDE_OBSERVABLE}"]
    lines += [f"detector D{index}" for index in range(2)]
    lines += [f"logical_observable L{index}" for index in range(num_observables)]
    return "\n".join(lines)


@pytest.fixture
def wide_decoder() -> DemAwareDecoder:
    return DemAwareDecoder.from_dem(_wide_dem(), decoder_type="bp_osd")


def test_observable_past_64_sets_its_own_bit(wide_decoder: DemAwareDecoder) -> None:
    mask = wide_decoder.decode_syndrome([0, 1]).observables_mask

    assert mask >> _WIDE_OBSERVABLE & 1, "observable 70 must set bit 70"
    assert not mask >> _WRAPPED_BIT & 1, "observable 70 must not wrap onto bit 6"
    assert mask == 1 << _WIDE_OBSERVABLE


def test_narrow_observables_are_unchanged(wide_decoder: DemAwareDecoder) -> None:
    # Values that fit in 64 bits must stay exactly what the previous u64
    # field held, so widening is not a behavior change for existing users.
    assert wide_decoder.decode_syndrome([1, 0]).observables_mask == 1


def test_repr_reports_wide_masks(wide_decoder: DemAwareDecoder) -> None:
    text = repr(wide_decoder.decode_syndrome([0, 1]))

    assert "DemAwareResult(" in text
    assert str(_WIDE_OBSERVABLE) in text
