from __future__ import annotations

import pytest

pytest.importorskip("guppylang")

from pecos.guppy.surface import (
    _guppy_module_cache_key,
    generate_guppy_source,
    generate_memory_experiment,
)
from pecos.qec.surface import GuppyRngMaskConfig, TwirlConfig
from pecos.qec.surface._twirl_sites import pauli_mask_round_tag
from pecos.qec.surface.patch import SurfacePatch


@pytest.fixture
def patch() -> SurfacePatch:
    return SurfacePatch.create(distance=3)


def test_no_twirl_source_has_no_rng_or_mask_tags(patch: SurfacePatch) -> None:
    src = generate_guppy_source(patch)

    assert "RNG(" not in src
    assert "random_int_bounded" not in src
    assert "seeded_pcg32_with_quantum_entropy(" not in src
    assert 'result("pauli_mask' not in src
    assert "for _t in range(comptime(num_rounds)):" in src
    assert 'result("final"' in src


def test_twirl_source_unrolls_rng_masks_and_runtime_paulis(patch: SurfacePatch) -> None:
    src = generate_guppy_source(
        patch,
        twirl=TwirlConfig(),
        rng=GuppyRngMaskConfig(seed=42),
        num_rounds=3,
    )

    assert "def _pcg32_next4(state: nat, inc: nat) -> tuple[nat, int]:" in src
    assert "def seeded_pcg32_with_quantum_entropy(seed: int) -> tuple[nat, nat]:" in src
    assert "entropy_q = qubit()" in src
    assert "if measure(entropy_q):" in src
    assert src.count("rng_state, rng_inc = seeded_pcg32_with_quantum_entropy(42)") == 2
    assert src.count('result("frame_mode:raw", True)') == 2
    assert "rng_state, m_0_0 = _pcg32_next4(rng_state, rng_inc)" in src
    assert "if m_0_0 == 1:" in src
    assert "    x(surf.data[0])" in src
    assert "if m_0_0 == 2:" in src
    assert "    y(surf.data[0])" in src
    assert "if m_0_0 == 3:" in src
    assert "    z(surf.data[0])" in src

    for r in range(2):
        assert src.count(f'result("{pauli_mask_round_tag(r)}"') == 2
    assert src.count("# === Round 2 (final, no twirl after) ===") == 2


def test_canonical_frame_output_emits_raw_sibling_tags(patch: SurfacePatch) -> None:
    src = generate_guppy_source(
        patch,
        twirl=TwirlConfig(frame_output="canonical"),
        rng=GuppyRngMaskConfig(seed=7),
        num_rounds=2,
    )

    assert 'result("frame_mode:canonical", True)' in src
    assert "fx_0 = False" in src
    assert "fz_0 = False" in src
    assert "sx0_raw = measure(ax0)" in src
    assert "sx0 = sx0_raw != sx0_flip" in src
    assert 'result("raw:sx0:bit:0", sx0_raw)' in src
    assert 'result("raw:final", final_raw)' in src
    assert 'result("final", array(final_0' in src


def test_twirl_validation_requires_rng_and_num_rounds(patch: SurfacePatch) -> None:
    with pytest.raises(ValueError, match="twirl and rng must be supplied together"):
        generate_guppy_source(patch, twirl=TwirlConfig(), num_rounds=2)
    with pytest.raises(ValueError, match="twirl and rng must be supplied together"):
        generate_guppy_source(patch, rng=GuppyRngMaskConfig(seed=42))
    with pytest.raises(ValueError, match="num_rounds is required when twirl is supplied"):
        generate_guppy_source(
            patch,
            twirl=TwirlConfig(),
            rng=GuppyRngMaskConfig(seed=42),
        )


def test_twirled_cache_key_includes_seed_rounds_and_frame_mode(patch: SurfacePatch) -> None:
    raw = _guppy_module_cache_key(
        patch,
        effective_budget=10,
        twirl=TwirlConfig(),
        rng=GuppyRngMaskConfig(seed=1),
        num_rounds=2,
    )
    raw_seed2 = _guppy_module_cache_key(
        patch,
        effective_budget=10,
        twirl=TwirlConfig(),
        rng=GuppyRngMaskConfig(seed=2),
        num_rounds=2,
    )
    canonical = _guppy_module_cache_key(
        patch,
        effective_budget=10,
        twirl=TwirlConfig(frame_output="canonical"),
        rng=GuppyRngMaskConfig(seed=1),
        num_rounds=2,
    )
    round3 = _guppy_module_cache_key(
        patch,
        effective_budget=10,
        twirl=TwirlConfig(),
        rng=GuppyRngMaskConfig(seed=1),
        num_rounds=3,
    )

    assert raw != _guppy_module_cache_key(patch, effective_budget=10)
    assert raw != raw_seed2
    assert raw != canonical
    assert raw != round3
    assert "s1" in raw
    assert "s2" in raw_seed2
    assert "frame-raw" in raw
    assert "frame-canonical" in canonical


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("frame_output", ["raw", "canonical"])
def test_twirled_memory_experiment_compiles(
    patch: SurfacePatch,
    basis: str,
    frame_output: str,
) -> None:
    fn = generate_memory_experiment(
        patch,
        num_rounds=2,
        basis=basis,
        twirl=TwirlConfig(frame_output=frame_output),
        rng=GuppyRngMaskConfig(seed=7),
    )
    assert fn is not None
