"""Parity checks between PECOS sim(...).classical(selene_engine()) and direct selene_sim.

These tests are intentionally focused on small Guppy surface-memory programs so we can
quickly detect drift between the PECOS QIS/Helios integration path and direct Selene.

For full surface-memory experiments, exact noiseless shot-by-shot raw-register parity is
not a valid target: the generated programs start from simple product states, so the
complementary stabilizer family is genuinely random in the first noiseless round and then
repeats after projection. The tests below therefore compare the deterministic pieces that
should match exactly and the qualitative round-to-round behavior that both backends should
share.
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import cx, h, measure, measure_array, qubit, x


@guppy
def tagged_bits_named_array() -> None:
    """Tiny named-array program to isolate raw result parity issues."""
    q0 = qubit()
    q1 = qubit()
    q2 = qubit()
    x(q0)
    x(q2)
    final = measure_array(array(q0, q1, q2))
    result("final", final)


def make_repeated_single_bit_results(num_rounds: int):
    """Create a tiny program that records the same named result repeatedly."""

    @guppy
    def repeated_single_bit_results() -> None:
        for _ in range(comptime(num_rounds)):
            q = qubit()
            bit = measure(q)
            result("synx", array(bit))

    return repeated_single_bit_results


def make_tiny_x_syndrome_memory(num_rounds: int):
    """Create a tiny memory-style circuit with fresh ancilla allocation each round."""

    @guppy
    def tiny_x_syndrome_memory() -> None:
        data = qubit()
        h(data)

        for _ in range(comptime(num_rounds)):
            anc = qubit()
            h(anc)
            cx(anc, data)
            h(anc)
            bit = measure(anc)
            result("synx", array(bit))

        h(data)
        final = measure_array(array(data))
        result("final", final)

    return tiny_x_syndrome_memory


def make_tiny_x_syndrome_memory_raw(num_rounds: int):
    """Create the same tiny circuit but without named outputs.

    This helps us distinguish "raw measured bits are wrong" from
    "named result collection is wrong".
    """

    @guppy
    def tiny_x_syndrome_memory_raw() -> None:
        data = qubit()
        h(data)

        for _ in range(comptime(num_rounds)):
            anc = qubit()
            h(anc)
            cx(anc, data)
            h(anc)
            _ = measure(anc)

        h(data)
        _ = measure(data)

    return tiny_x_syndrome_memory_raw


@guppy
def alloc_reuse_probe() -> None:
    """Measure |1>, then allocate again and verify the fresh qubit is |0>."""
    q = qubit()
    x(q)
    b1 = measure(q)
    result("m1", array(b1))

    q2 = qubit()
    b2 = measure(q2)
    result("m2", array(b2))


def _require_selene_runtime() -> Any:
    import pecos

    try:
        return pecos.selene_engine()
    except RuntimeError as exc:
        if "Failed to load Selene runtime" in str(exc):
            pytest.skip("Selene runtime not available in this test environment")
        raise


def _configure_selene_caches() -> None:
    tmpdir = Path(tempfile.gettempdir())
    os.environ.setdefault("ZIG_GLOBAL_CACHE_DIR", str(tmpdir / "pecos_zig_global_cache"))
    os.environ.setdefault("ZIG_LOCAL_CACHE_DIR", str(tmpdir / "pecos_zig_local_cache"))


def _collect_selene_named_results(instance: Any, *, n_qubits: int, n_shots: int, p: float, seed: int):
    from selene_sim import DepolarizingErrorModel, SimpleRuntime, Stim

    results: dict[str, list[int] | list[list[int]]] = defaultdict(list)
    try:
        for shot_results in instance.run_shots(
            simulator=Stim(),
            n_qubits=n_qubits,
            n_shots=n_shots,
            error_model=DepolarizingErrorModel(
                p_1q=p,
                p_2q=p,
                p_meas=p,
                p_init=p,
            ),
            runtime=SimpleRuntime(),
            random_seed=seed,
            n_processes=1,
        ):
            shot_rows: dict[str, list[int]] = defaultdict(list)
            for name, values in shot_results:
                shot_rows[name].extend(int(v) for v in values)
            for name, values in shot_rows.items():
                # Match ShotMap.to_dict(): one-bit registers become a flat list across
                # shots, while vector-valued registers remain nested by shot.
                if len(values) == 1:
                    results[name].append(values[0])
                else:
                    results[name].append(values)
    finally:
        delete_files = getattr(instance, "delete_files", None)
        if callable(delete_files):
            try:
                delete_files()
            except Exception:
                pass

    return dict(results)


def _round_blocks_repeat(row: list[int], num_rounds: int) -> bool:
    if len(row) % num_rounds != 0:
        return False
    block_size = len(row) // num_rounds
    blocks = [
        tuple(row[round_idx * block_size : (round_idx + 1) * block_size])
        for round_idx in range(num_rounds)
    ]
    return all(block == blocks[0] for block in blocks[1:])


def _run_surface_memory_via_sim(
    *,
    distance: int,
    num_rounds: int,
    basis: str,
    shots: int,
    p: float,
    seed: int,
) -> dict[str, list[list[int]]]:
    import pecos
    from pecos.guppy import get_num_qubits, make_surface_code

    _require_selene_runtime()

    program = make_surface_code(distance=distance, num_rounds=num_rounds, basis=basis)
    noise_model = pecos.depolarizing_noise().with_uniform_probability(p)
    return (
        pecos.sim(program)
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(get_num_qubits(distance))
        .noise(noise_model)
        .seed(seed)
        .run(shots)
        .to_shot_map()
        .to_dict()
    )


def _run_surface_memory_via_selene_sim(
    *,
    distance: int,
    num_rounds: int,
    basis: str,
    shots: int,
    p: float,
    seed: int,
) -> dict[str, list[int] | list[list[int]]]:
    pytest.importorskip("selene_sim")

    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from pecos.guppy import get_num_qubits, make_surface_code
    from selene_sim import build

    _configure_selene_caches()

    program = make_surface_code(distance=distance, num_rounds=num_rounds, basis=basis)
    hugr_bytes = compile_guppy_to_hugr(program)
    instance = build(hugr_bytes, name=f"surface_d{distance}_{basis.lower()}_r{num_rounds}")

    return _collect_selene_named_results(
        instance,
        n_qubits=get_num_qubits(distance),
        n_shots=shots,
        p=p,
        seed=seed,
    )


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_surface_memory_selene_backends_return_same_register_shapes(basis: str) -> None:
    """Both gate-level Selene-backed entry points should at least agree on output shape."""
    sim_results = _run_surface_memory_via_sim(
        distance=3,
        num_rounds=2,
        basis=basis,
        shots=2,
        p=0.0,
        seed=123,
    )
    selene_results = _run_surface_memory_via_selene_sim(
        distance=3,
        num_rounds=2,
        basis=basis,
        shots=2,
        p=0.0,
        seed=123,
    )

    assert set(sim_results) == {"final", "synx", "synz"}
    assert set(selene_results) == {"final", "synx", "synz"}

    for key in ("final", "synx", "synz"):
        assert len(sim_results[key]) == 2
        assert len(selene_results[key]) == 2
        assert len(sim_results[key][0]) == len(selene_results[key][0])


def test_named_bool_array_matches_between_selene_backends() -> None:
    """A simple named bool-array program should agree exactly across gate backends."""
    import pecos
    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from selene_sim import build

    _require_selene_runtime()
    _configure_selene_caches()

    sim_results = (
        pecos.sim(pecos.Guppy(tagged_bits_named_array))
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(3)
        .noise(pecos.depolarizing_noise().with_uniform_probability(0.0))
        .seed(123)
        .run(1)
        .to_shot_map()
        .to_dict()
    )

    instance = build(compile_guppy_to_hugr(tagged_bits_named_array), name="tagged_bits_named_array")
    selene_results = _collect_selene_named_results(instance, n_qubits=3, n_shots=1, p=0.0, seed=123)

    assert sim_results == selene_results == {"final": [[1, 0, 1]]}


def test_repeated_named_bool_array_matches_between_selene_backends() -> None:
    """Repeated named outputs should accumulate identically across gate backends."""
    import pecos
    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from selene_sim import build

    _require_selene_runtime()
    _configure_selene_caches()

    program = make_repeated_single_bit_results(3)

    sim_results = (
        pecos.sim(pecos.Guppy(program))
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(1)
        .noise(pecos.depolarizing_noise().with_uniform_probability(0.0))
        .seed(123)
        .run(1)
        .to_shot_map()
        .to_dict()
    )

    instance = build(compile_guppy_to_hugr(program), name="repeated_single_bit_results")
    selene_results = _collect_selene_named_results(instance, n_qubits=1, n_shots=1, p=0.0, seed=123)

    assert sim_results == selene_results == {"synx": [[0, 0, 0]]}


def test_tiny_syndrome_memory_matches_between_selene_backends() -> None:
    """A tiny syndrome-extraction pattern should match before we blame the full surface program."""
    import pecos
    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from selene_sim import build

    _require_selene_runtime()
    _configure_selene_caches()

    program = make_tiny_x_syndrome_memory(2)

    sim_results = (
        pecos.sim(pecos.Guppy(program))
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(2)
        .noise(pecos.depolarizing_noise().with_uniform_probability(0.0))
        .seed(123)
        .run(1)
        .to_shot_map()
        .to_dict()
    )

    instance = build(compile_guppy_to_hugr(program), name="tiny_x_syndrome_memory")
    selene_results = _collect_selene_named_results(instance, n_qubits=2, n_shots=1, p=0.0, seed=123)

    assert sim_results == selene_results


def test_alloc_reuse_probe_matches_between_selene_backends() -> None:
    """Fresh allocation after a `|1>` measurement should still return `|0>`."""
    import pecos
    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from selene_sim import build

    _require_selene_runtime()
    _configure_selene_caches()

    sim_results = (
        pecos.sim(pecos.Guppy(alloc_reuse_probe))
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(1)
        .noise(pecos.depolarizing_noise().with_uniform_probability(0.0))
        .seed(123)
        .run(1)
        .to_shot_map()
        .to_dict()
    )

    instance = build(compile_guppy_to_hugr(alloc_reuse_probe), name="alloc_reuse_probe")
    selene_results = _collect_selene_named_results(instance, n_qubits=1, n_shots=1, p=0.0, seed=123)

    assert sim_results == selene_results == {"m1": [1], "m2": [0]}


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_surface_memory_selene_backends_match_noiseless_deterministic_family(basis: str) -> None:
    """The stabilizer family commuting with the prepared basis should stay all-zero."""
    sim_results = _run_surface_memory_via_sim(
        distance=3,
        num_rounds=6,
        basis=basis,
        shots=2,
        p=0.0,
        seed=123,
    )
    selene_results = _run_surface_memory_via_selene_sim(
        distance=3,
        num_rounds=6,
        basis=basis,
        shots=2,
        p=0.0,
        seed=123,
    )

    det_key = "synx" if basis == "X" else "synz"
    assert sim_results[det_key] == selene_results[det_key]
    for row in sim_results[det_key]:
        assert all(bit == 0 for bit in row)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_surface_memory_noiseless_complementary_family_repeats_after_projection(basis: str) -> None:
    """After the first noiseless round, repeated rounds should replay the same syndrome block."""
    num_rounds = 6
    comp_key = "synz" if basis == "X" else "synx"

    for runner in (_run_surface_memory_via_sim, _run_surface_memory_via_selene_sim):
        results = runner(
            distance=3,
            num_rounds=num_rounds,
            basis=basis,
            shots=5,
            p=0.0,
            seed=123,
        )
        for row in results[comp_key]:
            assert _round_blocks_repeat(row, num_rounds)
