"""A non-WASM runtime panic must not contaminate a later batch."""

from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.builtins import nat, panic, result
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).resolve().parents[4] / "examples/guppy_wasm/worker_state.wat"


@wasm_module(str(WASM_PATH))
class WorkerFailureCounter:
    @wasm
    def increment(self) -> int: ...


@guppy
def abort_after_mutation() -> None:
    module = WorkerFailureCounter(nat(0))
    value = module.increment()
    module.discard()
    result("count", value)
    if value == 1:
        panic("worker-state abort", 1)


@pytest.mark.parametrize("backend", ["selene", "pecos"])
@pytest.mark.parametrize("async_workers", [0, 2])
def test_runtime_panic_after_wasm_mutation_leaves_fresh_batches(backend, tmp_path, async_workers):
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
    if backend == "selene":
        from pecos_selene_wasm import build
        from selene_sim import Quest
        from selene_sim.exceptions import SelenePanicError

        runner = build(
            abort_after_mutation,
            modules,
            build_dir=tmp_path,
            wasm_state_scope="worker",
            wasm_async_workers=async_workers,
        )
        for _ in range(2):
            with pytest.raises(SelenePanicError, match="worker-state abort"):
                [dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=2)]
    else:
        pc = pytest.importorskip("pecos")
        runner = (
            pc.sim(
                pc.Guppy(
                    abort_after_mutation,
                    wasm_modules=modules,
                    wasm_state_scope="worker",
                    wasm_async_workers=async_workers,
                ),
            )
            .qubits(1)
            .quantum(pc.state_vector())
            .workers(1)
            .build()
        )
        for _ in range(2):
            with pytest.raises(RuntimeError, match="worker-state abort"):
                runner.run(2)
