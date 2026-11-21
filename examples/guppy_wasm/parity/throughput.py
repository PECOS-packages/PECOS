"""Public dependent-call workloads for separating setup cost from call scaling."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).resolve().parent.parent / "counter.wat"


@wasm_module(str(WASM_PATH))
class Counter:
    @wasm
    def increment(self) -> int: ...


@guppy
def calls_100() -> None:
    counter = Counter(nat(0))
    value = 0
    for _ in range(100):
        value = counter.increment()
    counter.discard()
    result("value", value)


@guppy
def calls_1000() -> None:
    counter = Counter(nat(0))
    value = 0
    for _ in range(1000):
        value = counter.increment()
    counter.discard()
    result("value", value)
