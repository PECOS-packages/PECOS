"""WASM contexts retain state within a shot and start fresh on each new shot."""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).with_name("counter.wat")


@wasm_module(str(WASM_PATH))
class Counter:
    @wasm
    def increment(self) -> int: ...

    @wasm
    def clear(self) -> None: ...

    @wasm
    def trap(self) -> None: ...


@guppy
def stateful_calls() -> None:
    counter = Counter(nat(0))
    counter.increment()
    counter.increment()
    value = counter.increment()
    result("three", value)
    counter.discard()

    counter = Counter(nat(0))
    result("four", counter.increment())
    other = Counter(nat(1))
    result("independent", other.increment())
    other.discard()

    counter.clear()
    result("cleared", counter.increment())
    counter.discard()


@guppy
def stateful_loop() -> None:
    """Currently rejected because of a public compiler external-call loop bug."""
    counter = Counter(nat(0))
    value = 0
    for _ in range(3):
        value = counter.increment()
    result("three", value)
    counter.discard()


@guppy
def increment_context() -> int:
    counter = Counter(nat(0))
    value = counter.increment()
    counter.discard()
    return value


@guppy
def context_helpers() -> None:
    """Order leases across helper calls and conditional regions."""
    first = increment_context()
    second = increment_context() if first == 1 else 0
    result("second", second)
    result("third", increment_context())


@guppy
def trapped() -> None:
    counter = Counter(nat(0))
    counter.trap()
    counter.discard()
