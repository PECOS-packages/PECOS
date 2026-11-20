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
    """Carry the result of each external call to the next iteration."""
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


@guppy
def loop_control() -> None:
    """Exercise zero, single, nested, and result-dependent loop iterations."""
    counter = Counter(nat(0))
    value = 99
    for _ in range(0):
        value = counter.increment()
    result("zero", value)
    for _ in range(1):
        value = counter.increment()
    result("one", value)
    for _ in range(2):
        for _ in range(3):
            value = counter.increment()
    result("nested", value)
    steps = 0
    while value < 10 and steps < 5:
        value = counter.increment()
        steps += 1
    result("condition", value)
    result("steps", steps)
    counter.discard()


@guppy
def loop_helpers() -> None:
    """Preserve effects before, inside and after helper calls in loops."""
    counter = Counter(nat(0))
    counter.increment()
    counter.discard()
    value = 0
    for _ in range(2):
        value = increment_context()
    result("helper", value)
    counter = Counter(nat(0))
    result("after", counter.increment())
    counter.discard()
