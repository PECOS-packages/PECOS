# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generative program fuzzing for the HUGR engine.

Each seed generates a random classical program that is simultaneously valid
Python and valid guppy: plain Python `exec` computes the reference value,
and the guppy version gates an X on a fresh qubit iff the engine computes
the same value. Every divergence -- a wrong arithmetic result, a
mis-executed loop, a dropped branch -- fails loudly as a 0 measurement.

Generated programs stay in Euclidean-safe territory (every reduction uses a
positive literal modulus, divisions use positive literal divisors, shift
operands are pre-masked non-negative) BY DESIGN: plain Python is only a
valid reference where its semantics coincide with HUGR's. The regimes
where they diverge -- negative divisors (floor vs Euclidean with an
UNSIGNED divisor bit pattern), arithmetic-vs-logical right shift on
negative operands, and shifts past the width -- are pinned separately with
hand-derived spec expectations in test_semantic_sweep.py.
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest
from pecos import Guppy, sim
from pecos_rslib import state_vector

MOD = 9973
SEEDS = range(20)


class _BodyGenerator:
    """Generate a random classical statement body over int variables."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.lines: list[str] = []
        self.vars = ["v0", "v1", "v2"]
        self.loop_depth = 0

    def operand(self) -> str:
        if self.rng.random() < 0.3:
            return str(self.rng.randint(0, 99))
        return self.rng.choice(self.vars)

    def emit(self, line: str, indent: int) -> None:
        self.lines.append("    " * indent + line)

    def assign(self, indent: int) -> None:
        target = self.rng.choice(self.vars)
        kind = self.rng.random()
        a, b = self.operand(), self.operand()
        if kind < 0.45:
            op = self.rng.choice(["+", "-", "*"])
            expr = f"({a} {op} {b}) % {MOD}"
        elif kind < 0.7:
            op = self.rng.choice(["//", "%"])
            divisor = self.rng.randint(2, 9)
            expr = f"({a} {op} {divisor}) % {MOD}"
        else:
            op = self.rng.choice(["<<", ">>"])
            shift = self.rng.randint(0, 6)
            expr = f"(({a} % 64) {op} {shift}) % {MOD}"
        self.emit(f"{target} = {expr}", indent)

    def branch(self, indent: int) -> None:
        a, b = self.rng.choice(self.vars), self.rng.choice(self.vars)
        cmp_op = self.rng.choice(["<", "<=", ">", ">=", "==", "!="])
        self.emit(f"if {a} {cmp_op} {b}:", indent)
        self.assign(indent + 1)
        if self.rng.random() < 0.5:
            self.emit("else:", indent)
            self.assign(indent + 1)

    def loop(self, indent: int) -> None:
        self.loop_depth += 1
        var = f"i{self.loop_depth}"
        bound = self.rng.randint(0, 4)
        self.emit(f"for {var} in range({bound}):", indent)
        target = self.rng.choice(self.vars)
        self.emit(f"{target} = ({target} + {var} + 1) % {MOD}", indent + 1)
        if self.loop_depth < 2 and self.rng.random() < 0.4:
            self.loop(indent + 1)
        elif self.rng.random() < 0.4:
            self.assign(indent + 1)

    def while_loop(self, indent: int) -> None:
        count = self.rng.randint(1, 5)
        target = self.rng.choice(self.vars)
        self.emit(f"w = {count}", indent)
        self.emit("while w > 0:", indent)
        self.emit(f"{target} = ({target} * 3 + w) % {MOD}", indent + 1)
        self.emit("w = w - 1", indent + 1)

    def generate(self) -> str:
        for i, name in enumerate(self.vars):
            self.emit(f"{name} = {self.rng.randint(0, MOD - 1)}", 1)
            del i
        for _ in range(self.rng.randint(4, 8)):
            pick = self.rng.random()
            if pick < 0.45:
                self.assign(1)
            elif pick < 0.65:
                self.branch(1)
            elif pick < 0.85:
                self.loop(1)
            else:
                self.while_loop(1)
        self.emit(f"acc = (v0 + 31 * v1 + 977 * v2) % {MOD}", 1)
        return "\n".join(self.lines)


def _reference_value(body: str) -> int:
    """Execute the generated body as plain Python and return acc."""
    source = "def _ref():\n" + body + "\n    return acc\n"
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 -- fuzz reference evaluation of generated code
    return namespace["_ref"]()


def _load_guppy_module(tmp_path: Path, seed: int, source: str):
    path = tmp_path / f"fuzz_prog_{seed}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"fuzz_prog_{seed}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.mark.parametrize("seed", SEEDS)
def test_fuzzed_program_matches_python_reference(seed: int, tmp_path: Path) -> None:
    rng = random.Random(seed)
    body = _BodyGenerator(rng).generate()
    expected = _reference_value(body)

    source = (
        "from guppylang import guppy\n"
        "from guppylang.std.quantum import measure, qubit, x\n"
        "\n"
        "\n"
        "@guppy\n"
        "def fuzz_prog() -> bool:\n"
        "    q = qubit()\n"
        f"{body}\n"
        f"    if acc == {expected}:\n"
        "        x(q)\n"
        "    return measure(q)\n"
    )
    module = _load_guppy_module(tmp_path, seed, source)

    results = sim(Guppy(module.fuzz_prog)).qubits(2).quantum(state_vector()).seed(7).run(2).to_dict()
    raw = results["measurements"]
    values = [m[-1] if isinstance(m, list) else m for m in raw]
    assert values == [1, 1], f"seed {seed}: engine diverged from reference\n{source}"
