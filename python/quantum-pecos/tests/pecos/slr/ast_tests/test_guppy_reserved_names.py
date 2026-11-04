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

"""User bindings cannot shadow names required by generated Guppy code."""

import ast
import re
from dataclasses import replace
from typing import ClassVar

import pytest
from pecos.slr import Block, CReg, For, Main, QAlloc, QReg, Repeat, Return, SlrConverter
from pecos.slr.ast import ast_to_guppy, slr_to_ast
from pecos.slr.ast.codegen.entry_wrapper import build_no_arg_entry_wrapper
from pecos.slr.ast.codegen.guppy import AstToGuppy, GuppyCodegenError
from pecos.slr.qeclib import qubit as qb
from pecos.slr.slr_converter import _load_and_compile_entry
from pecos.slr.vars import LoopVar

IMPORTED_NAMES = sorted(
    alias.asname or alias.name
    for node in ast.parse(SlrConverter(Main()).guppy()).body
    if isinstance(node, ast.ImportFrom)
    for alias in node.names
)


def _helper(q):
    class Helper(Block):
        block_inputs: ClassVar[dict[str, str]] = {"data": "live_preserved"}

        def __init__(self) -> None:
            super().__init__()
            self.data = q
            self.extend(qb.H(q[0]))

    return Helper()


def _reproducer(case: str, label: str):
    q, c = QReg("q", 1), CReg("c", 1)
    loop = For(label, 0, 1 if case == "h-after" else 2)
    if case == "h-after":
        operations = [loop.Do(), qb.H(q[0])]
    elif case == "h-before-conservative":
        # Deliberate conservative rejection: Guppy can compile this ordering,
        # but the codegen rule does not perform flow analysis of enclosing loops.
        operations = [qb.H(q[0]), loop.Do(qb.X(q[0]))]
    elif case == "h-inside":
        operations = [loop.Do(qb.H(q[0]))]
    elif case == "qubit":
        operations = [qb.Measure(q) > c, loop.Do(), qb.PZ(q[0])]
    elif case == "array":
        operations = [loop.Do(), Return(q, c)]
    else:
        operations = [loop.Do(), _helper(q)]
    return Main(q, c, *operations)


@pytest.mark.parametrize("case", ["h-after", "h-before-conservative", "h-inside", "qubit", "array", "helper_0"])
@pytest.mark.parametrize("method", ["guppy", "hugr"])
def test_reported_loop_name_collisions(case: str, method: str) -> None:
    label = "h" if case.startswith("h-") else case
    message = f"For loop label {label!r} collides with generated Guppy name {label!r}. Rename the loop label."
    with pytest.raises(GuppyCodegenError, match=re.escape(message)):
        getattr(SlrConverter(_reproducer(case, label)), method)()


@pytest.mark.parametrize("case", ["h-after", "h-before-conservative", "h-inside", "qubit", "array", "helper_0"])
def test_renamed_loop_reproducers_compile(case: str) -> None:
    assert SlrConverter(_reproducer(case, "iteration")).hugr()


@pytest.mark.parametrize(
    "name",
    ["_measurement_0", "_measurement_future", "_call_ret_0", "_repeat_0", "_comprehension_0"],
)
@pytest.mark.parametrize("binding", ["loop", "classical", "quantum", "allocator"])
def test_temporary_prefixes_are_reserved(name: str, binding: str) -> None:
    declarations = {
        "loop": For(name, 0, 1).Do(),
        "classical": CReg(name, 1),
        "quantum": QReg(name, 1),
        "allocator": QAlloc(1, name=name),
    }
    rename = "loop label" if binding == "loop" else "register"
    with pytest.raises(GuppyCodegenError, match=rf"{re.escape(repr(name))}.*generated Guppy name.*Rename the {rename}"):
        SlrConverter(Main(declarations[binding])).guppy()


@pytest.mark.parametrize("binding", ["classical", "quantum", "allocator"])
def test_helper_function_name_is_reserved_for_storage(binding: str) -> None:
    q = QReg("q", 1)
    declarations = {
        "classical": CReg("helper_0", 1),
        "quantum": QReg("helper_0", 1),
        "allocator": QAlloc(1, name="helper_0"),
    }
    with pytest.raises(GuppyCodegenError, match=r"'helper_0'.*generated Guppy name 'helper_0'.*Rename the register"):
        SlrConverter(Main(q, declarations[binding], _helper(q))).guppy()


@pytest.mark.parametrize("binding", ["classical", "quantum", "allocator"])
def test_storage_name_with_generated_gate_use(binding: str) -> None:
    q = QReg("q", 1)
    bindings = {"classical": CReg("h", 1), "quantum": QReg("h", 1), "allocator": QAlloc(1, name="h")}
    with pytest.raises(
        GuppyCodegenError,
        match=r"'h'.*generated Guppy name 'h'.*Rename the register",
    ):
        SlrConverter(Main(q, bindings[binding], qb.H(q[0]))).guppy()


def test_implicit_measurement_register_name_is_reserved() -> None:
    q, c = QReg("q", 1), CReg("h", 1)
    with pytest.raises(
        GuppyCodegenError,
        match=r"Classical register 'h'.*generated Guppy name 'h'.*Rename the register",
    ):
        SlrConverter(Main(q, qb.H(q[0]), qb.Measure(q) > c)).guppy()


@pytest.mark.parametrize("kind", ["bit", "qubit"])
@pytest.mark.parametrize("generated_use", [False, True])
def test_block_input_collision_depends_on_its_own_body(kind: str, *, generated_use: bool) -> None:
    q, c = QReg("q", 2), CReg("c", 1)

    class Helper(Block):
        block_inputs: ClassVar[dict[str, str]] = {"h": "live_preserved", "data": "live_preserved"}

        def __init__(self) -> None:
            super().__init__()
            self.h = c[0] if kind == "bit" else q[0]
            self.data = q[1]
            if generated_use:
                self.extend(qb.H(q[1]))
            if kind == "bit":
                self.extend(c[0].set(c[0]))

    converter = SlrConverter(Main(q, c, qb.H(q[1]), Helper()))
    if generated_use:
        message = "Block input 'h' in block 'helper_0' collides with generated Guppy name 'h'. Rename the block input."
        with pytest.raises(GuppyCodegenError, match=re.escape(message)):
            converter.guppy()
    else:
        assert converter.hugr()


@pytest.mark.parametrize("generated_use", [False, True])
def test_reserved_names_follow_emitted_import_aliases(*, generated_use: bool) -> None:
    class ExtraImport(AstToGuppy):
        def _imports(self):
            return [*super()._imports(), "from guppylang.std.quantum.functional import h as added_gate"]

        def _emit_gate(self, node):
            return [line.replace("h(", "added_gate(") for line in super()._emit_gate(node)]

    q = QReg("q", 1)
    program = slr_to_ast(Main(q, For("added_gate", 0, 1).Do(), *([qb.H(q[0])] if generated_use else [])))
    if generated_use:
        with pytest.raises(
            GuppyCodegenError,
            match=r"'added_gate'.*generated Guppy name 'added_gate'.*Rename the loop label",
        ):
            ExtraImport().generate(program)
    else:
        assert ExtraImport().generate(program)


@pytest.mark.parametrize("name", ["helper", "_measurement"])
def test_generated_qubit_locals_avoid_reserved_names(name: str) -> None:
    q = QReg(name, 1)
    converter = SlrConverter(Main(q, _helper(q)))
    assert f"    {name}_0, =" not in converter.guppy()
    assert converter.hugr()


@pytest.mark.parametrize("name", [*IMPORTED_NAMES, "main"])
def test_unused_import_or_function_may_be_a_loop_label(name: str) -> None:
    assert SlrConverter(Main(For(name, 0, 1).Do())).hugr()


@pytest.mark.parametrize("name", [name for name in IMPORTED_NAMES if name != "array"])
def test_unused_import_may_be_a_classical_register(name: str) -> None:
    # Initializing a CReg calls array(); other imports are unused here.
    c = CReg(name, 1)
    assert SlrConverter(Main(c, c.set(1), c[0].set(c[0]), Return(c))).hugr()


@pytest.mark.parametrize("name", [name for name in IMPORTED_NAMES if name != "discard"])
@pytest.mark.parametrize("kind", ["register", "allocator"])
def test_unused_import_may_be_a_quantum_binding(name: str, kind: str) -> None:
    # Cleanup calls discard(); array/qubit/owned in the signature are not local reads.
    q = QReg(name, 1) if kind == "register" else QAlloc(1, name=name)
    assert SlrConverter(Main(q)).hugr()


@pytest.mark.parametrize("name", ["array", "discard"])
def test_generated_storage_operation_still_rejects_shadowing(name: str) -> None:
    binding = CReg(name, 1) if name == "array" else QReg(name, 1)
    with pytest.raises(GuppyCodegenError, match=rf"{name!r}.*generated Guppy name {name!r}.*Rename the register"):
        SlrConverter(Main(binding)).guppy()


def test_classical_gate_name_is_local_to_main() -> None:
    q, h, z = QReg("q", 1), CReg("h", 1), CReg("z", 2)
    program = Main(q, h, z, h.set(True), _helper(q), h.set(h), Return(h, z))
    assert SlrConverter(program).hugr()


def test_classical_z_with_h_gate_compiles() -> None:
    q, z = QReg("q", 1), CReg("z", 2)
    assert SlrConverter(Main(q, z, qb.H(q[0]), Return(z))).hugr()


def test_loop_gate_name_can_be_read_as_a_user_value() -> None:
    c, h = CReg("c", 1), LoopVar("h")
    assert SlrConverter(Main(c, For(h, 0, 2).Do(c[0].set(h == 1)), Return(c))).hugr()


@pytest.mark.parametrize("binding", ["loop", "classical", "quantum", "allocator"])
def test_unused_helper_name_may_be_bound(binding: str, tmp_path) -> None:
    q = QReg("q", 1)
    bindings = {
        "loop": For("helper_0", 0, 1).Do(),
        "classical": CReg("helper_0", 1),
        "quantum": QReg("helper_0", 1),
        "allocator": QAlloc(1, name="helper_0"),
    }
    program = slr_to_ast(Main(q, bindings[binding], _helper(q)))
    # Retain the emitted helper definition, but remove its call from main.
    program = replace(program, body=program.body[:-1])
    source = ast_to_guppy(program) + build_no_arg_entry_wrapper(program)[0]
    path = tmp_path / "unused_helper.py"
    path.write_text(source)
    assert _load_and_compile_entry(path, "_slr_unused_helper")


@pytest.mark.parametrize(
    "case",
    ["classical-for", "classical-repeat", "quantum-for", "allocator-for", "nested-for", "sibling-for"],
)
def test_builtin_range_cannot_be_bound_and_called(case: str) -> None:
    loop = For("i", 0, 1).Do()
    programs = {
        "classical-for": Main(CReg("range", 1), loop),
        "classical-repeat": Main(CReg("range", 1), Repeat(2).block()),
        "quantum-for": Main(QReg("range", 1), loop),
        "allocator-for": Main(QAlloc(1, name="range"), loop),
        "nested-for": Main(For("range", 0, 1).Do(loop)),
        "sibling-for": Main(For("range", 0, 1).Do(), loop),
    }
    with pytest.raises(GuppyCodegenError, match=r"'range'.*generated Guppy name 'range'.*Rename the"):
        SlrConverter(programs[case]).guppy()


def test_builtin_range_in_empty_register_initializer_is_checked() -> None:
    with pytest.raises(GuppyCodegenError, match=r"Classical register 'range'.*generated Guppy name 'range'"):
        SlrConverter(Main(CReg("range", 0))).guppy()


@pytest.mark.parametrize("name", ["class", "True", "lambda", "a b"])
@pytest.mark.parametrize("binding", ["classical", "quantum", "allocator", "loop", "implicit-measurement"])
def test_invalid_user_binding_names_are_codegen_errors(name: str, binding: str) -> None:
    q = QReg("q", 1)
    bindings = {
        "classical": CReg(name, 1),
        "quantum": QReg(name, 1),
        "allocator": QAlloc(1, name=name),
        "loop": For(name, 0, 1).Do(),
        "implicit-measurement": qb.Measure(q) > CReg(name, 1),
    }
    message = f"{name!r} must be a valid identifier that is not a Python keyword. Rename the"
    with pytest.raises(GuppyCodegenError, match=re.escape(message)):
        SlrConverter(Main(q, bindings[binding])).guppy()


@pytest.mark.parametrize("name", ["class", "True", "lambda", "a b", "_repeat_0", "_comprehension_0"])
@pytest.mark.parametrize("kind", ["bit", "qubit"])
def test_block_input_name_validation_reports_its_block(name: str, kind: str) -> None:
    q, c = QReg("q", 1), CReg("c", 1)

    class Helper(Block):
        block_inputs: ClassVar[dict[str, str]] = {name: "live_preserved"}

        def __init__(self) -> None:
            super().__init__()
            setattr(self, name, c[0] if kind == "bit" else q[0])

    message = f"Block input {name!r} in block 'helper_0'"
    with pytest.raises(GuppyCodegenError, match=re.escape(message) + r".*Rename the block input"):
        SlrConverter(Main(q, c, Helper())).guppy()


def test_empty_array_comprehensions_use_distinct_temporaries() -> None:
    c, u = CReg("c", 0), LoopVar("_")
    converter = SlrConverter(Main(c, For(u, 0, 2).Do(c.set(0)), Return(c)))
    function = ast.parse(converter.guppy()).body[-1]
    targets = [node.target.id for node in ast.walk(function) if isinstance(node, ast.comprehension)]
    assert len(targets) == 2
    assert len(set(targets)) == 2
    assert all(name.startswith("_comprehension_") for name in targets)
    assert converter.hugr()


def test_entry_array_comprehensions_use_distinct_temporaries() -> None:
    program = Main(QReg("_", 1), QReg("q", 1))
    wrapper, _ = build_no_arg_entry_wrapper(slr_to_ast(program))
    targets = [node.target.id for node in ast.walk(ast.parse(wrapper)) if isinstance(node, ast.comprehension)]
    assert len(targets) == 2
    assert len(set(targets)) == 2
    assert all(name.startswith("_comprehension_") for name in targets)
    assert SlrConverter(program).hugr()


@pytest.mark.parametrize("kind", ["bit", "qubit"])
def test_block_input_cannot_shadow_builtin_call(kind: str) -> None:
    q, c = QReg("q", 1), CReg("c", 1)

    class Helper(Block):
        block_inputs: ClassVar[dict[str, str]] = {"range": "live_preserved"}

        def __init__(self) -> None:
            super().__init__()
            self.range = c[0] if kind == "bit" else q[0]
            self.extend(For("i", 0, 2).Do())

    message = (
        "Block input 'range' in block 'helper_0' collides with generated Guppy name 'range'. Rename the block input."
    )
    with pytest.raises(GuppyCodegenError, match=re.escape(message)):
        SlrConverter(Main(q, c, Helper())).guppy()


@pytest.mark.parametrize("name", ["range", "match", "case"])
def test_unused_builtin_and_soft_keywords_can_be_bound(name: str) -> None:
    c = CReg(name, 1)
    assert SlrConverter(Main(c, Return(c))).hugr()


def test_repeat_temporary_avoids_an_emitted_helper_name() -> None:
    q = QReg("q", 1)

    class _Repeat(Block):
        block_inputs: ClassVar[dict[str, str]] = {"data": "live_preserved"}

        def __init__(self) -> None:
            super().__init__()
            self.data = q
            self.extend(qb.H(q[0]))

    converter = SlrConverter(Main(q, Repeat(2).block(), _Repeat()))
    source = converter.guppy()
    assert "def _repeat_0(" in source
    assert "for _repeat_0 in" not in source
    assert converter.hugr()
