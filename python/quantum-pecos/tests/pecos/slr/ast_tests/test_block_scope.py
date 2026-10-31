"""Program storage in ordinary blocks and classical scope in Guppy functions."""

from typing import ClassVar

import pytest
from pecos.slr import Block, CReg, For, If, Main, Parallel, Permute, QAlloc, QReg, Repeat, SlrConverter, While
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen.guppy import GuppyCodegenError, ast_to_guppy
from pecos.slr.ast.codegen.qasm import AstToQasm
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    AssignOp,
    BitTypeExpr,
    BlockDecl,
    BlockInput,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    Program,
    RegisterDecl,
    ResourceEffect,
    ReturnOp,
    SlotRef,
    VarExpr,
)
from pecos.slr.qeclib import qubit as qb
from pecos.slr.vars import LoopVar, Vars
from pecos_rslib import Qasm, sim, state_vector


def test_vars_keeps_arguments() -> None:
    a, b = QReg("a", 2), CReg("b", 3)
    assert Vars(a, b).vars == [a, b]


def test_vars_rejects_non_var() -> None:
    with pytest.raises(TypeError, match=r"Unrecognized variable type.*int"):
        Vars(1)


def test_vars_subclass_source_class() -> None:
    class Registers(Vars):
        def __init__(self, *args) -> None:
            super().__init__(*args)
            self.source_class = type(self).__name__

    a = CReg("a", 2)
    block = Block(Registers(a))
    assert block.vars.vars == [a]
    assert block.vars.var_source_classes == {"a": "Registers"}


def test_nested_vars_reproducer() -> None:
    q, c, local = QReg("q", 2), CReg("c", 2), CReg("bitswap", 3)
    program = Main(q, c, Block(Vars(local), local.set(5)))
    assert RegisterDecl(name="bitswap", size=3) in slr_to_ast(program).declarations
    assert "creg bitswap[3];" in SlrConverter(program).qasm()


def _run(program: Main) -> dict:
    return sim(Qasm.from_string(SlrConverter(program).qasm())).quantum(state_vector()).run(4).to_dict()


@pytest.mark.parametrize("region", ["block", "if", "for"])
def test_nested_declaration_matches_main_simulation(region) -> None:
    q, c, local = QReg("q", 1), CReg("c", 1), CReg("local", 3)

    def program(*, nested: bool) -> Main:
        body = Block(*([local] if nested else []), local.set(5))
        if region == "if":
            body = If(c == 0).Then(body)
        elif region == "for":
            body = For("i", 0, 2).Do(body)
        else:
            body = Block(Block(body))
        return Main(q, c, *([] if nested else [local]), body)

    actual, expected = _run(program(nested=True)), _run(program(nested=False))
    assert actual["local"] == expected["local"] == [5] * 4
    assert actual["c"] == expected["c"]


def test_nested_bitswap_avoids_temporary_collision() -> None:
    q, c, local = QReg("q", 2), CReg("c", 2), CReg("bitswap", 3)
    program = Main(q, c, c.set(1), Permute([c[0], c[1]], [c[1], c[0]]), Block(local, local.set(5)))
    code = SlrConverter(program).qasm()
    assert "creg bitswap[1];" not in code
    assert "creg bitswap[3];" in code
    result = _run(program)
    assert result["bitswap"] == [5] * 4
    assert result["c"] == [2] * 4


@pytest.mark.parametrize("kind", [CReg, QReg])
def test_nested_conflicting_sizes(kind) -> None:
    program = Main(Block(kind("local", 2)), Block(kind("local", 3)))
    with pytest.raises(ValueError, match=r"local.*2.*3.*rename one or use one size"):
        slr_to_ast(program)


@pytest.mark.parametrize("kind", [CReg, QReg])
def test_nested_same_name_size_is_one_declaration(kind) -> None:
    local = kind("local", 2)
    program = Main(Block(local), Block(local, kind("local", 2)))
    declarations = slr_to_ast(program).declarations
    assert len(declarations) == 1
    assert declarations[0].name == "local"


def test_same_name_children_of_different_parents_conflict() -> None:
    left, right = QAlloc(2, name="left"), QAlloc(2, name="right")
    program = Main(left, right, Block(left.child(1, name="local")), Block(right.child(1, name="local")))
    with pytest.raises(ValueError, match=r"'local' is declared twice as different storage.*rename one"):
        slr_to_ast(program)


def test_declarations_depth_first_all_block_kinds() -> None:
    c = CReg("c", 1)
    program = Main(
        c,
        Block(CReg("a", 1), Block(CReg("b", 1))),
        If(c == 0).Then(CReg("then", 1)).Else(CReg("otherwise", 1)),
        For("i", 0, 1).Do(CReg("for_reg", 1)),
        While(c == 0).Do(CReg("while_reg", 1)),
        Repeat(2).block(CReg("repeat_reg", 1)),
        Parallel(CReg("parallel_reg", 1), Block(QAlloc(2, name="nested_alloc"))),
        CReg("last_root", 1),
    )
    assert [d.name for d in slr_to_ast(program).declarations] == [
        "c",
        "last_root",
        "a",
        "b",
        "then",
        "otherwise",
        "for_reg",
        "while_reg",
        "repeat_reg",
        "parallel_reg",
        "nested_alloc",
    ]


def test_nested_child_allocator_offsets() -> None:
    root = QAlloc(5, name="root")
    first = root.child(2, name="first")
    child = root.child(3, name="child")
    grandchild = child.child(2, name="grandchild")
    program = Main(root, first, Block(child, Block(grandchild)))
    ast = slr_to_ast(program)
    assert ast.declarations == (
        AllocatorDecl(name="root", capacity=5, offset=0),
        AllocatorDecl(name="first", capacity=2, parent="root", offset=0),
        AllocatorDecl(name="child", capacity=3, parent="root", offset=2),
        AllocatorDecl(name="grandchild", capacity=2, parent="child", offset=0),
    )
    # Inspect the backend's computed physical mapping even without prepared slots.
    generator = AstToQasm()
    generator.generate(ast)
    assert generator.context.allocator_offsets == {"root": 0, "first": 0, "child": 2, "grandchild": 2}


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["classical", "quantum", "allocator"])
def test_block_inputs_cannot_declare_registers(nested, kind) -> None:
    local = {"classical": CReg("local", 2), "quantum": QReg("local", 2), "allocator": QAlloc(2, name="local")}[kind]

    class Inner(Block):
        block_inputs: ClassVar[dict[str, str]] = {}

        def __init__(self) -> None:
            super().__init__(If(True).Then(Block(Block(local))) if nested else local)

    with pytest.raises(
        ValueError,
        match=r"Inner.*local.*blocks with block_inputs cannot declare registers yet.*Main.*ordinary Block",
    ):
        slr_to_ast(Main(Inner()))


def test_guppy_symbolic_capture_reproducer() -> None:
    qr, e, c, x = QReg("qr", 2), CReg("e", 2), CReg("c", 2), LoopVar("x")

    class Inner(Block):
        block_inputs: ClassVar[dict[str, str]] = {"z": "live_preserved"}

        def __init__(self) -> None:
            super().__init__()
            self.z = qr[0]
            self.extend(qb.H(qr[0]), c[x].set(e[x]))

    program = Main(qr, e, c, For(x, 0, 2).Do(Inner()))
    with pytest.raises(
        GuppyCodegenError,
        match=r"[cex].*block 'inner_0'.*cannot capture outer registers or loop variables",
    ):
        SlrConverter(program).guppy()


@pytest.mark.parametrize("reference", ["missing", "missing[0]", "missing[x]", "c[x]", "c[1+1]"])
@pytest.mark.parametrize("target", [False, True])
def test_guppy_unresolved_reference_in_main(reference, target) -> None:
    op = (
        AssignOp(target=reference, value=LiteralExpr(value=0))
        if target
        else AssignOp(target="c", value=VarExpr(name=reference))
    )
    program = Program(name="main", declarations=(RegisterDecl(name="c", size=2),), body=(op,))
    with pytest.raises(
        GuppyCodegenError,
        match=r"Unresolved classical reference.*in main; it names no declared register or enclosing loop variable",
    ):
        ast_to_guppy(program)


@pytest.mark.parametrize("use", ["condition", "return", "angle"])
def test_guppy_expression_capture_rendering_paths(use) -> None:
    expr = VarExpr(name="missing")
    operations = {
        "condition": IfStmt(condition=expr, then_body=()),
        "return": ReturnOp(values=(expr,)),
        "angle": GateOp(gate=GateKind.RZ, targets=(SlotRef(allocator="q", index=0),), params=(expr,)),
    }
    program = Program(name="main", declarations=(AllocatorDecl(name="q", capacity=1),), body=(operations[use],))
    with pytest.raises(
        GuppyCodegenError,
        match=r"'missing' in main; it names no declared register",
    ):
        ast_to_guppy(program)


def test_guppy_loop_variable_does_not_escape() -> None:
    loop = ForStmt(variable="x", start=LiteralExpr(value=0), stop=LiteralExpr(value=2))
    program = Program(
        name="main",
        declarations=(RegisterDecl(name="c", size=2),),
        body=(loop, AssignOp(target="c[x]", value=LiteralExpr(value=0))),
    )
    with pytest.raises(
        GuppyCodegenError,
        match=r"'c\[x\]' in main; it names no declared register or enclosing loop variable",
    ):
        ast_to_guppy(program)


# Compatibility: these uses were valid before scope validation.
def test_compat_plain_block_symbolic_bits_compile() -> None:
    qr, e, c, x = QReg("qr", 2), CReg("e", 2), CReg("c", 2), LoopVar("x")
    program = Main(qr, e, c, For(x, 0, 2).Do(Block(qb.H(qr[0]), c[x].set(e[x]))))
    assert SlrConverter(program).hugr() is not None


def test_compat_block_inner_loop_bit_input_compiles() -> None:
    c, x = CReg("c", 1), LoopVar("x")

    class Inner(Block):
        block_inputs: ClassVar[dict[str, str]] = {"bit": "live_preserved"}

        def __init__(self) -> None:
            super().__init__()
            self.bit = c[0]
            self.extend(For(x, 0, 1).Do(c[0].set("bit[x]")))

    assert SlrConverter(Main(c, Inner())).hugr() is not None


def test_compat_guppy_shadowed_loop_variables_and_scalars() -> None:
    c, x = CReg("c", 2), LoopVar("x")
    program = Main(c, For(x, 0, 2).Do(For(x, 0, 1).Do(c[x].set(c[x])), c[x].set(c[x])), c.set(c))
    assert "c = c" in SlrConverter(program).guppy()


@pytest.mark.parametrize("reference", ["c", "c[0]", "c[x]", "bit[x]", "x"])
@pytest.mark.parametrize("target", [False, True])
def test_guppy_unresolved_reference_in_block(reference, target) -> None:
    op = (
        AssignOp(target=reference, value=LiteralExpr(value=0))
        if target
        else AssignOp(target="bit", value=VarExpr(name=reference))
    )
    decl = BlockDecl(
        name="inner",
        inputs=(BlockInput(name="bit", type_expr=BitTypeExpr(), effect=ResourceEffect.LIVE_PRESERVED),),
        body=(op,),
    )
    program = Program(name="main", declarations=(RegisterDecl(name="c", size=2),), block_decls=(decl,))
    with pytest.raises(GuppyCodegenError, match=r"Unresolved classical reference.*block 'inner'.*cannot capture"):
        ast_to_guppy(program)


@pytest.mark.parametrize("nested", [False, True])
def test_parallel_optimizer_preserves_declarations(nested) -> None:
    q, local = QReg("q", 1), CReg("local", 2)
    body = (local, local.set(2), qb.X(q[0]), qb.H(q[0]))
    region = Parallel(Block(*body)) if nested else Parallel(*body)
    program = Main(q, region)
    assert "creg local[2];" in SlrConverter(program).qasm()
    assert _run(program)["local"] == [2] * 4


def _placement_program(shape, *, parent_first=False):
    root = QAlloc(3, name="root")
    c = CReg("c", 2)
    first = root.child(1, name="first")
    if shape in {"p9", "p9b"}:
        mid = root.child(2, name="mid")
        target = mid.child(1, name="target")
        allocators = (root, first, mid, target)
        regions = (first, target, mid) if shape == "p9b" else (first, Block(target), Block(mid))
    else:
        target = root.child(1, name="target")
        last = root.child(1, name="last")
        allocators = (root, first, target, last)
        regions = (first, Block(target), last)
    ops = (qb.X(first[0]), qb.X(target[0]), qb.Z(target[0]))
    measures = (qb.Measure(first[0]) > c[0], qb.Measure(target[0]) > c[1])
    if parent_first:
        program = Main(*allocators, c, *ops, *measures)
    elif shape == "p7":
        program = Main(root, c, last, Parallel(Block(first, ops[0]), target, *ops[1:]), *measures)
    else:
        program = Main(root, c, *regions, *ops, *measures)
    expected = (
        {"root": 0, "first": 0, "mid": 1, "target": 1}
        if shape in {"p9", "p9b"}
        else {
            "root": 0,
            "first": 0,
            "target": 1,
            "last": 2,
        }
    )
    return program, expected, allocators


@pytest.mark.parametrize("shape", ["p9", "p9b", "p1b", "p7"])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("backend", ["qasm", "stim", "qir", "quantum_circuit"])
def test_round2_child_placement(shape, optimized, backend) -> None:
    from pecos.slr.ast.codegen.qir import AstToQir
    from pecos.slr.ast.codegen.quantum_circuit import AstToQuantumCircuit
    from pecos.slr.ast.codegen.stim import AstToStim
    from pecos.slr.transforms import ParallelOptimizer

    program, expected, allocators = _placement_program(shape)
    transformed = ParallelOptimizer().transform(program) if optimized else program
    generator = {"qasm": AstToQasm, "stim": AstToStim, "qir": AstToQir, "quantum_circuit": AstToQuantumCircuit}[
        backend
    ]()
    output = generator.generate(slr_to_ast(transformed))
    assert generator.context.allocator_offsets == expected
    for allocator in allocators[1:]:
        assert expected[allocator.name] == expected[allocator.parent.name] + allocator.parent_indices[0]
    if backend == "stim":
        assert [t.value for op in output if op.name == "X" for t in op.targets_copy()] == [0, 1]
        assert output.compile_sampler().sample(shots=4).tolist() == [[True, True]] * 4
    elif backend == "qir":
        calls = [line.strip() for line in output.splitlines() if "call void @__quantum__qis__x__body" in line]
        assert calls == [
            "call void @__quantum__qis__x__body(ptr null)",
            "call void @__quantum__qis__x__body(ptr inttoptr (i64 1 to ptr))",
        ]
    elif backend == "quantum_circuit":
        assert sorted(qubit for gate, locations, _ in output.items() if gate == "X" for qubit in locations) == [0, 1]
    code = SlrConverter(program, optimize_parallel=optimized).qasm()
    assert "x root[0];" in code
    assert "x root[1];" in code
    baseline, _, _ = _placement_program(shape, parent_first=True)
    result = sim(Qasm.from_string(code)).quantum(state_vector()).run(4).to_dict()
    assert result == _run(baseline)
    assert result["c"] == [3] * 4


def test_round2_noncontiguous_child_is_rejected() -> None:
    root = QAlloc(4, name="root")
    first = root.child(1, name="first")
    root.child(1, name="middle")
    first.release()
    scattered = root.child(2, name="scattered")
    with pytest.raises(ValueError, match=r"scattered.*contiguous"):
        slr_to_ast(Main(root, scattered))
    assert scattered.parent_indices == (0, 2)


def test_round2_root_block_inputs_is_inline() -> None:
    q, c = QReg("q", 1), CReg("c", 1)

    class Root(Block):
        block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

        def __init__(self) -> None:
            super().__init__(q, c, qb.X(q[0]), qb.Measure(q[0]) > c[0])
            self.q = q

    ast = slr_to_ast(Root())
    assert ast.declarations == (AllocatorDecl(name="q", capacity=1, offset=0), RegisterDecl(name="c", size=1))
    assert not ast.block_decls
    assert "creg c[1];" in SlrConverter(Root()).qasm()


@pytest.mark.parametrize("ordinary", ["before", "after", "bound_only"])
def test_round2_function_vars_may_repeat_inputs(ordinary) -> None:
    q = QReg("q", 1)

    class Inner(Block):
        block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

        def __init__(self) -> None:
            super().__init__(Block(q, qb.H(q[0])))
            self.q = q

    parts = {"before": (Block(q), Inner()), "after": (Inner(), Block(q)), "bound_only": (Inner(),)}
    program = Main(*parts[ordinary])
    assert slr_to_ast(program).declarations == (AllocatorDecl(name="q", capacity=1, offset=0),)
    assert SlrConverter(program).hugr() is not None


@pytest.mark.parametrize("ordinary_first", [False, True])
def test_round2_function_vars_may_repeat_ordinary_registers(ordinary_first) -> None:
    c = CReg("c", 1)

    class Inner(Block):
        block_inputs: ClassVar[dict[str, str]] = {}

        def __init__(self) -> None:
            super().__init__(Block(c, c[0].set(1)))

    parts = (Block(c), Inner()) if ordinary_first else (Inner(), Block(c))
    assert slr_to_ast(Main(*parts)).declarations == (RegisterDecl(name="c", size=1),)
    assert _run(Main(*parts))["c"] == [1] * 4


@pytest.mark.parametrize("scalar", ["integer", "real", "real64"])
def test_round2_numeric_scalars_are_literals(scalar) -> None:
    import numpy as np

    value = {"integer": np.int64(2), "real": np.float32(0.5), "real64": np.float64(0.5)}[scalar]
    c = CReg("c", 2)
    expression = slr_to_ast(Main(c, c.set(value))).body[0].value
    assert isinstance(expression, LiteralExpr)
    assert type(expression.value) is (int if scalar == "integer" else float)
    if scalar == "integer":
        q = QReg("q", 1)
        assert SlrConverter(Main(q, For("i", 0, value).Do(qb.H(q[0])))).hugr() is not None


@pytest.mark.parametrize("backend", ["qasm", "qir", "stim", "quantum_circuit"])
@pytest.mark.parametrize("child", [False, True])
def test_round2_undeclared_quantum_reference(backend, child) -> None:
    root = QAlloc(2, name="root")
    missing = root.child(1, name="missing") if child else QReg("missing", 1)
    c = CReg("c", 1)
    program = Main(root, c, qb.H(missing[0]), qb.Measure(missing[0]) > c[0])
    with pytest.raises(ValueError, match=r"missing.*declare"):
        getattr(SlrConverter(program), backend)()


@pytest.mark.parametrize(
    "use",
    ["measurement", "bit_target", "register_target", "bit_value", "register_value", "condition"],
)
def test_round2_qasm_undeclared_classical_reference(use) -> None:
    q, c, missing = QReg("q", 1), CReg("c", 1), CReg("missing", 1)
    ops = {
        "measurement": qb.Measure(q[0]) > missing[0],
        "bit_target": missing[0].set(1),
        "register_target": missing.set(1),
        "bit_value": c[0].set(missing[0]),
        "register_value": c.set(missing),
        "condition": If(missing == 0).Then(qb.X(q[0])),
    }
    with pytest.raises(ValueError, match=r"missing.*declare"):
        SlrConverter(Main(q, c, ops[use])).qasm()


def test_round2_qasm_inline_bitswap_is_rejected() -> None:
    q, c, inline = QReg("q", 1), CReg("c", 2), CReg("bitswap", 1)
    program = Main(q, c, c.set(1), Permute([c[0], c[1]], [c[1], c[0]]), qb.Measure(q[0]) > inline[0])
    with pytest.raises(ValueError, match=r"bitswap.*declare"):
        SlrConverter(program).qasm()


@pytest.mark.parametrize("backend", ["qasm", "qir"])
def test_round2_undeclared_measurement_output(backend) -> None:
    q, missing = QReg("q", 1), CReg("missing", 1)
    with pytest.raises(ValueError, match=r"missing.*declare"):
        getattr(SlrConverter(Main(q, qb.Measure(q[0]) > missing[0])), backend)()


def test_round2_parallel_allocator_declarations() -> None:
    from pecos.slr.transforms import ParallelOptimizer

    root = QAlloc(2, name="root")
    child = root.child(1, name="child")
    program = Main(Parallel(root, Block(child, qb.X(child[0])), qb.Z(child[0])))
    ast = slr_to_ast(ParallelOptimizer().transform(program))
    assert [decl.name for decl in ast.declarations] == ["root", "child"]
    assert "x root[0];" in SlrConverter(program).qasm()


@pytest.mark.parametrize("region", ["block", "if"])
def test_round2_nested_register_selene_oracle(region) -> None:
    from pecos import Hugr, selene_engine
    from pecos import sim as pecos_sim
    from pecos.slr import Print, Return

    def run(nested):
        q, c, flag = QReg("q", 1), CReg("c", 2), CReg("flag", 1)
        inner = Block(*([c] if nested else []), qb.X(q[0]), qb.Measure(q[0]) > c[0], qb.PZ(q[0]), c[1].set(1))
        body = If(flag[0] == 0).Then(inner) if region == "if" else Block(inner)
        program = Main(q, flag, *([] if nested else [c]), body, Print(c), Return(c))
        hugr_bytes = SlrConverter(program).hugr().to_str().encode("utf-8")
        return pecos_sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(1).seed(42).run(4).to_dict()

    result = run(True)
    assert result == run(False)
    assert result["result.c"] == [[True, True]] * 4


def test_round2_placement_survives_ast_consumers() -> None:
    from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
    from pecos.slr.ast.optimizations import optimize
    from pecos.slr.ast.pretty_print import pretty_print
    from pecos.slr.ast.serialize import ast_to_json, json_to_ast

    program, _, _ = _placement_program("p9")
    ast = slr_to_ast(program)
    assert "offset=1" in pretty_print(ast)
    restored = json_to_ast(ast_to_json(ast))
    optimized = optimize(restored, level=3).program
    flattened = flatten_block_calls(optimized)
    assert flattened.declarations == ast.declarations
    assert {decl.name: decl.offset for decl in flattened.declarations if isinstance(decl, AllocatorDecl)} == {
        "root": 0,
        "first": 0,
        "target": 0,
        "mid": 1,
    }


def test_round2_zero_capacity_parent_is_retained() -> None:
    root = QAlloc(0, name="root")
    child = root.child(0, name="child")
    assert slr_to_ast(Main(root, child)).declarations[1].parent == "root"


@pytest.mark.parametrize("kind", [QReg, CReg])
def test_round2_root_duplicate_names_conflict(kind) -> None:
    with pytest.raises(ValueError, match=r"q.*2.*3.*rename one or use one size"):
        slr_to_ast(Main(kind("q", 2), kind("q", 3)))


def test_round2_same_parent_different_offsets_conflict() -> None:
    root = QAlloc(2, name="root")
    first = root.child(1, name="child")
    second = root.child(1, name="child")
    with pytest.raises(ValueError, match=r"child.*different storage"):
        slr_to_ast(Main(root, first, Block(second)))


@pytest.mark.parametrize("whole", [False, True])
def test_round2_qasm_undeclared_permute_diagnostic(whole) -> None:
    c, missing = CReg("c", 1), CReg("missing", 1)
    refs = [c, missing] if whole else [c[0], missing[0]]
    with pytest.raises(ValueError, match=r"missing.*declare it"):
        SlrConverter(Main(c, Permute(refs, refs[::-1]))).qasm()


def test_compat_guppy_inline_measurement_has_separate_storage() -> None:
    from pecos import Hugr, selene_engine
    from pecos import sim as pecos_sim
    from pecos.slr import Print

    q, c, inline = QReg("q", 1), CReg("c", 2), CReg("bitswap", 1)
    program = Main(
        q,
        c,
        c[0].set(True),
        Permute([c[0], c[1]], [c[1], c[0]]),
        qb.Measure(q[0]) > inline[0],
        Print(c),
        Print(inline[0]),
    )
    hugr_bytes = SlrConverter(program).hugr().to_str().encode("utf-8")
    result = pecos_sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(1).seed(42).run(4).to_dict()
    assert result["result.c"] == [[False, True]] * 4
    assert result["result.bitswap_0"] == [False] * 4


@pytest.mark.parametrize("child", [False, True])
def test_compat_guppy_undeclared_allocator_is_rejected(child) -> None:
    root = QAlloc(2, name="root")
    missing = root.child(1, name="missing") if child else QReg("missing", 1)
    with pytest.raises(GuppyCodegenError, match="missing"):
        SlrConverter(Main(root, qb.H(missing[0]))).guppy()


@pytest.mark.parametrize("backend", ["qasm", "qir", "stim", "quantum_circuit"])
@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_round3_fragment_quantum_reference(backend, child, optimized) -> None:
    root = QAlloc(2, name="root")
    missing = root.child(1, name="missing") if child else QReg("missing", 1)
    c = CReg("c", 1)
    fragment = Block(root, c, qb.H(missing[0]), qb.Measure(missing[0]) > c[0])
    assert slr_to_ast(fragment).is_fragment
    output = getattr(SlrConverter(fragment, optimize_parallel=optimized), backend)()
    if backend == "qasm":
        assert "h missing[0];" in output
        assert "measure missing[0] -> c[0];" in output
        assert "qreg missing" not in output
    elif backend == "stim":
        assert str(output) == "H 2\nM 2"
    elif backend == "qir":
        assert "call void @__quantum__qis__h__body(ptr inttoptr (i64 2 to ptr))" in output
    else:
        assert output.metadata["qvars"][2] == ("missing", 0)
        assert [(gate, locations) for gate, locations, _ in output.items()] == [("H", {2}), ("Measure", {2})]


@pytest.mark.parametrize(
    ("use", "expected"),
    [
        ("measurement", "measure q[0] -> missing[0];"),
        ("bit_target", "missing[0] = 1;"),
        ("register_target", "missing = 1;"),
        ("bit_value", "c[0] = missing[0];"),
        ("register_value", "c = missing;"),
        ("condition", "if(missing == 0) x q[0];"),
    ],
)
def test_round3_fragment_qasm_classical_reference(use, expected) -> None:
    q, c, missing = QReg("q", 1), CReg("c", 1), CReg("missing", 1)
    ops = {
        "measurement": qb.Measure(q[0]) > missing[0],
        "bit_target": missing[0].set(1),
        "register_target": missing.set(1),
        "bit_value": c[0].set(missing[0]),
        "register_value": c.set(missing),
        "condition": If(missing == 0).Then(qb.X(q[0])),
    }
    output = SlrConverter(Block(q, c, ops[use])).qasm()
    assert expected in output.splitlines()
    assert "creg missing" not in output


@pytest.mark.parametrize("backend", ["qasm", "stim", "quantum_circuit"])
def test_round3_fragment_measurement_output(backend) -> None:
    q, missing = QReg("q", 1), CReg("missing", 1)
    output = getattr(SlrConverter(Block(q, qb.Measure(q[0]) > missing[0])), backend)()
    if backend == "qasm":
        assert "measure q[0] -> missing[0];" in output
        assert "creg missing" not in output
    elif backend == "stim":
        assert str(output) == "M 0"
    else:
        assert list(output.items()) == [("Measure", {0}, {"var_output": {0: ("missing", 0)}})]


def test_round3_fragment_qasm_inline_bitswap() -> None:
    q, c, inline = QReg("q", 1), CReg("c", 2), CReg("bitswap", 1)
    fragment = Block(q, c, c.set(1), Permute([c[0], c[1]], [c[1], c[0]]), qb.Measure(q[0]) > inline[0])
    output = SlrConverter(fragment).qasm()
    assert "measure q[0] -> bitswap[0];" in output


@pytest.mark.parametrize("fragment", [False, True])
def test_round3_qasm_caller_owned_barrier(fragment) -> None:
    from pecos.slr import Barrier

    q, other = QReg("q", 1), QReg("other", 1)
    program = (Block if fragment else Main)(Barrier(q, q, other), qb.H(q[0]))
    if fragment:
        assert SlrConverter(program).qasm().splitlines()[2:] == ["barrier q, other;", "h q[0];"]
    else:
        with pytest.raises(ValueError, match=r"undeclared allocator 'q'.*declare it before using Barrier"):
            SlrConverter(program).qasm()


@pytest.mark.parametrize("backend", ["qasm", "qir", "stim", "quantum_circuit"])
@pytest.mark.parametrize("declare_root", [False, True])
def test_round3_fragment_declared_child_placement(backend, declare_root) -> None:
    from pecos.slr.ast.codegen.qir import AstToQir
    from pecos.slr.ast.codegen.quantum_circuit import AstToQuantumCircuit
    from pecos.slr.ast.codegen.stim import AstToStim

    root = QAlloc(4, name="root")
    root.child(1, name="padding")
    mid = root.child(3, name="mid")
    mid.child(1, name="padding_mid")
    leaf = mid.child(1, name="leaf")
    fragment = Block(*([root] if declare_root else []), leaf, mid, qb.X(leaf[0]))
    generator = {
        "qasm": AstToQasm,
        "qir": AstToQir,
        "stim": AstToStim,
        "quantum_circuit": AstToQuantumCircuit,
    }[backend]()
    generator.generate(slr_to_ast(fragment))
    assert generator.context.get_root_allocator("leaf") == "root"
    assert generator.context.get_absolute_index("leaf", 0) == 2
    assert generator.context.get_root_allocator("external") == "external"
    assert generator.context.get_absolute_index("external", 3) == 3
    if backend == "qasm":
        assert "x root[2];" in SlrConverter(fragment).qasm()

    with pytest.raises(ValueError, match=r"Undeclared allocator 'root'.*declare"):
        generator.generate(slr_to_ast(Main(leaf, mid, qb.X(leaf[0]))))

    # Reusing a generator for a complete program must restore strict checking.
    with pytest.raises(ValueError, match=r"Undeclared allocator 'external'.*declare"):
        generator.generate(slr_to_ast(Main(qb.H(QReg("external", 1)[0]))))


def test_compat_fragment_qir_requires_classical_storage() -> None:
    q, missing = QReg("q", 1), CReg("missing", 1)
    with pytest.raises(ValueError, match=r"missing.*no QIR storage"):
        SlrConverter(Block(q, qb.Measure(q[0]) > missing[0])).qir()


@pytest.mark.parametrize("whole", [False, True])
def test_compat_fragment_qasm_permute_requires_sizes(whole) -> None:
    c, missing = CReg("c", 1), CReg("missing", 1)
    refs = [c, missing] if whole else [c[0], missing[0]]
    message = "known whole registers of equal size" if whole else r"missing\[0\] is not a declared slot"
    with pytest.raises(ValueError, match=message):
        SlrConverter(Block(c, Permute(refs, refs[::-1]))).qasm()


@pytest.mark.parametrize("rule", ["conflicting_size", "noncontiguous_child", "function_local"])
def test_compat_fragment_declaration_rules_remain_strict(rule) -> None:
    if rule == "conflicting_size":
        fragment = Block(CReg("local", 1), Block(CReg("local", 2)))
        message = r"local.*conflicting sizes 1 and 2"
    elif rule == "noncontiguous_child":
        root = QAlloc(3, name="root")
        first = root.child(1)
        root.child(1)
        first.release()
        fragment = Block(root, root.child(2, name="local"))
        message = r"local.*contiguous"
    else:

        class Inner(Block):
            block_inputs: ClassVar[dict[str, str]] = {}

            def __init__(self) -> None:
                super().__init__(CReg("local", 1))

        fragment = Block(Inner())
        message = r"Inner.*local.*cannot declare registers"
    with pytest.raises(ValueError, match=message):
        slr_to_ast(fragment)


@pytest.mark.parametrize("backend", ["qasm", "qir", "stim", "quantum_circuit"])
@pytest.mark.parametrize("legacy_json", [False, True])
def test_round4_child_offsets_must_be_explicit(backend, legacy_json) -> None:
    import json

    from pecos.slr.ast.codegen.qir import AstToQir
    from pecos.slr.ast.codegen.quantum_circuit import AstToQuantumCircuit
    from pecos.slr.ast.codegen.stim import AstToStim
    from pecos.slr.ast.serialize import ast_to_json, json_to_ast

    declarations = (
        AllocatorDecl(name="q", capacity=3),
        AllocatorDecl(name="padding", capacity=1, parent="q", **({"offset": 0} if legacy_json else {})),
        AllocatorDecl(name="child", capacity=2, parent="q", **({"offset": 1} if legacy_json else {})),
    )
    program = Program(
        name="main",
        declarations=declarations,
        body=tuple(
            GateOp(gate=GateKind.X, targets=(SlotRef(allocator=name, index=0),)) for name in ("padding", "child")
        ),
    )
    if legacy_json:
        data = json.loads(ast_to_json(program))
        for decl in data["declarations"]:
            decl.pop("offset")
        program = json_to_ast(json.dumps(data))
    assert program.declarations[1].offset is None
    generator = {"qasm": AstToQasm, "qir": AstToQir, "stim": AstToStim, "quantum_circuit": AstToQuantumCircuit}[
        backend
    ]()
    with pytest.raises(ValueError, match=r"padding.*set offset=.*parent.child"):
        generator.generate(program)


@pytest.mark.parametrize("bad", ["past_parent", "negative", "root_offset", "cycle", "missing_offset"])
def test_round4_allocator_layout_guards(bad) -> None:
    from pecos.slr.ast.codegen._allocator_layout import allocator_offsets

    root = AllocatorDecl(name="q", capacity=3)
    cases = {
        "past_parent": ((root, AllocatorDecl(name="child", capacity=2, parent="q", offset=2)), "child.*outside"),
        "negative": ((root, AllocatorDecl(name="child", capacity=1, parent="q", offset=-1)), "child.*outside"),
        "root_offset": ((AllocatorDecl(name="q", capacity=3, offset=1),), "q.*offset zero"),
        "cycle": (
            (
                AllocatorDecl(name="a", capacity=1, parent="b", offset=0),
                AllocatorDecl(name="b", capacity=1, parent="a", offset=0),
            ),
            "a.*cyclic",
        ),
        "missing_offset": ((root, AllocatorDecl(name="child", capacity=1, parent="q")), "child.*set offset="),
    }
    declarations, message = cases[bad]
    with pytest.raises(ValueError, match=message):
        allocator_offsets(Program(name="main", declarations=declarations))


@pytest.mark.parametrize("backend", ["stim", "quantum_circuit"])
def test_round4_inline_measurement_outputs(backend) -> None:
    q, m = QReg("q", 2), CReg("m", 2)
    ops = (qb.X(q[0]), qb.Measure(q) > m)
    inline, declared = SlrConverter(Main(q, *ops)), SlrConverter(Main(q, m, *ops))
    if backend == "stim":
        actual, records = inline.stim_with_records()
        expected, declared_records = declared.stim_with_records()
        assert actual == inline.stim() == expected
        assert records == declared_records == (("m", 0), ("m", 1))
        assert actual.compile_sampler().sample(shots=4).tolist() == [[True, False]] * 4
    else:
        actual, expected = inline.quantum_circuit(), declared.quantum_circuit()
        assert list(actual.items()) == list(expected.items())
        assert [params["var_output"] for gate, _, params in actual.items() if gate == "Measure"] == [
            {0: ("m", 0)},
            {1: ("m", 1)},
        ]
        assert actual.metadata["qvars"] == expected.metadata["qvars"]


def test_round4_qasm_inline_measurement_diagnostic() -> None:
    q = QReg("q", 2)
    with pytest.raises(ValueError, match=r"'m'.*QASM needs every classical register declared.*declare it in Main"):
        SlrConverter(Main(q, qb.X(q[0]), qb.Measure(q) > CReg("m", 2))).qasm()


@pytest.mark.parametrize("backend", ["qasm", "qir", "stim", "quantum_circuit"])
def test_round4_direct_child_constructor_placement(backend) -> None:
    def build(direct):
        root = QAlloc(4, name="root")
        first = QAlloc(1, parent=root, name="first") if direct else root.child(1, name="first")
        child = QAlloc(3, parent=root, name="child") if direct else root.child(3, name="child")
        leaf = QAlloc(1, parent=child, name="leaf") if direct else child.child(1, name="leaf")
        assert root.available == 0
        assert child.available == 2
        return Main(root, first, child, leaf, qb.X(first[0]), qb.X(leaf[0])), (first, child, leaf)

    direct, direct_children = build(True)
    method, method_children = build(False)
    assert [child.parent_indices for child in direct_children] == [child.parent_indices for child in method_children]
    assert [child.parent_indices for child in direct_children] == [(0,), (1, 2, 3), (0,)]
    actual, expected = getattr(SlrConverter(direct), backend)(), getattr(SlrConverter(method), backend)()
    if backend == "quantum_circuit":
        assert list(actual.items()) == list(expected.items())
        assert [locations for gate, locations, _ in actual.items() if gate == "X"] == [{0}, {1}]
    else:
        assert str(actual) == str(expected)
    assert "x root[1];" in SlrConverter(direct).qasm()


@pytest.mark.parametrize("bad", ["negative", "full", "released"])
def test_round4_direct_child_constructor_validation(bad) -> None:
    def construct(direct):
        root = QAlloc(2, name="root")
        size = -1 if bad == "negative" else 1
        if bad == "full":
            root.child(2)
        elif bad == "released":
            root.release()
        with pytest.raises((ValueError, RuntimeError)) as error:
            QAlloc(size, parent=root) if direct else root.child(size)
        assert root.available == (0 if bad == "full" else 2)
        return type(error.value), str(error.value)

    assert construct(True) == construct(False)


def test_round4_released_child_slots_can_be_reused() -> None:
    root = QAlloc(2, name="root")
    first = QAlloc(2, parent=root, name="first")
    gate = qb.X(first[0])
    first.release()
    assert root.available == 2
    second = QAlloc(2, parent=root, name="second")
    assert second.parent_indices == first.parent_indices == (0, 1)
    program = Main(root, first, gate, second, qb.X(second[1]))
    code = SlrConverter(program).qasm()
    assert "x root[0];" in code
    assert "x root[1];" in code
    root.release()
    assert second.is_released
