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
        AllocatorDecl(name="root", capacity=5),
        AllocatorDecl(name="first", capacity=2, parent="root"),
        AllocatorDecl(name="child", capacity=3, parent="root"),
        AllocatorDecl(name="grandchild", capacity=2, parent="child"),
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
