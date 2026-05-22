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

"""AST to QIR (Quantum Intermediate Representation) code generator.

This module transforms AST nodes into QIR using LLVM IR generation.
QIR is an LLVM-based intermediate representation for quantum programs.

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.codegen import AstToQir

    ast = slr_to_ast(slr_program)
    generator = AstToQir()
    llvm_ir = generator.generate(ast)
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
from pecos.slr.ast.codegen._prep_tail import prep_tail
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    AssignOp,
    BarrierOp,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    PrintOp,
    RegisterDecl,
    RepeatStmt,
    ReturnOp,
    SlotRef,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import (
        Expression,
        Program,
        Statement,
    )

# Optional LLVM dependency - imported at module level for efficiency
try:
    from pecos_rslib_llvm import ir as llvm_ir

    LLVM_AVAILABLE = True
except ImportError:
    llvm_ir = None  # type: ignore[assignment]
    LLVM_AVAILABLE = False

# Mapping from AST GateKind to QIR gate names
GATE_TO_QIR: dict[GateKind, str] = {
    # Single-qubit Paulis
    GateKind.X: "x",
    GateKind.Y: "y",
    GateKind.Z: "z",
    # Hadamard
    GateKind.H: "h",
    # Phase gates
    GateKind.T: "t",
    GateKind.Tdg: "t__adj",
    # Square root gates - mapped to S variants
    GateKind.SZ: "s",
    GateKind.SZdg: "s__adj",
    # Rotation gates
    GateKind.RX: "rx",
    GateKind.RY: "ry",
    GateKind.RZ: "rz",
    # Two-qubit gates -- only the qir-qis ALLOWED_QIS_FNS that
    # actually execute through `qir_to_qis -> selene`. The native
    # Quantinuum 2q is `rzz` (parameterized); `__quantum__qis__zz__body`
    # is NOT in the allowlist (cf. ~/Repos/qir-qis/src/lib.rs:59).
    # SZZ/SZZdg/SXX/SXXdg/SYY/SYYdg are lowered via `_GATE_DECOMP`
    # to RZZ + 1q Cliffords (verified up-to-phase + end-to-end).
    GateKind.CX: "cnot",
    GateKind.CZ: "cz",
    GateKind.RZZ: "rzz",
}

# Decomposition table: a sequence of (primitive_kind, qubit_idx_tuple,
# params_tuple) steps. Each step's qubit_idx_tuple indexes into the
# input gate's `targets`; params_tuple is the (constant) angles for a
# parameterized primitive (RZZ, RY, RX, RZ). Every entry was found by
# extracting the gate's authoritative unitary from
# `pecos.simulators.StateVec` (or the canonical matrix oracle in
# `tests/pecos/integration/state_sim_tests/gate_matrix_def.py`),
# searching/deriving a decomposition into ONLY the qir-qis ALLOWED
# primitive set (`h, x, y, z, s, s__adj, t, t__adj, rx, ry, rz, rzz,
# rxy, cnot, cz`), verifying it equal up to a GLOBAL PHASE
# (unobservable for measurement-terminated circuits) to the PECOS
# unitary, AND verifying it end-to-end through `qir_to_qis -> selene`
# with discriminating deterministic identities (a no-op lowering
# would fail). For *Clifford* gates the selene Stim backend can
# verify; non-Clifford gates (e.g. CH, T-decompositions, arbitrary-
# angle rotations) use the selene Quest statevector backend.
# Sequences are in CIRCUIT order (first applied first). Decompositions
# minimize 2q-gate count first (2q ops are the hardware cost driver).
# A decomposition step's params slot is either:
#   - a tuple of constant floats (most common, e.g. SZZ -> RZZ(pi/2)), or
#   - a callable that takes the *input* gate's params (e.g. CRZ(theta))
#     and returns the step's params (e.g. (theta/2,) on RZ, (-theta/2,)
#     on RZZ). This lets parameterized controlled-rotation gates thread
#     their angle through a 2q-minimal decomposition.
_DecompParams = tuple[float, ...] | Callable[[tuple[float, ...]], tuple[float, ...]]
_DecompStep = tuple[GateKind, tuple[int, ...], _DecompParams]
_GATE_DECOMP: dict[GateKind, tuple[_DecompStep, ...]] = {
    # ---- single-qubit Clifford sqrt + face rotations ----
    GateKind.SX: ((GateKind.H, (0,), ()), (GateKind.SZ, (0,), ()), (GateKind.H, (0,), ())),
    GateKind.SXdg: ((GateKind.H, (0,), ()), (GateKind.SZdg, (0,), ()), (GateKind.H, (0,), ())),
    GateKind.SY: ((GateKind.H, (0,), ()), (GateKind.X, (0,), ())),
    GateKind.SYdg: ((GateKind.H, (0,), ()), (GateKind.Z, (0,), ())),
    GateKind.F: ((GateKind.SZdg, (0,), ()), (GateKind.H, (0,), ())),
    GateKind.Fdg: ((GateKind.H, (0,), ()), (GateKind.SZ, (0,), ())),
    GateKind.F4: ((GateKind.H, (0,), ()), (GateKind.SZdg, (0,), ())),
    GateKind.F4dg: ((GateKind.SZ, (0,), ()), (GateKind.H, (0,), ())),
    # ---- two-qubit Clifford gates ----
    # SZZ/SZZdg directly via the native parameterized rzz.
    GateKind.SZZ: ((GateKind.RZZ, (0, 1), (math.pi / 2,)),),
    GateKind.SZZdg: ((GateKind.RZZ, (0, 1), (-math.pi / 2,)),),
    # SXX = (H⊗H)·SZZ·(H⊗H); SXXdg with -π/2.
    GateKind.SXX: (
        (GateKind.H, (0,), ()),
        (GateKind.H, (1,), ()),
        (GateKind.RZZ, (0, 1), (math.pi / 2,)),
        (GateKind.H, (0,), ()),
        (GateKind.H, (1,), ()),
    ),
    GateKind.SXXdg: (
        (GateKind.H, (0,), ()),
        (GateKind.H, (1,), ()),
        (GateKind.RZZ, (0, 1), (-math.pi / 2,)),
        (GateKind.H, (0,), ()),
        (GateKind.H, (1,), ()),
    ),
    # SYY = (S⊗S)·(H⊗H)·SZZ·(H⊗H)·(Sdg⊗Sdg) since Y = S·X·S† and
    # XX = (H⊗H)·ZZ·(H⊗H). SYYdg with -π/2.
    GateKind.SYY: (
        (GateKind.SZdg, (0,), ()),
        (GateKind.SZdg, (1,), ()),
        (GateKind.H, (0,), ()),
        (GateKind.H, (1,), ()),
        (GateKind.RZZ, (0, 1), (math.pi / 2,)),
        (GateKind.H, (0,), ()),
        (GateKind.H, (1,), ()),
        (GateKind.SZ, (0,), ()),
        (GateKind.SZ, (1,), ()),
    ),
    GateKind.SYYdg: (
        (GateKind.SZdg, (0,), ()),
        (GateKind.SZdg, (1,), ()),
        (GateKind.H, (0,), ()),
        (GateKind.H, (1,), ()),
        (GateKind.RZZ, (0, 1), (-math.pi / 2,)),
        (GateKind.H, (0,), ()),
        (GateKind.H, (1,), ()),
        (GateKind.SZ, (0,), ()),
        (GateKind.SZ, (1,), ()),
    ),
    # CY = Sdg(target); CX(control,target); S(target).
    GateKind.CY: (
        (GateKind.SZdg, (1,), ()),
        (GateKind.CX, (0, 1), ()),
        (GateKind.SZ, (1,), ()),
    ),
    # CH = (I_c x Ry(-pi/4)_t) . CX(c,t) . (I_c x Ry(pi/4)_t) -- 1 CX
    # (the 2q-minimal Clifford+rotation form; conjugation by Ry maps
    # X to H since Ry(-pi/4) X Ry(pi/4) = cos(-pi/4) X - sin(-pi/4) Z
    # = (X+Z)/sqrt(2) = H). The PECOS oracle CH() in gate_matrix_def
    # uses a Clifford+T 2-CX form; ours matches it up to global phase
    # (max_err 3e-14) and matches textbook block-diag(I,H) exactly.
    GateKind.CH: (
        (GateKind.RY, (1,), (math.pi / 4,)),
        (GateKind.CX, (0, 1), ()),
        (GateKind.RY, (1,), (-math.pi / 4,)),
    ),
    # ---- parameterized controlled rotations ----
    # CRZ(theta) = (RZ(theta/2) o RZ(theta/2)) . RZZ(-theta/2): 1 RZZ,
    # 2 single-qubit RZ. The RZ on the control absorbs the e^{i theta/2}
    # phase that PECOS's R*(theta) all carry (otherwise it would be a
    # c=1-only relative phase, which is observable). Verified against
    # gate_matrix_def.CRZ(theta) for 5 random angles.
    GateKind.CRZ: (
        (GateKind.RZZ, (0, 1), lambda p: (-p[0] / 2,)),
        (GateKind.RZ, (0,), lambda p: (p[0] / 2,)),
        (GateKind.RZ, (1,), lambda p: (p[0] / 2,)),
    ),
    # CRX(theta) = (I o H) . CRZ(theta) . (I o H): conjugate CRZ by H
    # on the target since H.Z.H = X. Same 1 RZZ.
    GateKind.CRX: (
        (GateKind.H, (1,), ()),
        (GateKind.RZZ, (0, 1), lambda p: (-p[0] / 2,)),
        (GateKind.RZ, (0,), lambda p: (p[0] / 2,)),
        (GateKind.RZ, (1,), lambda p: (p[0] / 2,)),
        (GateKind.H, (1,), ()),
    ),
    # CRY(theta) = (I o (S.H)) . CRZ(theta) . (I o (H.Sdg)): conjugate
    # CRZ by (S.H) on the target since S.X.Sdg = Y (and H.Z.H = X).
    # Same 1 RZZ.
    GateKind.CRY: (
        (GateKind.SZdg, (1,), ()),
        (GateKind.H, (1,), ()),
        (GateKind.RZZ, (0, 1), lambda p: (-p[0] / 2,)),
        (GateKind.RZ, (0,), lambda p: (p[0] / 2,)),
        (GateKind.RZ, (1,), lambda p: (p[0] / 2,)),
        (GateKind.H, (1,), ()),
        (GateKind.SZ, (1,), ()),
    ),
}

# Gates with rotation parameters
PARAMETERIZED_GATES = {GateKind.RX, GateKind.RY, GateKind.RZ, GateKind.RZZ}

# Two-qubit gates
TWO_QUBIT_GATES = {GateKind.CX, GateKind.CZ, GateKind.RZZ}


def _param_to_radians(p: object) -> float:
    """Resolve a gate angle param to a signed-radians float.

    Accepts a `LiteralExpr` wrapping either a typed `Angle` or a bare
    number, or a raw number (decomposition steps thread raw floats).
    Typed angles use the signed principal value so the float-based
    decomposition arithmetic (`-theta/2`) avoids the global-phase flip
    that the unsigned `[0, 2pi)` form would introduce at the wrap point.
    """
    from pecos.slr.angle import Angle  # noqa: PLC0415  (avoid import cycle)

    value = p.value if isinstance(p, LiteralExpr) else p
    if isinstance(value, Angle):
        return value.value.to_radians_signed()
    return float(value)


def _require_typed_angle_params(node: GateOp, backend: str) -> None:
    """Fail loud if a parameterized user gate has a non-`Angle` param.

    Enforces the typed-AST-dialect contract uniformly across backends: a
    parameterized `GateOp` reaching codegen from the user / direct-AST path
    must carry typed `Angle` literals (`rad(...)` / `turns(...)`), not bare
    floats. (Internal decomposition steps thread raw floats but reach the
    per-gate emitters directly, not this top-level entry.)
    """
    from pecos.slr.angle import Angle  # noqa: PLC0415  (avoid import cycle)

    if not node.gate.is_parameterized:
        return
    for p in node.params:
        if not (isinstance(p, LiteralExpr) and isinstance(p.value, Angle)):
            gate_name = getattr(node.gate, "name", node.gate)
            msg = (
                f"{backend} codegen: parameterized gate {gate_name!r} requires typed `Angle` "
                f"params (use `rad(...)` / `turns(...)` in SLR); got {p!r}."
            )
            raise NotImplementedError(msg)


@dataclass
class QirCodeGenContext:
    """Context for QIR code generation."""

    qubit_map: dict[tuple[str, int], int] = field(default_factory=dict)
    qubit_count: int = 0
    creg_map: dict[str, int] = field(default_factory=dict)  # name -> size
    qreg_sizes: dict[str, int] = field(default_factory=dict)  # name -> capacity
    measurement_count: int = 0
    allocator_parents: dict[str, str | None] = field(default_factory=dict)
    allocator_offsets: dict[str, int] = field(default_factory=dict)
    # Static logical permutation, mirroring the Guppy linearity
    # tracker's `.permute()` (compile-time relabel; QIR/Selene have no
    # runtime permute intrinsic). Maps a logical (reg, index) ref to
    # the (reg, index) whose storage it should resolve to. Consulted
    # at every qubit-ref and classical-bit-ref lowering.
    permutation_map: dict[tuple[str, int], tuple[str, int]] = field(default_factory=dict)

    def get_root_allocator(self, name: str) -> str:
        """Get the root allocator for a given allocator name."""
        current = name
        while self.allocator_parents.get(current) is not None:
            current = self.allocator_parents[current]
        return current

    def get_absolute_index(self, allocator: str, index: int) -> int:
        """Get the absolute index in the root allocator."""
        offset = self.allocator_offsets.get(allocator, 0)
        return offset + index

    def get_qubit_index(self, allocator: str, index: int) -> int:
        """Get the global qubit index for an allocator slot.

        For child allocators, translates to root allocator with computed offset.
        """
        # Resolve any active logical permutation first (identity until
        # a Permute runs; decl-time pre-population sees the empty map,
        # so real qubits are still allocated 1:1).
        allocator, index = self.permutation_map.get((allocator, index), (allocator, index))

        # Translate to root allocator and absolute index
        root = self.get_root_allocator(allocator)
        abs_index = self.get_absolute_index(allocator, index)

        key = (root, abs_index)
        if key not in self.qubit_map:
            self.qubit_map[key] = self.qubit_count
            self.qubit_count += 1
        return self.qubit_map[key]


class AstToQir:
    """Transforms AST programs into QIR using recursive descent.

    Generates LLVM IR suitable for QIR-compatible execution environments.

    Usage:
        generator = AstToQir()
        llvm_ir = generator.generate(ast_program)
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self.context = QirCodeGenContext()
        self._module = None
        self._builder = None
        self._types = None
        self._main_func = None
        self._gate_cache: dict[str, Any] = {}
        self._creg_ptrs: dict[str, Any] = {}
        self._creg_funcs = None

    def generate(self, program: Program) -> str:
        """Generate QIR (LLVM IR) for a program.

        Args:
            program: The AST Program to generate code for.

        Returns:
            QIR as an LLVM IR string.

        Raises:
            ImportError: If LLVM dependencies are not available.
        """
        if not LLVM_AVAILABLE:
            msg = "LLVM dependencies not available. Install with 'pip install pecos[qir]'"
            raise ImportError(msg)

        program = flatten_block_calls(program)

        self.context = QirCodeGenContext()
        self._gate_cache = {}
        self._creg_ptrs = {}

        # Setup LLVM module
        self._module = llvm_ir.Module(name="ast_qir_module")

        # Setup types
        qubit_ty = self._module.context.get_identified_type("Qubit")
        result_ty = self._module.context.get_identified_type("Result")

        self._types = {
            "void": llvm_ir.VoidType(),
            "bool": llvm_ir.IntType(1),
            "int": llvm_ir.IntType(64),
            "double": llvm_ir.DoubleType(),
            "qubit_ptr": qubit_ty.as_pointer(),
            "result_ptr": result_ty.as_pointer(),
            "tag": llvm_ir.IntType(8).as_pointer(),
        }

        # Setup creg helper functions
        self._setup_creg_funcs()

        # Standard QIR classical model. A measurement lowers to a
        # static `%Result*` slot -> `__quantum__qis__mz__body` ->
        # `__quantum__rt__read_result` -> `store` into a per-CReg mutable
        # `[N x i1]` entry-block `alloca` buffer. `%Result*`
        # is the existing `result_ptr` type.
        self._mz_body = self._declare_function(
            "__quantum__qis__mz__body",
            self._types["void"],
            [self._types["qubit_ptr"], self._types["result_ptr"]],
        )
        self._read_result = self._declare_function(
            "__quantum__rt__read_result",
            self._types["bool"],
            [self._types["result_ptr"]],
        )

        # Setup main function
        main_fnty = llvm_ir.FunctionType(self._types["void"], [])
        self._main_func = llvm_ir.Function(self._module, main_fnty, name="main")
        entry_block = self._main_func.append_basic_block(name="entry")
        self._builder = llvm_ir.IRBuilder(entry_block)

        # Setup operator map
        self._setup_op_map()

        # Process declarations
        self._process_declarations(program)

        # Process body statements
        for stmt in program.body:
            self._process_statement(stmt)

        # Generate results output
        self._generate_results()

        # Return void
        self._builder.ret_void()

        # Return the LLVM IR with attributes
        return self._finalize_module()

    def _setup_creg_funcs(self) -> None:
        """Declare the standard classical-output runtime function.

        The static `[N x i1]` CReg model replaced the bespoke
        `create_creg`/`get_creg_bit`/
        `set_creg_bit`/`get_int_from_creg`/`set_creg_to_int`/`mz_to_creg_bit`
        runtime helpers with native `alloca`/`store`/`load`/`gep`/`zext`,
        so only the standard `__quantum__rt__int_record_output` remains.
        """
        self._creg_funcs = {
            "int_result": self._declare_function(
                "__quantum__rt__int_record_output",
                self._types["void"],
                [self._types["int"], self._types["tag"]],
            ),
        }

    def _declare_function(self, name: str, ret_ty: Any, arg_tys: list) -> Any:
        """Declare an LLVM function."""
        fnty = llvm_ir.FunctionType(ret_ty, arg_tys)
        return llvm_ir.Function(self._module, fnty, name=name)

    def _setup_op_map(self) -> None:
        """Setup binary operator mapping."""
        # CReg comparisons are UNSIGNED (the CReg's `[N x i1]` buffer
        # packs to an i64 unsigned bit pattern via `_pack_creg`). The
        # `icmp_unsigned` choice matters only when bit 63 of a 64-bit
        # CReg is set; for narrower CRegs (or for bit-level `c[i]`
        # comparisons that zext to i64 with the top 63 bits zero)
        # signed and unsigned agree. Switching is safe -- no existing
        # test asserts the signed semantics on a high-bit-set 64-bit
        # CReg, and unsigned is the correct interpretation of the
        # bit-pattern semantics SLR exposes.
        self._op_map = {
            BinaryOp.EQ: lambda lhs, rhs: self._builder.icmp_unsigned("==", lhs, rhs),
            BinaryOp.NE: lambda lhs, rhs: self._builder.icmp_unsigned("!=", lhs, rhs),
            BinaryOp.LT: lambda lhs, rhs: self._builder.icmp_unsigned("<", lhs, rhs),
            BinaryOp.GT: lambda lhs, rhs: self._builder.icmp_unsigned(">", lhs, rhs),
            BinaryOp.LE: lambda lhs, rhs: self._builder.icmp_unsigned("<=", lhs, rhs),
            BinaryOp.GE: lambda lhs, rhs: self._builder.icmp_unsigned(">=", lhs, rhs),
            BinaryOp.MUL: self._builder.mul,
            BinaryOp.DIV: self._builder.udiv,
            BinaryOp.XOR: self._builder.xor,
            BinaryOp.AND: self._builder.and_,
            BinaryOp.OR: self._builder.or_,
            BinaryOp.ADD: self._builder.add,
            BinaryOp.SUB: self._builder.sub,
            BinaryOp.RSHIFT: self._builder.lshr,
            BinaryOp.LSHIFT: self._builder.shl,
        }

    def _process_declarations(self, program: Program) -> None:
        """Process declarations to allocate qubits and classical registers."""
        # First pass: collect allocator parent info
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self.context.allocator_parents[decl.name] = decl.parent

        if program.allocator:
            self.context.allocator_parents[program.allocator.name] = program.allocator.parent

        # Calculate offsets for child allocators
        self._calculate_allocator_offsets(program)

        # Process allocator declarations - only allocate for root allocators
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self.context.qreg_sizes[decl.name] = decl.capacity
                if decl.parent is None:
                    for i in range(decl.capacity):
                        self.context.get_qubit_index(decl.name, i)
            elif isinstance(decl, RegisterDecl):
                # A CReg is a mutable `[N x i1]` buffer in the
                # entry block (declarations are processed at entry, before
                # any control flow, so the builder is positioned there),
                # zero-initialised so unmeasured/unset bits read 0. The
                # record pack is a single `i64` for `int_record_output`, so
                # the model caps at 64 bits. A >64-bit CReg must fail LOUD
                # here -- silently dropping its storage/output (the old
                # `create_creg` `size < 64` behaviour) is a miscompile.
                # Fail BEFORE recording any state (no partial creg_map).
                if decl.size > 64:
                    msg = (
                        f"QIR codegen: CReg {decl.name!r} has {decl.size} bits, "
                        "but the static classical model packs each CReg "
                        "into a single i64 for "
                        "`__quantum__rt__int_record_output` (64-bit cap). "
                        ">64-bit CRegs are not supported by the QIR backend."
                    )
                    raise NotImplementedError(msg)
                self.context.creg_map[decl.name] = decl.size
                arr_ty = llvm_ir.ArrayType(self._types["bool"], decl.size)
                creg_ptr = self._builder.alloca(arr_ty, decl.name)
                self._builder.store(creg_ptr, llvm_ir.Constant(arr_ty))
                self._creg_ptrs[decl.name] = creg_ptr

        if program.allocator:
            self.context.qreg_sizes[program.allocator.name] = program.allocator.capacity
        if program.allocator and program.allocator.parent is None:
            for i in range(program.allocator.capacity):
                self.context.get_qubit_index(program.allocator.name, i)

    def _calculate_allocator_offsets(self, program: Program) -> None:
        """Calculate the offset of each child allocator within its parent."""
        parent_next_offset: dict[str, int] = {}

        # Root allocators have offset 0
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl) and decl.parent is None:
                self.context.allocator_offsets[decl.name] = 0

        if program.allocator and program.allocator.parent is None:
            self.context.allocator_offsets[program.allocator.name] = 0

        # Process child allocators in declaration order
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl) and decl.parent is not None:
                parent = decl.parent
                if parent not in parent_next_offset:
                    parent_next_offset[parent] = 0

                parent_offset = self.context.allocator_offsets.get(parent, 0)
                self.context.allocator_offsets[decl.name] = parent_offset + parent_next_offset[parent]
                parent_next_offset[parent] += decl.capacity

    def _process_statement(self, stmt: Statement) -> None:
        """Process a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._process_gate(stmt)
        elif isinstance(stmt, MeasureOp):
            self._process_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._process_prepare(stmt)
        elif isinstance(stmt, BarrierOp):
            self._process_barrier(stmt)
        elif isinstance(stmt, AssignOp):
            self._process_assign(stmt)
        elif isinstance(stmt, IfStmt):
            self._process_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._process_while(stmt)
        elif isinstance(stmt, ForStmt):
            self._process_for(stmt)
        elif isinstance(stmt, RepeatStmt):
            self._process_repeat(stmt)
        elif isinstance(stmt, ParallelBlock):
            self._process_parallel(stmt)
        elif isinstance(stmt, PermuteOp):
            self._process_permute(stmt)
        elif isinstance(stmt, ReturnOp):
            self._process_return(stmt)
        elif isinstance(stmt, PrintOp):
            # Classical-output streaming (`Print` -> Guppy `result(...)`)
            # is unimplemented in the QIR backend. Silently dropping it
            # loses observable program output -- fail LOUD instead.
            msg = (
                "QIR codegen does not support Print (classical output "
                "streaming is unimplemented; silently dropping it would "
                "lose observable program output)."
            )
            raise NotImplementedError(msg)

    def _process_return(self, node: ReturnOp) -> None:
        """Validate that returned CLASSICAL registers have QIR storage.

        `_generate_results` records every Main-declared CReg, but a
        CReg surfaced ONLY via `Return(creg)` (never measured /
        assigned / read) reaches no other `_require_creg` site, so
        an inline / local-scope returned CReg produced ZERO recorded
        output for an explicit `Return` -- the build succeeded and
        validated, the program just silently returned nothing
        (same silent-output-loss class as the four
        point-of-use sites). Qubit returns record no classical
        output and are skipped via per-value provenance
        (`ReturnOp.value_kinds` from `_convert_return`), so a
        `Return(qreg)` is not false-rejected AND a returned inline
        CReg whose name collides with a declared QReg is still
        validated (a name-membership skip was unsound).
        """
        # Provenance comes from `_convert_return` (it knows the real
        # QReg/CReg object), NOT from a name-membership guess: a
        # returned inline CReg can share a declared QReg's name, which
        # a `qubit_map`-name skip silently mistook for a qubit return
        # and dropped. Unknown kind
        # ("" -- e.g. a directly-constructed ReturnOp) falls back to
        # "classical", the fail-loud-safe default.
        kinds = node.value_kinds
        for i, value in enumerate(node.values):
            kind = kinds[i] if i < len(kinds) else "classical"
            if isinstance(value, str):
                if kind == "quantum":
                    continue  # qubit-register return: no classical record
                self._require_creg(value)
            elif isinstance(value, BitRef):
                self._require_creg(value.register)
            elif isinstance(value, BitExpr):
                self._require_creg(value.ref.register)

    def _process_gate(self, node: GateOp) -> None:
        """Process a gate operation."""
        # Fail-loud arity guard (defense-in-depth vs the angle-first
        # mis-order footgun): a parameterized gate with fewer targets
        # than its qubit arity would otherwise SILENTLY emit no call
        # (the per-target emit loops just iterate zero/too-few times),
        # or hit a raw IndexError in the decomposition path. The SLR
        # `QGate.__call__` already rejects the mis-ordered call at the
        # source; this guards a malformed GateOp reaching codegen from
        # any other path. Multi-target (parallel) application is fine
        # (len >= arity).
        if node.gate.is_parameterized and len(node.targets) < node.gate.arity:
            gate_name = getattr(node.gate, "name", node.gate)
            msg = (
                f"QIR codegen: parameterized gate {gate_name!r} has "
                f"{len(node.targets)} qubit target(s) but needs at least "
                f"{node.gate.arity} (a mis-ordered `gate(qubit, angle)` call "
                "drops the qubit). Call it as `gate(angle, qubit...)`."
            )
            raise NotImplementedError(msg)

        # Typed-angle guard: a user/direct-AST parameterized gate's params
        # must be typed `Angle` literals (matches the Guppy backend and the
        # typed-AST-dialect contract). Internal decomposition steps thread
        # raw floats but reach `_process_*_gate` directly, bypassing here.
        _require_typed_angle_params(node, "QIR")

        qir_name = GATE_TO_QIR.get(node.gate)
        if qir_name is None and node.gate in _GATE_DECOMP:
            # A gate with no direct QIR primitive but a verified
            # decomposition into the qir-qis ALLOWED primitive set.
            # Emit each step in circuit order, routing its qubits
            # through the input gate's `targets` and threading params
            # (constant for non-parameterized gates like SZZ -> RZZ(pi/2);
            # callable on the input gate's params for parameterized
            # gates like CRZ(theta) -> RZZ(-theta/2)). LiteralExpr
            # bracket-params are unwrapped to floats here so the
            # callable can do arithmetic on them; non-literal
            # expressions (VarExpr / BinaryExpr at gate-param position)
            # are not yet supported for parameterized decomposition
            # (out of scope; classical-var lowering covers it).
            input_params_raw = tuple(node.params or ())
            for prim_kind, idxs, params_spec in _GATE_DECOMP[node.gate]:
                prim_targets = tuple(node.targets[i] for i in idxs)
                if callable(params_spec):
                    try:
                        input_params_resolved = tuple(_param_to_radians(p) for p in input_params_raw)
                    except (AttributeError, TypeError) as exc:
                        msg = (
                            f"Parameterized decomposition of gate {node.gate.name} requires literal "
                            f"params; got non-literal: {input_params_raw}"
                        )
                        raise NotImplementedError(msg) from exc
                    prim_params = params_spec(input_params_resolved)
                else:
                    prim_params = params_spec
                prim_node = replace(node, gate=prim_kind, targets=prim_targets, params=prim_params)
                if prim_kind in TWO_QUBIT_GATES:
                    self._process_two_qubit_gate(prim_node, GATE_TO_QIR[prim_kind])
                else:
                    self._process_single_qubit_gate(prim_node, GATE_TO_QIR[prim_kind])
            return
        if qir_name is None:
            # A gate with no
            # GATE_TO_QIR entry was SILENTLY DROPPED -- valid QIR,
            # wrong semantics, qir-qis-uncatchable. Fail
            # loud instead of miscompiling. Gates with a real QIR
            # lowering should be added to GATE_TO_QIR (a feature);
            # until then a program using one must not be silently
            # mis-emitted.
            gate_name = getattr(node.gate, "name", node.gate)
            msg = (
                f"QIR codegen: gate {gate_name!r} has no QIR lowering "
                "(not in GATE_TO_QIR). Emitting QIR without it would be "
                "a silent miscompile; it is not supported by the QIR "
                "backend."
            )
            raise NotImplementedError(msg)

        if node.gate in TWO_QUBIT_GATES:
            self._process_two_qubit_gate(node, qir_name)
        else:
            self._process_single_qubit_gate(node, qir_name)

    def _process_single_qubit_gate(self, node: GateOp, qir_name: str) -> None:
        """Process a single-qubit gate."""
        # Get or create gate function
        gate_func = self._get_or_create_gate(
            qir_name,
            has_params=node.gate in PARAMETERIZED_GATES,
            num_qubits=1,
        )

        for target in node.targets:
            qubit_ptr = self._get_qubit_ptr(target)

            args = []
            if node.gate in PARAMETERIZED_GATES and node.params:
                # An angle param reaches here as a `LiteralExpr` wrapping a
                # typed `Angle` (or a raw float from a decomposition step);
                # resolve to signed radians for the QIR `double`.
                args.extend(llvm_ir.Constant(self._types["double"], _param_to_radians(p)) for p in node.params)
            args.append(qubit_ptr)

            self._builder.call(gate_func, args, name="")

    def _process_two_qubit_gate(self, node: GateOp, qir_name: str) -> None:
        """Process a two-qubit gate."""
        gate_func = self._get_or_create_gate(
            qir_name,
            has_params=node.gate in PARAMETERIZED_GATES,
            num_qubits=2,
        )

        if len(node.targets) >= 2:
            q0_ptr = self._get_qubit_ptr(node.targets[0])
            q1_ptr = self._get_qubit_ptr(node.targets[1])

            args = []
            if node.gate in PARAMETERIZED_GATES and node.params:
                # An angle param reaches here as a `LiteralExpr` wrapping a
                # typed `Angle` (or a raw float from a decomposition step);
                # resolve to signed radians for the QIR `double`.
                args.extend(llvm_ir.Constant(self._types["double"], _param_to_radians(p)) for p in node.params)
            args.extend([q0_ptr, q1_ptr])

            self._builder.call(gate_func, args, name="")

    def _get_or_create_gate(
        self,
        qir_name: str,
        *,
        has_params: bool,
        num_qubits: int,
    ) -> Any:
        """Get or create a QIR gate function declaration."""
        cache_key = f"{qir_name}_{has_params}_{num_qubits}"
        if cache_key in self._gate_cache:
            return self._gate_cache[cache_key]

        # Build argument types
        arg_tys = []
        if has_params:
            arg_tys.append(self._types["double"])
        arg_tys.extend([self._types["qubit_ptr"]] * num_qubits)

        # Build mangled name
        suffix = "__body" if "adj" not in qir_name else ""
        mangled_name = f"__quantum__qis__{qir_name}{suffix}"

        fnty = llvm_ir.FunctionType(self._types["void"], arg_tys)
        gate_func = llvm_ir.Function(self._module, fnty, name=mangled_name)

        self._gate_cache[cache_key] = gate_func
        return gate_func

    def _get_qubit_ptr(self, target: SlotRef) -> Any:
        """Get a qubit pointer for a target."""
        qubit_index = self.context.get_qubit_index(target.allocator, target.index)
        return llvm_ir.Constant(self._types["int"], qubit_index).inttoptr(
            self._types["qubit_ptr"],
        )

    def _require_creg(self, reg_name: str) -> None:
        """Fail LOUD if `reg_name` has no entry-block CReg storage.

        Only CRegs declared at Main scope (`program.declarations`)
        get an `alloca [N x i1]` in `_process_declarations`. An
        inline / local-scope CReg -- e.g. one created in a block and
        only surfaced via `Return(creg)` -- has no storage, so every
        measure/assign/read against it used to be SILENTLY skipped
        (the store dropped, a read folded to constant 0) and the
        explicit returned value vanished from the QIS records (the
        `docs.inline_measure_creg` defect surfaced). Mirror the
        fail-loud doctrine: a silent miscompile must become a loud
        `NotImplementedError`, not a buried wrong answer.
        """
        if reg_name not in self._creg_ptrs:
            msg = (
                f"QIR codegen: classical register {reg_name!r} is "
                "used/measured/returned but was not declared at Main "
                "scope, so it has no QIR storage. Inline / local-scope "
                "CRegs are not supported by the QIR backend (their "
                "values would be silently dropped from the recorded "
                f"output). Declare {reg_name!r} in Main(...)."
            )
            raise NotImplementedError(msg)

    def _creg_bit_ptr(self, reg_name: str, index: int) -> Any:
        """`getelementptr [N x i1], [N x i1]* %creg, i64 0, i64 index`.

        Emitted at point-of-use (not cached across blocks) so the pointer
        always dominates its uses under control flow.
        """
        # Resolve any active logical permutation (classical bits, like
        # qubits, are relabelled by a Permute -- mirrors Guppy's
        # mem_swap-based classical permute).
        reg_name, index = self.context.permutation_map.get((reg_name, index), (reg_name, index))
        return self._builder.gep(
            self._creg_ptrs[reg_name],
            [
                llvm_ir.Constant(self._types["int"], 0),
                llvm_ir.Constant(self._types["int"], index),
            ],
            name="",
        )

    def _as_i1(self, value: Any) -> Any:
        """Coerce an evaluated expression to `i1` (bit-store / predicate)."""
        if value.type == self._types["bool"]:
            return value
        return self._builder.icmp_signed(
            "!=",
            value,
            llvm_ir.Constant(self._types["int"], 0),
        )

    def _as_i64(self, value: Any) -> Any:
        """Coerce an evaluated expression to `i64` (the canonical width)."""
        if value.type == self._types["bool"]:
            return self._builder.zext(value, self._types["int"])
        return value

    def _process_measure(self, node: MeasureOp) -> None:
        """Process a measurement operation.

        Every measured target emits `__quantum__qis__mz__body(q, %Result*)`
        against a static result slot; `read_result` + `store` into the CReg
        buffer only when a classical result target exists (qir-qis accepts
        `mz` without `read_result`, but rejects `read_result` before `mz`).
        """
        for i, target in enumerate(node.targets):
            self.context.measurement_count += 1
            slot = self.context.measurement_count - 1
            qubit_ptr = self._get_qubit_ptr(target)
            result_ptr = llvm_ir.Constant(self._types["int"], slot).inttoptr(
                self._types["result_ptr"],
            )
            self._builder.call(self._mz_body, [qubit_ptr, result_ptr], name="")

            if i < len(node.results):
                result = node.results[i]
                self._require_creg(result.register)
                bit = self._builder.call(
                    self._read_result,
                    [result_ptr],
                    name="",
                )
                self._builder.store(
                    self._creg_bit_ptr(result.register, result.index),
                    bit,
                )

    def _process_prepare(self, node: PrepareOp) -> None:
        """Process a prepare/reset operation (Z-reset + canonical basis tail)."""
        tail = prep_tail(node.basis)
        if node.slots is None:
            # prepare_all is a pre-existing no-op gap; a NON-PZ
            # prepare_all silently doing nothing would be a basis
            # miscompile -- fail loud rather than extend the gap.
            if tail:
                msg = f"QIR codegen: prepare_all with non-PZ basis {node.basis!r} is not supported"
                raise NotImplementedError(msg)
            return

        reset_func = self._get_or_create_gate("reset", has_params=False, num_qubits=1)
        tail_funcs = [(self._get_or_create_gate(GATE_TO_QIR[gk], has_params=False, num_qubits=1)) for gk in tail]

        for slot in node.slots:
            qubit_ptr = self._get_qubit_ptr(
                SlotRef(allocator=node.allocator, index=slot),
            )
            self._builder.call(reset_func, [qubit_ptr], name="")
            for func in tail_funcs:
                self._builder.call(func, [qubit_ptr], name="")

    def _process_barrier(self, node: BarrierOp) -> None:
        """Process a barrier operation."""
        # Collect all qubits involved
        qubits = []
        if node.allocators:
            for alloc in node.allocators:
                qubits.extend(
                    self._get_qubit_ptr(SlotRef(allocator=key[0], index=key[1]))
                    for key in self.context.qubit_map
                    if key[0] == alloc
                )

        if not qubits:
            return

        # Create barrier function if needed
        barrier_name = f"__quantum__qis__barrier{len(qubits)}__body"
        fnty = llvm_ir.FunctionType(
            self._types["void"],
            [self._types["qubit_ptr"]] * len(qubits),
        )
        barrier_func = llvm_ir.Function(self._module, fnty, name=barrier_name)
        self._builder.call(barrier_func, qubits, name="")

    def _process_assign(self, node: AssignOp) -> None:
        """Process an assignment operation."""
        if isinstance(node.target, BitRef):
            reg_name = node.target.register
            self._require_creg(reg_name)
            rhs = self._as_i1(self._eval_expression(node.value))
            self._builder.store(
                self._creg_bit_ptr(reg_name, node.target.index),
                rhs,
            )
        elif isinstance(node.target, str):
            # Whole-CReg `c.set(int)` (converter.py:928/930 -> target=str).
            # Unpack the i64 value bit-by-bit into the buffer.
            reg_name = node.target
            self._require_creg(reg_name)
            size = self.context.creg_map.get(reg_name, 0)
            val = self._as_i64(self._eval_expression(node.value))
            for i in range(size):
                shifted = self._builder.lshr(
                    val,
                    llvm_ir.Constant(self._types["int"], i),
                )
                self._builder.store(
                    self._creg_bit_ptr(reg_name, i),
                    self._builder.trunc(shifted, self._types["bool"]),
                )

    def _eval_expression(self, expr: Expression) -> Any:
        """Evaluate an expression to an LLVM value."""
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, bool):
                return llvm_ir.Constant(self._types["int"], 1 if expr.value else 0)
            if isinstance(expr.value, int):
                return llvm_ir.Constant(self._types["int"], expr.value)
            if isinstance(expr.value, float):
                return llvm_ir.Constant(self._types["double"], expr.value)
            return llvm_ir.Constant(self._types["int"], expr.value)

        if isinstance(expr, BitExpr):
            reg_name = expr.ref.register
            self._require_creg(reg_name)
            # `load i1, gep c[i]` then `zext -> i64` (canonical width).
            bit = self._builder.load(
                self._creg_bit_ptr(reg_name, expr.ref.index),
                name="",
            )
            return self._builder.zext(bit, self._types["int"])

        if isinstance(expr, VarExpr):
            # A classical `VarExpr` denotes a whole CReg used as a
            # scalar (e.g. `If(m == 0)`, `o.set(m + n)`, Steane
            # `smid_flag_x`). SLR has no scalar-integer classical type
            # (verified: `pecos.slr.vars` exposes only Reg/CReg/Bit/
            # SymbolicElem/LoopVar). A `LoopVar` used as a symbolic index
            # is resolved by `For` unrolling; a `LoopVar` appearing as a
            # bare classical scalar (e.g. `If(i == 0)`) is NOT substituted
            # by `_process_for` and reaches here as `VarExpr(name="i")`,
            # where `_require_creg` fails it loud (no CReg named `i`) --
            # never silent-0. The lowering
            # is the existing i64 pack (`OR_i (zext c[i] << i)`)
            # factored out of `_generate_results` as `_pack_creg`. A
            # `VarExpr` whose name is not a declared Main-scope CReg
            # fails LOUD via `_require_creg`, preserving the
            # anti-silent-0 guarantee.
            self._require_creg(expr.name)
            return self._pack_creg(expr.name)

        if isinstance(expr, BinaryExpr):
            left = self._eval_expression(expr.left)
            right = self._eval_expression(expr.right)
            if expr.op in self._op_map:
                return self._op_map[expr.op](left, right)
            return left

        if isinstance(expr, UnaryExpr):
            operand = self._eval_expression(expr.operand)
            if expr.op == UnaryOp.NEG:
                return self._builder.neg(operand)
            if expr.op == UnaryOp.NOT:
                return self._builder.not_(operand)
            return operand

        # Any unhandled expression type silently evaluating to constant
        # 0 is a value miscompile qir-qis cannot catch (the
        # fail-loud class -- same smell as the VarExpr arm above). Every
        # currently-reachable type is handled above; a new/unhandled one
        # must fail LOUD, not lower as 0.
        msg = (
            f"QIR codegen: unsupported classical expression "
            f"{type(expr).__name__} (it must not be silently evaluated "
            "as 0 -- that would be a value miscompile)."
        )
        raise NotImplementedError(msg)

    def _process_if(self, node: IfStmt) -> None:
        """Process an if statement."""
        pred = self._as_i1(self._eval_expression(node.condition))

        if node.else_body:
            with self._builder.if_else(pred) as (then, otherwise):
                with then:
                    for stmt in node.then_body:
                        self._process_statement(stmt)
                with otherwise:
                    for stmt in node.else_body:
                        self._process_statement(stmt)
        else:
            with self._builder.if_then(pred):
                for stmt in node.then_body:
                    self._process_statement(stmt)

    def _process_while(self, node: WhileStmt) -> None:
        """`While` is not supported by the QIR backend.

        Unbounded iteration / fixed-point linear state through an
        unknown iteration count is out of scope for the sound emitter;
        the AST->Guppy path also rejects `While`.
        Fail LOUD here -- the previous single-pass approximation
        silently dropped the loop condition and all iterations (a
        miscompile qir-qis cannot catch, since one pass is valid QIR).
        """
        msg = (
            "QIR codegen does not support While loops (unbounded "
            "iteration / fixed-point linear state is out of scope for "
            "the QIR backend; a single-pass approximation would be a "
            "silent miscompile)."
        )
        raise NotImplementedError(msg)

    def _static_int_bound(self, expr: Any, which: str) -> int:
        """Resolve a static integer `For` bound.

        The converter wraps integer range bounds in `LiteralExpr`
        (`converter.py` `_convert_for`), so the bound is never a raw
        `int`. A non-literal / non-int bound is a symbolic/dynamic
        `For`, which is unsupported -- fail LOUD rather than silently
        drop the loop body (the previous `isinstance(int)` guard was
        always false, so EVERY `For` body was silently dropped).
        """
        if isinstance(expr, LiteralExpr) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return expr.value
        msg = (
            f"QIR codegen: For loop {which} bound is not a static integer "
            f"({type(expr).__name__}); only fixed-bound `For(i, <int>, "
            "<int>)` is supported (symbolic/dynamic For is out of scope -- "
            "and must not silently drop the loop body)."
        )
        raise NotImplementedError(msg)

    def _process_for(self, node: ForStmt) -> None:
        """Unroll a static fixed-bound `For` (v1-supported)."""
        start = self._static_int_bound(node.start, "start")
        stop = self._static_int_bound(node.stop, "stop")
        step = 1 if node.step is None else self._static_int_bound(node.step, "step")
        if step == 0:
            msg = "QIR codegen: For loop step is 0 (infinite loop); only a non-zero static step is supported."
            raise NotImplementedError(msg)
        for _ in range(start, stop, step):
            for stmt in node.body:
                self._process_statement(stmt)

    def _process_repeat(self, node: RepeatStmt) -> None:
        """Process a repeat loop by unrolling."""
        if isinstance(node.count, int):
            for _ in range(node.count):
                for stmt in node.body:
                    self._process_statement(stmt)

    def _process_parallel(self, node: ParallelBlock) -> None:
        """Process a parallel block."""
        for stmt in node.body:
            self._process_statement(stmt)

    def _expand_permute_ref(self, ref: str) -> list[tuple[str, int]]:
        """Expand a Permute ref string to logical (reg, index) pairs.

        `name[idx]` -> a single element; bare `name` -> every element
        of the register (QReg capacity or CReg size). Mirrors the
        Guppy codegen's `_expand_permute_ref`.
        """
        if ref.endswith("]") and "[" in ref:
            name, idx = ref[:-1].split("[", 1)
            return [(name, int(idx))]
        if ref in self.context.qreg_sizes:
            return [(ref, i) for i in range(self.context.qreg_sizes[ref])]
        if ref in self.context.creg_map:
            return [(ref, i) for i in range(self.context.creg_map[ref])]
        msg = f"QIR codegen: unknown Permute ref {ref!r}"
        raise NotImplementedError(msg)

    def _process_permute(self, node: PermuteOp) -> None:
        """Realize a Permute as a static logical relabel.

        QIR (and the Selene runtime) have no permute intrinsic, so --
        exactly like the legacy gen_qir permutation_map and the Guppy
        linearity tracker's `.permute()` -- a permutation is realized
        at compile time by relabelling which storage each logical
        (reg, index) ref resolves to. Build the source->target logical
        mapping, require it bijective over the same ref set, then
        compose it ATOMICALLY into the standing permutation_map
        (snapshot old, then `map[s] = old.get(t, t)`). Every
        qubit-ref and classical-bit-ref lowering consults
        permutation_map, so subsequent refs hit the permuted storage.
        Works uniformly for whole-register and element-wise, QReg and
        CReg.
        """
        if len(node.sources) != len(node.targets):
            msg = "QIR codegen: Permute source/target length mismatch"
            raise NotImplementedError(msg)

        # Accumulate the expanded refs as LISTS first and validate
        # BEFORE building the dict: a dict would silently collapse a
        # duplicate expanded source (e.g.
        # `Permute([a[0], a[0]], [b[0], a[0]])`) so a genuinely
        # non-bijective Permute would compile -- a silent miscompile.
        src_all: list[tuple[str, int]] = []
        tgt_all: list[tuple[str, int]] = []
        for source, target in zip(node.sources, node.targets, strict=True):
            src_refs = self._expand_permute_ref(source)
            tgt_refs = self._expand_permute_ref(target)
            if len(src_refs) != len(tgt_refs):
                msg = f"QIR codegen: Permute element count mismatch for {source!r} -> {target!r}"
                raise NotImplementedError(msg)
            src_all.extend(src_refs)
            tgt_all.extend(tgt_refs)

        if len(src_all) != len(set(src_all)):
            msg = "QIR codegen: Permute has a duplicate source ref (not a permutation)"
            raise NotImplementedError(msg)
        if len(tgt_all) != len(set(tgt_all)):
            msg = "QIR codegen: Permute has a duplicate target ref (not a permutation)"
            raise NotImplementedError(msg)
        if set(src_all) != set(tgt_all):
            msg = "QIR codegen: Permute must be bijective over the same ref set"
            raise NotImplementedError(msg)

        mapping: dict[tuple[str, int], tuple[str, int]] = dict(zip(src_all, tgt_all, strict=True))

        # Human-readable comment mirroring the legacy gen_qir format
        # (rendered from the post-substitution sources so it stays
        # correct inside a flattened BlockCall).
        if node.add_comment and node.sources:
            if node.whole_register and len(node.sources) >= 2:
                self._builder.comment(f"; Permutation: {node.sources[0]} <-> {node.sources[1]}")
            else:
                pairs = ", ".join(f"{s} -> {t}" for s, t in zip(node.sources, node.targets, strict=True))
                self._builder.comment(f"; Permutation: {pairs}")

        # Compose ATOMICALLY (Guppy `.permute` semantics): a whole
        # register swap arrives as sources=(a,b)/targets=(b,a), so a
        # sequential apply would cancel to a no-op; snapshotting the
        # old map first applies the relabel exactly once.
        old = dict(self.context.permutation_map)
        for s_ref, t_ref in mapping.items():
            self.context.permutation_map[s_ref] = old.get(t_ref, t_ref)

    def _pack_creg(self, reg_name: str) -> Any:
        """Pack a CReg's `[N x i1]` buffer into a single i64 value.

        `OR_i (zext c[i] << i)` -- this is the canonical SLR CReg-as-
        integer lowering. Used by both `_generate_results` (for the
        `__quantum__rt__int_record_output` call) and by
        `_eval_expression(VarExpr)` (a whole-CReg scalar
        reference in `If(m == 0)` / `o.set(m + n)` / etc.). Sharing
        the pack ensures the record-output and VarExpr interpretations
        of `m` are bit-identical -- the same packed i64.
        """
        c_int: Any = llvm_ir.Constant(self._types["int"], 0)
        for i in range(self.context.creg_map.get(reg_name, 0)):
            bit = self._builder.load(
                self._creg_bit_ptr(reg_name, i),
                name="",
            )
            widened = self._builder.zext(bit, self._types["int"])
            if i:
                widened = self._builder.shl(
                    widened,
                    llvm_ir.Constant(self._types["int"], i),
                )
            c_int = self._builder.or_(c_int, widened)
        return c_int

    def _generate_results(self) -> None:
        """Generate result output calls."""
        for reg_name in self._creg_ptrs:
            # Create tag for the register name
            reg_name_bytes = bytearray(reg_name.encode("utf-8"))
            tag_type = llvm_ir.ArrayType(llvm_ir.IntType(8), len(reg_name))
            reg_tag = llvm_ir.GlobalVariable(self._module, tag_type, reg_name)
            reg_tag.initializer = llvm_ir.Constant(tag_type, reg_name_bytes)
            reg_tag.global_constant = True
            reg_tag.linkage = "private"

            # Pack the [N x i1] buffer into one i64 (shared with the
            # VarExpr lowering).
            c_int = self._pack_creg(reg_name)

            reg_tag_gep = reg_tag.gep(
                (
                    llvm_ir.Constant(llvm_ir.IntType(32), 0),
                    llvm_ir.Constant(llvm_ir.IntType(32), 0),
                ),
            )
            self._builder.call(
                self._creg_funcs["int_result"],
                [c_int, reg_tag_gep],
                name="",
            )

    def _finalize_module(self) -> str:
        """Finalize the module and return LLVM IR with attributes."""
        ll_text = self._fix_internal_consts(str(self._module))
        mod_w_attr = ll_text.replace("@main()", "@main() #0")

        mod_w_attr += '\nattributes #0 = { "entry_point"'
        # adaptive_profile: PECOS emits measurement-conditioned `If` ->
        # adaptive, not base profile.
        mod_w_attr += ' "qir_profiles"="adaptive_profile"'
        mod_w_attr += ' "output_labeling_schema"="labeled"'
        mod_w_attr += f' "required_num_qubits"="{self.context.qubit_count}"'
        mod_w_attr += f' "required_num_results"="{self.context.measurement_count}" }}'

        # QIR module flags (Adaptive Profile). pecos_rslib_llvm.ir.Module has
        # no named-metadata API, so append as raw IR text -- same approach as
        # the entry attributes above. The emitted module carries no `!`
        # metadata, so !0..!4 are collision-free. The static classical
        # model sets dynamic_*/arrays = false (it keeps the static
        # %Result + mutable local-buffer model; flags must match).
        mod_w_attr += "\n!llvm.module.flags = !{!0, !1, !2, !3, !4}"
        mod_w_attr += '\n!0 = !{i32 1, !"qir_major_version", i32 1}'
        mod_w_attr += '\n!1 = !{i32 7, !"qir_minor_version", i32 0}'
        mod_w_attr += '\n!2 = !{i32 1, !"dynamic_qubit_management", i1 false}'
        mod_w_attr += '\n!3 = !{i32 1, !"dynamic_result_management", i1 false}'
        mod_w_attr += '\n!4 = !{i32 1, !"arrays", i1 false}'
        return mod_w_attr

    def _fix_internal_consts(self, llvm_ir: str) -> str:
        """Fix internal constants in LLVM IR."""
        return re.sub('([@%])"([^"]+)"', r"\1\2", llvm_ir)


def ast_to_qir(program: Program) -> str:
    """Convert an AST Program to QIR (LLVM IR).

    Convenience function for simple code generation.

    Args:
        program: The AST Program to convert.

    Returns:
        QIR as an LLVM IR string.
    """
    generator = AstToQir()
    return generator.generate(program)
