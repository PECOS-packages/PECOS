# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Six-state preparation and frame-aware transversal 6.6.6 Clifford gadgets.

References: Lavasani et al., arXiv:2610.24077v1, Logical Memory Experiment
Setup, Clifford Randomized Benchmarking, and Supplement B. This implements
the static circuit ingredients, not the paper's hardware or RB benchmark.
See also Lacroix et al., arXiv:2412.14256v1, Sec. IV, on detector relations
across logical Cliffords. Its hardware-specific extraction circuit is distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

from pecos.qec.color.paper_codes import (
    _single,
    bell_flag_projection,
    color_code_implementation,
    color_code_spec,
)
from pecos.qec.generic.gadgets import Gadget, GadgetProgram, Gate, NamedParity, _name
from pecos.qec.generic.scheduled import SchedulePolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.qec.color.paper_codes import ColorCodeImplementation, CSSCodeSpec
    from pecos.qec.generic.gadgets import GadgetCall, GadgetFlow
    from pecos.qec.generic.scheduled import ScheduledCircuit


class LogicalState(Enum):
    """The six eigenstates of the encoded Pauli operators."""

    ZERO = "+Z"
    ONE = "-Z"
    PLUS = "+X"
    MINUS = "-X"
    PLUS_I = "+Y"
    MINUS_I = "-Y"

    @property
    def preparation(self) -> tuple[str, ...]:
        """Clifford gates, in application order, taking encoded |0> to this state."""
        sign = ("X",) if self.value[0] == "-" else ()
        rotations = {"Z": (), "X": ("H",), "Y": ("H", "S")}
        return (*sign, *rotations[self.value[1]])

    @property
    def unpreparation(self) -> tuple[str, ...]:
        """Inverse preparation, for homogeneous memory success observables."""
        return tuple("S_DAG" if gate == "S" else gate for gate in reversed(self.preparation))


def _canonical_666(code: CSSCodeSpec) -> None:
    if code.family != "666" or code != color_code_spec("666", code.distance):
        msg = "Logical Clifford gadgets require the canonical triangular 666 code"
        raise ValueError(msg)


def logical_clifford_gadget(code: CSSCodeSpec, gate: str) -> Gadget:
    """Return a reusable transversal H/S/S_DAG/X/Z/CX gadget.

    Single-block ports are ``data[i]``. CX ports are ``control[i]`` and
    ``target[i]`` and require identical code layouts. S acts as +S on the
    encoded qubit: S on A, S_DAG on B, where each face is balanced and
    |A|-|B|=1. Its inverse reverses those physical phases. Boundary frames are
    the ordinary physical convention; H here is an actual logical operation.
    """
    _canonical_666(code)
    if gate not in {"H", "S", "S_DAG", "X", "Z", "CX"}:
        msg = f"Unsupported logical Clifford: {gate}"
        raise ValueError(msg)
    if gate == "CX":
        roles = tuple(f"{port}[{q}]" for port in ("control", "target") for q in range(code.num_data))
        layer = tuple(Gate("CX", (f"control[{q}]", f"target[{q}]")) for q in range(code.num_data))
    else:
        roles = code.data_roles
        if gate in {"S", "S_DAG"}:
            sublattice = tuple((x - 3 * y) % 6 for x, y in code.data_coordinates)
            if set(sublattice) != {0, 2} or sublattice.count(0) - sublattice.count(2) != 1:
                msg = "The phase-gate construction requires the canonical signed bipartition"
                raise ValueError(msg)
            if any(sum(1 if sublattice[q] == 0 else -1 for q in face) for face in code.faces):
                msg = "Every stabilizer must have balanced phase-gate support"
                raise ValueError(msg)
            inverse = "S_DAG" if gate == "S" else "S"
            layer = tuple(Gate(gate if sublattice[q] == 0 else inverse, (role,)) for q, role in enumerate(roles))
        else:
            layer = tuple(Gate(gate, (role,)) for role in roles)
    return Gadget(f"logical:{gate}", roles, roles, roles, (layer,), citations=(code.citation,))


@dataclass(frozen=True)
class ColorFrame:
    """Named check references and logical sign corrections for one live block.

    ``x_sign``/``z_sign`` are deviations from the intended ideal logical state,
    not that state's eigenvalues. A Y readout uses their XOR. Check references
    include initialization outcomes; extraction updates both input/output maps.
    """

    check_x: tuple[NamedParity, ...]
    check_z: tuple[NamedParity, ...]
    x_sign: NamedParity = field(default_factory=NamedParity)
    z_sign: NamedParity = field(default_factory=NamedParity)

    def __post_init__(self) -> None:
        """Snapshot check references and validate the two CSS check families."""
        object.__setattr__(self, "check_x", tuple(self.check_x))
        object.__setattr__(self, "check_z", tuple(self.check_z))
        if len(self.check_x) != len(self.check_z):
            msg = "Color-frame X/Z check counts must match"
            raise ValueError(msg)
        if any(not isinstance(p, NamedParity) for p in (*self.check_x, *self.check_z, self.x_sign, self.z_sign)):
            msg = "Color-frame references must be NamedParity expressions"
            raise TypeError(msg)

    def after(self, gate: str) -> ColorFrame:
        """Propagate sign corrections through a supported one-block Clifford."""
        if gate == "H":
            return ColorFrame(self.check_z, self.check_x, self.z_sign, self.x_sign)
        if gate in {"S", "S_DAG"}:
            return ColorFrame(
                tuple(x ^ z for x, z in zip(self.check_x, self.check_z, strict=True)),
                self.check_z,
                self.x_sign ^ self.z_sign,
                self.z_sign,
            )
        if gate in {"X", "Z"}:
            return self
        msg = f"Unsupported one-block frame operation: {gate}"
        raise ValueError(msg)

    def cx(self, target: ColorFrame) -> tuple[ColorFrame, ColorFrame]:
        """Propagate control/target sign references through transversal CX."""
        control = replace(
            self,
            check_x=tuple(a ^ b for a, b in zip(self.check_x, target.check_x, strict=True)),
            x_sign=self.x_sign ^ target.x_sign,
        )
        target = replace(
            target,
            check_z=tuple(a ^ b for a, b in zip(self.check_z, target.check_z, strict=True)),
            z_sign=self.z_sign ^ target.z_sign,
        )
        return control, target


class ColorCliffordExperiment:
    """Compose prepared 666 blocks, extraction rounds, Clifford gates, and readout.

    Every block uses the same code and extraction implementation. Block state
    tracks syndrome references, logical signs, and the next extraction phase;
    the generic gadget compiler tracks physical carrier relabeling. This is a
    static protocol builder and does not infer ideal logical outcome relations.
    """

    def __init__(self, implementation: ColorCodeImplementation, blocks: Sequence[str] = ("memory",)) -> None:
        """Declare blocks and their distinct data/ancilla role namespaces."""
        _canonical_666(implementation.code)
        if implementation != color_code_implementation(implementation.code, implementation.extraction):
            msg = "Clifford experiments require a canonical extraction implementation"
            raise ValueError(msg)
        if isinstance(blocks, str):
            msg = "blocks must be a sequence of distinct names"
            raise TypeError(msg)
        self._blocks = tuple(blocks)
        for block in self._blocks:
            _name(block)
            if ":" in block:
                msg = "Block names must not contain ':'; it separates generated protocol scopes"
                raise ValueError(msg)
        if not self._blocks or len(set(self._blocks)) != len(self._blocks):
            msg = "blocks must be nonempty and unique"
            raise ValueError(msg)
        self._implementation = implementation
        self._program = GadgetProgram(
            tuple(f"{block}:{role}" for block in self._blocks for role in implementation.roles),
        )
        self._frames: dict[str, ColorFrame] = {}
        self._phases: dict[str, int] = {}
        self._rounds: dict[str, int] = {}
        self._prepared: set[str] = set()

    @property
    def calls(self) -> tuple[GadgetCall, ...]:
        """Inspect the bound gadget sequence."""
        return self._program.calls

    @property
    def detectors(self) -> tuple[GadgetFlow, ...]:
        """Inspect named detector relations."""
        return self._program.detectors

    @property
    def observables(self) -> tuple[GadgetFlow, ...]:
        """Inspect declared logical outcome relations."""
        return self._program.observables

    def _live(self, block: str) -> ColorFrame:
        if block not in self._blocks:
            msg = f"Unknown block: {block}"
            raise ValueError(msg)
        if block not in self._frames:
            msg = f"Block {block} must be prepared and not yet read out"
            raise ValueError(msg)
        return self._frames[block]

    def frame(self, block: str) -> ColorFrame:
        """Inspect an immutable snapshot of the block's current sign references."""
        return self._live(block)

    def next_phase(self, block: str) -> int:
        """Inspect the phase that the block's next extraction cycle will use."""
        self._live(block)
        return self._phases[block]

    def _append(self, block: str, gadget: Gadget, label: str) -> GadgetCall:
        return self._program.append(gadget, label=label, bindings={r: f"{block}:{r}" for r in gadget.roles})

    def prepare(self, block: str, state: LogicalState | str = LogicalState.ZERO, *, start_phase: int = 0) -> None:
        """Bell-flag prepare logical zero, then rotate to any signed Pauli state."""
        if block not in self._blocks or block in self._prepared:
            msg = f"Block {block} is unknown or already prepared"
            raise ValueError(msg)
        state = LogicalState(state)
        impl = self._implementation
        if type(start_phase) is not int or not 0 <= start_phase < impl.period:
            msg = "start_phase is outside this implementation's period"
            raise ValueError(msg)
        code, roles = impl.code, impl.roles
        nf = len(code.faces)
        preparation = Gadget.from_operations(
            "prepare:Z",
            roles,
            (_single("R", range(code.num_data)),),
            inputs=(),
            outputs=code.data_roles,
            outcomes=(),
            citations=(code.citation,),
        )
        projection = Gadget.from_operations(
            "bell_flag:Z",
            roles,
            bell_flag_projection(code, "Z"),
            inputs=code.data_roles,
            outputs=code.data_roles,
            outcomes=(*(f"syndrome[{f}]" for f in range(nf)), *(f"flag[{f}]" for f in range(nf))),
            citations=(code.citation,),
        )
        self._append(block, preparation, f"{block}:prepare")
        measured = self._append(block, projection, f"{block}:initialize")
        for face in range(nf):
            self._program.detector(f"{block}:init:flag:{face}", measured[f"flag[{face}]"])
        self._frames[block] = ColorFrame(tuple(measured[f"syndrome[{f}]"] for f in range(nf)), (NamedParity(),) * nf)
        self._phases[block], self._rounds[block] = start_phase, 0
        self._prepared.add(block)
        for gate in state.preparation:
            self.gate(gate, block)

    def gate(self, gate: str, block: str, target: str | None = None) -> None:
        """Apply a transversal Clifford and propagate check/logical sign references."""
        frame = self._live(block)
        gadget = logical_clifford_gadget(self._implementation.code, gate)
        if gate == "CX":
            if target is None or target == block:
                msg = "CX requires a distinct live target block"
                raise ValueError(msg)
            target_frame = self._live(target)
            bindings = {
                f"{port}[{q}]": f"{name}:data[{q}]"
                for port, name in (("control", block), ("target", target))
                for q in range(self._implementation.code.num_data)
            }
            self._program.append(gadget, label=f"gate[{len(self.calls)}]:CX:{block}:{target}", bindings=bindings)
            self._frames[block], self._frames[target] = frame.cx(target_frame)
        else:
            if target is not None:
                msg = "Only CX accepts a target block"
                raise ValueError(msg)
            self._append(block, gadget, f"gate[{len(self.calls)}]:{gate}:{block}")
            self._frames[block] = frame.after(gate)

    def extract(self, block: str, rounds: int = 1) -> None:
        """Append extraction cycles with continuous phase and carrier tracking."""
        self._live(block)
        if type(rounds) is not int or rounds < 1:
            msg = "rounds must be a positive integer"
            raise ValueError(msg)
        nf = len(self._implementation.code.faces)
        for _ in range(rounds):
            frame = self._frames[block]
            rd = self._rounds[block]
            gadget = self._implementation.gadget(self._phases[block])
            current = self._append(block, gadget, f"{block}:round[{rd}]")
            for flow in gadget.flows:
                if flow.name.startswith("constraint["):
                    self._program.detector(f"{block}:round:{rd}:{flow.name}", current.flow(flow.name))
            for face in range(nf):
                for axis, checks in (("X", frame.check_x), ("Z", frame.check_z)):
                    self._program.detector(
                        f"{block}:round:{rd}:{axis}:{face}",
                        current.flow(f"input:{axis}[{face}]") ^ checks[face],
                    )
            self._frames[block] = ColorFrame(
                tuple(current.flow(f"output:X[{f}]") for f in range(nf)),
                tuple(current.flow(f"output:Z[{f}]") for f in range(nf)),
                frame.x_sign ^ current.flow("logical:X"),
                frame.z_sign ^ current.flow("logical:Z"),
            )
            self._phases[block] = (self._phases[block] + 1) % self._implementation.period
            self._rounds[block] += 1

    def readout(self, block: str, basis: str = "Z") -> NamedParity:
        """Measure a logical X/Y/Z operator and return its frame-corrected parity.

        This is a logical outcome, not a success/failure bit. Negative-state
        readout returns one. To construct a homogeneous memory failure parity,
        undo preparation first and measure Z, as color_state_memory_program does.
        """
        self._live(block)
        if basis not in {"X", "Y", "Z"}:
            msg = "readout basis must be X, Y, or Z"
            raise ValueError(msg)
        for gate in {"Z": (), "X": ("H",), "Y": ("S_DAG", "H")}[basis]:
            self.gate(gate, block)
        frame = self._frames[block]
        impl = self._implementation
        code = impl.code
        gadget = Gadget.from_operations(
            f"readout:{basis}",
            impl.roles,
            (_single("M", range(code.num_data)),),
            inputs=code.data_roles,
            outputs=(),
            outcomes=code.data_roles,
            citations=(code.citation,),
        )
        readout = self._append(block, gadget, f"{block}:readout")
        for face, support in enumerate(code.faces):
            parity = frame.check_z[face]
            for q in support:
                parity ^= readout[code.data_roles[q]]
            self._program.detector(f"{block}:final:{basis}:{face}", parity)
        logical = frame.z_sign
        for q in code.logical_support:
            logical ^= readout[code.data_roles[q]]
        del self._frames[block]
        return logical

    def observable(self, name: str, parity: NamedParity) -> None:
        """Declare a logical outcome relation, including cross-block correlations."""
        self._program.observable(name, parity)

    def compile(self, schedule: SchedulePolicy = SchedulePolicy.REFERENCE) -> ScheduledCircuit:
        """Compile a complete experiment using the shared gadget/backend pipeline."""
        if not self._prepared:
            msg = "Prepare at least one block before compiling"
            raise ValueError(msg)
        return self._program.compile(schedule)

    def describe(self) -> str:
        """Inspect the authored circuit and all named detector/observable relations."""
        return self._program.describe()


def color_state_memory_program(
    implementation: ColorCodeImplementation,
    rounds: int,
    state: LogicalState | str = LogicalState.ZERO,
    *,
    start_phase: int = 0,
) -> ColorCliffordExperiment:
    """Compose a 666 six-state memory with inverse-preparation readout.

    This follows arXiv:2610.24077v1's prepare-Z/rotate/extract/unrotate protocol.
    The resulting observable is a homogeneous failure parity for all six states.
    Existing color_memory X/Z recipes retain their original reference circuits.
    """
    state = LogicalState(state)
    if type(rounds) is not int or rounds < 1:
        msg = "rounds must be a positive integer"
        raise ValueError(msg)
    program = ColorCliffordExperiment(implementation)
    program.prepare("memory", state, start_phase=start_phase)
    program.extract("memory", rounds)
    for gate in state.unpreparation:
        program.gate(gate, "memory")
    program.observable(f"memory:{state.value}", program.readout("memory"))
    return program


def color_state_memory(
    implementation: ColorCodeImplementation,
    rounds: int,
    state: LogicalState | str = LogicalState.ZERO,
    *,
    start_phase: int = 0,
    schedule: SchedulePolicy = SchedulePolicy.REFERENCE,
) -> ScheduledCircuit:
    """Compile a six-state memory through the named Clifford experiment builder."""
    return color_state_memory_program(implementation, rounds, state, start_phase=start_phase).compile(schedule)
