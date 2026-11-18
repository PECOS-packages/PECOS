# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Named, static gadget authoring and composition above physical lowering.

Composition validates lifetimes and physical Hadamard-frame conventions.
Declared check flows are annotations; explicit PauliFlow contracts can be checked
with audit_flows. No Guppy knowledge is needed for authoring or flow verification.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pecos.qec.generic.scheduled import (
    CircuitOp,
    MeasurementParity,
    OperationOrigin,
    ScheduledCircuit,
    SchedulePolicy,
    _detector_coordinates,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pecos.qec.generic.gadget_flows import GadgetFlowAudit, PauliFlow


def _name(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or not value.isprintable() or "/" in value:
        msg = f"Names must be nonempty printable strings without '/': {value!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class NamedParity:
    """An XOR of named outcomes; repeated terms cancel automatically."""

    terms: tuple[str, ...] = ()
    constant: bool = False

    def __post_init__(self) -> None:
        """Canonicalize XOR expressions without exposing binary masks."""
        if isinstance(self.terms, str):
            object.__setattr__(self, "terms", (self.terms,))
        if type(self.constant) is not bool or any(not isinstance(t, str) or not t for t in self.terms):
            msg = "Parity terms must be nonempty names and constant must be bool"
            raise ValueError(msg)
        object.__setattr__(self, "terms", tuple(sorted(t for t, count in Counter(self.terms).items() if count % 2)))

    def __xor__(self, other: NamedParity | bool | int) -> NamedParity:
        """Combine two parity expressions."""
        if type(other) in (bool, int):
            if other not in (0, 1):
                msg = "Parity constants must be 0 or 1"
                raise ValueError(msg)
            other = NamedParity(constant=bool(other))
        if not isinstance(other, NamedParity):
            return NotImplemented
        return NamedParity((*self.terms, *other.terms), self.constant ^ other.constant)

    def __rxor__(self, other: bool | int) -> NamedParity:
        """Allow 1 ^ parity as well as parity ^ 1."""
        return self ^ other

    def __str__(self) -> str:
        """Render a readable parity expression."""
        return " ^ ".join((*self.terms, *(("1",) if self.constant else ()))) or "0"


@dataclass(frozen=True)
class NamedPauli:
    """A named code check or logical operator over named data roles."""

    name: str
    terms: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Check role uniqueness and Pauli vocabulary."""
        _name(self.name)
        object.__setattr__(self, "terms", tuple(tuple(t) for t in self.terms))
        for role, axis in self.terms:
            _name(role)
            if axis not in {"X", "Y", "Z"}:
                msg = f"Invalid Pauli axis: {axis}"
                raise ValueError(msg)
        if len({role for role, _ in self.terms}) != len(self.terms):
            msg = "A Pauli operator may mention each role only once"
            raise ValueError(msg)

    def __str__(self) -> str:
        """Display the operator on its named support."""
        return f"{self.name} = " + (" * ".join(f"{axis}({role})" for role, axis in self.terms) or "I")


@dataclass(frozen=True)
class Gate:
    """A primitive on named roles; M must declare a local outcome name."""

    name: str
    roles: tuple[str, ...]
    outcome: str | None = None

    def __post_init__(self) -> None:
        """Validate the supported physical vocabulary and measurement naming."""
        object.__setattr__(self, "roles", (self.roles,) if isinstance(self.roles, str) else tuple(self.roles))
        for role in self.roles:
            _name(role)
        CircuitOp(self.name, tuple(range(len(self.roles))))
        if len(set(self.roles)) != len(self.roles):
            msg = "A gate cannot alias its operands"
            raise ValueError(msg)
        if (self.name == "M") != (self.outcome is not None):
            msg = "Exactly measurement gates must name an outcome"
            raise ValueError(msg)
        if self.outcome is not None:
            _name(self.outcome)


@dataclass(frozen=True)
class GadgetFlow:
    """A named check, frame-sign, or constraint expressed in local outcomes.

    The label identifies its meaning (for example ``input:X[0]``). This is an
    author-declared semantic relation, independently verified for the paper gadgets.
    """

    name: str
    parity: NamedParity
    coords: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Snapshot optional detector annotation coordinates."""
        object.__setattr__(self, "coords", _detector_coordinates(self.coords))


@dataclass(frozen=True)
class Gadget:
    """Reusable named circuit with explicit live ports and boundary frames.

    ``inputs``/``outputs`` are the roles live at entry/exit. Other roles are scratch
    ports. Hadamard-frame tuples describe physical basis conventions, not Pauli
    correction bits. Measurement-dependent signs belong in ``flows``.
    """

    name: str
    roles: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    moments: tuple[tuple[Gate, ...], ...]
    input_h_frame: tuple[str, ...] = ()
    output_h_frame: tuple[str, ...] = ()
    flows: tuple[GadgetFlow, ...] = ()
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate local lifetimes, moments, outcomes, and frame references."""
        _name(self.name)
        for field in ("roles", "inputs", "outputs", "input_h_frame", "output_h_frame", "flows", "citations"):
            if isinstance(getattr(self, field), str):
                msg = f"Gadget {field} must be a sequence, not a bare string"
                raise TypeError(msg)
            object.__setattr__(self, field, tuple(getattr(self, field)))
        object.__setattr__(self, "moments", tuple(tuple(m) for m in self.moments))
        for field in ("roles", "inputs", "outputs", "input_h_frame", "output_h_frame"):
            values = getattr(self, field)
            for value in values:
                _name(value)
            if len(set(values)) != len(values) or not set(values) <= set(self.roles):
                msg = f"Invalid or duplicate {field} in gadget {self.name}"
                raise ValueError(msg)
        if not set(self.input_h_frame) <= set(self.inputs) or not set(self.output_h_frame) <= set(self.outputs):
            msg = "Basis frames must reference live input/output ports"
            raise ValueError(msg)
        live = set(self.inputs)
        outcomes = []
        for moment in self.moments:
            touched: set[str] = set()
            for gate in moment:
                if not set(gate.roles) <= set(self.roles) or touched.intersection(gate.roles):
                    msg = f"Unknown or overlapping roles in gadget {self.name}: {gate.roles}"
                    raise ValueError(msg)
                touched.update(gate.roles)
                if gate.name == "R":
                    if gate.roles[0] in live:
                        msg = f"Gadget {self.name} prepares an already-live role: {gate.roles[0]}"
                        raise ValueError(msg)
                    live.add(gate.roles[0])
                else:
                    if not set(gate.roles) <= live:
                        msg = f"Gadget {self.name} uses non-live roles: {gate.roles}"
                        raise ValueError(msg)
                    if gate.name == "M":
                        live.remove(gate.roles[0])
                        outcomes.append(gate.outcome)
        if live != set(self.outputs):
            msg = f"Gadget {self.name} output contract disagrees with its live roles"
            raise ValueError(msg)
        if len(set(outcomes)) != len(outcomes):
            msg = f"Duplicate outcome name in gadget {self.name}"
            raise ValueError(msg)
        flow_names = set()
        for flow in self.flows:
            _name(flow.name)
            if flow.name in flow_names or not set(flow.parity.terms) <= set(outcomes):
                msg = f"Duplicate flow or unknown outcome in gadget flow {flow.name}"
                raise ValueError(msg)
            flow_names.add(flow.name)

    @property
    def outcomes(self) -> tuple[str, ...]:
        """Local measurement names in canonical circuit order."""
        return tuple(g.outcome for m in self.moments for g in m if g.outcome is not None)

    @classmethod
    def from_operations(
        cls,
        name: str,
        roles: Sequence[str],
        moments: Sequence[Sequence[CircuitOp]],
        *,
        inputs: Sequence[str],
        outputs: Sequence[str],
        outcomes: Sequence[str],
        input_h_frame: Sequence[str] = (),
        output_h_frame: Sequence[str] = (),
        flows: Sequence[GadgetFlow] = (),
        citations: Sequence[str] = (),
    ) -> Gadget:
        """Adapt an existing indexed construction without changing its moments."""
        if sum(op.name == "M" for m in moments for op in m) != len(outcomes):
            msg = "Supply exactly one outcome name per measurement"
            raise ValueError(msg)
        names = iter(outcomes)
        return cls(
            name,
            tuple(roles),
            tuple(inputs),
            tuple(outputs),
            tuple(
                tuple(
                    Gate(op.name, tuple(roles[q] for q in op.qubits), next(names) if op.name == "M" else None)
                    for op in moment
                )
                for moment in moments
            ),
            tuple(input_h_frame),
            tuple(output_h_frame),
            tuple(flows),
            tuple(citations),
        )

    def describe(self) -> str:
        """Show ports, basis conventions, named operations, and check relations."""
        lines = [
            f"{self.name}: {len(self.moments)} reference moments",
            f"  live inputs: {', '.join(self.inputs) or '(none)'}",
            f"  live outputs: {', '.join(self.outputs) or '(none)'}",
            f"  input H frame: {', '.join(self.input_h_frame) or 'I'}",
            f"  output H frame: {', '.join(self.output_h_frame) or 'I'}",
        ]
        lines.extend(f"  {flow.name} = {flow.parity}" for flow in self.flows)
        for layer, moment in enumerate(self.moments):
            operations = (
                f"{gate.name}({', '.join(gate.roles)})" + (f" -> {gate.outcome}" if gate.outcome else "")
                for gate in moment
            )
            lines.append(f"  moment {layer}: " + ("; ".join(operations) or "(idle)"))
        return "\n".join(lines)

    def audit_flows(self, contracts: Sequence[PauliFlow]) -> GadgetFlowAudit:
        """Verify explicitly supplied signed Pauli relations on arbitrary inputs."""
        from pecos.qec.generic.gadget_flows import audit_gadget_flows  # noqa: PLC0415

        return audit_gadget_flows(self, contracts)


@dataclass(frozen=True)
class GadgetCall:
    """One scoped use of a gadget with its role bindings."""

    label: str
    gadget: Gadget
    bindings: tuple[tuple[str, str], ...]

    def __getitem__(self, outcome: str) -> NamedParity:
        """Refer to a named measurement in this particular call."""
        if outcome not in self.gadget.outcomes:
            raise KeyError(outcome)
        return NamedParity((f"{self.label}/{outcome}",))

    def flow(self, name: str) -> NamedParity:
        """Resolve a local check/frame relation into program-scoped outcomes."""
        for flow in self.gadget.flows:
            if flow.name == name:
                return NamedParity(tuple(f"{self.label}/{term}" for term in flow.parity.terms), flow.parity.constant)
        raise KeyError(name)


class GadgetProgram:
    """Compose static gadgets with checked lifetimes, bindings, and basis frames.

    Append returns a scoped call whose named outcomes can be XORed into detectors
    and observables. Compilation consumes every live qubit and lowers to the shared
    ScheduledCircuit backend. Appending a gadget does not introduce a timing barrier.
    """

    def __init__(self, roles: Sequence[str]) -> None:
        """Declare the experiment's role namespace in physical allocation order."""
        if isinstance(roles, str):
            msg = "Program roles must be a sequence, not a bare string"
            raise TypeError(msg)
        self._roles = tuple(roles)
        for role in self._roles:
            _name(role)
        if not roles or len(set(roles)) != len(roles):
            msg = "Program roles must be nonempty and unique"
            raise ValueError(msg)
        self._calls: list[GadgetCall] = []
        self._live: set[str] = set()
        self._h_frame: set[str] = set()
        self._outcomes: list[str] = []
        self._detectors: list[GadgetFlow] = []
        self._observables: list[GadgetFlow] = []

    @property
    def roles(self) -> tuple[str, ...]:
        """Named experiment roles in allocation order."""
        return self._roles

    @property
    def calls(self) -> tuple[GadgetCall, ...]:
        """Composition in authoring order."""
        return tuple(self._calls)

    @property
    def detectors(self) -> tuple[GadgetFlow, ...]:
        """Named detector expressions before event-index lowering."""
        return tuple(self._detectors)

    @property
    def observables(self) -> tuple[GadgetFlow, ...]:
        """Named logical expressions before event-index lowering."""
        return tuple(self._observables)

    def append(self, gadget: Gadget, *, label: str, bindings: Mapping[str, str] | None = None) -> GadgetCall:
        """Append a bound gadget atomically after validating its entry contract."""
        _name(label)
        if any(call.label == label for call in self._calls):
            msg = f"Duplicate gadget call label: {label}"
            raise ValueError(msg)
        bound = dict(bindings) if bindings is not None else {r: r for r in gadget.roles}
        if set(bound) != set(gadget.roles) or not set(bound.values()) <= set(self.roles):
            msg = f"Bindings for {label} must map every gadget role to a declared program role"
            raise ValueError(msg)
        if len(set(bound.values())) != len(bound):
            msg = f"Bindings for {label} alias distinct gadget roles"
            raise ValueError(msg)
        touched = set(bound.values())
        incoming = {bound[r] for r in gadget.inputs}
        if self._live & touched != incoming:
            msg = f"Live-port mismatch at {label}: expected {sorted(incoming)}, found {sorted(self._live & touched)}"
            raise ValueError(msg)
        expected_frame = {bound[r] for r in gadget.input_h_frame}
        if self._h_frame & incoming != expected_frame:
            msg = f"Basis-frame mismatch at {label}; insert an explicit frame-conversion gadget"
            raise ValueError(msg)
        call = GadgetCall(label, gadget, tuple((r, bound[r]) for r in gadget.roles))
        self._calls.append(call)
        self._outcomes.extend(f"{label}/{name}" for name in gadget.outcomes)
        self._live = (self._live - touched) | {bound[r] for r in gadget.outputs}
        self._h_frame = (self._h_frame - touched) | {bound[r] for r in gadget.output_h_frame}
        return call

    def _parity(
        self,
        target: list[GadgetFlow],
        name: str,
        parity: NamedParity,
        coords: Sequence[float] | None = None,
    ) -> None:
        _name(name)
        if any(flow.name == name for flow in target) or not set(parity.terms) <= set(self._outcomes):
            msg = f"Duplicate label or unknown measurement in parity {name}: {parity}"
            raise ValueError(msg)
        target.append(GadgetFlow(name, parity, coords))

    def detector(self, name: str, parity: NamedParity, *, coords: Sequence[float] | None = None) -> None:
        """Declare a named XOR with optional decoder coordinates.

        Coordinates describe the authored detector, not the selected physical
        schedule. Rescheduling preserves them verbatim, including any time axis.
        """
        self._parity(self._detectors, name, parity, coords)

    def observable(self, name: str, parity: NamedParity) -> None:
        """Declare a logical observable as a named XOR."""
        self._parity(self._observables, name, parity)

    def compile(self, schedule: SchedulePolicy = SchedulePolicy.REFERENCE) -> ScheduledCircuit:
        """Lower names and provenance, then apply the selected scheduling policy."""
        if self._live:
            msg = f"Complete the experiment before compiling; live roles: {sorted(self._live)}"
            raise ValueError(msg)
        at = {name: index for index, name in enumerate(self.roles)}
        events = {name: index for index, name in enumerate(self._outcomes)}
        moments = []
        citations = []
        for call in self._calls:
            bound = dict(call.bindings)
            citations.extend(c for c in call.gadget.citations if c not in citations)
            for layer, moment in enumerate(call.gadget.moments):
                moments.append(
                    tuple(
                        CircuitOp(
                            gate.name,
                            tuple(at[bound[r]] for r in gate.roles),
                            OperationOrigin(call.label, call.gadget.name, layer, index),
                        )
                        for index, gate in enumerate(moment)
                    ),
                )

        def lower(flow: GadgetFlow) -> MeasurementParity:
            return MeasurementParity(
                tuple(sorted(events[t] for t in flow.parity.terms)),
                flow.name,
                flow.parity.constant,
                flow.coords,
            )

        return ScheduledCircuit(
            len(self.roles),
            tuple(moments),
            tuple(map(lower, self.detectors)),
            tuple(map(lower, self.observables)),
            tuple(citations),
            role_names=self.roles,
            measurement_names=tuple(self._outcomes),
        ).rescheduled(schedule)

    def describe(self) -> str:
        """Summarize the authored sequence and its detector/logical expressions."""
        lines = [f"Program: {len(self.roles)} roles, {len(self._calls)} gadget calls"]
        lines.extend(f"  {c.label}: {c.gadget.name} ({len(c.gadget.outcomes)} outcomes)" for c in self.calls)
        lines.extend(
            f"  detector {flow.name} = {flow.parity}" + (f" coords={flow.coords}" if flow.coords is not None else "")
            for flow in self.detectors
        )
        lines.extend(f"  observable {flow.name} = {flow.parity}" for flow in self.observables)
        return "\n".join(lines)
