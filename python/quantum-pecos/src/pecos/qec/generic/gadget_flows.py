# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Signed Pauli-flow contracts for open, named Clifford gadgets.

Verification uses exact membership in Stim's signed flow-generator group, not
sampled ideal memories. Only the explicitly supplied contracts are checked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from pecos.qec.generic.gadgets import NamedParity, NamedPauli, _name

if TYPE_CHECKING:
    from collections.abc import Sequence

    import stim

    from pecos.qec.generic.gadgets import Gadget


@dataclass(frozen=True)
class PauliFlow:
    """Input Pauli -> output Pauli times the sign of a named outcome parity.

    None means identity. Operators use the gadget's named ports in their declared
    boundary H frames. Negative signs are expressed by ``parity ^ 1``, including
    flows with no measurements. This does not assert a full channel specification.
    """

    name: str
    input: NamedPauli | None = None
    output: NamedPauli | None = None
    parity: NamedParity = field(default_factory=NamedParity)

    def __post_init__(self) -> None:
        """Validate explicit operators and a signed parity without parsing names."""
        _name(self.name)
        if any(p is not None and not isinstance(p, NamedPauli) for p in (self.input, self.output)):
            msg = "Flow input and output must be NamedPauli operators or None"
            raise TypeError(msg)
        if not isinstance(self.parity, NamedParity):
            msg = "Flow parity must be a NamedParity"
            raise TypeError(msg)

    def describe(self) -> str:
        """Render the relation in named boundary roles."""
        def operator(pauli: NamedPauli | None) -> str:
            return " * ".join(f"{axis}({role})" for role, axis in pauli.terms) if pauli and pauli.terms else "I"

        return f"{self.name}: {operator(self.input)} -> {operator(self.output)}; sign parity = {self.parity}"


@dataclass(frozen=True)
class GadgetFlowAudit:
    """Immutable evidence for a specified list of ideal signed flow contracts."""

    _report_json: str = field(repr=False)

    def to_dict(self) -> dict:
        """Return detached contracts, carrier bindings, and generator witnesses."""
        return json.loads(self._report_json)

    @property
    def all_valid(self) -> bool:
        """Whether every requested signed contract holds."""
        return all(row["valid"] for row in self.to_dict()["contracts"])

    def require_valid(self) -> None:
        """Fail with the names of invalid requested contracts."""
        failed = [r["name"] for r in self.to_dict()["contracts"] if not r["valid"]]
        if failed:
            msg = "Invalid signed Pauli flows: " + ", ".join(failed)
            raise ValueError(msg)

    def describe(self) -> str:
        """Display requested relations and separate sign from support failures."""
        report = self.to_dict()
        lines = [f"{report['gadget']['name']}: {len(report['contracts'])} signed flow contracts; "
                 f"all valid={self.all_valid}"]
        for row in report["contracts"]:
            status = "valid" if row["valid"] else "wrong sign" if row["support_valid"] else "unsupported relation"
            lines.append(f"  {status}: {row['description']}")
        lines.append("Checks only the supplied ideal contracts; not completeness, fault tolerance, timing, or noise")
        return "\n".join(lines)


def _vector(flow: stim.Flow, n: int, measurements: int) -> int:
    """Binary input/output X/Z and measurement support; signs handled separately."""
    result = 0
    for side, pauli in enumerate((flow.input_copy(), flow.output_copy())):
        for q in range(len(pauli)):
            axis = pauli[q]
            if axis in (1, 2):
                result ^= 1 << (2 * n * side + q)
            if axis in (2, 3):
                result ^= 1 << (2 * n * side + n + q)
    for index in flow.measurements_copy():
        result ^= 1 << (4 * n + (index % measurements))
    return result


def audit_gadget_flows(gadget: Gadget, contracts: Sequence[PauliFlow]) -> GadgetFlowAudit:
    """Check named contracts by signed generator membership on an open circuit.

    Input ports are arbitrary quantum inputs, not prepared in a chosen basis.
    RELABEL permutes carrier bindings without emitting a SWAP. Boundary H frames
    conjugate X/Z and negate Y. The gadget remains unchanged for all backends.
    """
    import stim  # noqa: PLC0415 - optional independent verification backend

    contracts = tuple(contracts)
    if not contracts or any(not isinstance(c, PauliFlow) for c in contracts):
        msg = "Provide a nonempty sequence of PauliFlow contracts"
        raise ValueError(msg)
    if len({c.name for c in contracts}) != len(contracts):
        msg = "Flow contract names must be unique"
        raise ValueError(msg)
    for contract in contracts:
        for side, ports in ((contract.input, gadget.inputs), (contract.output, gadget.outputs)):
            if side is not None and not {r for r, _ in side.terms} <= set(ports):
                msg = f"Flow {contract.name} references a role outside its live boundary ports"
                raise ValueError(msg)
        if not set(contract.parity.terms) <= set(gadget.outcomes):
            msg = f"Flow {contract.name} references an unknown local outcome"
            raise ValueError(msg)

    n = len(gadget.roles)
    initial = {role: q for q, role in enumerate(gadget.roles)}
    carriers = initial.copy()
    circuit = stim.Circuit()
    # Include unused passthrough ports in the analysis dimension only.
    if n:
        circuit.append("I", [n - 1])
    for moment in gadget.moments:
        for gate in moment:
            if gate.name == "RELABEL":
                a, b = gate.roles
                carriers[a], carriers[b] = carriers[b], carriers[a]
            else:
                circuit.append(gate.name, [carriers[r] for r in gate.roles])
        circuit.append("TICK")
    events = {name: i for i, name in enumerate(gadget.outcomes)}

    def operator(pauli: NamedPauli | None, bindings: dict, frame: tuple[str, ...]) -> stim.PauliString:
        result = stim.PauliString(n)
        for role, axis in (() if pauli is None else pauli.terms):
            physical_axis = axis
            if role in frame:
                if axis == "Y":
                    result.sign *= -1
                physical_axis = {"X": "Z", "Y": "Y", "Z": "X"}[axis]
            result[bindings[role]] = physical_axis
        return result

    generators = circuit.flow_generators()
    basis = {}
    for index, generator in enumerate(generators):
        vector, used = _vector(generator, n, len(events)), 1 << index
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (vector, used)
                break
            old, combination = basis[pivot]
            vector ^= old
            used ^= combination

    rows = []
    for contract in contracts:
        incoming = operator(contract.input, initial, gadget.input_h_frame)
        outgoing = operator(contract.output, carriers, gadget.output_h_frame)
        if contract.parity.constant:
            outgoing.sign *= -1
        target = stim.Flow(input=incoming, output=outgoing, measurements=[events[t] for t in contract.parity.terms])
        vector, used = _vector(target, n, len(events)), 0
        while vector and (pivot := vector.bit_length() - 1) in basis:
            old, combination = basis[pivot]
            vector ^= old
            used ^= combination
        support_valid = vector == 0
        indices = [i for i in range(len(generators)) if (used >> i) & 1] if support_valid else []
        product = stim.Flow()
        for index in indices:
            product *= generators[index]
        valid = support_valid and (product.input_copy().sign * product.output_copy().sign
                                   == incoming.sign * outgoing.sign)
        rows.append({"name": contract.name, "contract": asdict(contract), "description": contract.describe(),
                     "physical_flow": str(target), "valid": valid, "support_valid": support_valid,
                     "generator_indices": indices, "generator_product": str(product) if support_valid else None})
    report = {"gadget": asdict(gadget), "contracts": rows, "backend": "stim", "backend_version": stim.__version__,
              "method": "Signed flow-generator membership with binary elimination and exact Pauli products",
              "analysis_circuit": str(circuit), "generators": [str(g) for g in generators],
              "input_carriers": {r: initial[r] for r in gadget.inputs},
              "output_carriers": {r: carriers[r] for r in gadget.outputs},
              "scope": "Supplied ideal Pauli contracts in the declared boundary H frames",
              "complete_channel_specification_verified": False, "fault_tolerance_verified": False,
              "noise_equivalence_verified": False}
    return GadgetFlowAudit(json.dumps(report))
