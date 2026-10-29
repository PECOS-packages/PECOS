# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Signed physical automorphisms and logical actions of C4-Helix."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from itertools import product

from pecos_rslib import CliffordRep, PauliStabilizerGroup, PauliString

from pecos.qec.helix.code import HelixCode, c4_helix_code
from pecos.qec.helix.compile import Matrix, logical_pauli_fixup, min_cost_words, parse_signed_logical_pauli


@dataclass(frozen=True)
class LogicalGate:
    """Physical operations in time order; swap denotes a virtual relabel.

    Indices name physical positions at the moment each op runs, so an op on a
    after swap(a, b) acts on the state that was at b. A relabel renderer must route
    later indices through the accumulated permutation instead of moving qubits.
    """

    name: str
    ops: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def is_automorphism(self) -> bool:
        """Whether only single-qubit gates and virtual relabels are used."""
        return all(len(qubits) == 1 or op == "swap" for op, qubits in self.ops)


def c4_helix_gates() -> dict[str, LogicalGate]:
    """Return the five paper gates, including the corrected five-block CZ."""
    swaps = tuple(("swap", (4 * b, 4 * b + 2)) for b in range(5))
    ops = {
        "SWAP": tuple(("swap", (i + 4, i + 12)) for i in range(4))
        + tuple(("swap", (i + 12, i + 16)) for i in range(4))
        + tuple(("swap", (i, i + 4)) for i in range(4))
        + swaps,
        "CNOT_SWAP": tuple(("swap", (4 * b, 4 * b + 3)) for b in range(5)) + swaps,
        "H_SWAP": tuple(("h", (q,)) for q in range(20)),
        "CZ": tuple(("szdg" if j in (0, 3) else "sz", (4 * b + j,)) for b in range(5) for j in range(4)),
        # Circuits flag all seven CZs: any partial flag set lowers gate-fault distance to five.
        "S0S1": (
            ("cz", (8, 19)),
            ("cz", (9, 17)),
            ("cz", (10, 18)),
            ("cz", (11, 16)),
            ("cz", (0, 1)),
            ("sz", (2,)),
            ("sz", (3,)),
            ("cz", (4, 5)),
            ("sz", (6,)),
            ("sz", (7,)),
            ("sz", (13,)),
            ("sz", (15,)),
            ("cz", (12, 14)),
        ),
    }
    return {name: LogicalGate(name, physical) for name, physical in ops.items()}


def logical_action(code: HelixCode, gate: LogicalGate) -> dict[str, str]:
    """Compute signed images of X0.., Z0.. modulo the signed stabilizer group.

    Canonical spelling orders factors by logical index, uses Y = iXZ,
    omits identity factors, and always includes a leading + or -.
    """
    clifford = CliffordRep.identity(code.n)
    arities = {"swap": 2, "x": 1, "y": 1, "z": 1, "h": 1, "sz": 1, "szdg": 1, "cz": 2, "cx": 2}
    for op, qubits in gate.ops:
        if op not in arities or len(qubits) != arities[op] or len(set(qubits)) != len(qubits):
            msg = f"Gate {gate.name}: invalid operation {(op, qubits)!r}"
            raise ValueError(msg)
        if any(type(q) is not int for q in qubits):
            msg = f"Gate {gate.name}: qubit indices must be ints, got {qubits!r}"
            raise TypeError(msg)
        if any(not 0 <= q < code.n for q in qubits):
            msg = f"Gate {gate.name}: invalid qubits {qubits!r} for n={code.n}"
            raise ValueError(msg)
        clifford = getattr(CliffordRep, op)(*qubits).compose(clifford)
    stabilizers = [PauliString.from_str(row) for row in code.stabilizers]
    group = PauliStabilizerGroup(stabilizers)
    for row in stabilizers:
        if not group.contains_with_phase(clifford.apply(row)):
            msg = f"Gate {gate.name} does not preserve signed stabilizer {row}"
            raise ValueError(msg)
    xs = [PauliString.from_str(row) for row in code.logical_x]
    zs = [PauliString.from_str(row) for row in code.logical_z]
    representatives = []
    for letters in product("IXYZ", repeat=code.k):
        representative = PauliString.I()
        for j, letter in enumerate(letters):
            if letter == "X":
                representative = representative * xs[j]
            elif letter == "Z":
                representative = representative * zs[j]
            elif letter == "Y":
                representative = representative * PauliString(phase=1) * xs[j] * zs[j]
        spelling = "".join(f"{letter}{j}" for j, letter in enumerate(letters) if letter != "I") or "I"
        representatives.append((spelling, representative))
    result = {}
    labels = [f"X{j}" for j in range(code.k)] + [f"Z{j}" for j in range(code.k)]
    for label, row in zip(labels, xs + zs, strict=True):
        image = clifford.apply(row)
        for spelling, representative in representatives:
            residual = image * representative
            if group.contains_with_phase(residual):
                result[label] = "+" + spelling
                break
            if group.contains_with_phase(-residual):
                result[label] = "-" + spelling
                break
        else:
            msg = f"Gate {gate.name}: cannot resolve logical image of {label}"
            raise ValueError(msg)
    return result


def symplectic_action(code: HelixCode, gate: LogicalGate) -> Matrix:
    """Return GF(2) rows in basis (x0.., z0..), images in columns.

    Gate A then gate B has matrix M_B @ M_A (modulo two).
    """
    action = logical_action(code, gate)
    labels = [f"X{j}" for j in range(code.k)] + [f"Z{j}" for j in range(code.k)]
    columns = [parse_signed_logical_pauli(action[label], code.k)[1] for label in labels]
    return tuple(zip(*columns, strict=True))


def logical_pauli_string(spelling: str) -> str:
    """Return the dense sign-free product of a canonical unsigned logical Pauli.

    Shared physical factors cancel; only the global phase is discarded.
    """
    if not isinstance(spelling, str):
        msg = f"Expected a canonical logical Pauli string, got {spelling!r}"
        raise TypeError(msg)
    code = c4_helix_code()
    _, bits = parse_signed_logical_pauli("+" + spelling, code.k)
    pauli = PauliString.I()
    for bit, row in zip(bits, code.logical_x + code.logical_z, strict=True):
        if bit:
            pauli = pauli * PauliString.from_str(row)
    return pauli.to_dense_str(code.n).lstrip("+-i")


def logical_pauli_ops(spelling: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return one physical op per nonidentity slot of a canonical unsigned logical Pauli.

    Products cancel shared factors; global phase is irrelevant to conjugation.
    """
    return tuple((axis.lower(), (q,)) for q, axis in enumerate(logical_pauli_string(spelling)) if axis != "I")


def logical_token_ops(token: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return the physical ops of a paper gate name or a nonidentity canonical logical Pauli."""
    if not isinstance(token, str):
        msg = f"Logical gate tokens must be strings, got {token!r}"
        raise TypeError(msg)
    gates = c4_helix_gates()
    if token in gates:
        return gates[token].ops
    try:
        ops = logical_pauli_ops(token)
    except ValueError as error:
        msg = f"Invalid logical gate token {token!r}"
        raise ValueError(msg) from error
    if not ops:
        msg = f"Invalid logical gate token {token!r}"
        raise ValueError(msg)
    return ops


@cache
def _clifford_words() -> dict[Matrix, tuple[str, ...]]:
    matrices = {name: symplectic_action(c4_helix_code(), gate) for name, gate in c4_helix_gates().items()}
    return min_cost_words(matrices, {name: int(name == "S0S1") for name in matrices})


def compile_logical_clifford(target: Mapping[str, str]) -> tuple[str, ...]:
    """Compile a signed two-logical-qubit Clifford with the fewest S0S1 gadgets.

    A final canonical Pauli token corrects signs when necessary; identity is omitted.
    """
    labels = ("X0", "X1", "Z0", "Z1")
    if set(target) != set(labels):
        msg = f"Expected keys {labels!r}, got {tuple(target)!r}"
        raise ValueError(msg)
    columns = []
    for label in labels:
        try:
            columns.append(parse_signed_logical_pauli(target[label], 2)[1])
        except (TypeError, ValueError) as error:
            msg = f"Invalid image for {label}: {error}"
            raise type(error)(msg) from error
    matrix = tuple(zip(*columns, strict=True))
    words = _clifford_words()
    if matrix not in words:
        msg = f"Target is not symplectic: {target!r}"
        raise ValueError(msg)
    word = words[matrix]
    gates = c4_helix_gates()
    ops = tuple(op for name in word for op in gates[name].ops)
    actual = logical_action(c4_helix_code(), LogicalGate("compiled", ops))
    fixup = logical_pauli_fixup(actual, target)
    return word if fixup == "I" else (*word, fixup)
