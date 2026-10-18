# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Paper fixtures and independent construction/compilation checks."""

import json
from collections import Counter
from pathlib import Path

import pytest
from pecos.qec.helix import (
    LogicalGate,
    c4_helix_code,
    c4_helix_gates,
    helix_code,
    logical_action,
    logical_pauli_fixup,
    min_cost_words,
    symplectic_action,
)
from pecos.testing import group_contains
from pecos_rslib import CliffordRep, PauliStabilizerGroup, PauliString, StabilizerCode

PAPER = json.loads((Path(__file__).parent / "fixtures" / "c4_helix_paper.json").read_text())
C4 = PAPER["c4_helix_20_2_6"]


def physical(ops, n=20):
    """Compose independently at the physical Clifford level."""
    result = CliffordRep.identity(n)
    for name, qubits in ops:
        result = getattr(CliffordRep, name)(*qubits).compose(result)
    return result


def row_space(rows):
    """Canonical GF(2) RREF, independent of package algebra."""
    vectors = set()
    for row in rows:
        vectors.add(
            sum(1 << j for j, p in enumerate(row) if p in "XY")
            | sum(1 << (len(row) + j) for j, p in enumerate(row) if p in "ZY"),
        )
    basis = {}
    for vector in sorted(vectors):
        value = vector
        for pivot in sorted(basis, reverse=True):
            if value & (1 << pivot):
                value ^= basis[pivot]
        if value:
            pivot = value.bit_length() - 1
            for previous, prior in basis.items():
                if prior & (1 << pivot):
                    basis[previous] = prior ^ value
            basis[pivot] = value
    return tuple(sorted(basis.values()))


@pytest.mark.parametrize(("factory", "key"), [(helix_code, "helix_10_2_3"), (c4_helix_code, "c4_helix_20_2_6")])
def test_code_data_and_parameters(factory, key):
    code = factory()
    expected = PAPER[key]
    assert (code.n, code.k, code.distance) == (expected["n"], expected["k"], expected["d"])
    assert code.stabilizers == tuple(expected["stabilizers"])
    assert code.logical_x == tuple(expected["logicals"][f"X{j}"] for j in range(2))
    assert code.logical_z == tuple(expected["logicals"][f"Z{j}"] for j in range(2))
    assert code.blocks == tuple(tuple(block) for block in expected.get("blocks", ()))
    assert code.name == ("Helix" if code.n == 10 else "C4-Helix")
    group = PauliStabilizerGroup.from_str("\n".join(code.stabilizers))
    analyzed = StabilizerCode(group, code.n)
    assert analyzed.num_qubits() == expected["n"]
    assert analyzed.num_logical_qubits() == expected["k"]
    assert analyzed.distance() == expected["d"]
    xs = [PauliString.from_str(row) for row in code.logical_x]
    zs = [PauliString.from_str(row) for row in code.logical_z]
    for j in range(2):
        for k in range(2):
            assert xs[j].anticommutes_with(zs[k]) == (j == k)
            assert xs[j].commutes_with(xs[k])
            assert zs[j].commutes_with(zs[k])
    for logical in xs + zs:
        assert all(logical.commutes_with(PauliString.from_str(row)) for row in code.stabilizers)


def test_csd_and_c4_concatenation():
    checks = [("10010", "01100"), ("01001", "00110"), ("10100", "00011"), ("01010", "10001"), ("00101", "11000")]
    doubled = ["".join("X" if bit == "1" else "I" for bit in x + z) for x, z in checks]
    doubled += ["".join("Z" if bit == "1" else "I" for bit in z + x) for x, z in checks]
    assert row_space(doubled) == row_space(PAPER["helix_10_2_3"]["stabilizers"])
    inner_x = ("XIIX", "IIXX")
    inner_z = ("IIZZ", "ZIIZ")
    concatenated = []
    for row in doubled:
        blocks = []
        for j in range(5):
            block = PauliString.I()
            for inner, outer in enumerate((j, j + 5)):
                if row[outer] == "X":
                    block = block * PauliString.from_str(inner_x[inner])
                if row[outer] == "Z":
                    block = block * PauliString.from_str(inner_z[inner])
            blocks.append(block.to_dense_str(4).lstrip("+"))
        concatenated.append("".join(blocks))
    concatenated += ["I" * (4 * b) + letter * 4 + "I" * (16 - 4 * b) for letter in "XZ" for b in range(5)]
    assert row_space(concatenated) == row_space(C4["stabilizers"])


@pytest.mark.parametrize("name", C4["signed_logical_images"])
def test_signed_images(name):
    assert logical_action(c4_helix_code(), c4_helix_gates()[name]) == C4["signed_logical_images"][name]


@pytest.mark.parametrize("name", C4["signed_logical_images"])
def test_signed_preservation_and_automorphisms(name):
    code = c4_helix_code()
    gate = c4_helix_gates()[name]
    assert gate.is_automorphism == (name != "S0S1")
    clifford = physical(gate.ops)
    generators = tuple("+" + row for row in code.stabilizers)
    for row in code.stabilizers:
        image = clifford.apply(PauliString.from_str(row)).to_dense_str(code.n)
        if image[0] not in "+-":
            image = "+" + image
        assert group_contains(generators, image)


def test_four_block_cz_rejected():
    gate = c4_helix_gates()["CZ"]
    broken = LogicalGate("four-block CZ", tuple((op, qs) for op, qs in gate.ops if qs[0] < 16))
    with pytest.raises(ValueError, match=r"four-block CZ.*signed stabilizer"):
        logical_action(c4_helix_code(), broken)


@pytest.fixture(scope="module")
def compiled():
    matrices = {name: symplectic_action(c4_helix_code(), gate) for name, gate in c4_helix_gates().items()}
    costs = {name: int(name == "S0S1") for name in matrices}
    return matrices, min_cost_words(matrices, costs)


def test_group_orders_and_cost_distribution(compiled):
    matrices, words = compiled
    automorphisms = {name: matrix for name, matrix in matrices.items() if name != "S0S1"}
    assert len(min_cost_words(automorphisms, dict.fromkeys(automorphisms, 0))) == C4["automorphism_group_order"]
    assert len(words) == C4["sp4_order"]
    assert Counter(word.count("S0S1") for word in words.values()) == {
        int(key): value for key, value in C4["s0s1_count_distribution"].items()
    }


def multiply(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(len(left))) % 2 for j in range(len(left)))
        for i in range(len(left))
    )


def pauli_ops(spelling, code):
    """Physical representative; global phase of Y is irrelevant to conjugation."""
    ops = []
    for j in range(code.k):
        for letter, rows in (("X", code.logical_x), ("Z", code.logical_z)):
            if f"{letter}{j}" in spelling or f"Y{j}" in spelling:
                ops.extend((p.lower(), (q,)) for q, p in enumerate(rows[j]) if p != "I")
    return tuple(ops)


def test_physical_round_trips(compiled):
    matrices, words = compiled
    code = c4_helix_code()
    gates = c4_helix_gates()
    sampled = sorted(words)[::13]
    assert len(sampled) >= 50
    group = PauliStabilizerGroup.from_str("\n".join(code.stabilizers))
    for index, element in enumerate(sampled):
        word = words[element]
        matrix = tuple(tuple(int(i == j) for j in range(4)) for i in range(4))
        for name in word:
            matrix = multiply(matrices[name], matrix)
        assert matrix == element
        ops = tuple(op for name in word for op in gates[name].ops)
        actual = logical_action(code, LogicalGate("word", ops))
        assert symplectic_action(code, LogicalGate("word", ops)) == element
        # Flip one selected generator sign: a valid Clifford target with the
        # same symplectic action. Ensure every target has a nontrivial sign.
        target = dict(actual)
        label = tuple(target)[index % 4]
        target[label] = ("-" if actual[label][0] == "+" else "+") + actual[label][1:]
        if all(value[0] == "+" for value in target.values()):
            target[label] = "-" + target[label][1:]
        correction = logical_pauli_fixup(actual, target)
        corrected = physical(ops + pauli_ops(correction, code))
        assert logical_action(code, LogicalGate("fixed", ops + pauli_ops(correction, code))) == target
        for label, representative in zip(("X0", "X1", "Z0", "Z1"), code.logical_x + code.logical_z, strict=True):
            image = corrected.apply(PauliString.from_str(representative))
            desired = PauliString.I()
            for j in range(code.k):
                x = PauliString.from_str(code.logical_x[j])
                z = PauliString.from_str(code.logical_z[j])
                if f"Y{j}" in target[label]:
                    desired = desired * PauliString(phase=1) * x * z
                elif f"X{j}" in target[label]:
                    desired = desired * x
                elif f"Z{j}" in target[label]:
                    desired = desired * z
            if target[label][0] == "-":
                desired = -desired
            assert group.contains_with_phase(image * desired)


def test_compiler_generic_and_validation():
    h = ((0, 1), (1, 0))
    identity = ((1, 0), (0, 1))
    assert min_cost_words({"b": h, "a": h}, {"b": 0, "a": 0}) == {identity: (), h: ("a",)}
    with pytest.raises(ValueError, match=r"nonnegative.*-1"):
        min_cost_words({"h": h}, {"h": -1})
    with pytest.raises(ValueError, match="not symplectic"):
        min_cost_words({"bad": ((0, 0), (0, 0))}, {"bad": 1})
    with pytest.raises(ValueError, match="Symplectic parts differ"):
        logical_pauli_fixup({"X0": "+X0", "Z0": "+Z0"}, {"X0": "+Z0", "Z0": "+X0"})
    assert logical_pauli_fixup({"X0": "+X0", "Z0": "+Z0"}, {"X0": "+X0", "Z0": "+Z0"}) == "I"
    assert logical_pauli_fixup({"X0": "+X0", "Z0": "+Z0"}, {"X0": "-X0", "Z0": "-Z0"}) == "Y0"


@pytest.mark.parametrize("size", [2, 4, 6])
def test_compiler_even_dimensions(size):
    identity = tuple(tuple(int(i == j) for j in range(size)) for i in range(size))
    permutation = list(range(size))
    permutation[0], permutation[size // 2] = permutation[size // 2], permutation[0]
    hadamard = tuple(identity[i] for i in permutation)
    assert min_cost_words({"h": hadamard}, {"h": 3}) == {identity: (), hadamard: ("h",)}


@pytest.mark.parametrize("image", ["X0", "+X2", "+X0Z0", "+Z1X0", "+foo"])
def test_fixup_rejects_invalid_images(image):
    with pytest.raises(ValueError, match="logical Pauli"):
        logical_pauli_fixup({"X0": image, "Z0": "+Z0"}, {"X0": "+X0", "Z0": "+Z0"})


def test_symplectic_columns_are_images():
    # CZ: X0 -> X0Z1 and X1 -> X1Z0, so its matrix is not symmetric and a
    # transposed convention would fail here.
    expected = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1))
    assert symplectic_action(c4_helix_code(), c4_helix_gates()["CZ"]) == expected


def test_sign_only_violation_rejected():
    code = c4_helix_code()
    gate = c4_helix_gates()["S0S1"]
    flipped = tuple(("szdg", qs) if (op, qs) == ("sz", (2,)) else (op, qs) for op, qs in gate.ops)
    assert flipped != gate.ops
    clifford = physical(flipped)
    group = PauliStabilizerGroup.from_str("\n".join(code.stabilizers))
    # The unsigned group survives; only a stabilizer sign is wrong.
    assert all(group.contains(clifford.apply(PauliString.from_str(row))) for row in code.stabilizers)
    with pytest.raises(ValueError, match=r"S0S1.*signed stabilizer"):
        logical_action(code, LogicalGate("S0S1", flipped))


def test_compiler_prefers_shorter_words_on_ties():
    rotation = ((1, 1), (1, 0))
    square = ((0, 1), (1, 1))
    words = min_cost_words({"a": rotation, "z": square}, {"a": 0, "z": 0})
    # ("a", "a") reaches square at equal cost and sorts first lexicographically.
    assert words[square] == ("z",)


def paper_permutation(listing):
    """Apply an operator-product listing rightmost first; return final contents."""
    contents = list(range(20))
    for a, b in reversed(listing):
        contents[a], contents[b] = contents[b], contents[a]
    return contents


def gate_permutation(gate):
    contents = list(range(20))
    for op, (a, b) in gate.ops:
        assert op == "swap"
        contents[a], contents[b] = contents[b], contents[a]
    return contents


def test_relabels_follow_paper_operator_products():
    gates = c4_helix_gates()
    swap_listing = (
        [(4 * i, 2 + 4 * i) for i in range(5)]
        + [(i, i + 4) for i in range(4)]
        + [(i + 12, i + 16) for i in range(4)]
        + [(i + 4, i + 12) for i in range(4)]
    )
    cnot_swap_listing = [(4 * i, 2 + 4 * i) for i in range(5)] + [(4 * i, 3 + 4 * i) for i in range(5)]
    assert gate_permutation(gates["SWAP"]) == paper_permutation(swap_listing)
    assert gate_permutation(gates["CNOT_SWAP"]) == paper_permutation(cnot_swap_listing)


def test_boolean_and_unknown_inputs_rejected():
    h = ((0, 1), (1, 0))
    with pytest.raises(TypeError, match="tuple matrix of int bits"):
        min_cost_words({"h": h, "g": ((0, True), (1, 0))}, {"h": 0, "g": 0})
    with pytest.raises(ValueError, match="2x2 matrix of bits"):
        min_cost_words({"h": h, "g": ((0, 2), (1, 0))}, {"h": 0, "g": 0})
    with pytest.raises(ValueError, match="nonnegative integer cost"):
        min_cost_words({"h": h}, {"h": True})
    with pytest.raises(ValueError, match="unknown generators"):
        min_cost_words({"h": h}, {"h": 0, "g": 1})
    with pytest.raises(TypeError, match="expected a tuple matrix"):
        min_cost_words({"h": [[0, 1], [1, 0]]}, {"h": 0})
    with pytest.raises(ValueError, match="invalid qubits"):
        logical_action(c4_helix_code(), LogicalGate("bool", (("h", (True,)),)))
