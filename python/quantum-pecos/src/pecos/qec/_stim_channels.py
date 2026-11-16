"""Faithful Stim representations of the supported inline Pauli-channel subset.

Stim is an optional export target, not the specification of TickCircuit.
Unsupported channels raise instead of dropping or approximating their payloads.
"""

from __future__ import annotations

from itertools import product
from math import fsum, isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.quantum import Gate, TickCircuit


def has_inline_noise(circuit: TickCircuit) -> bool:
    """Recognize payloads and explicit zero-noise intent."""
    return circuit.get_meta("pecos_noise_source") == "inline" or any(
        gate.is_channel() for _, gate in circuit.gate_batches()
    )


def channel_to_stim(gate: Gate) -> str | None:
    """Export one categorical Pauli channel, retaining mutually exclusive branches.

    PAULI_CHANNEL_2 arguments use IX,IY,IZ,XI,...,ZZ with the first letter on
    the first target, per https://github.com/quantumlib/Stim/blob/main/doc/gates.md.
    Read physical qubit labels from the payload, not an assumed integer encoding.
    """
    qubits = tuple(int(q) for q in gate.qubits)
    if len(qubits) not in (1, 2) or len(set(qubits)) != len(qubits):
        msg = "Stim export supports inline Pauli channels on one or two distinct qubits"
        raise ValueError(msg)
    words = {"".join(word): [] for word in product("IXYZ", repeat=len(qubits))}
    for probability, terms in gate.channel_mixed_pauli_terms():
        if not isfinite(probability) or not 0 <= probability <= 1:
            msg = "Inline channel probabilities must be finite and in [0, 1]"
            raise ValueError(msg)
        word = ["I"] * len(qubits)
        seen = set()
        for pauli, qubit in terms:
            if pauli not in "IXYZ" or len(pauli) != 1 or qubit not in qubits or qubit in seen:
                msg = "Invalid Pauli support in inline channel payload"
                raise ValueError(msg)
            seen.add(qubit)
            word[qubits.index(qubit)] = pauli
        words["".join(word)].append(probability)
    probabilities = {word: fsum(values) for word, values in words.items()}
    if abs(fsum(probabilities.values()) - 1) > 1e-12:
        msg = "Inline channel probabilities must sum to 1"
        raise ValueError(msg)
    probabilities.pop("I" * len(qubits))
    active = [(word, p) for word, p in probabilities.items() if p > 0]
    if not active:
        return None
    if len(active) == 1:
        word, p = active[0]
        support = [(pauli, q) for pauli, q in zip(word, qubits, strict=True) if pauli != "I"]
        if len(support) == 1:
            pauli, q = support[0]
            return f"{pauli}_ERROR({p!r}) {q}"
        targets = " ".join(f"{pauli}{q}" for pauli, q in support)
        return f"CORRELATED_ERROR({p!r}) {targets}"
    targets = " ".join(str(q) for q in qubits)
    values = list(probabilities.values())
    # Exact equality is intentional: an almost uniform channel stays categorical.
    if all(p == values[0] for p in values):
        p = values[0] * len(values)
        return f"DEPOLARIZE{len(qubits)}({p!r}) {targets}"
    args = ",".join(repr(p) for p in values)
    return f"PAULI_CHANNEL_{len(qubits)}({args}) {targets}"
