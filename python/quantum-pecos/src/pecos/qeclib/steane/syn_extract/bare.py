"""Bare syndrome extraction implementations for the Steane code.

This module provides syndrome extraction without flag qubits for detecting
errors in the Steane 7-qubit quantum error correction code.
"""

from itertools import cycle
from typing import Any

from pecos.qeclib.generic.check import Check
from pecos.slr import Block, Comment, CReg, QReg


def poly2qubits(poly: list[Any], data: QReg) -> list[Any]:
    """Convert polygon node IDs to qubit references.

    Args:
        poly: Polygon representation with node IDs and color.
        data: Quantum register containing the data qubits.

    Returns:
        List of qubit references corresponding to the polygon nodes.
    """
    return [data[q] for q in poly]


class SynExtractBare(Block):
    """Bare syndrome extraction for Steane code without flag qubits."""

    def __init__(self, data: QReg, ancillas: QReg, checks: list, syn: CReg) -> None:
        """Initialize bare syndrome extraction.

        Args:
            data: Data qubit register.
            ancillas: Ancilla qubit register.
            checks: List of check operators to apply.
            syn: Classical register for syndrome storage.

        Raises:
            ValueError: If syndrome register length doesn't match expected size.
        """
        if len(syn) != 2 * len(checks) or len(syn) != 6:
            msg = f"Expected syndrome register of length 6 (2 * {len(checks)} checks), got {len(syn)}"
            raise ValueError(msg)
        a = cycle(range(len(ancillas)))
        s = iter(range(len(syn)))

        super().__init__()

        pauli = "Z"
        for c in checks:
            data_ids = c[:-1]
            syn_id = next(s)
            anc_id = next(a)
            self.extend(
                Comment(f"Check['{pauli}', {data_ids}] -> {syn}[{syn_id}]"),
                Check(
                    d=poly2qubits(c, data),
                    paulis=pauli,
                    a=ancillas[anc_id],
                    out=syn[syn_id],
                    with_barriers=False,
                ),
            )

        pauli = "X"
        for c in checks:
            data_ids = c[:-1]
            syn_id = next(s)
            anc_id = next(a)
            self.extend(
                Comment(f"Check['{pauli}', {data_ids}] -> {syn}[{syn_id}]"),
                Check(
                    d=poly2qubits(c, data),
                    paulis=pauli,
                    a=ancillas[anc_id],
                    out=syn[syn_id],
                    with_barriers=False,
                ),
            )
