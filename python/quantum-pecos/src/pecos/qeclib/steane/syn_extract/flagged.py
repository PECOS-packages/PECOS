"""Flagged syndrome extraction implementations for the Steane code.

This module provides syndrome extraction with flag qubits for detecting
and diagnosing errors during the syndrome extraction process in the
Steane 7-qubit quantum error correction code.
"""

from itertools import cycle
from typing import Any

from pecos.qeclib.generic.check_1flag import Check1Flag
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


class SynExtractFlagged(Block):
    """Flagged syndrome extraction for Steane code with flag qubits for error detection."""

    def __init__(
        self,
        data: QReg,
        ancillas: QReg,
        flag_qubits: QReg,
        checks: list,
        syn: CReg,
        flag_bits: CReg,
    ) -> None:
        """Initialize flagged syndrome extraction.

        Args:
            data: Data qubit register.
            ancillas: Ancilla qubit register.
            flag_qubits: Flag qubit register for hook error detection.
            checks: List of check operators to apply.
            syn: Classical register for syndrome storage.
            flag_bits: Classical register for flag bit storage.

        Raises:
            ValueError: If register lengths don't match expected sizes.
        """
        if not (len(syn) == len(flag_bits) == 2 * len(checks) == 6):
            msg = (
                f"Expected syndrome and flag registers of length 6 (2 * {len(checks)} checks), "
                f"got syn={len(syn)}, flag_bits={len(flag_bits)}"
            )
            raise ValueError(msg)
        a = cycle(range(len(ancillas)))
        f = cycle(range(len(flag_qubits)))
        s = iter(range(len(syn)))
        fb = iter(range(len(flag_bits)))

        super().__init__()

        pauli = "Z"
        for c in checks:
            data_ids = c[:-1]
            syn_id = next(s)
            anc_id = next(a)
            flag_qubit_id = next(f)
            flag_bit_id = next(fb)
            self.extend(
                Comment(f"Check['{pauli}', {data_ids}] -> {syn}[{syn_id}]"),
                Check1Flag(
                    d=poly2qubits(c, data),
                    ops=pauli,
                    a=ancillas[anc_id],
                    flag=flag_qubits[flag_qubit_id],
                    out=syn[syn_id],
                    out_flag=flag_bits[flag_bit_id],
                    with_barriers=False,
                ),
            )

        pauli = "X"
        for c in checks:
            data_ids = c[:-1]
            syn_id = next(s)
            anc_id = next(a)
            flag_qubit_id = next(f)
            flag_bit_id = next(fb)
            self.extend(
                Comment(f"Check['{pauli}', {data_ids}] -> {syn}[{syn_id}]"),
                Check1Flag(
                    d=poly2qubits(c, data),
                    ops=pauli,
                    a=ancillas[anc_id],
                    flag=flag_qubits[flag_qubit_id],
                    out=syn[syn_id],
                    out_flag=flag_bits[flag_bit_id],
                    with_barriers=False,
                ),
            )
