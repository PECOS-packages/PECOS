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
        checks: list,
        syn_x: CReg,
        syn_z: CReg,
        flag_bits_x: CReg,
        flag_bits_z: CReg,
    ) -> None:
        """Initialize flagged syndrome extraction.

        Args:
            data: Data qubit register.
            ancillas: Ancilla qubit register.
            checks: List of check operators to apply.
            syn_x: Classical register for X syndrome storage.
            syn_z: Classical register for Z syndrome storage.
            flag_bits_x: Classical register for X flag bit storage.
            flag_bits_z: Classical register for Z flag bit storage.

        Raises:
            ValueError: If register lengths don't match expected sizes.
        """
        if not (
            len(syn_x)
            == len(syn_z)
            == len(flag_bits_x)
            == len(flag_bits_z)
            == len(checks)
            == 3
        ):
            msg = (
                f"Expected syndrome and flag registers of length 3 ({len(checks)} checks), "
                f"got syn_x={len(syn_x)}, syn_z={len(syn_z)}, "
                f"flag_bits_x={len(flag_bits_x)}, flag_bits_z={len(flag_bits_z)}"
            )
            raise ValueError(msg)
        a = cycle(range(len(ancillas)))
        sx = iter(range(len(syn_x)))
        sz = iter(range(len(syn_z)))
        fbx = iter(range(len(flag_bits_x)))
        fbz = iter(range(len(flag_bits_z)))

        super().__init__()

        pauli = "Z"
        for c in checks:
            data_ids = c[:-1]
            syn_id = next(sz)
            anc_id = next(a)
            flag_qubit_id = next(a)
            flag_bit_id = next(fbz)
            self.extend(
                Comment(f"Check['{pauli}', {data_ids}] -> {syn_z}[{syn_id}]"),
                Check1Flag(
                    d=poly2qubits(c, data),
                    ops=pauli,
                    a=ancillas[anc_id],
                    flag=ancillas[flag_qubit_id],
                    out=syn_z[syn_id],
                    out_flag=flag_bits_z[flag_bit_id],
                    with_barriers=False,
                ),
            )

        pauli = "X"
        for c in checks:
            data_ids = c[:-1]
            syn_id = next(sx)
            anc_id = next(a)
            flag_qubit_id = next(a)
            flag_bit_id = next(fbx)
            self.extend(
                Comment(f"Check['{pauli}', {data_ids}] -> {syn_x}[{syn_id}]"),
                Check1Flag(
                    d=poly2qubits(c, data),
                    ops=pauli,
                    a=ancillas[anc_id],
                    flag=ancillas[flag_qubit_id],
                    out=syn_x[syn_id],
                    out_flag=flag_bits_x[flag_bit_id],
                    with_barriers=False,
                ),
            )
