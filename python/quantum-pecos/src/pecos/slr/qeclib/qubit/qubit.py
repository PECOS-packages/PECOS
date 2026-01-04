"""Physical qubit gate implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.qeclib.qubit import (
    measures,
    preps,
    sq_hadamards,
    sq_paulis,
    sq_sqrt_paulis,
    tq_cliffords,
)

if TYPE_CHECKING:
    from pecos.slr import Bit, Qubit

# TODO: accept multiple arguments like the underlying implementations


class PhysicalQubit:
    """Collection of physical qubit gate operations."""

    @staticmethod
    def x(*qargs: Qubit) -> sq_paulis.X:
        """Pauli X gate."""
        return sq_paulis.X(*qargs)

    @staticmethod
    def y(*qargs: Qubit) -> sq_paulis.Y:
        """Pauli Y gate."""
        return sq_paulis.Y(*qargs)

    @staticmethod
    def z(*qargs: Qubit) -> sq_paulis.Z:
        """Pauli Z gate."""
        return sq_paulis.Z(*qargs)

    @staticmethod
    def sz(*qargs: Qubit) -> sq_sqrt_paulis.SZ:
        """Sqrt of Pauli Z gate."""
        return sq_sqrt_paulis.SZ(*qargs)

    @staticmethod
    def h(*qargs: Qubit) -> sq_hadamards.H:
        """Hadamard gate."""
        return sq_hadamards.H(*qargs)

    @staticmethod
    def cx(*qargs: Qubit) -> tq_cliffords.CX:
        """Controlled-X gate."""
        return tq_cliffords.CX(*qargs)

    @staticmethod
    def cy(*qargs: Qubit) -> tq_cliffords.CY:
        """Controlled-Y gate."""
        return tq_cliffords.CY(*qargs)

    @staticmethod
    def cz(*qargs: Qubit) -> tq_cliffords.CZ:
        """Controlled-Z gate."""
        return tq_cliffords.CZ(*qargs)

    @staticmethod
    def pz(*qargs: Qubit) -> preps.Prep:
        """Measurement gate."""
        return preps.Prep(*qargs)

    @staticmethod
    def mz(
        qubits: tuple[Qubit, ...],
        outputs: Bit | tuple[Bit, ...],
    ) -> measures.Measure:
        """Measurement gate."""
        return measures.Measure(*qubits) > outputs
