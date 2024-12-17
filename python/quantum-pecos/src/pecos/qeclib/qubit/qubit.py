from pecos.slr import Qubit
from pecos.qeclib.qubit import sq_paulis, sq_sqrt_paulis, sq_hadamards, tq_cliffords, measures

# TODO accept multiple arguments like the underlying implementations

class PhysicalQubit:

    @staticmethod
    def x(qubit: Qubit):
        """Pauli X gate"""
        return sq_paulis.X(qubit)

    @staticmethod
    def y(qubit: Qubit):
        """Pauli Y gate"""
        return sq_paulis.Y(qubit)

    @staticmethod
    def z(qubit: Qubit):
        """Pauli Z gate"""
        return sq_paulis.Z(qubit)

    @staticmethod
    def sz(qubit: Qubit):
        """Sqrt of Pauli Z gate"""
        return sq_sqrt_paulis.SZ(qubit)

    @staticmethod
    def cx(ctrl: Qubit, trgt: Qubit):
        """Controlled-X gate"""
        return tq_cliffords.CX(ctrl, trgt)

    @staticmethod
    def h(qubit: Qubit):
        """Hadamard gate"""
        return sq_hadamards.H(qubit)

    @staticmethod
    def mz(self, qubit: Qubit):
        """Measurement gate"""
        return measures.Measure(qubit)
