"""Test partial array consumption in SLR."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import Block, CReg, Main, QReg, SlrConverter


class MeasureAncillas(Block):
    """Block for measuring ancilla qubits."""

    def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
        """Initialize measurement block.

        Args:
            data: Data qubit register
            ancilla: Ancilla qubit register
            syndrome: Syndrome measurement register
        """
        super().__init__()
        self.data = data
        self.ancilla = ancilla
        self.syndrome = syndrome
        self.ops = [
            qubit.CX(data[0], ancilla[0]),
            Measure(ancilla) > syndrome,
        ]


prog = Main(
    data := QReg("data", 2),
    ancilla := QReg("ancilla", 1),
    syndrome := CReg("syndrome", 1),
    result := CReg("result", 2),
    MeasureAncillas(data, ancilla, syndrome),
    qubit.H(data[0]),
    Measure(data) > result,
)

print("Generated Guppy code:")
print("=" * 50)
print(SlrConverter(prog).guppy())
