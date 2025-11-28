from pecos.slr import Block, CReg, Main, QReg, SlrConverter
from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure

class MeasureAncillas(Block):
    def __init__(self, data, ancilla, syndrome):
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
