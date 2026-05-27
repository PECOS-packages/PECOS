"""Module containing the Color488Patch class for working with color 488 quantum error correction codes."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from pecos.slr import Bit, Block, CReg, QReg, Vars
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.color488.color488 import Color488
from pecos.slr.qeclib.color488.gates_sq import hadamards, sqrt_paulis
from pecos.slr.qeclib.color488.gates_tq import transversal_tq
from pecos.slr.qeclib.color488.meas.destructive_meas import MeasureZ, SynMeasProcessing
from pecos.slr.qeclib.color488.plot_layout import plot_layout
from pecos.slr.qeclib.color488.syn_extract.bare import SynExtractBareParallel


class Color488Patch(Vars):
    """A patch of the color 488 quantum error correction code.

    This class represents a logical qubit encoded in the color 488 code,
    providing methods for syndrome extraction, logical operations, and measurements.
    """

    def __init__(self, name: str, distance: int, num_ancillas: int) -> None:
        """Initialize a Color488Patch.

        Args:
            name: The name of the patch.
            distance: The code distance.
            num_ancillas: The number of ancilla qubits to use.
        """
        self.name = name
        self.distance = distance
        self.num_ancillas = num_ancillas
        self.layout = Color488(distance)
        self.num_data = self.layout.num_data_qubits()

        if self.num_ancillas >= self.num_data:
            msg = f"Number of ancillas ({self.num_ancillas}) must be less than number of data qubits ({self.num_data})"
            raise ValueError(msg)

        self.d = QReg(f"{name}_d__", self.num_data)
        # self.syn = QReg(f"{name}_syn", self.num_data-1)
        self.a = QReg(f"{name}_a__", self.num_ancillas)
        self.meas = CReg(f"{name}_meas__", self.num_data)

        self.vars = [
            self.d,
            # self.syn,
            self.a,
            self.meas,
        ]

    def plot_layout(
        self,
        *,
        numbered_qubits: bool = False,
        numbered_poly: bool = False,
    ) -> "plt":
        """Plot the layout of the color 488 code.

        Args:
            numbered_qubits: Whether to number the qubits in the plot.
            numbered_poly: Whether to number the stabilizer polygons in the plot.

        Returns:
            The plot of the color 488 code layout.
        """
        return plot_layout(
            self.layout,
            numbered_qubits=numbered_qubits,
            numbered_poly=numbered_poly,
        )

    def syn_extract_bare(self, syn: CReg) -> Block:
        """Extract syndromes using bare ancilla-based extraction.

        Args:
            syn: The classical register to store the syndrome values.

        Returns:
            Block: A block containing the syndrome extraction circuit.
        """
        syn_input_len = len(syn)
        syn_len = self.num_data - 1
        if syn_len != syn_input_len:
            msg = f"Syndrome register length mismatch: expected {syn_len}, got {syn_input_len}"
            raise ValueError(msg)

        _, poly = self.layout.get_layout()
        return SynExtractBareParallel(self.d, self.a, poly, syn)

    def prep_z_bare(self, syndromes: list[CReg]) -> Block:
        """Prepare the patch in the Z basis and extract syndromes.

        Args:
            syndromes: List of classical registers to store syndrome values for multiple rounds.

        Returns:
            Block: A block containing the preparation and syndrome extraction circuits.
        """
        block = Block()
        block.extend(
            qb.PZ(self.d),
        )
        for s in syndromes:
            block.extend(
                self.syn_extract_bare(s),
            )
        return block

    def meas_z(self, meas: CReg = None, *, syn: CReg = None, log: Bit = None) -> Block:
        """Destructive Z basis measurement."""
        block = Block(
            MeasureZ(self.d, self.meas),
        )

        if meas is not None:
            block.extend(
                meas.set(self.meas),
            )

        if syn is not None:
            _, poly = self.layout.get_layout()
            syn_indices = [check[:-1] for check in poly]

            block.extend(
                SynMeasProcessing(
                    self.meas,
                    syn_indices,
                    syn,
                ),
            )

        if log is not None:

            num_data = self.num_data
            d = self.distance

            print("num_data", self.num_data, "distance", self.distance)

            log_indices = list(range(num_data - d, num_data))

            print("log_indices", log_indices)

            # block.extend(
            #     SynMeasProcessing(
            #         self.meas,
            #         log_indices,
            #         log,
            #     )
            # )

        return block

    def cx(self, target: "Color488Patch") -> Block:
        """Logical CX."""
        return transversal_tq.CX(self.d, target.d)

    def cy(self, target: "Color488Patch") -> Block:
        """Logical CX."""
        return transversal_tq.CY(self.d, target.d)

    def cz(self, target: "Color488Patch") -> Block:
        """Logical CX."""
        return transversal_tq.CZ(self.d, target.d)

    def szz(self, target: "Color488Patch") -> Block:
        """Logical CX."""
        return transversal_tq.SZZ(self.d, target.d)

    def h(self) -> Block:
        """Logical H."""
        return hadamards.H(self.d)

    def sz(self) -> Block:
        """Logical SZ."""
        return sqrt_paulis.SZ(self.d)
