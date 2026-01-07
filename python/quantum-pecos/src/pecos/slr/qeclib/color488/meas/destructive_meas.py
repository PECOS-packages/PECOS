"""Module containing destructive measurement operations for color 488 codes."""

from pecos.slr import Bit, Block, CReg, QReg
from pecos.slr.qeclib import qubit as qb


class MeasureZ(Block):
    """Measure in the logical Z basis."""

    def __init__(
        self,
        data: QReg,
        meas: CReg,
        # syn_idxes
        # log_idxes
        # syn
        # log
        # meas = None # optional
    ) -> None:
        """Initialize MeasureZ block for logical Z basis measurement.

        Args:
            data: Register containing the data qubits to be measured.
            meas: Classical register to store the measurement results.
        """
        super().__init__()

        if len(data) != len(meas):
            msg = f"Data and measurement registers must have the same length: {len(data)} != {len(meas)}"
            raise ValueError(msg)

        self.extend(
            qb.Measure(data) > meas,
        )

        # TODO: Extract the syndromes and logical outcome


class SynMeasProcessing(Block):
    """Basic syndrome extraction from measurement outcomes."""

    def __init__(
        self,
        meas: CReg,
        syn_indices: list[list[int]],
        syn: CReg,
    ) -> None:
        """Initialize syndrome measurement processing.

        Args:
            meas: Classical register containing measurement outcomes.
            syn_indices: List of lists containing qubit indices for each syndrome.
            syn: Classical register to store the extracted syndromes.
        """
        super().__init__()

        if len(syn_indices) != len(syn) / 2:
            msg = (
                f"Number of syndrome indices must equal half the syndrome register length: "
                f"{len(syn_indices)} != {len(syn) / 2}"
            )
            raise ValueError(
                msg,
            )

        for i, s in enumerate(syn_indices):
            for j in s:
                self.extend(
                    syn.set(syn[i] ^ meas[j]),
                )


class RawLogMeasProcessing(Block):
    """Basic measuring process for raw logical outcome."""

    def __init__(
        self,
        meas: CReg,
        log_indices: list[int],
        log: Bit,
    ) -> None:
        """Initialize raw logical measurement processing.

        Args:
            meas: Classical register containing measurement outcomes.
            log_indices: List of qubit indices that form the logical operator.
            log: Bit to store the logical measurement outcome.
        """
        super().__init__()

        for i in log_indices:
            self.extend(log.set(log[i] ^ meas[i]))
