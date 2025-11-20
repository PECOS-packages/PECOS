"""Bare syndrome extraction implementations for Color488 codes."""

from itertools import chain, cycle, repeat
from typing import Any

import pecos as pc
from pecos.qeclib.generic.check import Check
from pecos.slr import Block, Comment, CReg, Parallel, QReg


def poly2qubits(poly: list[Any], data: QReg) -> list[Any]:
    """Convert polygon node IDs to qubit references.

    Args:
        poly: Polygon representation with node IDs and color.
        data: Quantum register containing the data qubits.

    Returns:
        List of qubit references corresponding to the polygon nodes.
    """
    return [data[q] for q in poly[:-1]]


class SynExtractBare(Block):
    """Bare syndrome extraction circuit without parallelization."""

    def __init__(self, data: QReg, ancillas: QReg, checks: list, syn: CReg) -> None:
        """Initialize bare syndrome extraction.

        Args:
            data: Data qubit register.
            ancillas: Ancilla qubit register.
            checks: List of check operators.
            syn: Classical register for syndrome storage.
        """
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


class SynExtractBareParallel(Block):
    """Bare syndrome extraction circuit with parallelization."""

    def __init__(self, data: QReg, ancillas: QReg, checks: list, syn: CReg) -> None:
        """Initialize parallel bare syndrome extraction.

        Args:
            data: Data qubit register.
            ancillas: Ancilla qubit register.
            checks: List of check operators.
            syn: Classical register for syndrome storage.
        """
        a = cycle(range(len(ancillas)))
        s = iter(range(len(syn)))

        super().__init__()

        annotations = Block()
        num_parallel_blocks = 2 * pc.ceil(len(checks) / len(ancillas))
        par_blocks = [Parallel() for _ in range(num_parallel_blocks)]

        # iterator for parallelizing circuits for one round of ancilla use
        par_iter = chain.from_iterable(repeat(obj, len(ancillas)) for obj in par_blocks)

        pauli = "Z"
        for c in checks:
            data_ids = c[:-1]
            syn_id = next(s)
            anc_id = next(a)
            annotations.extend(
                Comment(f"Check['{pauli}', {data_ids}] -> {syn}[{syn_id}]"),
            )
            par = next(par_iter)
            par.extend(
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
            annotations.extend(
                Comment(f"Check['{pauli}', {data_ids}] -> {syn}[{syn_id}]"),
            )
            par = next(par_iter)
            par.extend(
                Check(
                    d=poly2qubits(c, data),
                    paulis=pauli,
                    a=ancillas[anc_id],
                    out=syn[syn_id],
                    with_barriers=False,
                ),
            )

        self.extend(
            annotations,
            Comment(),
        )

        for p in par_blocks:
            self.extend(
                Comment(),
                p,
            )
