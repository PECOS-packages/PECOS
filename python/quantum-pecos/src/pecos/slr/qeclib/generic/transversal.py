"""Generic transversal gate implementations."""

from collections.abc import Callable

from pecos.slr import Block, QReg


def transversal_tq(tq_gate: Callable, q1: QReg, q2: QReg) -> Block:
    """Apply a two-qubit gate transversally across two quantum registers.

    Args:
        tq_gate: Two-qubit gate to apply.
        q1: First quantum register.
        q2: Second quantum register.

    Returns:
        Block containing the transversal gate operations.

    Raises:
        ValueError: If the two registers have different lengths.
    """
    if len(q1) != len(q2):
        msg = f"Registers must have the same length, got {len(q1)} and {len(q2)}"
        raise ValueError(msg)

    block = Block()

    for i in range(len(q1)):
        block.extend(tq_gate(q1[i], q2[i]))

    return block
