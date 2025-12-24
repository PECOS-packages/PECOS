"""Demonstration of ParallelOptimizer transformation with QASM output."""

from pecos.qeclib import qubit as qb
from pecos.slr import Block, Main, Parallel, QReg, SlrConverter


def main() -> None:
    """Demonstrate the exact transformation with QASM output."""
    # Create the program with three Bell state preparations
    prog = Main(
        q := QReg("q", 6),
        Parallel(
            Block(
                qb.H(q[0]),
                qb.CX(q[0], q[1]),
            ),
            Block(
                qb.H(q[2]),
                qb.CX(q[2], q[3]),
            ),
            Block(
                qb.H(q[4]),
                qb.CX(q[4], q[5]),
            ),
        ),
    )

    print("=== Original Program Structure ===")
    print("Parallel(")
    print("    Block(H(q[0]), CX(q[0], q[1])),")
    print("    Block(H(q[2]), CX(q[2], q[3])),")
    print("    Block(H(q[4]), CX(q[4], q[5]))")
    print(")")
    print()

    # Generate QASM without optimization
    print("=== QASM without optimization ===")
    qasm_unopt = SlrConverter(prog, optimize_parallel=False).qasm()
    for line in qasm_unopt.split("\n"):
        if line.strip() and not line.startswith(
            ("OPENQASM", "include", "qreg", "creg"),
        ):
            print(line)
    print()

    # Generate QASM with optimization (default)
    print("=== QASM with optimization (default) ===")
    qasm_opt = SlrConverter(prog).qasm()
    for line in qasm_opt.split("\n"):
        if line.strip() and not line.startswith(
            ("OPENQASM", "include", "qreg", "creg"),
        ):
            print(line)
    print()

    print("=== Optimized Program Structure ===")
    print("Block(")
    print("    Parallel(H(q[0]), H(q[2]), H(q[4])),      # All H gates")
    print("    Parallel(CX(q[0],q[1]), CX(q[2],q[3]), CX(q[4],q[5]))  # All CX gates")
    print(")")
    print()

    print("The optimization groups operations by type, allowing all H gates")
    print("to execute in parallel, followed by all CX gates in parallel.")


# Demo code:
main()
