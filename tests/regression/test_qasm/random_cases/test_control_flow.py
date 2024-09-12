from pecos.qeclib import qubit as qb
from pecos.slr import Block, CReg, If, Main, QReg, Repeat


def test_phys_teleport(compare_qasm):
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
    )

    compare_qasm(prog, filename="phys.teleport")


def test_phys_tele_block_block(compare_qasm):
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
        Block(
            qb.H(q[0]),
            Block(
                qb.H(q[1]),
            ),
        ),
    )

    compare_qasm(prog, filename="phys.tele_block_block")


def test_phys_tele_if(compare_qasm):
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
        If(c == 0).Then(
            qb.H(q[0]),
        ),
    )

    compare_qasm(prog, filename="phys.tele_if")


def test_phys_tele_if_block_block(compare_qasm):
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
        If(c == 0).Then(
            qb.H(q[0]),
            Block(
                qb.H(q[1]),
            ),
        ),
    )

    compare_qasm(prog, filename="phys.tele_if_block_block")


def test_phys_tele_block_telep_block(compare_qasm):
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        c2 := CReg("m2", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
        Block(
            qb.Prep(q),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q) > c2,
            Block(
                qb.H(q[0]),
            ),
        ),
    )

    compare_qasm(prog, filename="phys.tele_block_telep_block")


def test_phys_repeat(compare_qasm):
    prog = Main(
        q := QReg("q", 2),
        c := CReg("m", 2),
        Repeat(3).block(
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q) > c,
        ),
    )

    compare_qasm(prog, filename="phys.tele_repeat")
