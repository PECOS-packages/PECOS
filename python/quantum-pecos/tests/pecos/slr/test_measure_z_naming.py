"""Measurement spelling compatibility across SLR's generated formats."""

import pytest
from pecos.slr import CReg, Main, Parallel, QReg, Return, SlrConverter
from pecos.slr.gen_codes import QASMGenerator, QIRGenerator, QuantumCircuitGenerator, StimGenerator
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure, MeasureZ
from pecos.slr.transforms import ParallelOptimizer


def _bell(measure, *, export=True):
    q, c = QReg("q", 2), CReg("c", 2)
    return Main(q, c, qb.PZ(q), qb.H(q[0]), qb.CX(q[0], q[1]), measure(q) > c, *([Return(c)] if export else []))


def test_deprecated_measure_warns_and_returns_measure_z() -> None:
    q, c = QReg("q", 1), CReg("c", 1)
    with pytest.warns(DeprecationWarning, match="^Measure is deprecated") as warnings:
        gate = Measure(q[0])
    assert warnings[0].filename == __file__
    assert type(gate) is MeasureZ
    assert gate.sym == "MeasureZ"
    with pytest.warns(DeprecationWarning, match="^Measure is deprecated"):
        assigned = qb.Measure(q[0]) > c[0]
    assert type(assigned) is MeasureZ
    assert assigned.cout == (c[0],)


def test_measure_z_parallel_group_order() -> None:
    q, c = QReg("q", 2), CReg("c", 2)
    program = Main(q, c, Parallel(qb.SZ(q[0]), qb.MeasureZ(q[1]) > c[1]))
    optimized = ParallelOptimizer().transform(program)
    assert [op.sym for op in optimized.ops[0].ops] == ["MeasureZ", "SZ"]


@pytest.mark.parametrize(
    ("target", "measurement"),
    [
        ("qasm", "measure q[0] -> c[0];"),
        ("qir", "call void @__quantum__qis__mz__body(ptr null, ptr null)"),
        ("stim", "M 0 1"),
        ("quantum_circuit", "'Measure':"),
        ("guppy", "c[0] = measure(q_0).read()"),
        ("phir", '"qop": "Measure"'),
    ],
)
def test_measure_alias_generator_equivalence(target: str, measurement: str) -> None:
    def generate(measure):
        converter = SlrConverter(_bell(measure))
        if target == "phir":
            return converter.quantum_circuit().to_phir_json()
        return str(getattr(converter, target)())

    expected = generate(MeasureZ)
    assert measurement in expected
    with pytest.warns(DeprecationWarning, match="^Measure is deprecated"):
        assert generate(Measure) == expected


@pytest.mark.parametrize(
    ("generator_type", "measurement"),
    [
        (QASMGenerator, "measure q -> c;"),
        (QIRGenerator, "call void @mz_to_creg_bit(ptr null, ptr %c, i64 0)"),
        (StimGenerator, "M 0 1"),
        (QuantumCircuitGenerator, "'Measure':"),
    ],
)
def test_measure_alias_legacy_generator_equivalence(generator_type, measurement: str) -> None:
    def generate(measure):
        generator = generator_type(_internal=True)
        generator.generate_block(_bell(measure, export=False))
        if generator_type is QuantumCircuitGenerator:
            return str(generator.get_circuit())
        return str(generator.get_output())

    expected = generate(MeasureZ)
    assert measurement in expected
    with pytest.warns(DeprecationWarning, match="^Measure is deprecated"):
        assert generate(Measure) == expected
