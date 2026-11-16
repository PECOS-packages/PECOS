"""PyPHIR angle units agree with the native PHIR-JSON command generator."""

import json
import math

import pytest
from pecos.reps.pyphir.pyphir import PyPHIR
from pecos_rslib import PhirJsonEngine


def program(unit: object, value: float = 1.0) -> dict:
    """Build the same one-qubit rotation for both readers."""
    return {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
            {"qop": "RZ", "args": [["q", 0]], "angles": [[value], unit]},
        ],
    }


@pytest.mark.parametrize(("unit", "scale"), [("rad", 1.0), ("pi", math.pi), ("deg", math.pi / 180)])
def test_units_match_native(unit: str, scale: float) -> None:
    """Both readers preserve the angle in radians for each accepted unit."""
    source = program(unit, 0.25)
    parsed = PyPHIR.from_phir(source)
    angle = next(op.angles[0] for op in parsed.ops if getattr(op, "angles", None))
    native = PhirJsonEngine(json.dumps(source)).process_program()[0]["params"]["theta"]
    assert angle == pytest.approx(0.25 * scale)
    assert angle == pytest.approx(native)


@pytest.mark.parametrize("unit", ["turn", "degrees", "", "RAD"])
def test_unknown_unit_is_rejected(unit: str) -> None:
    """Unknown units must name the bad value and accepted alternatives."""
    with pytest.raises(ValueError, match="rad, pi, or deg") as error:
        PyPHIR.from_phir(program(unit))
    assert repr(unit) in str(error.value)


@pytest.mark.parametrize("unit", [None, 1, []])
def test_wrong_unit_type_is_rejected(unit: object) -> None:
    """Unit types have a TypeError rather than implicit radians."""
    with pytest.raises(TypeError, match="use rad, pi, or deg"):
        PyPHIR.from_phir(program(unit))
