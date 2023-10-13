import json
from pathlib import Path

# from pecos.foreign_objects.wasmer import WasmerObj
from pecos.engines.hybrid_engine import HybridEngine
from pecos.error_models.generic_error_model import GenericErrorModel


def test_example1():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS."""
    file = Path("./phir_examples/add.wasm")
    print("f1>>", file)
    # wasm = WasmerObj("./phir_examples/add.wasm")

    prog_path = Path(__file__).parent / "phir_examples/example1_no_wasm.json"
    phir = json.load(Path.open(prog_path))
    # phir = json.load(open(prog_path))

    HybridEngine().run(program=phir, shots=10)

    sim = HybridEngine(error_model=GenericErrorModel())
    sim.run(
        program=phir,
        shots=10,
        error_params={
            "p1": 2e-1,
            "p2": 2e-1,
            "p_meas": 2e-1,
            "p_init": 1e-1,
            "p1_error_model": {
                "X": 0.25,
                "Y": 0.25,
                "Z": 0.25,
                "L": 0.25,
            },
        },
    )
