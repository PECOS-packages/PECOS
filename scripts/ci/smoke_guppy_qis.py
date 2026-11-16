"""Exercise an installed PECOS wheel through the Guppy -> QIS -> Selene path."""

import sys
from pathlib import Path

import pecos
import pecos_rslib
from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit


class QisSmokeError(RuntimeError):
    """Raised when the installed-wheel QIS smoke test fails."""


@guppy
def coin() -> bool:
    q = qubit()
    h(q)
    return measure(q).read()


bundle = Path(pecos_rslib.__file__).parent / "_native"
for library in ("pecos_qis_ffi", "selene_simple_runtime", "selene_soft_rz_runtime"):
    filename = (
        f"{library}.dll" if sys.platform == "win32" else f"lib{library}.{'dylib' if sys.platform == 'darwin' else 'so'}"
    )
    if not (bundle / filename).is_file():
        message = f"installed wheel is missing {filename} in {bundle}"
        raise QisSmokeError(message)

for runtime in ("selene_simple_runtime", "selene_soft_rz_runtime"):
    results = pecos.sim(pecos.Guppy(coin)).classical(pecos.selene_engine(runtime)).qubits(1).seed(42).run(4).to_dict()
    shot_count = len(results["measurement_0"])
    if shot_count != 4:
        message = f"expected 4 QIS results from {runtime}, received {shot_count}"
        raise QisSmokeError(message)
