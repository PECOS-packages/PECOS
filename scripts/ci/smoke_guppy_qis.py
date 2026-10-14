"""Exercise an installed PECOS wheel through the Guppy -> QIS -> Selene path."""

import pecos
from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit


class QisSmokeError(RuntimeError):
    """Raised when the installed-wheel QIS smoke test fails."""


@guppy
def coin() -> bool:
    q = qubit()
    h(q)
    return measure(q).read()


results = pecos.sim(pecos.Guppy(coin)).classical(pecos.selene_engine()).qubits(1).seed(42).run(4).to_dict()
shot_count = len(results["measurement_0"])
if shot_count != 4:
    message = f"expected 4 QIS results, received {shot_count}"
    raise QisSmokeError(message)
