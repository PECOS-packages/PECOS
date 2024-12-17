import guppylang
from guppylang import GuppyModule
from guppylang import guppy
from guppylang.std.quantum import qubit, measure, cx, h, x
from guppylang.std import quantum
from .steane import Steane
from . import steane
guppylang.enable_experimental_features()

module = GuppyModule("steane_std")
module.load_all(guppylang.std.quantum)
module.load_all(steane.module)


@guppy(module)
def st_new() -> Steane:
    st = Steane(
        [qubit() for _ in range(7)], # data =  # TODO: Support kwargs...
        [qubit() for _ in range(3)],
        qubit(),
        0,
    )

    return st

@guppy(module)
def st_x(code: "Steane") -> None:
    for i in range(len(code.d)):
        x(code.d[i])
    # TODO: Why doesn't this work?
    # for d in self.d:
    #     x(d)

    # quantum.x doesn't work TODO: why???
