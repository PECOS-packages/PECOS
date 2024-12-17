import guppylang
from guppylang import GuppyModule
from guppylang import guppy
from guppylang.std.quantum import qubit, cx, h, discard
from guppylang.std.builtins import owned

# from pecos.qeclib.guppy.steane.preps.encoding_circ import encoding_circ_mod

module = GuppyModule("steane_struct")
module.load_all(guppylang.std.quantum)
module.load_all(guppylang.std.builtins)
# module.load_all(encoding_circ_mod)

guppylang.enable_experimental_features()

@guppy.struct(module)
class Steane:
    d: list[qubit]
    c: int
    # c: int = 0 # TODO: let this work...

    @guppy(module)
    def method(self: "Steane") -> None:
        # d: list[qubit] = self.data  # TODO: let this work...
        h(self.d[0])
        cx(self.d[0], self.d[1])

    # TODO: support classmethods
    # @classmethod
    # @guppy(module)
    # def new(cls) -> "QubitPair":
    #     pair = QubitPair(qubit(), qubit())
    #     pair.method()
    #     return pair

    # # TODO: WHy I am getting a mismatch error when I use `x()` instead of some other name?
    # @guppy(module)
    # def x(self: "Steane") -> None:
    #     for i in range(len(self.d)):
    #         x(self.d[i])
    #     # TODO: Why doesn't this work?
    #     # for d in self.d:
    #     #     x(d)

@guppy(module)
def st_discard(code: Steane @ owned) -> None:
    for q in code.d:
        discard(q)


@guppy(module)
def st_new() -> "Steane":
    return Steane(
        [qubit() for _ in range(7)],
        0
    )
