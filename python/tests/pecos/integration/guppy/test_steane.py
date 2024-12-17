from guppylang import guppy
import guppylang
from guppylang import GuppyModule
from guppylang.std.quantum import x, measure, discard, cx, qubit
from guppylang.std.builtins import result
import guppylang
from pecos.qeclib.guppy.steane import steane

guppylang.enable_experimental_features()



import pecos

def test_steane_guppy():
    st = steane.module

    mod = GuppyModule("mod")
    mod.load_all(st)
    mod.load_all(guppylang.std.quantum)
    mod.load_all(pecos.qeclib.guppy.steane.std)


    @guppy(mod)
    def main() -> None:
        st = Steane
        a = new()
        b = new()
        a.method()

        c = 0

        for q in a.d:
            discard(q)
        for q in b.d:
            discard(q)

        for q in a.a:
            discard(q)
        for q in b.a:
            discard(q)

    mod.compile_hugr()