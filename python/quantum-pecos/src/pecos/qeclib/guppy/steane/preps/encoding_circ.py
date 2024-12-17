from guppylang import guppy, GuppyModule
from guppylang.std.quantum import reset, cx, h, qubit
import guppylang
from guppylang.std import builtins

# from pecos.qeclib.guppy.steane.mod import steane_module

encoding_circ_mod = GuppyModule("encoding_circ_mod")

encoding_circ_mod.load_all(guppylang.std.quantum)
encoding_circ_mod.load_all(guppylang.std.builtins)
guppylang.enable_experimental_features()

@guppy(encoding_circ_mod)
def encoding_circuit(q: list[qubit]) -> None:
    # Encoding circuit
    # ---------------
    reset(q[0])
    reset(q[1])
    reset(q[2])
    reset(q[3])
    reset(q[4])
    reset(q[5])

    # q[6] is the input qubit

    cx(q[6], q[5])

    h(q[1])
    cx(q[1], q[0])

    h(q[2])
    cx(q[2], q[4])

    # ---------------
    h(q[3])
    cx(q[3], q[5])
    cx(q[2], q[0])
    cx(q[6], q[4])

    # ---------------
    cx(q[2], q[6])
    cx(q[3], q[4])
    cx(q[1], q[5])

    # ---------------
    cx(q[1], q[6])
    cx(q[3], q[0])