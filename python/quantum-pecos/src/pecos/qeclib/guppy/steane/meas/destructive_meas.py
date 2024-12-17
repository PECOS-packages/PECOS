from guppylang import guppy
from guppylang.std.quantum import reset, cx, h, qubit

from pecos.qeclib.guppy.steane.mod import steane_module
from pecos.qeclib.guppy.func_helpers import measure_to_bit, list_insert_int

@guppy(steane_module)
def measure_z(q: list[qubit], meas: list[int], log_raw_reg: list[int], log_raw_index) -> None:
    # barrier q;

    measure_to_bit(q, 0, meas, 0)
    measure_to_bit(q, 1, meas, 1)
    measure_to_bit(q, 2, meas, 2)
    measure_to_bit(q, 3, meas, 3)
    measure_to_bit(q, 4, meas, 4)
    measure_to_bit(q, 5, meas, 5)
    measure_to_bit(q, 6, meas, 6)

    # determine raw logical output
    # ============================
    list_insert_int(log_raw_reg, 0, (meas[4] ^ meas[5]) ^ meas[6])
