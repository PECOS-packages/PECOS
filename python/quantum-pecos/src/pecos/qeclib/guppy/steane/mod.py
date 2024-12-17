# import guppylang
import guppylang.std.quantum
import guppylang.std.builtins
from guppylang import GuppyModule
from pecos.qeclib.guppy.func_helpers import func_helpers
from pecos.qeclib.guppy.steane.preps.encoding_circ import encoding_circ_mod


steane_module = GuppyModule("steanemodule")

steane_module.load_all(guppylang.std.quantum)
steane_module.load_all(guppylang.std.builtins)
steane_module.load_all(encoding_circ_mod)
steane_module.load_all(func_helpers)
guppylang.enable_experimental_features()
