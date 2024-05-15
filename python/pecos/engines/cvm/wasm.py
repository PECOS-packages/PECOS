# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import pickle
from pathlib import Path

from pecos.engines.cvm.binarray2 import BinArray2 as BinArray
from pecos.engines.cvm.sim_func import sim_exec
from pecos.engines.cvm.wasm_vms.pywasm import read_pywasm
from pecos.engines.cvm.wasm_vms.pywasm3 import read_pywasm3
from pecos.engines.cvm.wasm_vms.wasmer import read_wasmer
from pecos.engines.cvm.wasm_vms.wasmtime import read_wasmtime
from pecos.errors import MissingCCOPError


def read_pickle(picklefile):
    """Read in either a file path or byte object meant to be a pickled class used to define the ccop."""
    if isinstance(picklefile, str):  # filename
        with Path.open(picklefile, "rb") as f:
            return pickle.load(f)  # noqa: S301
    else:
        return pickle.loads(picklefile)  # byte object  # noqa: S301


def get_ccop(circuit):
    if circuit.metadata.get("ccop"):
        ccop = circuit.metadata["ccop"]
        ccop_type = circuit.metadata["ccop_type"]

        if ccop_type is None:
            ccop_type = "wasmtime"

        # Set self.ccop
        # ------------------------------------------------
        if ccop_type in {"py", "python"}:
            ccop = read_pickle(ccop)

        elif ccop_type == "wasmtime":
            ccop = read_wasmtime(ccop)

        elif ccop_type == "pywasm":
            ccop = read_pywasm(ccop)

        elif ccop_type == "pywasm3":
            ccop = read_pywasm3(ccop)

        elif ccop_type in {"wasmer", "wasmer_cl"}:
            ccop = read_wasmer(ccop, compiler="wasmer_cl")

        elif ccop_type == "wasmer_llvm":
            ccop = read_wasmer(ccop, compiler=ccop_type)

        elif ccop_type in {"obj", "object"}:
            pass

        else:
            msg = f'Got ccop object but ccop_type "{ccop_type}" is unknown or not supported!'
            raise Exception(msg)

        # Call the CCOP object initialization method.
        ccop.exec("init", [])

    else:
        ccop = None

    return ccop


def eval_cfunc(runner, params, output):
    func = params["func"]
    assign_vars = params["assign_vars"]
    args = params["args"]

    valargs = []
    for sym in args:
        valargs.append((sym, output[sym]))

    try:
        if runner.debug and func.startswith("sim_"):
            vals = sim_exec(func, runner, valargs)

        else:
            vals = runner.ccop.exec(func, valargs, debug=runner.debug)

    except AttributeError:
        ccop = runner.circuit.metadata["ccop"]
        ccop_type = runner.circuit.metadata["ccop_type"]

        if ccop is None:
            msg = f"Wasm ({ccop_type}) function not found: {func} with args: {args}"
            raise MissingCCOPError(msg) from AttributeError

        msg = f"Classical coprocessor object not assigned or missing exec method. Wasm-type = {ccop_type}"
        raise MissingCCOPError(msg) from AttributeError

    if assign_vars:
        if len(assign_vars) == 1:
            a_obj = output[assign_vars[0]]
            if runner.debug and func.startswith("sim_"):
                output[assign_vars[0]] = vals
            else:
                b = BinArray(a_obj.size, int(vals))
                a_obj.set(b)

        else:
            for asym, b in zip(assign_vars, vals):
                a_obj = output[asym]

                if runner.debug and func.startswith("sim_"):
                    output[asym] = b
                else:
                    if isinstance(b, int):
                        b = BinArray(a_obj.size, int(b))
                        a_obj.set(b)
                    else:
                        msg = "Only int return values are supported currently"
                        raise NotImplementedError(msg)
