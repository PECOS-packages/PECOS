# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.qeclib.steane.steane_class import Steane
from pecos.slr import Barrier, CReg, If, Main
from pecos.slr.gen_codes.gen_guppy import GuppyGenerator
# from pecos.slr.gen_codes.gen_hugr import HUGRGenerator

# ruff: noqa: INP001
# Turn of Black's formatting to allow for newline spacing below:
# fmt: off


def telep(prep_basis: str, meas_basis: str) -> Main:
    """A simple example of creating a logical teleportation circuit.

    Args:
        prep_basis (str):  A string indicating what Pauli basis to prepare the state in. Acceptable inputs include:
            "+X"/"X", "-X", "+Y"/"Y", "-Y", "+Z"/"Z", and "-Z".
        meas_basis (str): A string indicating what Pauli basis the measure out the logical qubit in. Acceptable inputs
            include: "X", "Y", and "Z".

    Returns:
        A logical program written in extended OpenQASM 2.0"""

    prog = Main(
        m_bell := CReg("m_bell", size=2),
        m_out := CReg("m_out", size=1),

        # Input state:
        sin := Steane("sin", default_rus_limit=2),

        smid := Steane("smid"),
        sout := Steane("sout"),

        # Create Bell state
        smid.pz(),  # prep logical qubit in |0>/|+Z> state with repeat-until-success initialization
        sout.pz(),
        Barrier(smid.d, sout.d),
        smid.h(),
        smid.cx(sout),  # CX with control on smid and target on sout

        smid.qec(),
        sout.qec(),

        # prepare input state in some Pauli basis state
        sin.p(prep_basis, rus_limit=3),
        sin.qec(),

        # entangle input with one of the logical qubits of the Bell pair
        sin.cx(smid),
        sin.h(),

        # Bell measurement
        sin.mz(m_bell[0]),
        smid.mz(m_bell[1]),

        # Corrections
        If(m_bell[1] == 0).Then(sout.x()),
        If(m_bell[0] == 0).Then(sout.z()),

        # Final output stored in `m_out[0]`
        sout.m(meas_basis, m_out[0]),
    )

    return prog


prog = telep("+X", "X")
tel_qasm_str = prog.qasm()  # Convert the program to extended OpenQASM 2.0
tel_guppy_str = prog.gen(GuppyGenerator())  # Convert the program to a guppy file

def t_gate(prep_basis: str, meas_basis: str) -> Main:
    """A simple example of teleporting a T gate on a state.

    Args:
        prep_basis (str):  A string indicating what Pauli basis to prepare the state in. Acceptable inputs include:
            "+X", "-X", "+Y", "-Y", "+Z", and "-Z".
        meas_basis (str): A string indicating what Pauli basis the measure out the logical qubit in. Acceptable inputs
            include: "X", "Y", and "Z".

    Returns:
        A logical program written in extended OpenQASM 2.0"""

    prog = Main(
        m_reject := CReg("m_reject", size=2),
        m_t := CReg("m_t", 1),
        m_out := CReg("m_out", size=2),

        # Input state:
        sin := Steane("sin", default_rus_limit=2),
        saux := Steane("saux"),

        sin.p(prep_basis, rus_limit=3),
        # Apply T to `sin` using the logical ancilla saux and indicating whether t prep failed.
        sin.t(saux, reject=m_reject[0], rus_limit=1),

        # Final output stored in `m_out[0]`
        sin.m(meas_basis, m_out[0]),

        # More explicitly, create the resource state and teleport the gate
        # ----------------------------------------------------------------
        # Prepare the resource state
        saux.prep_t_plus_state(reject=m_reject[1], rus_limit=2),

        # Prepare the input state
        sin.p(prep_basis),

        # teleport the T gate onto the input state
        sin.cx(saux),
        saux.mz(m_t[0]),
        If(sin.t_meas == 1).Then(sin.sz()),

        # Final output stored in `m_out[1]`
        sin.m(meas_basis, m_out[1]),
    )

    return prog

prog = t_gate("+X", "X")
t_qasm_str = prog.qasm()  # Convert the program to extended OpenQASM 2.0
t_guppy_str = prog.gen(GuppyGenerator())  # Convert the program to a guppy file
# hugr_json = prog.Gen(HUGRGenerator())

# fmt: on
