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

"""Code generators for SLR programs.

This package provides code generators that transform SLR programs into
various target formats including QASM, QIR, Stim, and QuantumCircuit.

.. deprecated::
    The direct generator classes in this module are deprecated.
    Use :func:`pecos.slr.generate` instead, which provides:

    - AST-based code generation
    - Validation before generation
    - Analysis passes (T-count, depth, etc.)
    - Optimization passes (gate cancellation, rotation merging, etc.)

    Example::

        from pecos.slr import Main, QReg, generate
        from pecos.slr.qeclib import qubit as qb

        prog = Main(q := QReg("q", 2), qb.H(q[0]), qb.CX(q[0], q[1]))
        qasm = generate(prog, "qasm")

For AST-based code generation, see :mod:`pecos.slr.ast.codegen`.
"""

from pecos.slr.gen_codes.gen_guppy import GuppyGenerator
from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos.slr.gen_codes.gen_qir import QIRGenerator
from pecos.slr.gen_codes.gen_quantum_circuit import QuantumCircuitGenerator
from pecos.slr.gen_codes.gen_stim import StimGenerator

__all__ = [
    "GuppyGenerator",
    "QASMGenerator",
    "QIRGenerator",
    "QuantumCircuitGenerator",
    "StimGenerator",
]
