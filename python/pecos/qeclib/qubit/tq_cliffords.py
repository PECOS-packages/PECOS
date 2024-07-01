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

from pecos.qeclib.qubit.metaclasses import TQCliffordGate


class CXGate(TQCliffordGate): ...


CNOT = CX = CXGate(qasm_sym="cx")


class CYGate(TQCliffordGate): ...


CY = CYGate(qasm_sym="cy")


class CZGate(TQCliffordGate): ...


CZ = CZGate(qasm_sym="cz")


class SXXGate(TQCliffordGate): ...


SXX = SXXGate()


class SYYGate(TQCliffordGate): ...


SYY = SYYGate()


class SZZGate(TQCliffordGate): ...


SZZ = SZZGate(qasm_sym="ZZ")


class SXXdgGate(TQCliffordGate): ...


SXXdg = SXXdgGate()


class SYYdgGate(TQCliffordGate): ...


SYYdg = SYYdgGate()


class SZZdgGate(TQCliffordGate): ...


SZZdg = SZZdgGate()
