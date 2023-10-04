# Copyright 2021 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


class Gate:
    def __init__(self, sym, size=None, qasm_def=None) -> None:
        self.sym = sym
        self.size = size
        self.qasm_def = qasm_def
        self.results = []

    def __call__(self, *qargs):
        if len(qargs) == 1 and isinstance(qargs[0], (list, set)):
            self.results = self._multi_call(*qargs)

        else:
            if self.size is not None and len(qargs) != self.size:
                msg = (
                    f"Supplying the incorrect number of qubits for gate to operate on. "
                    f"#args {len(qargs)} != {self.size}"
                )
                raise Exception(msg)

            qs = ", ".join(str(a) for a in qargs)
            self.results = [f"{self.sym} {qs}"]

        return str(self)

    def __str__(self) -> str:
        return "\n".join(self.results)

    def _multi_call(self, *qargs):
        qargs = qargs[0]

        if len(qargs) == 0:
            msg = "List of qubits is empty"
            raise Exception(msg)

        results = []
        for loc in qargs:
            if not isinstance(loc, tuple):
                loc = (loc,)

            if self.size is not None and len(loc) != self.size:
                msg = (
                    f"Supplying the incorrect number of qubits for gate to operate on. "
                    f"#args {len(qargs)} != {self.size}"
                )
                raise Exception(msg)

            qs = ", ".join(str(a) for a in loc)
            results.append(f"{self.sym} {qs}")

        return results


class GateOld:
    def __init__(self, sym, size=None, qasm_def=None) -> None:
        self.sym = sym
        self.size = size
        self.qasm_def = qasm_def
        self.locs = []

    def __call__(self, *qargs):
        if len(qargs) == 1 and isinstance(qargs[0], (list, set)):
            return self._multi_call(*qargs)

        if self.size is not None and len(qargs) != self.size:
            msg = f"Supplying the incorrect number of qubits for gate to operate on. #args {len(qargs)} != {self.size}"
            raise Exception(msg)

        qs = ", ".join(str(a) for a in qargs)
        return f"{self.sym} {qs}"

    def _multi_call(self, *qargs):
        qargs = qargs[0]

        if len(qargs) == 0:
            msg = "List of qubits is empty"
            raise Exception(msg)

        results = []
        for loc in qargs:
            if not isinstance(loc, tuple):
                loc = (loc,)

            if self.size is not None and len(loc) != self.size:
                msg = (
                    f"Supplying the incorrect number of qubits for gate to operate on. "
                    f"#args {len(qargs)} != {self.size}"
                )
                raise Exception(msg)

            qs = ", ".join(str(a) for a in loc)
            results.append(f"{self.sym} {qs}")
        return "\n".join(results)


class ArgGate(Gate):
    def __init__(self, sym, size=None, num_args=None, qasm_def=None) -> None:
        super().__init__(sym, size, qasm_def)
        self.num_args = num_args

    def __call__(self, params, *qargs):
        if isinstance(params, (str, float, int)):
            params = (params,)

        if len(qargs) == 1 and isinstance(qargs[0], (list, set)):
            return self._multi_call(params, *qargs)

        if self.size is not None and len(qargs) != self.size:
            msg = "Supplying the incorrect number of qubits for gate to operate on."
            raise Exception(msg)

        if self.num_args is not None and len(params) != self.num_args:
            msg = "Supplying supplying the wrong number of gate parameters."
            raise Exception(msg)

        args = ", ".join(str(a) for a in params)
        qs = ", ".join(str(a) for a in qargs)
        return f"{self.sym}({args}) {qs}"

    def _multi_call(self, params, *qargs):
        args = ", ".join(str(a) for a in params)
        qargs = qargs[0]

        if len(qargs) == 0:
            msg = "List of qubits is empty"
            raise Exception(msg)

        results = []
        for loc in qargs:
            if not isinstance(loc, tuple):
                loc = (loc,)

            if self.size is not None and len(loc) != self.size:
                msg = "Supplying the incorrect number of qubits for gate to operate on."
                raise Exception(msg)

            qs = ", ".join(str(a) for a in loc)
            results.append(f"{self.sym}({args}) {qs}")
        return "\n".join(results)


class MeasGate(Gate):
    def __init__(self) -> None:
        super().__init__(sym="measure", size=1)
        self.qreg = None
        self.creg = None

    def __call__(self, qreg):
        self.qreg = qreg
        return self

    def __gt__(self, creg):
        self.creg = creg

        if isinstance(self.qreg, list):
            return self._multi_call(self.qreg, self.creg)

        return f"measure {self.qreg} -> {creg}"

    def _multi_call(self, qargs, cargs):
        if not isinstance(qargs, list):
            msg = "Both quantum and classical arguments should be a list."
            raise TypeError(msg)

        results = []

        if isinstance(cargs, list):
            if len(qargs) != len(cargs):
                msg = "The number of quantum and classical arguments must be the same."
                raise Exception(msg)

            for qloc, cloc in zip(qargs, cargs):
                results.append(f"measure {qloc} -> {cloc}")
        else:
            for i, qloc in enumerate(qargs):
                results.append(f"measure {qloc} -> {cargs}[{i}]")

        return "\n".join(results)


class ResetGate(Gate):
    def __init__(self) -> None:
        super().__init__(sym="reset", size=1)

    def __call__(self, *qargs):
        if len(qargs) == 1 and isinstance(qargs[0], (list, set)):
            return self._multi_call(*qargs)

        return "\n".join([f"reset {a};" for a in qargs])

    def _multi_call(self, *qargs):
        qargs = qargs[0]

        results = []
        for loc in qargs:
            if not isinstance(loc, tuple):
                loc = (loc,)

            qregs = ", ".join([str(a) for a in loc])
            results.append(f"reset {qregs}")

        return "\n".join(results)
