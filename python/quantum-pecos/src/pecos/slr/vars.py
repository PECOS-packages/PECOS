# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


from pecos.slr.cops import SET, PyCOp

# TODO: Make it a VarDef


class Vars:
    """A collection of variables."""

    def __init__(self, *_args) -> None:
        self.vars = []
        # Store the source class name for code generation
        self.source_class = None

    def extend(self, vars_obj: "Vars") -> None:
        if isinstance(vars_obj, Vars):
            self.vars.extend(vars_obj.vars)
            # Preserve source class information if available
            if hasattr(vars_obj, "source_class") and vars_obj.source_class:
                # Store mapping of variables to their source class
                if not hasattr(self, "var_source_classes"):
                    self.var_source_classes = {}
                for var in vars_obj.vars:
                    if hasattr(var, "sym"):
                        self.var_source_classes[var.sym] = vars_obj.source_class
        else:
            msg = f"Was expecting a Vars object. Instead got type: {type(vars_obj)}"
            raise TypeError(msg)

    def append(self, op) -> bool:
        if isinstance(op, Var):
            self.vars.append(op)
            return True
        return False

    def extend_vars(self, vargs) -> None:
        for v in vargs:
            if not self.append(v):
                msg = f"Unrecognized variable type: {type(v)}"
                raise TypeError(msg)

    def get(self, sym: str):
        for v in self.vars:
            if v.sym == sym:
                return v
        return None

    def __iter__(self):
        return iter(self.vars)


class Var: ...


class Reg(Var):
    def __init__(self, sym: str, size: int, elem_type: type["Elem"]) -> None:
        self.sym = sym
        self.size = size
        self.elems = []
        self.elem_type = elem_type

        for i in range(size):
            self.elems.append(self.new_elem(i))

    def new_elem(self, item):
        return self.elem_type(self, item)

    def set(self, other):
        return SET(self, other)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.elems[item]

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        if self.sym is not None:
            repr_str = f"{repr_str}:{self.sym}"
        return f"<{repr_str} at {hex(id(self))}>"

    def __str__(self) -> str:
        return self.sym


class Elem(Var):
    def __init__(self, reg: Reg, idx: int) -> None:
        super().__init__()

        self.reg = reg
        self.index = idx

    def set(self, other):
        return SET(self, other)

    def __getitem__(self, item: int):
        msg = f"'{self.__class__.__name__}' object is not subscriptable"
        raise TypeError(msg)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.index} of {self.reg.sym}>"

    def __str__(self) -> str:
        return f"{self.reg.sym}[{self.index}]"


class QReg(Reg):
    def __init__(self, sym: str, size: int) -> None:
        super().__init__(sym, size, elem_type=Qubit)


class Qubit(Elem):
    def __init__(self, reg: QReg, idx: int) -> None:
        super().__init__(reg, idx)


class CReg(Reg, PyCOp):
    def __init__(self, sym: str, size: int, *, result: bool = True) -> None:
        """
        Representation for a collection of bits.

        Args:
            sym:
            size:
            result: Whether this register is a result register (default True)
        """
        super().__init__(sym, size, elem_type=Bit)
        self.result = result


class Bit(Elem, PyCOp):
    def __init__(self, reg: CReg, idx: int) -> None:
        super().__init__(reg, idx)
