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
from __future__ import annotations

from pecos.slr.fund import Node
from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos.slr.vars import Var, Vars


class Block(Node):
    """A collection of other operations and blocks."""

    def __init__(self, *args, ops=None, vargs=None, allow_no_ops=True):
        self.ops = []
        self.vars = Vars()

        if args and ops:
            msg = "Can not use both *args for ops and the ops keyword argument."
            raise Exception(msg)

        elif args:
            ops = args

        if vargs is not None:
            self.vars.extend_vars(vargs)

        if ops is None and not allow_no_ops:
            msg = "Missing operations!"
            raise Exception(msg)

        if ops is not None:
            self.extend(*ops)

    def extend(self, *stmts):
        """Adds more ops to the Block."""

        for s in stmts:
            if isinstance(s, Var):
                self.vars.append(s)
            elif isinstance(s, Vars):
                self.vars.extend(s)
            else:
                self.ops.append(s)

        return self

    def __iter__(self):
        for op in self.ops:
            if hasattr(op, "ops"):
                yield from op.iter()
            else:
                yield op

    def iter(self):
        yield from self.__iter__()

    def gen(self, target: object | str, add_versions=True):
        if isinstance(target, str):
            if target == "qasm":
                target = QASMGenerator(add_versions=add_versions)
            else:
                msg = f"Code gen target '{target}' is not supported."
                raise NotImplementedError(msg)

        target.generate_block(self)
        return target.get_output()

    def qasm(self, add_versions=True):
        return self.gen("qasm", add_versions=add_versions)
