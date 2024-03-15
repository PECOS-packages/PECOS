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


from pecos.slr.block import Block


class CondBlock(Block):
    def __init__(self, *args, cond=None, ops=None):
        if cond is None and args:
            if len(args) > 1:
                msg = "Expected a single condition."
                raise Exception(msg)
            cond = args[0]
            args = ()

        super().__init__(*args, ops=ops, vargs=None, allow_no_ops=True)
        self.cond = cond

    def extend(self, *ops):
        raise NotImplementedError

    def _extend(self, *ops):
        super().extend(*ops)
        return self


class If(CondBlock):
    def __init__(self, *args, cond=None, then_block=None, else_block=None):
        super().__init__(*args, cond=cond, ops=then_block)
        self.else_block = None if else_block is None else self.Else(else_block)

    def Then(self, *args):
        self._extend(*args)
        return self

    def Else(self, *args):
        raise NotImplementedError

    def qasm(self):
        qasm = []

        cond = self.cond.qasm()

        for op in self.ops:
            for q in op.qasm().split("\n"):
                # TODO: Make this more eloquent....
                q = q.strip()
                if q != "" and not q.startswith("//"):
                    for qi in q.split(";"):
                        qi = qi.strip()
                        if qi != "" and not qi.startswith("//"):
                            qasm.append(f"if({cond}) {qi};")
                else:
                    qasm.append(q)

        return "\n".join(qasm)


class Repeat(CondBlock):
    def __init__(self, cond=None):
        if not isinstance(cond, int):
            msg = f"Condition for Repeat block must be an int! Got type: {type(cond)}"
            raise TypeError(msg)
        super().__init__(cond=cond)

    def block(self, *args):
        super()._extend(*args)
        return self

    def qasm(self):
        qasm = []
        for _ in range(self.cond):
            for op in self.ops:
                qasm.append(op.qasm())
        return "\n".join(qasm)
