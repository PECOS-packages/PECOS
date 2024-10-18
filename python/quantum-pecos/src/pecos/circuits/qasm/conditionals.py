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

'''class CIf:
"""Represents a if expression""".

def __init__(self, cond, expr, else_expr=None):
self.cond = cond
self.expr = expr
self.else_expr = else_expr

def __str__(self):
return f'if({str(self.cond)}) {str(self.expr)}'
'''


class CIf:
    """Represents a if expression."""

    def __init__(self, cond, *expr, expect=None) -> None:
        self.cond = cond

        if expect is not None:
            self.expect = f" expect({expect})"
        else:
            self.expect = ""

        if len(expr) == 1 and isinstance(expr[0], (tuple, list)):
            self.expr = [expr[0]]
        else:
            self.expr = expr

    def __str__(self) -> str:
        ifs = []

        for e in self.expr:
            for n in str(e).split("\n"):
                ifs.append(f"if{self.expect}({str(self.cond)}) {n}")

        return "\n".join(ifs)


class CIfExpect:
    """Represents a if expression with expression."""

    def __init__(self, expect, cond, *expr) -> None:
        self.cond = cond

        if expect is not None:
            self.expect = f" expect({expect})"
        else:
            self.expect = ""

        if len(expr) == 1 and isinstance(expr[0], (tuple, list)):
            self.expr = [expr[0]]
        else:
            self.expr = expr

    def __str__(self) -> str:
        ifs = []

        for e in self.expr:
            for n in str(e).split("\n"):
                ns = n.strip()

                if len(ns) > 0 and not ns.startswith("//"):
                    ifs.append(f"if({str(self.cond)}) {n}")
                else:
                    ifs.append(ns)

        return "\n".join(ifs)
