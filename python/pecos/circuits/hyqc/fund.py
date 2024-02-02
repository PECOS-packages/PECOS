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

"""Fundamental definitions for HYQC."""


class Statement:
    """Statements are something that do something."""


class Expression:
    """Expressions are something that evaluate to a values."""


class Block:
    """A sequence of statements."""

    def __init__(self, *stmts) -> None:
        self.statements = []
        self.extend(*stmts)

    def extend(self, *args):
        self.statements.extend(args)

    def __call__(self, *stmts):
        self.extend(*stmts)
