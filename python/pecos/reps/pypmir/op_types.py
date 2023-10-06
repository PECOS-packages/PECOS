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


class Op:
    def __init__(
        self,
        name: str,
        args: list | None = None,
        returns: list | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.name = name
        self.args = args
        self.returns = returns
        self.metadata = metadata


class QOp(Op):
    """Quantum operation."""

    def __init__(self, name: str, args: list, returns: list | None = None, metadata: dict | None = None) -> None:
        super().__init__(
            name=name,
            args=args,
            returns=returns,
            metadata=metadata,
        )


class COp(Op):
    """Classical operation."""

    def __init__(self, name: str, args: list, returns: list | None = None, metadata: dict | None = None) -> None:
        super().__init__(
            name=name,
            args=args,
            returns=returns,
            metadata=metadata,
        )


class FFCall(COp):
    pass


class MOp(Op):
    """Machine operation."""


class EMOp(Op):
    """Error model operation."""


class SOp(Op):
    """Simulation model."""