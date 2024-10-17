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

from pecos.reps.pypmir.instr_type import Instr


class Data(Instr):
    """General data type."""


class DefineVar(Data):
    def __init__(
        self,
        data_type: str | type,
        variable: str,
        metadata: dict | None = None,
    ) -> None:
        super().__init__(metadata=metadata)
        self.data_type = data_type
        self.variable = variable


class CVarDefine(DefineVar):
    def __init__(
        self,
        data_type: str | type,
        variable: str,
        cvar_id: int,
        size: int,
        metadata: dict | None = None,
    ) -> None:
        super().__init__(data_type=data_type, variable=variable, metadata=metadata)
        self.size = size
        self.cvar_id = cvar_id


class QVarDefine(DefineVar):
    def __init__(
        self,
        data_type: str | type,
        variable: str,
        size: int,
        qubit_ids: list[int],
        metadata: dict | None = None,
    ) -> None:
        super().__init__(data_type=data_type, variable=variable, metadata=metadata)
        self.size = size
        self.qubit_ids = qubit_ids


class ExportVar(Data):
    def __init__(
        self,
        variables: list[str],
        to: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        super().__init__(metadata=metadata)
        self.variables = variables
        self.to = to
