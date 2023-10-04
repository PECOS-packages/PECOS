# Copyright 2020 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.error_models.parent_class_error_gen import ParentErrorModel


class FakeErrorModel(ParentErrorModel):
    def __init__(self, error_circuits) -> None:
        super().__init__()
        self.error_circuits = error_circuits
        self.leaked_qubits = set()

    def start(self, *args, **kwargs):
        return self.error_circuits

    def generate_tick_errors(self, *args, **kwargs):
        return self.error_circuits
