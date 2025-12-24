# Copyright 2023 The PECOS developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Test the use of the external phir package for validating and using PHIR."""

import json
from pathlib import Path

from pecos.typing import PhirModel

this_dir = Path(__file__).parent


def test_spec_example() -> None:
    """Test PHIR specification example for dependency validation."""
    # From https://github.com/CQCL/phir/blob/main/phir_spec_qasm.md#overall-phir-example-with-quantinuums-extended-openqasm-20
    data = json.load(Path.open(this_dir / "phir/spec_example.phir.json"))

    PhirModel.model_validate(data)
