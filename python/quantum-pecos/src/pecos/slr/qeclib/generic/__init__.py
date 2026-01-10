"""Generic quantum error correction operations.

This package provides generic operations that can be used across different QEC codes.
"""

# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos.slr.gen_codes.generator import Generator
from pecos.slr.gen_codes.language import Language

# QIRGenerator requires llvmlite which is optional
try:
    from pecos.slr.gen_codes.gen_qir import QIRGenerator
except ImportError:
    QIRGenerator = None
