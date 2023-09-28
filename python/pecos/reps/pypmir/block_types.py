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

from typing import List, Optional

from .op_types import COp


class Block:
    """General block type"""

    def __init__(self, metadata: Optional[dict] = None):
        self.metadata = metadata


class SeqBlock(Block):
    """A generic sequence block"""

    def __init__(self,
                 ops: list,
                 metadata: Optional[dict] = None) -> None:

        super().__init__(metadata=metadata)
        self.ops = ops


class IfBlock(Block):
    """If/else block"""

    def __init__(self,
                 condition: "COp",
                 true_branch: List["COp"],
                 false_branch: Optional[list] = None,
                 metadata: Optional[dict] = None):
        super().__init__(metadata=metadata)
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
