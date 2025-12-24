# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


from typing import NoReturn

from pecos.slr.block import Block
from pecos.slr.cond_block import CondBlock


class While(CondBlock):
    """While loop block.

    Usage:
        While(condition).Do(
            # operations
        )
    """

    def __init__(self, *args, cond=None):
        super().__init__(*args, cond=cond)

    def Do(self, *args):
        """Add operations to the while loop body."""
        self._extend(*args)
        return self


class For(Block):
    """For loop block with iteration variable.

    Usage:
        For(i, range(n)).Do(
            # operations that can use i
        )

        For(i, start, stop).Do(
            # operations
        )

        For(i, start, stop, step).Do(
            # operations
        )
    """

    def __init__(self, var, *args, **kwargs):
        """Initialize For loop.

        Args:
            var: Loop variable name (string or symbol)
            *args: Either a range object or start, stop, [step] values
            **kwargs: Additional keyword arguments passed to parent Block
        """
        # Extract any Block-specific kwargs
        ops = kwargs.pop("ops", None)
        allow_no_ops = kwargs.pop("allow_no_ops", True)
        super().__init__(ops=ops, allow_no_ops=allow_no_ops, **kwargs)
        self.var = var

        # Parse arguments
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            # For(i, range(n)) or For(i, iterable)
            self.iterable = args[0]
            self.start = None
            self.stop = None
            self.step = None
        elif len(args) == 2:
            # For(i, start, stop)
            self.iterable = None
            self.start = args[0]
            self.stop = args[1]
            self.step = 1
        elif len(args) == 3:
            # For(i, start, stop, step)
            self.iterable = None
            self.start = args[0]
            self.stop = args[1]
            self.step = args[2]
        else:
            msg = f"Invalid arguments for For loop: {args}"
            raise ValueError(msg)

    def Do(self, *args):
        """Add operations to the for loop body."""
        super().extend(*args)
        return self

    def extend(self, *ops) -> NoReturn:
        """Prevent direct extend - use Do() instead."""
        msg = "Use Do() to add operations to For loop"
        raise NotImplementedError(msg)
