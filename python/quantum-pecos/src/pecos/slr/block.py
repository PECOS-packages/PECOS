# Copyright 2023-2024 The PECOS Developers
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

from pecos.slr.fund import Node
from pecos.slr.types import ArrayType, ReturnNotSet
from pecos.slr.vars import Var, Vars


class Block(Node):
    """A collection of other operations and blocks.

    Subclasses can declare their return types using the `block_returns` class attribute:

    Example:
        from pecos.slr.types import Array, QubitType

        class PrepEncodingFTZero(Block):
            block_returns = (Array[QubitType, 2], Array[QubitType, 7])

            def __init__(self, data, ancilla, init_bit):
                super().__init__()
                # ... implementation ...

    Note:
        Use `block_returns = None` to explicitly indicate a block returns nothing (procedural).
        If `block_returns` is not set, it defaults to ReturnNotSet, indicating the return
        type hasn't been declared.
    """

    # Subclasses override this to declare return types
    # Default to ReturnNotSet sentinel (not None, which means "returns nothing")
    block_returns = ReturnNotSet

    def __init__(
        self,
        *args,
        ops=None,
        vargs=None,
        allow_no_ops=True,
        block_name=None,
    ) -> None:
        self.ops = []
        self.vars = Vars()
        # Preserve the original block type name for code generation
        self.block_name = block_name or self.__class__.__name__
        self.block_module = self.__class__.__module__

        # Process return type annotation if present
        # Check against ReturnNotSet sentinel, not None (None means "returns nothing")
        if (
            hasattr(self.__class__, "block_returns")
            and self.__class__.block_returns is not ReturnNotSet
        ):
            self.__slr_return_type__ = self._process_return_annotation(
                self.__class__.block_returns,
            )

        if args and ops:
            msg = "Can not use both *args for ops and the ops keyword argument."
            raise Exception(msg)

        if args:
            ops = args

        if vargs is not None:
            self.vars.extend_vars(vargs)

        if ops is None and not allow_no_ops:
            msg = "Missing operations!"
            raise Exception(msg)

        if ops is not None:
            self.extend(*ops)

    def extend(self, *stmts):
        """Adds more ops to the Block."""
        for s in stmts:
            if isinstance(s, Var):
                self.vars.append(s)
            elif isinstance(s, Vars):
                self.vars.extend(s)
            else:
                self.ops.append(s)

        return self

    def __iadd__(self, other):
        """Implements += operator. For lists/tuples, calls extend(*other). For single items, calls extend(other)."""
        if isinstance(other, list | tuple):
            return self.extend(*other)
        return self.extend(other)

    def __iter__(self):
        for op in self.ops:
            if hasattr(op, "ops"):
                yield from op.iter()
            else:
                yield op

    def iter(self):
        yield from iter(self)

    def _process_return_annotation(self, returns):
        """Process the returns annotation into a structured format.

        Args:
            returns: Either a single ArrayType or a tuple of ArrayTypes

        Returns:
            A tuple of array sizes, e.g., (2, 7) for returns=(Array[Qubit, 2], Array[Qubit, 7])
        """
        if isinstance(returns, ArrayType):
            # Single return value
            return (returns.size,)
        if isinstance(returns, tuple):
            # Multiple return values
            sizes = []
            for ret_type in returns:
                if isinstance(ret_type, ArrayType):
                    sizes.append(ret_type.size)
                else:
                    msg = f"Expected ArrayType in returns annotation, got {type(ret_type)}"
                    raise TypeError(msg)
            return tuple(sizes)
        msg = f"Expected ArrayType or tuple of ArrayTypes, got {type(returns)}"
        raise TypeError(msg)

    def get_return_statement(self):
        """Find the Return() statement in this block's operations.

        Returns:
            The Return operation if found, None otherwise.
        """
        # Check for Return statement in ops
        for op in self.ops:
            if type(op).__name__ == "Return":
                return op
        return None

    def get_return_vars(self):
        """Get the variables being returned by this block.

        Looks for a Return() statement in the block's operations.

        Returns:
            Tuple of variables being returned, or None if no Return statement found.
        """
        return_stmt = self.get_return_statement()
        if return_stmt:
            return return_stmt.return_vars
        return None

    def validate_return_annotation(self):
        """Validate that the Return() statement matches the block_returns annotation.

        Raises:
            TypeError: If the Return() statement doesn't match the annotation.
        """
        return_vars = self.get_return_vars()
        if return_vars is None:
            # No Return statement - that's okay, we fall back to old inference
            return

        # Check if we have a block_returns annotation
        if not hasattr(self, "__slr_return_type__"):
            # No annotation - that's okay too
            return

        # Both exist - validate they match in count
        if len(return_vars) != len(self.__slr_return_type__):
            msg = (
                f"Return statement has {len(return_vars)} variables but "
                f"block_returns annotation specifies {len(self.__slr_return_type__)} return values"
            )
            raise TypeError(msg)

    def check_return_annotation_recommended(self) -> tuple[bool, str]:
        """Check if this block should have a Return() statement and block_returns annotation.

        This is a diagnostic helper to identify blocks that would benefit from explicit
        return annotations for better type checking.

        Returns:
            tuple[bool, str]: (should_have_annotation, reason)
                - should_have_annotation: True if annotation is recommended
                - reason: Human-readable explanation

        Example:
            >>> block = MyBlock()
            >>> should_annotate, reason = block.check_return_annotation_recommended()
            >>> if should_annotate:
            ...     print(f"Consider adding Return() statement: {reason}")
            ...
        """
        has_annotation = hasattr(self, "__slr_return_type__")
        has_return = self.get_return_statement() is not None

        # Already fully annotated - great!
        if has_annotation and has_return:
            return (
                False,
                "Block already has both block_returns and Return() statement",
            )

        # Check if block has vars that suggest it returns something
        # Note: self.vars is a Vars object, need to check if it has any variables
        if hasattr(self, "vars") and list(self.vars):
            var_count = len(list(self.vars))
            if not has_annotation and not has_return:
                return (
                    True,
                    f"Block has {var_count} variable(s) in self.vars but no Return() "
                    "statement or block_returns annotation",
                )
            if has_annotation and not has_return:
                return (
                    True,
                    "Block has block_returns annotation but no Return() statement - "
                    "add Return() for explicit variable mapping",
                )
            if has_return and not has_annotation:
                return (
                    True,
                    "Block has Return() statement but no block_returns annotation - "
                    "add block_returns for type declaration",
                )

        # No obvious signs this block returns anything
        return (False, "Block appears to be procedural (no return values)")
