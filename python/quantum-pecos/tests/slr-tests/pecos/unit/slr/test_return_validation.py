"""Test Return() statement validation and diagnostics."""

import pytest
from pecos.slr import Block, QReg
from pecos.slr.misc import Return
from pecos.slr.types import Array, QubitType


class BlockWithBoth(Block):
    """Block with both block_returns and Return() statement."""

    block_returns = (Array[QubitType, 7],)

    def __init__(self, q: QReg) -> None:
        """Initialize block with return statement."""
        super().__init__()
        self.extend(Return(q))


class BlockWithAnnotationOnly(Block):
    """Block with block_returns but no Return() statement."""

    block_returns = (Array[QubitType, 7],)

    def __init__(self, q: QReg) -> None:  # noqa: ARG002
        """Initialize block without return statement."""
        super().__init__()


class BlockWithReturnOnly(Block):
    """Block with Return() but no block_returns annotation."""

    def __init__(self, q: QReg) -> None:
        """Initialize block with return but no annotation."""
        super().__init__()
        self.extend(Return(q))


class BlockWithNeither(Block):
    """Block with neither annotation nor Return() statement."""

    def __init__(self, q: QReg) -> None:  # noqa: ARG002
        """Initialize procedural block."""
        super().__init__()


class TestReturnValidation:
    """Test validation of Return() statements and block_returns annotations."""

    def test_validate_matching_return_annotation(self) -> None:
        """Test that matching Return() and block_returns validates successfully."""
        q = QReg("q", 7)
        block = BlockWithBoth(q)

        # Should not raise
        block.validate_return_annotation()

    def test_validate_mismatched_count_raises(self) -> None:
        """Test that mismatched counts raise TypeError."""

        class MismatchedBlock(Block):
            block_returns = (Array[QubitType, 7],)

            def __init__(self, q: QReg, a: QReg) -> None:
                super().__init__()
                self.extend(Return(q, a))  # 2 vars but annotation says 1

        q = QReg("q", 7)
        a = QReg("a", 2)

        block = MismatchedBlock(q, a)
        with pytest.raises(TypeError) as exc_info:
            block.validate_return_annotation()

        assert "Return statement has 2 variables" in str(exc_info.value)
        assert "annotation specifies 1 return values" in str(exc_info.value)

    def test_validate_no_return_statement(self) -> None:
        """Test that missing Return() statement doesn't raise during validation."""
        q = QReg("q", 7)
        block = BlockWithAnnotationOnly(q)

        # Should not raise - validation allows missing Return()
        block.validate_return_annotation()

    def test_validate_no_annotation(self) -> None:
        """Test that missing block_returns doesn't raise during validation."""
        q = QReg("q", 7)
        block = BlockWithReturnOnly(q)

        # Should not raise - validation allows missing annotation
        block.validate_return_annotation()


class TestReturnDiagnostics:
    """Test diagnostic helper for Return() annotations."""

    def test_check_fully_annotated_block(self) -> None:
        """Test that fully annotated block returns False (no action needed)."""
        q = QReg("q", 7)
        block = BlockWithBoth(q)

        should_annotate, reason = block.check_return_annotation_recommended()

        assert not should_annotate
        assert "already has both" in reason.lower()

    def test_check_block_with_vars_needing_annotation(self) -> None:
        """Test that block with vars but no annotation/Return() is detected."""

        class BlockWithVars(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__(vargs=q)  # Add vars to block

        q = QReg("q", 7)
        block = BlockWithVars(q)

        should_annotate, reason = block.check_return_annotation_recommended()

        # Block has vars but no annotation or Return()
        assert should_annotate
        assert "variable(s) in self.vars" in reason

    def test_check_block_annotation_needs_return(self) -> None:
        """Test that block with annotation and vars but no Return() is detected."""

        class BlockAnnotationNeedsReturn(Block):
            block_returns = (Array[QubitType, 7],)

            def __init__(self, q: QReg) -> None:
                super().__init__(vargs=q)  # Add vars to block

        q = QReg("q", 7)
        block = BlockAnnotationNeedsReturn(q)

        should_annotate, reason = block.check_return_annotation_recommended()

        # Block has annotation and vars but no Return()
        assert should_annotate
        assert "Return() statement" in reason or "Return()" in reason

    def test_check_block_return_needs_annotation(self) -> None:
        """Test that block with Return() and vars but no annotation is detected."""

        class BlockReturnNeedsAnnotation(Block):
            def __init__(self, q: QReg) -> None:
                super().__init__(vargs=q)  # Add vars to block
                self.extend(Return(q))

        q = QReg("q", 7)
        block = BlockReturnNeedsAnnotation(q)

        should_annotate, reason = block.check_return_annotation_recommended()

        # Block has Return() and vars but no annotation
        assert should_annotate
        assert "block_returns" in reason

    def test_check_procedural_block(self) -> None:
        """Test that procedural block without vars returns False."""
        q = QReg("q", 7)
        block = BlockWithNeither(q)

        should_annotate, reason = block.check_return_annotation_recommended()

        # Block has no vars, no annotation, no Return() - appears procedural
        assert not should_annotate
        assert "procedural" in reason.lower()


class TestReturnStatementAccess:
    """Test accessing Return() statements from blocks."""

    def test_get_return_statement_exists(self) -> None:
        """Test getting Return() statement when it exists."""
        q = QReg("q", 7)
        block = BlockWithBoth(q)

        return_stmt = block.get_return_statement()

        assert return_stmt is not None
        assert type(return_stmt).__name__ == "Return"

    def test_get_return_statement_missing(self) -> None:
        """Test getting Return() statement when it doesn't exist."""
        q = QReg("q", 7)
        block = BlockWithNeither(q)

        return_stmt = block.get_return_statement()

        assert return_stmt is None

    def test_get_return_vars_exists(self) -> None:
        """Test getting return variables when Return() exists."""
        q = QReg("q", 7)
        block = BlockWithBoth(q)

        return_vars = block.get_return_vars()

        assert return_vars is not None
        assert len(return_vars) == 1
        assert return_vars[0] is q

    def test_get_return_vars_missing(self) -> None:
        """Test getting return variables when Return() doesn't exist."""
        q = QReg("q", 7)
        block = BlockWithNeither(q)

        return_vars = block.get_return_vars()

        assert return_vars is None

    def test_get_return_vars_multiple(self) -> None:
        """Test getting multiple return variables."""

        class MultiReturnBlock(Block):
            block_returns = (Array[QubitType, 2], Array[QubitType, 7])

            def __init__(self, ancilla: QReg, data: QReg) -> None:
                super().__init__()
                self.extend(Return(ancilla, data))

        ancilla = QReg("a", 2)
        data = QReg("d", 7)
        block = MultiReturnBlock(ancilla, data)

        return_vars = block.get_return_vars()

        assert return_vars is not None
        assert len(return_vars) == 2
        assert return_vars[0] is ancilla
        assert return_vars[1] is data
