"""Test improved HUGR error messages."""

from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, Main, QReg


def test_place_not_used_error() -> None:
    """Test improved error message for unconsumed quantum registers."""
    # Test the error handler directly since our IR generator adds cleanup
    from pecos.slr.gen_codes.guppy.hugr_error_handler import HugrErrorHandler

    # Code that would fail with PlaceNotUsedError
    bad_code = """
@guppy
def main() -> None:
    q = array(quantum.qubit() for _ in range(3))
    c = quantum.measure(q[0])
    # q[1] and q[2] are not consumed
"""

    # Create a mock error that guppy would produce
    mock_error = Exception(
        "PlaceNotUsedError: Variable(name='q') has not been fully consumed",
    )

    handler = HugrErrorHandler(bad_code)
    error_msg = handler.analyze_error(mock_error)

    # print("\nImproved error message:")
    # print(error_msg)

    # Check that the error message is helpful
    assert "PlaceNotUsedError" in error_msg
    assert "was not consumed" in error_msg
    assert "Add a measurement" in error_msg
    assert "Example fix:" in error_msg
    assert "quantum.measure" in error_msg


def test_move_out_of_subscript_error() -> None:
    """Test improved error for array subscript after consumption."""
    # This would generate code that tries to access array elements after measure_array
    # Let's create a scenario that would cause this error

    # Note: This is a hypothetical test - the actual code generation might avoid this error
    # But we can test the error handler directly

    from pecos.slr.gen_codes.guppy.hugr_error_handler import HugrErrorHandler

    bad_code = """
@guppy
def main() -> None:
    q = array(quantum.qubit() for _ in range(3))
    c = quantum.measure_array(q)
    # This would fail - accessing q[0] after q is consumed
    x = q[0]
"""

    # Create a mock error
    mock_error = Exception("MoveOutOfSubscriptError: Cannot move out of subscript")

    handler = HugrErrorHandler(bad_code)
    error_msg = handler.analyze_error(mock_error)

    # Check the error message has helpful content
    assert "MoveOutOfSubscriptError" in error_msg
    assert "Cannot move out of array subscript" in error_msg
    assert "Example fix:" in error_msg
    assert "measure" in error_msg


def test_name_conflict_error() -> None:
    """Test error message for variable name conflicts."""
    Main(
        result := QReg("result", 2),  # Conflicts with result() function
        Measure(result) > CReg("c", 2),
    )

    # Note: The current generator renames this automatically
    # But we can test the error handler

    from pecos.slr.gen_codes.guppy.hugr_error_handler import HugrErrorHandler

    mock_error = TypeError("NotCallableError: 'result' is not callable")

    handler = HugrErrorHandler("")
    error_msg = handler.analyze_error(mock_error)

    assert "NotCallableError" in error_msg
    assert "not callable" in error_msg
    assert "conflicts with a function name" in error_msg


def test_already_used_error() -> None:
    """Test error message for using consumed resources."""
    from pecos.slr.gen_codes.guppy.hugr_error_handler import HugrErrorHandler

    code_with_double_use = """
@guppy
def main() -> None:
    q = quantum.qubit()
    _ = quantum.measure(q)
    _ = quantum.measure(q)  # Error: q already consumed
"""

    mock_error = Exception("AlreadyUsedError: Variable(name='q') has already been used")

    handler = HugrErrorHandler(code_with_double_use)
    error_msg = handler.analyze_error(mock_error)

    # print("\nError message for already used:")
    # print(error_msg)

    assert "AlreadyUsedError" in error_msg
    assert "already been consumed" in error_msg
    assert "can only be used once" in error_msg
