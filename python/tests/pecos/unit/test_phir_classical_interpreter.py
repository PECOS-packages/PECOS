import numpy as np
import pytest
from pecos.classical_interpreters.phir_classical_interpreter import (
    PHIRClassicalInterpreter,
)

# Note: This test assumes the get_bit method has been updated to include bounds checking.
# If you're implementing the int() conversion approach instead, this test should be removed.


@pytest.fixture
def interpreter():
    """Create and initialize a PHIRClassicalInterpreter with essential test data."""
    interpreter = PHIRClassicalInterpreter()

    # Set up test variables
    interpreter.csym2id = {
        "u8_var": 0,
        "u64_var": 1,
    }

    # Test patterns: alternating bits for u8, highest bit set for u64
    interpreter.cenv = [
        np.uint8(0b10101010),  # u8_var with alternating bits
        np.uint64(0x8000000000000000),  # u64_var with only bit 63 set
    ]

    interpreter.cid2dtype = [
        np.uint8,
        np.uint64,
    ]

    return interpreter


def test_get_bit_basic_functionality(interpreter):
    """Test basic bit retrieval functionality."""
    # Test alternating 0s and 1s in the 8-bit variable
    assert interpreter.get_bit("u8_var", 0) == 0
    assert interpreter.get_bit("u8_var", 1) == 1
    assert interpreter.get_bit("u8_var", 7) == 1


def test_get_bit_highest_bit(interpreter):
    """Test accessing the highest bit of a 64-bit value, which is most likely to cause issues."""
    # This is the critical test for the potential overflow issue
    assert interpreter.get_bit("u64_var", 63) == 1

    # Verify lower bits are 0
    assert interpreter.get_bit("u64_var", 0) == 0
    assert interpreter.get_bit("u64_var", 62) == 0


def test_get_bit_out_of_bounds(interpreter):
    """Test that attempting to access bits beyond the data type width raises an error."""
    # Test with specific error message patterns matching the implementation
    with pytest.raises(
        ValueError,
        match=r"Bit index 8 out of range for.*uint8.* \(max 7\)",
    ):
        interpreter.get_bit("u8_var", 8)  # u8 has bits 0-7 only

    with pytest.raises(
        ValueError,
        match=r"Bit index 64 out of range for.*uint64.* \(max 63\)",
    ):
        interpreter.get_bit("u64_var", 64)  # u64 has bits 0-63 only

    # Test with an extremely large index
    with pytest.raises(
        ValueError,
        match=r"Bit index 1000 out of range for.*uint64.* \(max 63\)",
    ):
        interpreter.get_bit("u64_var", 1000)
