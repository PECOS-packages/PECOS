import guppylang
from guppylang import guppy
from guppylang.std.quantum import qubit, measure_return, reset, x
from guppylang.std.builtins import owned, result
from guppylang import GuppyModule

func_helpers = GuppyModule("func_helpers")
func_helpers.load_all(guppylang.std.quantum)
func_helpers.load_all(guppylang.std.builtins)
guppylang.enable_experimental_features()

@guppy(func_helpers)
def bool_list2int(bool_list: list[int]) -> int:
    """
    Convert a list of booleans to an integer using bit manipulation.

    Args:
        bool_list (list): A list containing only booleans (True, False).

    Returns:
        int: The integer value represented by the boolean list.
    """

    result = 0
    for bit in bool_list:
        result = (result << 1) | bit  # Shift result left by 1 and add the current bit (True=1, False=0)
    return result


@guppy(func_helpers)
def int2bool_list(value: int, size: int) -> list[int]:
    """
    Convert an integer to a list of booleans representing its binary form.

    Args:
        value (int): The integer to convert.
        size (int): The size of the boolean list (number of bits).

    Returns:
        list: A list of booleans representing the binary form of the integer.

    Raises:
        ValueError: If the integer cannot be represented in the given size.
    """

    bool_list = [((value >> (size - 1 - i)) & 1) for i in range(size)]
    return bool_list

@guppy(func_helpers)
def list_insert_int(lst: list[int], index: int, value: int) -> list[int]:
    """Replace a value at a specific index in a list using only looping, pop, and append."""
    temp: list[int] = []

    for _ in range(len(lst)):
        elem = lst.pop()
        temp.append(elem)

    for i in range(len(temp)):
        if i == index:
            lst.append(value)
        else:
            lst.append(temp[len(temp) - 1 - i])  # Access elements in reverse order

    return lst

@guppy(func_helpers)
def list_insert_bool(lst: list[int], index: int, value: bool) -> list[int]:
    """Replace a value at a specific index in a list using only looping, pop, and append."""

    if value:
        temp_value = 1
    else:
        temp_value = 0

    return list_insert_int(lst, index, temp_value)

@guppy(func_helpers)
def measure_qlist(lst: list[qubit], index: int) -> bool:
    value: bool = False
    temp: list[qubit] = []
    for i in range(len(lst)):
        q = lst.pop()
        if i == index:
            value = measure_return(q)
            reset(q)
        temp.append(q)
    for q in temp:
        lst.append(q)
    return value

@guppy(func_helpers)
def measure_to_bit(qs: list[qubit], qindex: int, cs: list[int], cindex: int) -> None:
    value: bool = measure_qlist(qs, qindex)
    list_insert_bool(cs, cindex, value)

@guppy(func_helpers)
def test_qubit(q: qubit) -> None:
    x(q)
    x(q)

# @guppy(func_helpers)
# def clist_result(clist: list[int], sym: str) -> None:
#     cint = bool_list2int(clist)
#     result(sym, cint)