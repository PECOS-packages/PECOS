"""Full-width Python operands, checked against Python integer arithmetic."""

import operator

import pytest
from hypothesis import given, settings, strategies as st
from pecos_rslib import BitInt as I
from pecos_rslib import BitUInt as U

WIDTHS = (1, 8, 63, 64, 65, 100, 130)
ARITHMETIC = (
    operator.add,
    operator.sub,
    operator.mul,
    operator.and_,
    operator.or_,
    operator.xor,
)
COMPARISONS = (
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
)


def wrap(value, width, cls):
    """Independent two's-complement oracle at the receiver's storage width."""
    bits = width + (cls is I)
    value %= 1 << bits
    if cls is I and value >= 1 << width:
        value -= 1 << bits
    return value


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (lambda: U(128, 0) == 2**100, False),
        (lambda: U(128, 2**100) == 2**100, True),
        (lambda: int(U(128, 0) + 2**100), 2**100),
        (lambda: int(U(128, 0) + I(8, -1)), 2**128 - 1),
        (lambda: int(U(128, 0) | ("1" + "0" * 70)), 2**70),
        (lambda: I(1, 1) == 5, False),
        (lambda: U(4, 15) == -1, False),
        (lambda: U(8, 0) < -1, False),
        (lambda: I(100, 0) == U(100, 2**70), False),
        (lambda: I(100, 5) == "101", True),
        (lambda: U(8, 5) == I(8, 5) == 5, True),
        (lambda: int(I(600, 1) << I(8, -1)), 0),
        (lambda: int(U(8, 1) << I(8, -1)), 0),
        (lambda: int(I(100, I(100, 2**70))), 2**70),
        (lambda: int(U(100, I(100, -1))), 2**100 - 1),
        (lambda: int(I(100, 0) + 2**63), 2**63),
        (lambda: I(64, 2**63) == 2**63, True),
        (lambda: int(I(100, 0) + U(64, 2**63)), 2**63),
        (lambda: int(U(100, -1)), 2**100 - 1),
        (lambda: int(U(100, 0) + (-1)), 2**100 - 1),
        (lambda: int(U(100, U(100, 2**70))), 2**70),
        (lambda: int(I(8, 0) + ("1" + "0" * 8)), -256),
        (lambda: int(I(8, 0) | ("1" * 9)), -1),
        (lambda: int(U(8, 1) << U(100, 2**70 + 3)), 8),
        (lambda: int(U(8, 12) // I(100, 2**70 + 3)), 4),
        (lambda: int(I(8, "1010")), 10),
    ],
)
def test_packet_rows(expression, expected):
    """Pin each non-mutating example in the packet, including unchanged controls."""
    assert expression() == expected


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize("width", [65, 100, 130, 65534])
def test_binary_full_width(cls, width):
    """Parsing a high bit must zero-extend into the extra signed storage bit."""
    binary = "1" + "0" * (width - 1)
    for value in (cls(binary), cls.from_binary(binary)):
        assert value.size == width
        assert int(value) == 1 << (width - 1)


def test_binary_width_limit():
    """The binding rejects oversized signed strings before the core can panic."""
    for constructor in (I, I.from_binary):
        with pytest.raises(ValueError, match="Binary string"):
            constructor("1" * 65535)
    assert int(U.from_binary("1" + "0" * 65534)) == 1 << 65534


@pytest.mark.parametrize(
    ("cls", "width", "value", "size"),
    [
        (I, 100, 2**70 + 5, 4),
        (U, 100, 2**70 + 5, 4),
        (I, 100, 2**64 + 5, 70),
        (I, 64, 2**63, 63),
        (I, 100, -1, 64),
    ],
)
def test_clamp_packet(cls, width, value, size):
    """Clamp retains low bits without a scalar conversion or a 63-bit cap."""
    result = cls(width, value)
    result.clamp(size)
    assert int(result) == value % (1 << size)
    assert result.size == width


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize("size", [0, 1, 63, 64, 65, 99, 100, 101])
def test_clamp_boundaries(cls, size):
    """Zero, word boundaries, and no-op widths all follow the same mask rule."""
    value = -(2**70) + 5 if cls is I else 2**99 + 2**70 + 5
    result = cls(100, value)
    result.clamp(size)
    assert int(result) == (value % (1 << size) if size < 100 else value)
    assert result.size == 100


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize("method", ["set", "set_clip"])
@pytest.mark.parametrize(
    ("operand", "mathematical"),
    [
        (I(8, -1), -1),
        (I(130, 2**100 + 5), 2**100 + 5),
        (U(130, 2**100 + 5), 2**100 + 5),
        ("1010", 10),
    ],
)
def test_set_full_width(cls, method, operand, mathematical):
    """Setter conversion handles signed wrappers before their __index__ path."""
    result = cls(100, 0)
    getattr(result, method)(operand)
    assert int(result) == wrap(mathematical, 100, cls)
    assert result.size == 100


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize("binary", ["", "0b", "102", "0B12", "-1", " 101", "1_0"])
def test_invalid_binary(cls, binary):
    """Invalid strings must retain ValueError on every inbound path."""
    for action in (
        lambda: cls(8, 0) + binary,
        lambda: cls(8, binary),
        lambda: cls(8, 0).set(binary),
        lambda: cls(8, 0).set_clip(binary),
        lambda: cls(8, 0) == binary,
        lambda: cls(8, 0) << binary,
    ):
        with pytest.raises(ValueError, match="Binary string"):
            action()


@pytest.mark.parametrize("cls", [I, U])
def test_unhashable(cls):
    """Mutable in place, so neither class can be a dict key or set member."""
    value = cls(8, 5)
    for action in (lambda: hash(value), lambda: {value}, lambda: {value: "x"}):
        with pytest.raises(TypeError, match="unhashable"):
            action()


@pytest.mark.parametrize("value", [0, 5, 2**63, 2**70])
def test_cross_class_equality(value):
    """Equal values of either class, at any sufficient width, compare equal."""
    values = [value, U(100, value), I(100, value), U(130, value), I(130, value)]
    assert all(a == b for a in values for b in values)


class IndexValue:
    """An integer protocol operand, including values beyond native scalars."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize("value", [5, -1, 2**63, 2**200 + 3, -(2**200) + 3])
def test_index_operand(cls, value):
    """__index__ uses the same full-width conversion as a Python int."""
    operand = IndexValue(value)
    assert int(cls(130, operand)) == wrap(value, 130, cls)
    assert int(cls(130, 0) + operand) == wrap(value, 130, cls)
    assert (cls(130, 0) < operand) == (0 < value)
    assert int(cls(8, 1) << operand) == wrap(1 << (value & 65535), 8, cls)
    result = cls(130, 0)
    result.set(operand)
    assert int(result) == wrap(value, 130, cls)


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize("operand", [object(), 1.5])
def test_unsupported_operand(cls, operand):
    """Unsupported operands raise TypeError, except equality, which is False."""
    for action in (
        lambda: cls(8, 0) + operand,
        lambda: cls(8, 0) < operand,
        lambda: cls(8, operand),
        lambda: cls(8, 0).set(operand),
    ):
        with pytest.raises(TypeError):
            action()
    assert not (cls(8, 0) == operand)
    assert cls(8, 0) != operand


@pytest.mark.parametrize("cls", [I, U])
def test_equality_with_a_float_is_false(cls):
    """A float operand has no integer value, so equality is False rather than an error."""
    assert not (cls(8, 1) == 1.0)
    assert cls(8, 1) not in [1.0]


@st.composite
def operands(draw):
    """Choose representation and width independently of the receiver."""
    value = draw(
        st.one_of(
            st.integers(-(2**260), 2**260),
            st.integers(2**63, 2**64 - 1),
            st.integers(-(2**260), -1),
            st.sampled_from([2**300 + 3, -(2**300) + 3]),
        )
    )
    kind = draw(st.sampled_from(["int", "binary", I, U]))
    if kind == "int":
        return value, value
    if kind == "binary":
        value = abs(value)
        prefix = draw(st.sampled_from(["", "0b", "0B"]))
        return prefix + format(value, "b"), value
    width = draw(st.sampled_from(WIDTHS))
    return kind(width, value), wrap(value, width, kind)


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize("width", WIDTHS)
@settings(max_examples=120, derandomize=True)
@given(value=st.integers(-(2**260), 2**260), operand=operands())
def test_operand_oracle(cls, width, value, operand):
    """All six operators and comparisons agree with independent integer values."""
    rhs, rhs_value = operand
    lhs = cls(width, value)
    lhs_value = wrap(value, width, cls)
    assert int(lhs) == lhs_value
    for op in ARITHMETIC:
        result = op(lhs, rhs)
        assert int(result) == wrap(op(lhs_value, rhs_value), width, cls)
        assert result.size == width
    for op in COMPARISONS:
        assert op(lhs, rhs) == op(lhs_value, rhs_value)


@pytest.mark.parametrize("cls", [I, U])
def test_wide_int_operand(cls):
    """A nonzero high word cannot disappear on inbound arithmetic conversion."""
    assert int(cls(130, 7) + (2**100 + 3)) == 2**100 + 10


def test_wide_wrapper_operand():
    """Cross-class operands preserve high words before receiver-width wrapping."""
    assert int(I(130, 7) + U(100, 2**70 + 3)) == 2**70 + 10
    assert int(U(130, 7) + I(100, 2**70 + 3)) == 2**70 + 10


def test_signed_wrapper_extension():
    """A signed negative wrapper fills the whole unsigned receiver."""
    assert int(U(128, 0) + I(8, -1)) == 2**128 - 1


def test_comparison_no_coercion():
    """Out-of-range values compare as integers, independent of receiver width."""
    assert not (I(1, 1) == 5)
    assert not (U(4, 15) == -1)
    assert U(8, 0) > -1
    assert I(64, 2**63) == 2**63


def test_signed_shift_count():
    """A narrow negative count contributes sixteen low bits, not its raw pattern."""
    assert int(I(600, 1) << I(8, -1)) == 0
    assert int(I(600, -2) >> I(8, -1)) == -1


def test_u64_range_signed_operand():
    """Unsigned scalar inputs above i64::MAX remain positive at a wide receiver."""
    for value in (2**63, 2**64 - 1):
        assert int(I(100, 0) + value) == value
        assert int(I(100, 0) + U(64, value)) == value


def test_int_word_roundtrip():
    """Byte conversion preserves distinct words and masks the partial top word."""
    for value in (2**129 + 2**100 + 2**64 + 5, -(2**200) + 2**100 + 5):
        for cls in (I, U):
            assert int(cls(130, value)) == wrap(value, 130, cls)


@pytest.mark.parametrize("cls", [I, U])
def test_binary_operand_prefix_and_length(cls):
    """Value strings accept prefixes and arbitrary lengths; width strings do not."""
    for prefix in ("", "0b", "0B"):
        binary = prefix + "1" + "0" * 65535 + "101"
        assert int(cls(8, binary)) == 5
        assert int(cls(8, 0) | binary) == 5
        value = cls(8, 0)
        value.set(binary)
        assert int(value) == 5
    for binary in ("0b101", "0B101"):
        for constructor in (cls, cls.from_binary):
            with pytest.raises(ValueError, match="Binary string"):
                constructor(binary)


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize("operand_cls", [I, U])
@pytest.mark.parametrize("width", [8, 130])
def test_copy_and_set_keep_declared_width(cls, operand_cls, width):
    """Same-class conversion preserves operand width until the store wraps it."""
    for source_width, value in ((8, -1), (130, 2**100 + 5)):
        operand = operand_cls(source_width, value)
        expected = wrap(wrap(value, source_width, operand_cls), width, cls)
        result = cls(width, operand)
        assert result.size == width
        assert int(result) == expected
        for method in ("set", "set_clip"):
            getattr(result, method)(operand)
            assert result.size == width
            assert int(result) == expected


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize(
    "operand",
    [
        2**100 + 3,
        I(130, 2**100 + 3),
        U(130, 2**100 + 3),
        "0b" + format(2**100 + 3, "b"),
    ],
)
def test_wide_shift_low_bits(cls, operand):
    """All count representations preserve their low bits, for both shifts."""
    assert int(cls(8, 1) << operand) == 8
    assert int(cls(8, 64) >> operand) == 8


def test_mixed_division_and_remainder():
    """Cross-class divisors wrap; same-class divisors retain their mathematical value."""
    assert int(U(8, 12) // I(100, 2**70 + 3)) == 4
    assert int(U(8, 14) % I(100, 2**70 + 3)) == 2
    assert int(U(8, 12) // U(100, 2**70 + 3)) == 0
    assert int(I(8, 12) // U(100, 2**70 + 3)) == 4
    assert int(I(8, 14) % U(100, 2**70 + 3)) == 2
    for cls, other in ((I, U), (U, I)):
        for op in (operator.floordiv, operator.mod):
            with pytest.raises(ZeroDivisionError):
                op(cls(8, 12), other(130, 2**100))


@pytest.mark.parametrize("cls", [I, U])
def test_reflected_operators(cls):
    """Reflected arithmetic shares conversion, while rsub retains its existing width."""
    for op in ARITHMETIC:
        assert int(op(2**100 + 3, cls(130, 5))) == wrap(op(2**100 + 3, 5), 130, cls)
    result = cls(8, 5).__rsub__(cls(130, 2**100 + 3))
    assert result.size == 130
    assert int(result) == 2**100 - 2


def test_set_packet_rows():
    """Pin the exact receiver widths of the packet's two setter examples."""
    unsigned = U(100, 0)
    unsigned.set(I(8, -1))
    assert int(unsigned) == 2**100 - 1
    signed = I(8, 0)
    signed.set("1010")
    assert int(signed) == 10


class FailingIndex:
    """An operand that claims integer support but fails to convert."""

    def __index__(self):
        raise TypeError("inside index")


class CountingIndex:
    """Records how many times the integer protocol is consulted."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


class StrangeEq(int):
    """An int subclass whose reflected comparisons return non-bool objects."""

    def __eq__(self, other):
        return ["eq"]

    def __ne__(self, other):
        return ["ne"]

    __hash__ = int.__hash__


class HiddenIndex:
    """Instance attribute lookup hides __index__, as the integer protocol ignores."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value

    def __getattribute__(self, name):
        if name == "__index__":
            raise AttributeError(name)
        return object.__getattribute__(self, name)


class NoisyGetattr:
    """No integer protocol, and a __getattr__ that raises for anything missing."""

    def __getattr__(self, name):
        raise RuntimeError(name)


@pytest.mark.parametrize("cls", [I, U])
def test_comparison_propagates_index_errors(cls):
    """Only a missing integer protocol means not comparable; its failures surface."""
    with pytest.raises(TypeError, match="inside index"):
        operator.eq(cls(8, 0), FailingIndex())


@pytest.mark.parametrize("cls", [I, U])
@pytest.mark.parametrize("value", [3, 2**100 + 3, -(2**100) + 3])
def test_shift_count_converts_once(cls, value):
    """A shift count consults __index__ exactly once, small or wide."""
    count = CountingIndex(value)
    assert int(cls(8, 1) << count) == wrap(1 << (value & 65535), 8, cls)
    assert count.calls == 1


@pytest.mark.parametrize("cls", [I, U])
def test_comparison_returns_bool(cls):
    """Comparison results are bools even when an int subclass answers oddly."""
    assert (cls(8, 1) == StrangeEq(1)) is True
    assert (cls(8, 1) != StrangeEq(2)) is True


@pytest.mark.parametrize("cls", [I, U])
def test_comparison_uses_the_type_level_index_protocol(cls):
    """Support is decided by the type, as operator.index decides it."""
    assert cls(8, 5) == HiddenIndex(5)
    assert not (cls(8, 0) == NoisyGetattr())
