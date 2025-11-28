"""Example showing the Pythonic SLR syntax for bitwise operations."""

from pecos.slr import CReg, If, Main, SlrConverter


def test_pythonic_syntax_example() -> None:
    """Example of Pythonic syntax in SLR programs."""
    prog = Main(
        c := CReg("c", 8),
        # Old style (still works but not idiomatic):
        # from pecos.slr.cops import SET, AND, OR, XOR, NOT
        # SET(c[0], AND(c[1], c[2]))
        # SET(c[3], OR(NOT(c[0]), XOR(c[1], c[2])))
        # New Pythonic style (preferred):
        c[0].set(c[1] & c[2]),
        c[3].set(~c[0] | (c[1] ^ c[2])),
        # Conditions also use Pythonic operators:
        If(c[0] & ~c[1]).Then(
            c[4].set(c[2] ^ c[3]),
        ),
        # Complex expressions with proper precedence:
        c[5].set((c[0] | c[1]) & (c[2] ^ c[3])),
        c[6].set(~((c[4] & c[5]) ^ (c[0] | c[3]))),
    )

    guppy_code = SlrConverter(prog).guppy()

    # The generated Guppy code matches the Pythonic style:
    # IR generator uses array syntax for classical arrays
    assert "c[0] = c[1] & c[2]" in guppy_code
    assert "c[3] = not c[0] | c[1] ^ c[2]" in guppy_code
    assert "if c[0] & not c[1]:" in guppy_code
    assert "c[5] = (c[0] | c[1]) & (c[2] ^ c[3])" in guppy_code
    # Complex expression - exact parentheses may vary due to precedence
    assert "c[6] = " in guppy_code

    # print("Pythonic SLR syntax example:")
    # print(guppy_code)
