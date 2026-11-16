"""Small GF(2) utilities from the direct-local reference."""


def bits(xs):
    value = 0
    for x in xs:
        value ^= 1 << x
    return value


def indices(value):
    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def rank(rows):
    pivots = {}
    for row in rows:
        while row:
            bit = row.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = row
                break
            row ^= pivots[bit]
    return len(pivots)
