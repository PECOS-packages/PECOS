"""Exact all-syndrome stabilizer/Choi instrument check for equal-ancilla tori.

For n data qubits, 0..2*n-1 are physical and 2*n..3*n-1 are algebraic
Choi reference legs. The latter never appear in the emitted physical circuit.
"""
from dataclasses import dataclass
from .linear_circuit import bits, indices, rank
from .cliffords import IMAGES


@dataclass
class P:
    x: int
    z: int
    k: int = 0
    r: int = 0
    n_data: int = 64

    def __mul__(self, b):
        assert self.n_data == b.n_data
        return P(self.x ^ b.x, self.z ^ b.z,
                 (self.k + b.k + 2 * (self.z & b.x).bit_count()) % 4,
                 self.r ^ b.r, self.n_data)

    @property
    def vec(self):
        return self.x | (self.z << (3 * self.n_data))

    def expr(self):
        assert self.k % 2 == 0
        return self.r | ((self.k // 2) << self.n_data)


def expression(indices):
    return bits(indices)


def make_p(axis, support, n_data=64):
    return P(bits(support) if axis == 'X' else 0,
             bits(support) if axis == 'Z' else 0, n_data=n_data)


def make_pivots(rows):
    piv = {}
    for p in rows:
        while p.vec:
            b = p.vec.bit_length() - 1
            if b not in piv:
                piv[b] = p
                break
            p = p * piv[b]
        else:
            assert p.k == 0 and p.r == 0, 'Inconsistent state stabilizers'
    return piv


def query(piv, p):
    while p.vec:
        b = p.vec.bit_length() - 1
        if b not in piv:
            raise ValueError('Target operator not in the branch Choi stabilizer group')
        p = p * piv[b]
    return p.expr()


def reduce_rows(v, rows):
    for b, r in rows:
        if (v >> b) & 1:
            v ^= r
    return v




def instrument(schedule, faces, logicals, n_data=None):
    n = len(schedule['measurement_order']) if n_data is None else n_data
    assert n > 0 and n % 2 == 0
    assert sorted(schedule['measurement_order']) == list(range(n, 2 * n))
    assert all(n <= q < 2 * n for key in ('h_prepare', 'h_measure')
               for q in schedule.get(key, []))
    for layer in schedule.get('layers', []):
        touched = [q for pair in layer for q in pair]
        assert all(len(pair) == 2 for pair in layer)
        assert len(touched) == len(set(touched)), 'CNOT layer is not a matching'
        assert all(0 <= q < 2 * n for q in touched)
    assert len(faces) == n // 2
    assert all(0 <= q < n for face in faces for q in face)
    assert len(logicals['x']) == len(logicals['z']) == 4
    assert all(0 <= q < n for ax in ('x', 'z') for lo in logicals[ax] for q in lo)
    record_mask = (1 << n) - 1
    syndrome_rank = n - 4
    ref = 2 * n

    def pauli(x=0, z=0, k=0, r=0):
        return P(x, z, k, r, n)

    def operator(axis, support):
        return make_p(axis, support, n)

    rows = [pauli((1 << q) | (1 << (q + ref))) for q in range(n)]
    rows += [pauli(z=(1 << q) | (1 << (q + ref))) for q in range(n)]
    rows += [pauli(z=1 << q) for q in range(n, 2 * n)]

    def h(qs):
        for p in rows:
            for q in qs:
                if (p.x >> q) & (p.z >> q) & 1:
                    p.k ^= 2
                if ((p.x ^ p.z) >> q) & 1:
                    p.x ^= 1 << q
                    p.z ^= 1 << q

    def cx(layer):
        for p in rows:
            for a, b in layer:
                if (p.x >> a) & 1:
                    p.x ^= 1 << b
                if (p.z >> b) & 1:
                    p.z ^= 1 << a

    if 'moments' in schedule:
        # Physical validity, complete boundary reset/readout, and matching
        # constraints are checked independently by challenge.validate_cycle.
        # These two conjugations are algebraic coordinate changes. The memory
        # exporter physically emits the corresponding initialization/readout
        # rotations once, and the SI1000 decorator charges those gates.
        h(schedule.get('data_h_frame', []))
        for moment in schedule['moments']:
            for gate in moment:
                name, qs = gate['gate'], gate['targets']
                if name == 'H':
                    h(qs)
                elif name == 'CZ':
                    targets = qs[1::2]
                    h(targets)
                    cx(list(zip(qs[::2], targets)))
                    h(targets)
                elif name in IMAGES:
                    table = IMAGES[name]
                    for p in rows:
                        for q in qs:
                            x, z = (p.x >> q) & 1, (p.z >> q) & 1
                            xx, zz, phase = table[x, z]
                            p.x ^= (x ^ xx) << q
                            p.z ^= (z ^ zz) << q
                            p.k = (p.k + phase) % 4
                elif name in {'R', 'M'}:
                    # Ancillas start reset; their common final readout is
                    # performed below. Disjoint boundary data gates commute
                    # with these R/M operations and are processed above.
                    continue
                else:
                    raise ValueError(f'Unsupported physical gate {name}')
        h(schedule.get('data_h_frame', []))
    else:
        h(schedule['h_prepare'])
        for layer in schedule['layers']:
            cx(layer)
        h(schedule['h_measure'])
    constraints = []
    nr = 0
    for j, q in enumerate(schedule['measurement_order']):
        anti = [i for i, p in enumerate(rows) if (p.x >> q) & 1]
        if anti:
            nr += 1
            pi = anti[0]
            p = rows[pi]
            for i in anti[1:]:
                rows[i] = rows[i] * p
            rows[pi] = pauli(z=1 << q, r=1 << j)
        else:
            constraints.append(query(make_pivots(rows), pauli(z=1 << q)) ^ (1 << j))
    cpiv = {}
    for r in constraints:
        while r & record_mask:
            b = (r & record_mask).bit_length() - 1
            if b not in cpiv:
                cpiv[b] = r
                break
            r ^= cpiv[b]
        else:
            assert r == 0
    cpiv = sorted(cpiv.items(), reverse=True)

    def red(r):
        return reduce_rows(r, cpiv)

    piv = make_pivots(rows)
    sin, sout, eqs = {}, {}, []
    for ax in 'XZ':
        sin[ax], sout[ax] = [], []
        for face in faces:
            a = red(query(piv, operator(ax, [q + ref for q in face])))
            b = red(query(piv, operator(ax, face)))
            sin[ax].append(a)
            sout[ax].append(b)
            f = bits(face)
            eqs.append(((f << n) if ax == 'X' else f, a ^ b))
    lsign = {}
    for ax in 'XZ':
        lsign[ax] = []
        for lo in logicals[ax.lower()]:
            e = red(query(piv, operator(ax, list(lo) + [q + ref for q in lo])))
            lsign[ax].append(e)
            f = bits(lo)
            eqs.append(((f << n) if ax == 'X' else f, e))
    ep = {}
    for lhs, rhs in eqs:
        while lhs:
            b = lhs.bit_length() - 1
            if b not in ep:
                ep[b] = lhs, rhs
                break
            ll, rr = ep[b]
            lhs ^= ll
            rhs ^= rr
        else:
            assert red(rhs) == 0, 'No data Pauli frame can explain the logical map'
    sol = [0] * (2 * n)
    for b, (lhs, rhs) in sorted(ep.items()):
        for j in indices(lhs ^ (1 << b)):
            rhs ^= sol[j]
        sol[b] = red(rhs)
    for lhs, rhs in eqs:
        for j in indices(lhs):
            rhs ^= sol[j]
        assert red(rhs) == 0

    # n-4 independent input/output face checks and 8 paired logical checks
    # fix a rank-2n stabilizer state on the data and the Choi reference.
    expected = []
    for ax in 'XZ':
        for face in faces:
            expected.append(operator(ax, face).vec)
            expected.append(operator(ax, [q + ref for q in face]).vec)
        for lo in logicals[ax.lower()]:
            expected.append(operator(ax, list(lo) + [q + ref for q in lo]).vec)
    assert rank(expected) == 2 * n
    srank = rank([red(r) & record_mask for ax in 'XZ' for r in sin[ax]])
    assert srank == syndrome_rank
    assert nr >= syndrome_rank
    return dict(schema='tessellate.symbolic_instrument/1', status='PASS',
                raw_measurement_count=n,
                independent_random_measurements_on_maximally_mixed_input=nr,
                record_constraints=[r for _, r in cpiv],
                input_face_record_maps=sin, output_face_record_maps=sout,
                pauli_frame_x=sol[:n], pauli_frame_z=sol[n:],
                logical_choi_signs=lsign, input_syndrome_rank=srank,
                choi_group_rank=2 * n,
                branch_alpha_squared=f'2**({syndrome_rank - nr})',
                branches_per_valid_syndrome=f'2**({nr - syndrome_rank})',
                branch_completeness=1)
