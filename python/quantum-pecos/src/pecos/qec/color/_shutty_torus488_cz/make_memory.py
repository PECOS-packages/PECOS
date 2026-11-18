"""Build complete ideal memories from exact single-cycle instrument maps.

Noise is added once, by challenge.decorate_si1000; no boundary channels here.
"""
from copy import deepcopy
from .instrument import instrument
from .linear_circuit import Inst, bits, indices


def sparse_expr(r, constraints):
    cand = [r]
    for c in constraints:
        cand += [a ^ c for a in cand[:]]
    return min(cand, key=lambda v: (v.bit_count(), v))


def sparse_maps(a):
    a = deepcopy(a)
    for key in ['input_face_record_maps', 'output_face_record_maps', 'logical_choi_signs']:
        for b in 'XZ':
            a[key][b] = [sparse_expr(r, a['record_constraints']) for r in a[key][b]]
    return a


def memory(geometry, schedules, basis, p=0, checked=None):
    assert schedules and basis in ('X', 'Z')
    assert p == 0, 'Decorate the final physical moments using challenge.decorate_si1000'
    assert all(set(s.get('data_h_frame', [])) == set(schedules[0].get('data_h_frame', []))
               for s in schedules), 'This exporter requires a common data frame between cycles'
    n = len(geometry['data_qubits'])
    faces = [f['data_support'] for f in geometry['plaquettes']]
    assert len(geometry['ancillas']) == n
    assert [row['index'] for row in geometry['data_qubits']] == list(range(n))
    assert [row['index'] for row in geometry['ancillas']] == list(range(n, 2 * n))
    assert checked is None or len(checked) == len(schedules)
    last_round = len(schedules) - 1
    audits = []
    for j, s in enumerate(schedules):
        a = checked[j] if checked is not None else instrument(
            s, faces, geometry['canonical_logicals'], n_data=n)
        assert a['status'] == 'PASS' and a['raw_measurement_count'] == n
        audits.append(sparse_maps(a))
    c, nm = [], 0
    detector_rows = []
    intrinsic_constraints = []

    def app(name, ts=(), args=(), tag=''):
        c.append(Inst(name, tuple(ts), tuple(args), tag))

    def tick():
        app('TICK')

    def moment(name, ts, tag=None, end=True):
        ts = list(ts)
        touched = set(ts)
        assert len(touched) == len(ts)
        assert all(0 <= q < 2 * n for q in ts)
        app(name, ts)
        if end:
            tick()

    def rec_expr(expr, offset):
        # Affine constants are absorbed into Stim's noiseless detector/logical
        # reference sample; they do not require an artificial physical gate.
        return bits(offset + j for j in indices(expr & ((1 << n) - 1)))

    def decl(name, row, args):
        if name == 'DETECTOR':
            detector_rows.append(row)
        app(name, [('r', j - nm) for j in indices(row)], args)

    def face_decl(ax, fi, rr, t):
        face = geometry['plaquettes'][fi]
        q = face['data_support'][0]
        co = geometry['data_qubits'][q]['coord']
        color = {'red': 0, 'green': 1, 'blue': 2}[face['color']] + (3 if ax == 'Z' else 0)
        decl('DETECTOR', rr, (*co, t, color))

    for row in geometry['data_qubits'] + geometry['ancillas']:
        app('QUBIT_COORDS', [row['index']], row['coord'])
    moment('R', range(n), tag='data_boundary:initial' if basis == 'Z' else None)
    initial_h = (set(range(n)) if basis == 'X' else set()) ^ set(schedules[0].get('data_h_frame', []))
    if initial_h:
        moment('H', sorted(initial_h))
    prev = prev_offset = None
    cycle_ticks = []
    for rd, (s, a) in enumerate(zip(schedules, audits)):
        cycle_ticks.append(sum(i.name == 'TICK' for i in c))
        offset = nm
        for relation in a['record_constraints']:
            # Intrinsic branch constraints are valid for every input syndrome
            # and every logical input. They contain no logical information.
            axes = {0 if s['measurement_order'][j] in s.get('h_measure', []) else 3
                    for j in indices(relation & ((1 << n) - 1))}
            intrinsic_constraints.append((rec_expr(relation, offset), rd,
                                          next(iter(axes)) if len(axes) == 1 else -1))
        terminal_readout_split = False
        if 'moments' in s:
            for physical_moment in s['moments'][:-1]:
                for gate in physical_moment:
                    app(gate['gate'], gate['targets'])
                tick()
            tail = [gate for gate in s['moments'][-1] if gate['gate'] != 'M']
            for gate in tail:
                app(gate['gate'], gate['targets'])
            if rd == last_round:
                if tail:
                    # Preserve the candidate's data operations concurrent with
                    # ancilla readout. Terminal data rotations/readout occur
                    # after that physical moment, with their own noise cost.
                    moment('M', s['measurement_order'])
                    terminal_readout_split = True
                final_h = (set(range(n)) if basis == 'X' else set()) ^ set(s.get('data_h_frame', []))
                if final_h:
                    moment('H', sorted(final_h))
        else:
            moment('R', s['measurement_order'])
            moment('H', s['h_prepare'])
            for layer in s['layers']:
                moment('CX', [q for gate in layer for q in gate])
            hm = list(s['h_measure']) + (list(range(n)) if rd == last_round and basis == 'X' else [])
            moment('H', hm)
        ms = ([] if terminal_readout_split else list(s['measurement_order'])) + (list(range(n)) if rd == last_round else [])
        moment('M', ms, tag=f'data_boundary:interround:{rd + 1}' if rd < last_round else None,
               end=rd < last_round)
        nm += n + (n if rd == last_round else 0)
        for j, e in enumerate(a['logical_choi_signs'][basis]):
            if e:
                decl('OBSERVABLE_INCLUDE', rec_expr(e, offset), (j,))
        if rd == 0:
            for fi in range(len(faces)):
                face_decl(basis, fi, rec_expr(a['input_face_record_maps'][basis][fi], offset), 0)
            other = 'Z' if basis == 'X' else 'X'
            for colors in [('blue', 'green'), ('blue', 'red')]:
                r = 0
                for fi, face in enumerate(geometry['plaquettes']):
                    if face['color'] in colors:
                        r ^= a['input_face_record_maps'][other][fi]
                decl('DETECTOR', rec_expr(r, offset), (0, 0, 0, -1))
        else:
            for fi in range(len(faces)):
                for ax in 'XZ':
                    r = (rec_expr(a['input_face_record_maps'][ax][fi], offset) ^
                         rec_expr(prev['output_face_record_maps'][ax][fi], prev_offset))
                    face_decl(ax, fi, r, rd)
        if rd == last_round:
            for fi, face in enumerate(faces):
                r = (rec_expr(a['output_face_record_maps'][basis][fi], offset) ^
                     bits(offset + n + q for q in face))
                face_decl(basis, fi, r, rd)
            for j, lo in enumerate(geometry['canonical_logicals'][basis.lower()]):
                decl('OBSERVABLE_INCLUDE', bits(offset + n + q for q in lo), (j,))
        prev, prev_offset = a, offset

    # The original d8 face presentation already spans all these relations.
    # Small periodic identifications can make one face-derived row redundant,
    # so explicitly append any intrinsic relation absent from that span.
    piv = {}

    def insert(row):
        while row:
            bit = row.bit_length() - 1
            if bit not in piv:
                piv[bit] = row
                return True
            row ^= piv[bit]
        return False

    for row in detector_rows:
        insert(row)
    for row, rd, axis in intrinsic_constraints:
        if insert(row):
            decl('DETECTOR', row, (0, 0, rd, axis))

    # Match Stim's adjacent-channel normalization to stabilize fault IDs.
    normalized = []
    for inst in c:
        if (normalized and inst.name == 'DEPOLARIZE1' and normalized[-1].name == inst.name
                and normalized[-1].args == inst.args and normalized[-1].tag == inst.tag):
            previnst = normalized[-1]
            assert not set(previnst.targets).intersection(inst.targets)
            normalized[-1] = Inst(inst.name, previnst.targets + inst.targets, inst.args, inst.tag)
        else:
            normalized.append(inst)
    return normalized, dict(cycle_start_ticks=cycle_ticks, cycle_instruments=audits)
