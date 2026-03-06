import os
import json
from pecos.simulators import SparseSim
from pecos import HybridEngine, QuantumCircuit


def get_path(file: str, folder: str) -> str:
    return os.path.join(os.path.dirname(__file__), folder, file)


def _make_wat_circuit(cvar_spec, gates):
    """Build a QuantumCircuit dict with the given cvars and gates."""
    return {
        'prog_type': 'PECOS.QuantumCircuit',
        'prog_metadata': {
            'cvar_spec': cvar_spec,
            'cvar_spec_type': {},
            'num_qubits': 1,
        },
        'gates': [
            {'sym': 'init |0>', 'qubits': [0], 'metadata': {'cond': None, 'start_init': True}},
            *gates,
        ],
    }


def _assign(var, value):
    """Create a classical assignment gate: var = value."""
    return {'sym': 'cop', 'qubits': [], 'metadata': {
        'expr': {'t': var, 'a': value, 'op': '='}, 'cond': None}}


def _wasm_call(func, assign_vars, args):
    """Create a WASM function call gate."""
    return {'sym': 'cop', 'qubits': [], 'metadata': {
        'cop_type': 'CFunc', 'wrapper': 'WASM',
        'func': func, 'assign_vars': assign_vars, 'args': args, 'cond': None}}


def _export(var):
    """Create an export gate."""
    return {'sym': 'cop', 'qubits': [], 'metadata': {
        'cop_type': 'ExportCVar', 'export': var, 'cond': None}}


def _run_wat_circuit(cvar_spec, gates, wat_file="test_values.wat"):
    """Build and run a circuit backed by a WAT file, return shot output."""
    wat_path = get_path(folder="wat", file=wat_file)
    with open(wat_path, "rb") as f:
        wat_bytes = f.read()

    qc_dict = _make_wat_circuit(cvar_spec, gates)
    qc = QuantumCircuit.from_json_str(json.dumps(qc_dict))
    qc.metadata["ccop"] = wat_bytes
    qc.metadata["ccop_type"] = "wasmtime"

    state = SparseSim(num_qubits=1)
    runner = HybridEngine()
    shot_output, _ = runner.run(state, qc, shot_id=0)
    return shot_output


def test_wat_1bit_identity():
    """A 1-bit register with value 1 should pass 1 to WASM and come back as 1.

    This is the core scenario that motivated the BitUInt refactor: previously,
    a 1-bit register set to 1 would return -1 via int() (signed two's complement),
    which WASM would interpret as 0xFFFFFFFF.
    """
    output = _run_wat_circuit(
        cvar_spec={'x': 1, 'y': 1},
        gates=[
            _assign('x', 1),
            _wasm_call('identity', ['y'], ['x']),
            _export('x'),
            _export('y'),
        ],
    )
    assert int(output['x']) == 1
    assert int(output['y']) == 1


def test_wat_add():
    """Two 8-bit values pass through a WASM add function correctly."""
    output = _run_wat_circuit(
        cvar_spec={'a': 8, 'b': 8, 'result': 8},
        gates=[
            _assign('a', 100),
            _assign('b', 50),
            _wasm_call('add', ['result'], ['a', 'b']),
            _export('a'),
            _export('b'),
            _export('result'),
        ],
    )
    assert int(output['a']) == 100
    assert int(output['b']) == 50
    assert int(output['result']) == 150


def test_wat_identity_preserves_value():
    """An 8-bit value round-trips through WASM identity unchanged."""
    output = _run_wat_circuit(
        cvar_spec={'v': 8, 'out': 8},
        gates=[
            _assign('v', 200),
            _wasm_call('identity', ['out'], ['v']),
            _export('v'),
            _export('out'),
        ],
    )
    assert int(output['v']) == 200
    assert int(output['out']) == 200


def test_wat_measurement_through_wasm():
    """A measurement result passes through WASM identity correctly.

    Apply X to get |1>, measure into a 1-bit register, pass through
    WASM identity, and verify the result is 1.
    """
    output = _run_wat_circuit(
        cvar_spec={'m': 1, 'out': 1},
        gates=[
            {'sym': 'X', 'qubits': [0], 'metadata': {'cond': None}},
            {'sym': 'measure Z', 'qubits': [0], 'metadata': {
                'cond': None, 'var_output': {'0': ['m', 0]}, 'mid_circuit': True}},
            {'sym': 'cop', 'qubits': [0], 'metadata': {
                'cop_type': 'Idle', 'cond': None, 'active_sym': 'MeasureZ'}},
            _wasm_call('identity', ['out'], ['m']),
            _export('m'),
            _export('out'),
        ],
    )
    assert int(output['m']) == 1
    assert int(output['out']) == 1


def test_wat_add_with_measurement():
    """A measurement result can be added with another register via WASM."""
    output = _run_wat_circuit(
        cvar_spec={'m': 1, 'offset': 8, 'result': 8},
        gates=[
            _assign('offset', 41),
            {'sym': 'X', 'qubits': [0], 'metadata': {'cond': None}},
            {'sym': 'measure Z', 'qubits': [0], 'metadata': {
                'cond': None, 'var_output': {'0': ['m', 0]}, 'mid_circuit': True}},
            {'sym': 'cop', 'qubits': [0], 'metadata': {
                'cop_type': 'Idle', 'cond': None, 'active_sym': 'MeasureZ'}},
            _wasm_call('add', ['result'], ['m', 'offset']),
            _export('m'),
            _export('result'),
        ],
    )
    assert int(output['m']) == 1
    assert int(output['result']) == 42  # 1 + 41
