import os
import json
import base64
from pecos.simulators import SparseSim
from pecos import HybridEngine, QuantumCircuit


def get_path(file: str, folder: str) -> str:
    return os.path.join(os.path.dirname(__file__), folder, file)

qc_dict = {'prog_type': 'PECOS.QuantumCircuit',
 'prog_metadata': {'cvar_spec': {'m': 1, 'e': 1,
   'c': 1,
   'd': 1,
   'a': 4,
   'b': 32},
  'cvar_spec_type': {},
  'num_qubits': 1},
 'gates': [
     {'sym': 'init |0>', 'qubits': [0], 'metadata': {'cond': None, 'start_init': True}},
  {'sym': 'cop', 'qubits': [], 'metadata': {'expr': {'t': 'b', 'a': 1, 'op': '='}, 'cond': None}},
  {'sym': 'cop', 'qubits': [], 'metadata': {'expr': {'t': 'd', 'a': 0, 'op': '='}, 'cond': None}},
  {'sym': 'cop', 'qubits': [], 'metadata': {'expr': {'t': 'c', 'a': 1, 'op': '='}, 'cond': None}},
  {'sym': 'X', 'qubits': [0], 'metadata': {'cond': None}},
  {'sym': 'measure Z', 'qubits': [0], 'metadata': {'cond': None, 'var_output': {'0': ['m', 0]}, 'mid_circuit': True}},
  {'sym': 'cop', 'qubits': [0], 'metadata': {'cop_type': 'Idle', 'cond': None, 'active_sym': 'MeasureZ'}},
  {'sym': 'cop', 'qubits': [], 'metadata': {'cop_type': 'CFunc', 'wrapper': 'WASM', 'func': 'meas_decoder', 'assign_vars': ['e'],
    'args': ['m', 'a', 'b', 'c', 'd'],
    'cond': None}},
  {'sym': 'cop',
   'qubits': [],
   'metadata': {'cop_type': 'CFunc',
    'wrapper': 'WASM',
    'func': 'global_reset',
    'assign_vars': [],
    'args': [],
    'cond': None}},
  {'sym': 'cop',
   'qubits': [],
   'metadata': {'cop_type': 'ExportCVar', 'export': 'm', 'cond': None}},
  {'sym': 'cop',
   'qubits': [],
   'metadata': {'cop_type': 'ExportCVar', 'export': 'e', 'cond': None}},
  {'sym': 'cop',
   'qubits': [],
   'metadata': {'cop_type': 'ExportCVar', 'export': 'c', 'cond': None}},
  {'sym': 'cop',
   'qubits': [],
   'metadata': {'cop_type': 'ExportCVar', 'export': 'd', 'cond': None}},
  {'sym': 'cop',
   'qubits': [],
   'metadata': {'cop_type': 'ExportCVar', 'export': 'a', 'cond': None}},
  {'sym': 'cop',
   'qubits': [],
   'metadata': {'cop_type': 'ExportCVar', 'export': 'b', 'cond': None}}]}


def test_running_wasm():
    qc_json = json.dumps(qc_dict)
    qc = QuantumCircuit.from_json_str(qc_json)

    wasm_path = get_path(folder="wasm", file="Five_One_Three_WASM_v3e.wasm")
    with open(wasm_path, "rb") as f:
        wasm_bytes = f.read()

    qc.metadata["ccop"] = wasm_bytes
    qc.metadata["ccop_type"] = "wasmtime"

    state = SparseSim(num_qubits=1)
    runner = HybridEngine()

    shot_output, _ = runner.run(
        state,
        qc,
        shot_id = 0
    )
