"""Extract native-CZ construction tables from the authors' dataset_0 circuits.

Run with the downloaded reference-data/dataset_0 directory as the argument.
Requires Stim only for this offline extraction, never for production factories.
"""

import hashlib
import json
import sys
from itertools import pairwise
from pathlib import Path

import stim


def extract(circuit):
    instructions = list(circuit)
    final_index = max(i for i, op in enumerate(instructions) if op.name == "M")
    data = [t.value for t in instructions[final_index].targets_copy()]
    active = sorted({t.value for op in instructions if op.name in {"H", "CZ", "M"} for t in op.targets_copy()})
    ancillas = sorted(set(active) - set(data))
    coords = {op.targets_copy()[0].value: op.gate_args_copy() for op in instructions if op.name == "QUBIT_COORDS"}
    starts = [0]
    seen_gates = False
    for i, op in enumerate(instructions[:final_index]):
        if op.name in {"H", "CZ"}:
            seen_gates = True
        if op.name == "R" and seen_gates:
            starts.append(i)
            seen_gates = False
    assert len(starts) == 3
    starts += [final_index, len(instructions)]
    chunks = []
    for part, (start, end) in enumerate(pairwise(starts)):
        moments, current, detectors, observable = [], [], [], []
        count = 0
        known_zero = set()
        if part == 0:
            moments.append([["R", [q]] for q in ancillas])
            known_zero.update(ancillas)
        for op in instructions[start:end]:
            targets = op.targets_copy()
            if op.name == "TICK":
                if current:
                    moments.append(current)
                    current = []
            elif op.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
                records = [count + t.value for t in targets]
                if op.name == "DETECTOR":
                    detectors.append({"records": records, "coords": op.gate_args_copy()})
                else:
                    assert op.gate_args_copy() == [0]
                    observable.extend(records)
            elif op.name in {"QUBIT_COORDS", "I"}:
                pass
            elif op.name == "CX":
                assert all(targets[i].is_sweep_bit_target for i in range(0, len(targets), 2))
            else:
                assert op.name in {"H", "CZ", "R", "M"}, op
                qubits = [t.value for t in targets]
                arity = 2 if op.name == "CZ" else 1
                for i in range(0, len(qubits), arity):
                    qs = qubits[i : i + arity]
                    if op.name == "R":
                        if qs[0] not in active or qs[0] in known_zero:
                            continue
                        known_zero.add(qs[0])
                    else:
                        known_zero.difference_update(qs)
                    current.append([op.name, qs])
                    count += op.name == "M"
        if current:
            moments.append(current)
        chunks.append({"moments": moments, "detectors": detectors, "observable": observable})
    return {
        "active": active,
        "data": data,
        "coords": {str(q): coords[q] for q in active},
        **dict(zip(("first", "middle", "last", "readout"), chunks, strict=True)),
    }


def main():
    root = Path(sys.argv[1])
    fixture_dir = Path(__file__).parent
    package = fixture_dir.parents[3] / "src/pecos/qec/color"
    tables = {}
    provenance = {"doi": "10.5281/zenodo.14238944", "paper": "https://arxiv.org/abs/2412.14256v1", "files": {}}
    for patch in ("d3_west", "d3_east", "d3_north", "d5"):
        for basis in "XZ":
            key = f"{patch}_{basis}"
            tables[key] = extract(stim.Circuit((root / patch / basis / "r03/circuit_ideal.stim").read_text()))
            for rounds in (3, 5):
                relative = Path(patch) / basis / f"r{rounds:02}/circuit_ideal.stim"
                data = (root / relative).read_bytes()
                name = f"{key}_r{rounds:02}.stim"
                (fixture_dir / name).write_bytes(data)
                provenance["files"][name] = {
                    "archive_path": f"google_superdense_color_code_d3_d5/dataset_0/{relative}",
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
    (package / "lacroix_templates.json").write_text(json.dumps(tables, separators=(",", ":")) + "\n")
    (fixture_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
