"""Generate reference BP+LSD corrections from the upstream ldpc package.

Not run by the test suite. Requires `ldpc` (2.4.1, the version PECOS vendors) and
`stim`. Usage: python gen_bp_lsd_upstream.py bp_lsd_upstream_ldpc.json
"""

import json
import sys
from pathlib import Path

import ldpc
import numpy as np
import stim
from ldpc import BpLsdDecoder

DISTANCE = 5
P = 0.008
SHOTS = 400
SEED = 20261112
SETTINGS = {
    "bp_method": "minimum_sum",
    "ms_scaling_factor": 0.625,
    "max_iter": 30,
    "schedule": "parallel",
    "bits_per_step": 1,
    "lsd_order": 0,
    "lsd_method": "LSD_0",
}


def main(out_path: str) -> None:
    """Write the check matrix, priors, syndromes and upstream corrections as JSON."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=DISTANCE,
        rounds=DISTANCE,
        after_clifford_depolarization=P,
        before_round_data_depolarization=P,
        before_measure_flip_probability=P,
        after_reset_flip_probability=P,
    )
    dem = circuit.detector_error_model(decompose_errors=False).flattened()
    columns, priors = [], []
    for instruction in dem:
        if instruction.type != "error":
            continue
        detectors = sorted(t.val for t in instruction.targets_copy() if t.is_relative_detector_id())
        if detectors:
            columns.append(detectors)
            priors.append(instruction.args_copy()[0])
    num_checks, num_bits = dem.num_detectors, len(columns)
    check_matrix = np.zeros((num_checks, num_bits), dtype=np.uint8)
    for column, detectors in enumerate(columns):
        check_matrix[detectors, column] = 1

    decoder = BpLsdDecoder(check_matrix, error_channel=priors, **SETTINGS)
    samples = circuit.compile_detector_sampler(seed=SEED).sample(SHOTS).astype(np.uint8)
    cases = []
    for syndrome in samples:
        if not syndrome.any():
            continue
        correction = decoder.decode(syndrome)
        if not ((check_matrix @ correction) % 2 == syndrome).all():
            msg = "upstream correction does not satisfy the syndrome"
            raise RuntimeError(msg)
        cases.append(
            {
                "syndrome": np.flatnonzero(syndrome).tolist(),
                "bp_converged": bool(decoder.converge),
                "correction": np.flatnonzero(correction).tolist(),
            },
        )
    reference = {
        "generator": Path(__file__).name,
        "ldpc_version": ldpc.__version__,
        "stim_version": stim.__version__,
        "circuit": (
            f"surface_code:rotated_memory_x d={DISTANCE} rounds={DISTANCE} p={P}, "
            "flattened DEM with decompose_errors=False, detector-free lines dropped, "
            "duplicate columns kept"
        ),
        "settings": SETTINGS,
        "num_checks": num_checks,
        "num_bits": num_bits,
        "columns": columns,
        "priors": priors,
        "cases": cases,
    }
    Path(out_path).write_text(json.dumps(reference) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
