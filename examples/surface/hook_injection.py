# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Sample heralded surface-code hook injection, conditioning on acceptance.

Run from the repository root, for example:
    .venv/bin/python examples/surface/hook_injection.py --shots 256 --noise 0.005
    .venv/bin/python examples/surface/hook_injection.py --teleport --basis Y

The reported logical parity is a raw readout statistic, not a decoded logical
error rate or a magic-state fidelity estimate. Noise affects the whole circuit,
including physical byproduct corrections, optional teleportation, and readout.
This simple uniform depolarizing model is not the paper's SI1000 model.
"""

import argparse
import json
import math

import pecos_rslib as prs
from pecos import sim, stab_vec
from pecos.guppy_gen import make_surface_hook_injection, make_surface_t_teleportation
from pecos.qec.surface import SurfacePatch


def main() -> None:
    """Execute independent attempts and report acceptance and conditional parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--verification-rounds", type=int, default=2)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.0, help="Uniform depolarizing probability, including SPAM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state", choices=("T", "TDG"), default="T")
    parser.add_argument("--basis", choices=("X", "Y", "Z"), default="X")
    parser.add_argument("--teleport", action="store_true", help="Consume the resource to apply T/TDG to logical +")
    args = parser.parse_args()
    if args.shots < 1 or not 0 <= args.noise <= 1:
        parser.error("shots must be positive and noise must be between zero and one")
    if args.distance < 3 or args.distance % 2 == 0 or args.verification_rounds < 1:
        parser.error("distance must be odd and >= 3; verification-rounds must be positive")
    patch = SurfacePatch.create(distance=args.distance)
    if args.teleport:
        program = make_surface_t_teleportation(
            patch,
            resource_preparation="hook",
            verification_rounds=args.verification_rounds,
            dagger=args.state == "TDG",
            readout_basis=args.basis,
        )
    else:
        program = make_surface_hook_injection(
            patch,
            args.verification_rounds,
            state=args.state,
            readout_basis=args.basis,
        )
    engine = prs.qis_engine().selene_runtime().interface(prs.qis_helios_interface())
    builder = (
        sim(program)
        .classical(engine)
        .quantum(stab_vec())
        .qubits(patch.geometry.num_qubits + (patch.geometry.num_data if args.teleport else 0))
        .seed(args.seed)
    )
    if args.noise:
        builder = builder.noise(prs.depolarizing_noise().with_uniform_probability(args.noise))
    results = builder.run(args.shots).to_dict()
    support = patch.geometry.logical_z.data_qubits if args.basis == "Z" else patch.geometry.logical_x.data_qubits
    signs = [
        (-1) ** (sum(bits[q] for q in support) % 2)
        for accepted, bits in zip(results["hook_accepted"], results["final_data"], strict=True)
        if accepted  # Rejected shots contain INVALID zero placeholders.
    ]
    mean = sum(signs) / len(signs) if signs else None
    # Estimated standard error for the conditional sample mean of +/-1.
    error = math.sqrt((1 - mean * mean) / (len(signs) - 1)) if len(signs) > 1 else None
    print(
        json.dumps(
            {
                "distance": args.distance,
                "verification_rounds": args.verification_rounds,
                "mode": "teleport" if args.teleport else "resource",
                "state": args.state,
                "readout_basis": args.basis,
                "noise_probability": args.noise,
                "seed": args.seed,
                "attempts": args.shots,
                "accepted": len(signs),
                "acceptance_rate": len(signs) / args.shots,
                "conditional_raw_parity_mean": mean,
                "estimated_standard_error": error,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
