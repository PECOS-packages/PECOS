# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Inspect a paper memory's emitted batch schedule with illustrative runtime times.

The serial simple-runtime durations below are demonstration inputs, not measured
hardware values or a reproduction of the papers' timing/noise models. References
are retained from the selected paper construction in the source and audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pecos.qec.color import (
    ColorExtraction,
    color_code_implementation,
    color_code_spec,
    color_memory,
    lacroix_memory,
    periodic_488_code,
    periodic_488_implementation,
    periodic_488_memory,
)
from pecos.qec.generic import GuppyOrdering, SchedulePolicy
from selene_simple_runtime_plugin import SimpleRuntimePlugin


def main() -> None:
    """Save an inspectable schedule receipt, raw evidence, and optional timeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("488", "swap666", "lacroix", "periodic"), default="488")
    parser.add_argument("--output", type=Path, default=Path("output/guppy-schedule"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--include-lifetimes",
        action="store_true",
        help="Tag preparations and bind reset-to-readout site windows; may change runtime scheduling",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare reference and candidate using tagged lifetimes and one runtime profile",
    )
    parser.add_argument(
        "--candidate",
        choices=("asap", "compact"),
        default="asap",
        help="Candidate scheduling policy for --compare (default: asap)",
    )
    args = parser.parse_args()
    if args.family == "488":
        plan = color_memory(color_code_implementation(color_code_spec("488", 3), ColorExtraction.BRICKWORK_CZ), 2, "Z")
    elif args.family == "swap666":
        impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
        plan = color_memory(impl, 3, "Z")
    elif args.family == "lacroix":
        plan = lacroix_memory("d3_west", 2, "Z")
    else:
        plan = periodic_488_memory(periodic_488_implementation(periodic_488_code(4), gate_set="native-cz"), 2, "Z")
    runtime = SimpleRuntimePlugin(duration_ns_reset=5, duration_ns_rxy=10, duration_ns_rzz=25, duration_ns_measure=40)
    if args.compare:
        policy = SchedulePolicy.ASAP_UNITARIES if args.candidate == "asap" else SchedulePolicy.COMPACT_UNITARIES
        comparison = plan.compare_guppy_schedules(runtime=runtime, seed=17, candidate_policy=policy)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "comparison.json").write_text(json.dumps(comparison.to_dict(), indent=2))
        (args.output / "comparison.txt").write_text(comparison.describe(include_windows=True) + "\n")
        (args.output / "diagnostics.txt").write_text(comparison.explain() + "\n")
        for name, variant, audit in (
            ("reference", plan, comparison.reference),
            (args.candidate, plan.rescheduled(policy), comparison.candidate),
        ):
            save_audit(
                variant,
                audit,
                runtime,
                args.output / name,
                f"{args.family} {name}",
                tagged=True,
                plot=args.plot,
            )
        print(comparison.describe())
        print(f"Artifacts: {args.output.resolve()}")
        return
    audit = plan.audit_guppy_schedule(
        runtime=runtime,
        ordering=GuppyOrdering.ALL_MOMENTS,
        seed=17,
        include_lifetimes=args.include_lifetimes,
    )
    save_audit(plan, audit, runtime, args.output, args.family, tagged=args.include_lifetimes, plot=args.plot)
    print(audit.describe())
    print(f"Artifacts: {args.output.resolve()}")


def save_audit(plan, audit, runtime, output: Path, family: str, *, tagged: bool, plot: bool) -> None:
    """Retain a capture and the exact tagged or untagged source used to generate it."""
    evidence = audit.evidence()
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(audit.to_dict(), indent=2))
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    (output / "source.py").write_text(
        plan.guppy_source(
            ordering=GuppyOrdering.ALL_MOMENTS,
            allocation_provenance=tagged,
        ),
    )
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "profile_scope": (
                    "Illustrative serial-runtime durations; not hardware calibrated or paper-matched noise"
                ),
                "plugin_library": str(runtime.library_file),
                "plugin_sha256": hashlib.sha256(runtime.library_file.read_bytes()).hexdigest(),
                "init_args": runtime.get_init_args(),
                "allocation_provenance": tagged,
            },
            indent=2,
        ),
    )
    if plot:
        plot_batches(evidence["runtime_batches"], output / "timeline.png", family, tagged=tagged)


def plot_batches(batches: list[dict], output: Path, family: str, *, tagged: bool = False) -> None:
    """Plot original batch intervals with normalized operation labels by physical site."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {
        "Reset": "#4778b7",
        "Rxy": "#e49736",
        "Rz": "#9567ac",
        "Rzz": "#4b996d",
        "Rpp": "#4b996d",
        "Measure": "#d16a73",
        "MeasureLeaked": "#d16a73",
        "Mixed": "#777777",
    }
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    sites, kinds = set(), set()
    for batch in batches:
        by_site = {}
        for operation in batch["normalized_operations"]:
            ((kind, data),) = operation.items()
            operands = [data["qubit_id"]] if "qubit_id" in data else [data["qubit_id_1"], data["qubit_id_2"]]
            for q in operands:
                by_site.setdefault(q, set()).add(kind)
        for q, names in by_site.items():
            sites.add(q)
            kind = next(iter(names)) if len(names) == 1 else "Mixed"
            kinds.add(kind)
            start, duration = batch["start_time_nanos"], batch["duration_nanos"]
            if duration:
                ax.broken_barh([(start, duration)], (q - 0.35, 0.7), facecolors=colors[kind])
            else:
                ax.plot(start, q, marker="|", color=colors[kind], markersize=6)
    scope = "tagged execution, " if tagged else ""
    ax.set(
        yticks=sorted(sites),
        ylabel="Runtime physical site",
        xlabel="Runtime-reported clock (ns)",
        title=f"{family}: {scope}illustrative serial runtime, not hardware calibrated",
    )
    ax.legend(
        handles=[Patch(color=colors[k], label=k) for k in sorted(kinds)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=min(6, max(1, len(kinds))),
    )
    ax.grid(axis="x", alpha=0.15)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
