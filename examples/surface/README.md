# Surface Sweep Examples

This directory contains the current rotated-surface-code memory sweep tooling:

- `native_dem_threshold_sweep.py`: runs X/Z memory experiments, fits a per-round
  logical error rate, and writes plots plus optional JSON/HTML reports.
- `surface_sweep_report.py`: rebuilds an HTML dashboard from already-generated
  sweep artifacts without rerunning simulations.

## Typical Workflow

Run these commands from the PECOS repo root using your normal project Python
environment.

```bash
python examples/surface/native_dem_threshold_sweep.py \
  --distances 3 5 7 9 \
  --error-rates 0.004 0.006 0.008 0.01 \
  --bases X Z \
  --shots 1000 \
  --sample-backend native_sampler \
  --save-json --save-html \
  --output-dir /tmp/pecos_surface_highshot_sweep
```

This assumes the standard PECOS Python/runtime setup is already available and
that the Selene/native-sampler pieces required by these examples are present.
Use the project's normal setup flow before running these examples if needed.

Run a full native-sampler sweep with the configuration we have been using:

Notes:

- The default duration schedule now uses about four evenly spaced integer round
  counts over `r in [2d, 3d]` for each distance.
- If we need to push a bit past `3d`, use `--duration-max-multiplier ...` or
  provide an explicit `--duration-multipliers ...` list.
- The fixed-`p` duration plots show fitted duration curves as lines and
  observed logical error rates as points with 95% Wilson intervals.

Open the generated report directly from the sweep run:

```bash
python examples/surface/native_dem_threshold_sweep.py \
  --distances 3 5 7 9 \
  --error-rates 0.004 0.006 0.008 0.01 \
  --bases X Z \
  --shots 1000 \
  --sample-backend native_sampler \
  --save-json --save-html --open-html \
  --output-dir /tmp/pecos_surface_highshot_sweep
```

## Rebuild A Report From Existing Artifacts

If the SVGs and JSON already exist, rebuild the dashboard without rerunning the
simulations:

```bash
python examples/surface/surface_sweep_report.py \
  --input-dir /tmp/pecos_surface_highshot_sweep
```

To rebuild and open it in the browser:

```bash
python examples/surface/surface_sweep_report.py \
  --input-dir /tmp/pecos_surface_highshot_sweep \
  --open
```

## Re-render Plots From Saved JSON

The JSON results file is the canonical source of truth -- the plots are
derived. If you want to regenerate plots later (for example to revisit
the data with different formats, or after the SVGs were deleted), pass
`--render-plots`:

```bash
python examples/surface/surface_sweep_report.py \
  --input-dir /tmp/pecos_surface_highshot_sweep \
  --render-plots --formats svg pdf --open
```

This reads `*_results.json`, reconstructs the in-memory data, and rewrites
every plot file before building the dashboard. Use this when you want to
keep only the JSON file long-term (it is small and fully replayable).

## Merge Multiple Sweep Shards

Run the same sweep multiple times (same distances, bases, error rates, rounds)
and merge the resulting JSON files for tighter confidence intervals. Each
shard stays on disk -- the merge is read-only.

```bash
# Run once overnight, seed 12345.
python examples/surface/native_dem_threshold_sweep.py \
  --distances 3 5 7 9 --error-rates 0.004 0.006 0.008 0.01 \
  --bases X Z --shots 5000 --sample-backend native_sampler \
  --seed 12345 --save-json --output-dir /tmp/sweep_mon

# Run again the next night, same config but fresh seed.
python examples/surface/native_dem_threshold_sweep.py \
  --distances 3 5 7 9 --error-rates 0.004 0.006 0.008 0.01 \
  --bases X Z --shots 5000 --sample-backend native_sampler \
  --seed 99999 --save-json --output-dir /tmp/sweep_tue

# Merge both shards: shots accumulate per SweepPoint key, fit summaries
# are re-derived from the combined points, and a fresh dashboard + PDF
# report get written to the chosen output directory.
mkdir -p /tmp/sweep_combined
python examples/surface/surface_sweep_report.py \
  --input-dir /tmp/sweep_combined \
  --json-files /tmp/sweep_mon/surface_threshold_sweep_results.json \
               /tmp/sweep_tue/surface_threshold_sweep_results.json \
  --render-plots --report-pdf --open
```

Passing multiple `--json-files` always triggers merge mode -- the script
re-renders every plot from the merged data (the existing SVGs in
``--input-dir`` are ignored and overwritten). The merged config in the
dashboard and PDF appendix records the contributing shard paths in
``source_shards`` for provenance.

## Generate A Single PDF Report

For archival or sharing, write a single multi-page PDF (cover page with
configuration + timing, then one plot per page):

```bash
# From a live sweep:
python examples/surface/native_dem_threshold_sweep.py \
  --distances 3 5 7 9 --error-rates 0.004 0.006 0.008 0.01 \
  --bases X Z --shots 5000 --sample-backend native_sampler \
  --save-json --save-report-pdf \
  --output-dir /tmp/sweep_out

# From existing artifacts (requires *_results.json in input-dir):
python examples/surface/surface_sweep_report.py \
  --input-dir /tmp/sweep_out --report-pdf
```

The PDF report is fully rebuildable from the JSON file alone, so you can
archive JSON + PDF and regenerate either from the other.

## Small Smoke Run

For a quick sanity check before launching a heavier sweep:

```bash
python examples/surface/native_dem_threshold_sweep.py \
  --distances 3 5 \
  --error-rates 0.006 \
  --bases X Z \
  --shots 200 \
  --sample-backend native_sampler \
  --save-json --save-html \
  --output-dir /tmp/pecos_surface_fitcurve_check
```

This is useful for validating that:

- Selene/native sampler execution is working
- plots and the HTML dashboard are being generated
- the fixed-`p` duration panels look sensible before spending time on a large
  run
