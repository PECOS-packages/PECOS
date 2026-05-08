#!/bin/bash
# Run all data generation shards sequentially, then analyze and build reports.
#
# Usage:
#   bash examples/surface/run_full_sweep.sh
#   bash examples/surface/run_full_sweep.sh --output-dir ~/Repos/pecos-data/reports/surface-code-decoder-comparison
#
# Each shard is skipped if its output already exists (re-run safe).
# Delete a shard's JSON to regenerate it.

set -euo pipefail

OUTPUT_DIR="${1:---output-dir}"
if [ "$OUTPUT_DIR" = "--output-dir" ]; then
    OUTPUT_DIR="${2:-/tmp/pecos_sweep}"
fi

DATA_DIR="$OUTPUT_DIR/data"
mkdir -p "$DATA_DIR"

GEN="uv run python examples/surface/generate_data.py"

# Common error rates
LOW_P="0.0004 0.0008 0.001 0.0015"
MID_P="0.002 0.004 0.006 0.008 0.010 0.012 0.014"
HIGH_P="0.016 0.018 0.020"
ALL_P="$LOW_P $MID_P $HIGH_P"

MULTI_ROUNDS="2.0 2.33 2.67 3.0"

run_shard() {
    local name="$1"
    shift
    local outdir="$DATA_DIR/$name"
    if [ -f "$outdir/data.json" ]; then
        echo "=== SKIP $name (already exists) ==="
        return
    fi
    echo ""
    echo "=== $name ==="
    echo ""
    $GEN "$@" --output-dir "$outdir"
}

# --- PyMatching: all distances, all error rates, multi-round ---
run_shard pm_all_d3579 \
    --distances 3 5 7 9 \
    --error-rates $ALL_P \
    --decoders pymatching \
    --shots 5000 \
    --duration-multipliers $MULTI_ROUNDS \
    --seed 42

# Extra shots at very low p for PyMatching (need more to resolve rare errors)
run_shard pm_lowp_extra \
    --distances 3 5 7 9 \
    --error-rates $LOW_P \
    --decoders pymatching \
    --shots 15000 \
    --duration-multipliers $MULTI_ROUNDS \
    --seed 142

# --- Tesseract: d=3,5,7 multi-round, d=9 single round ---
run_shard ts_d357 \
    --distances 3 5 7 \
    --error-rates $MID_P $HIGH_P \
    --decoders tesseract \
    --shots 5000 \
    --duration-multipliers $MULTI_ROUNDS \
    --seed 200

run_shard ts_d9 \
    --distances 9 \
    --error-rates $MID_P \
    --decoders tesseract \
    --shots 5000 \
    --duration-multipliers 2.0 \
    --seed 300

# --- MWPF: d=3,5,7 single round ---
run_shard mwpf_d357 \
    --distances 3 5 7 \
    --error-rates $MID_P $HIGH_P \
    --decoders mwpf \
    --shots 5000 \
    --duration-multipliers 2.0 \
    --seed 400

# --- BP+OSD: d=3,5,7 single round ---
run_shard bposd_d357 \
    --distances 3 5 7 \
    --error-rates $MID_P $HIGH_P \
    --decoders bp_osd \
    --shots 5000 \
    --duration-multipliers 2.0 \
    --seed 500

# --- Analyze ---
echo ""
echo "=== ANALYZE ==="
echo ""
uv run python examples/surface/analyze_data.py \
    "$DATA_DIR"/*/data.json \
    -o "$OUTPUT_DIR"

# --- Build reports ---
echo ""
echo "=== BUILD REPORTS ==="
echo ""
uv run python examples/surface/build_report.py \
    "$OUTPUT_DIR/analysis.json" \
    --html --pdf --markdown \
    -o "$OUTPUT_DIR"

echo ""
echo "Done. Reports in $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/report.*
