#!/usr/bin/env bash
# MemWeaver A/B on LoCoMo, both arms through the same harness.
#
#   scripts/run_locomo_ab.sh [--samples N] [--repeats N] [--out DIR] [--ablations]
#
# Defaults to sample 0 only, 3 repeats per arm (MemWeaver decides at
# temperature 0.7, so single runs are not comparable - report mean +/- std).
#
# Requires OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL in the environment.
set -euo pipefail

SAMPLES=1
REPEATS=3
OUT="results"
ABLATIONS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --samples)   SAMPLES="$2"; shift 2 ;;
        --repeats)   REPEATS="$2"; shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        --ablations) ABLATIONS=1; shift ;;
        -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."
mkdir -p "$OUT"

echo "== preflight =="
python scripts/preflight_llm.py

if [[ ! -f test_ref/locomo10.json ]]; then
    echo "== dataset =="
    python scripts/fetch_locomo10.py
fi

COMMON=(--dataset test_ref/locomo10.json --num-samples "$SAMPLES"
        --llm-judge --parallel-questions)

for run in $(seq 1 "$REPEATS"); do
    echo "== arm A (baseline) run $run/$REPEATS =="
    python test_locomo10.py "${COMMON[@]}" --no-memweaver \
        --result-file "$OUT/baseline_run$run.json" 2>&1 \
        | tee "$OUT/baseline_run$run.log" | tail -40

    echo "== arm B (memweaver) run $run/$REPEATS =="
    python test_locomo10.py "${COMMON[@]}" --memweaver \
        --result-file "$OUT/memweaver_run$run.json" 2>&1 \
        | tee "$OUT/memweaver_run$run.log" | tail -40
done

if [[ "$ABLATIONS" == "1" ]]; then
    echo "== ablation: no weaving =="
    python test_locomo10.py "${COMMON[@]}" --memweaver --no-weaving \
        --result-file "$OUT/mw_no_weaving.json" 2>&1 \
        | tee "$OUT/mw_no_weaving.log" | tail -40

    echo "== ablation: no sweep =="
    python test_locomo10.py "${COMMON[@]}" --memweaver --no-sweep \
        --result-file "$OUT/mw_no_sweep.json" 2>&1 \
        | tee "$OUT/mw_no_sweep.log" | tail -40
fi

echo "== comparison =="
python scripts/compare_locomo_results.py \
    "$OUT"/baseline_run*.json --memweaver "$OUT"/memweaver_run*.json \
    | tee "$OUT/comparison.txt"

if [[ "$ABLATIONS" == "1" ]]; then
    for ablation in no_weaving no_sweep; do
        echo "== comparison: memweaver vs $ablation =="
        python scripts/compare_locomo_results.py \
            "$OUT"/memweaver_run*.json --memweaver "$OUT/mw_$ablation.json" \
            | tee "$OUT/comparison_$ablation.txt"
    done
fi
