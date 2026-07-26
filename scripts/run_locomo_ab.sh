#!/usr/bin/env bash
# MemWeaver A/B on LoCoMo, both arms through the same harness.
#
#   scripts/run_locomo_ab.sh [--samples N] [--repeats N] [--out DIR] [--ablations]
#
# Defaults to sample 0 only, 3 repeats per arm (MemWeaver decides at
# temperature 0.7, so single runs are not comparable - report mean +/- std).
#
# Both arms run with the same cross-encoder reranker, which is the baseline parity
# discipline from the design doc (section 10). --ablations additionally runs the
# per-mechanism rows, including --no-expand-rerank.
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

    echo "== ablation: no recontext =="
    python test_locomo10.py "${COMMON[@]}" --memweaver --no-recontext \
        --result-file "$OUT/mw_no_recontext.json" 2>&1 \
        | tee "$OUT/mw_no_recontext.log" | tail -40

    echo "== ablation: no profiles =="
    python test_locomo10.py "${COMMON[@]}" --memweaver --no-profiles \
        --result-file "$OUT/mw_no_profiles.json" 2>&1 \
        | tee "$OUT/mw_no_profiles.log" | tail -40

    echo "== ablation: no expansion (C3 attribution, reranker kept) =="
    python test_locomo10.py "${COMMON[@]}" --memweaver --no-expansion \
        --result-file "$OUT/mw_no_expansion.json" 2>&1 \
        | tee "$OUT/mw_no_expansion.log" | tail -40

    echo "== ablation: no rerank (inherited component dropped) =="
    python test_locomo10.py "${COMMON[@]}" --memweaver --no-rerank \
        --result-file "$OUT/mw_no_rerank.json" 2>&1 \
        | tee "$OUT/mw_no_rerank.log" | tail -40

    echo "== reference: pure SimpleMem (no reranker) =="
    python test_locomo10.py "${COMMON[@]}" --no-memweaver --no-expand-rerank \
        --result-file "$OUT/simplemem_pure.json" 2>&1 \
        | tee "$OUT/simplemem_pure.log" | tail -40
fi

echo "== comparison =="
python scripts/compare_locomo_results.py \
    "$OUT"/baseline_run*.json --memweaver "$OUT"/memweaver_run*.json \
    | tee "$OUT/comparison.txt"

if [[ "$ABLATIONS" == "1" ]]; then
    for ablation in no_weaving no_sweep no_recontext no_profiles no_expansion no_rerank; do
        echo "== comparison: memweaver vs $ablation =="
        python scripts/compare_locomo_results.py \
            "$OUT"/memweaver_run*.json --memweaver "$OUT/mw_$ablation.json" \
            | tee "$OUT/comparison_$ablation.txt"
    done
fi
