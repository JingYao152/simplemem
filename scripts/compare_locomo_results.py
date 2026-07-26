#!/usr/bin/env python3
"""Diff two test_locomo10.py result files (MemWeaver A/B).

    python scripts/compare_locomo_results.py baseline.json memweaver.json

Prints per-category answer quality and evidence-based retrieval hit rate side by
side, with category 2 (temporal questions, the P0 target) highlighted. Pass
several result files per arm to average repeated runs - MemWeaver runs at
temperature 0.7, so results are reported as mean +/- std over runs.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

CATEGORY_NAMES = {
    "1": "single-hop",
    "2": "temporal",
    "3": "multi-hop",
    "4": "open-domain",
    "5": "adversarial",
}

QUALITY_METRICS = ("llm_judge_score", "f1", "rougeL_f", "bert_f1")
RETRIEVAL_METRICS = (
    "retrieval_hit_any",
    "retrieval_hit_all",
    "retrieval_coverage",
    "retrieval_session_hit_all",
)


def load_runs(paths: List[Path]) -> List[dict]:
    runs = []
    for path in paths:
        with path.open() as handle:
            runs.append(json.load(handle))
    return runs


def arm_label(runs: List[dict], fallback: str) -> str:
    arms = {run.get("summary", {}).get("arm") for run in runs}
    arms.discard(None)
    label = " / ".join(sorted(arms)) if arms else fallback
    return f"{label} (n={len(runs)} run{'s' if len(runs) > 1 else ''})"


def collect(runs: List[dict], scope: str, metric: str) -> List[float]:
    """Per-run means of one metric, for the overall scope or one category."""
    values = []
    for run in runs:
        block = run.get("aggregated_metrics", {}).get(scope, {})
        stats = block.get(metric)
        if stats and stats.get("count"):
            values.append(stats["mean"])
    return values


def render(values: List[float]) -> str:
    if not values:
        return "     -    "
    if len(values) == 1:
        return f"{values[0]:10.4f}"
    return f"{statistics.mean(values):.4f}±{statistics.stdev(values):.3f}"


def delta(left: List[float], right: List[float]) -> str:
    if not left or not right:
        return "    -   "
    difference = statistics.mean(right) - statistics.mean(left)
    return f"{difference:+8.4f}"


def print_block(
    title: str,
    metrics: tuple,
    scopes: List[str],
    arm_a: List[dict],
    arm_b: List[dict],
) -> None:
    print(f"\n{title}")
    print(f"  {'scope':<22} {'metric':<24} {'A':>10} {'B':>10} {'B-A':>9}")
    print("  " + "-" * 78)
    for scope in scopes:
        label = scope
        if scope.startswith("category_"):
            number = scope.split("_")[1]
            label = f"cat{number} {CATEGORY_NAMES.get(number, '')}".strip()
        for metric in metrics:
            values_a = collect(arm_a, scope, metric)
            values_b = collect(arm_b, scope, metric)
            if not values_a and not values_b:
                continue
            marker = " <-- P0 target" if scope == "category_2" and metric in (
                "llm_judge_score",
                "retrieval_hit_any",
            ) else ""
            print(
                f"  {label:<22} {metric:<24} {render(values_a):>10} "
                f"{render(values_b):>10} {delta(values_a, values_b):>9}{marker}"
            )


def scopes_of(runs: List[dict]) -> List[str]:
    found = set()
    for run in runs:
        found.update(run.get("aggregated_metrics", {}).keys())
    categories = sorted(scope for scope in found if scope.startswith("category_"))
    return (["overall"] if "overall" in found else []) + categories


def print_write_stats(arm_b: List[dict]) -> None:
    totals: Dict[str, List[float]] = {}
    for run in arm_b:
        for name, value in (run.get("summary", {}).get("write_stats") or {}).items():
            totals.setdefault(name, []).append(value)
    if not totals:
        return

    print("\nMemWeaver write-side health indicators (arm B)")
    for name, values in totals.items():
        print(f"  {name:26s}: {statistics.mean(values):.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, nargs="+", help="Arm A result file(s)")
    parser.add_argument(
        "--memweaver",
        type=Path,
        nargs="+",
        required=False,
        help="Arm B result file(s); defaults to the last positional path",
    )
    args = parser.parse_args()

    if args.memweaver:
        baseline_paths, memweaver_paths = args.baseline, args.memweaver
    elif len(args.baseline) >= 2:
        *baseline_paths, last = args.baseline
        memweaver_paths = [last]
    else:
        parser.error("provide two result files, or use --memweaver")

    arm_a = load_runs(baseline_paths)
    arm_b = load_runs(memweaver_paths)

    print("=" * 80)
    print("LoCoMo A/B comparison")
    print(f"  A: {arm_label(arm_a, 'arm A')}  [{', '.join(map(str, baseline_paths))}]")
    print(f"  B: {arm_label(arm_b, 'arm B')}  [{', '.join(map(str, memweaver_paths))}]")
    print("=" * 80)

    scopes = scopes_of(arm_a + arm_b)
    print_block("Answer quality", QUALITY_METRICS, scopes, arm_a, arm_b)
    print_block(
        "Retrieval hit rate (QA evidence)", RETRIEVAL_METRICS, scopes, arm_a, arm_b
    )

    print("\nCost / latency")
    for key in ("avg_retrieval_time", "avg_answer_time", "avg_total_time"):
        values_a = [run["summary"][key] for run in arm_a if key in run.get("summary", {})]
        values_b = [run["summary"][key] for run in arm_b if key in run.get("summary", {})]
        if values_a or values_b:
            print(
                f"  {key:<22} {render(values_a):>10} {render(values_b):>10} "
                f"{delta(values_a, values_b):>9}"
            )

    print_write_stats(arm_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
