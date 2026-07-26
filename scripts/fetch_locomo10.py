#!/usr/bin/env python3
"""Download the LoCoMo-10 dataset used by test_locomo10.py.

    python scripts/fetch_locomo10.py [--output test_ref/locomo10.json]

Source: https://github.com/snap-research/locomo
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


LOCOMO10_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DEFAULT_OUTPUT = Path("test_ref") / "locomo10.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=LOCOMO10_URL)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the output file already exists",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"{args.output} already exists (use --force to re-download)")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url} -> {args.output}")
    with urllib.request.urlopen(args.url) as response:
        payload = response.read()

    try:
        samples = json.loads(payload)
    except json.JSONDecodeError as error:
        print(f"Downloaded file is not valid JSON: {error}", file=sys.stderr)
        return 1

    args.output.write_bytes(payload)

    questions = sum(len(sample.get("qa", [])) for sample in samples)
    sessions = sum(
        1
        for sample in samples
        for key in sample.get("conversation", {})
        if key.endswith("_date_time")
    )
    print(
        f"Saved {len(samples)} conversations, {sessions} sessions, "
        f"{questions} questions to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
