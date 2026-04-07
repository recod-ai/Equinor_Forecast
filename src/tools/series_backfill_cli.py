# src/tools/series_backfill_cli.py
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Set

# Ensure project root imports work if running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.series_backfill import run_backfill  # noqa: E402


def _parse_campaigns(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    return {c.strip() for c in raw.split(",") if c.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing Series Store Parquet artifacts from successful job JSONs."
    )
    parser.add_argument(
        "--results-root", type=Path, required=True,
        help="Root directory containing JSON results (e.g., src/experiment_configs/<GROUP>/results/)"
    )
    parser.add_argument(
        "--series-store-root", type=Path, required=True,
        help="Root directory for Series Store Parquet (Hive-partitioned)."
    )
    parser.add_argument(
        "--campaigns", type=str, default=None,
        help="Optional comma-separated list of campaigns to include."
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Number of threads for parallel backfill (default: 8)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Audit only; do not write Parquet files."
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)."
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s - %(message)s")
    logger = logging.getLogger("series_backfill_cli")

    campaigns = _parse_campaigns(args.campaigns)

    results = run_backfill(
        results_root=args.results_root,
        series_store_root=args.series_store_root,
        campaigns=campaigns,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        logger=logger,
    )

    # Aggregate summary
    counts = Counter(r.action for r in results)
    total = len(results)

    print("\n--- Series Backfill Summary ---")
    print(f"Artifacts discovered (missing): {total}")
    print(f"  Created:       {counts.get('CREATED', 0)}")
    print(f"  Would create:  {counts.get('DRY_RUN_CREATE', 0)}")
    print(f"  Failed:        {counts.get('FAILED', 0)}")
    print("--------------------------------")

    # Exit with 1 if any failures
    sys.exit(1 if counts.get("FAILED", 0) > 0 else 0)


if __name__ == "__main__":
    main()
