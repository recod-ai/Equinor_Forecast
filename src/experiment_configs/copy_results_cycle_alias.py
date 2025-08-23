#!/usr/bin/env python3
"""
Copy result folders from a source "results" into a destination "results",
renaming with the next available `..._cycle_N` to avoid overwrites.

- Reserves cycle numbers per base within the same run (no duplicate targets).
- Works in terminal and Jupyter/IPython (unknown args ignored).
- Skips non-directories automatically.
"""

from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CYCLE_RE = re.compile(r"^(?P<base>.+?)_cycle_(?P<num>\d+)$")


def parse_base_and_cycle(name: str) -> Tuple[str, Optional[int]]:
    """Return (base, cycle_number or None) from a dir name."""
    m = CYCLE_RE.match(name)
    if not m:
        return name, None
    return m.group("base"), int(m.group("num"))


def existing_cycles_for_base(base: str, dest_results: Path) -> List[int]:
    """Find all existing cycle numbers for a given base in destination."""
    numbers: List[int] = []
    pattern = re.compile(rf"^{re.escape(base)}_cycle_(\d+)$")
    for p in dest_results.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                try:
                    numbers.append(int(m.group(1)))
                except ValueError:
                    pass
    return sorted(numbers)


def next_cycle_seed(base: str, dest_results: Path) -> int:
    """Return the first free cycle number in dest for this base."""
    cycles = existing_cycles_for_base(base, dest_results)
    return (cycles[-1] if cycles else 0) + 1


def plan_operations(src_results: Path, dest_results: Path):
    """
    Build a list of (src_dir, dest_dir) copy operations.
    Guarantees unique destination names by reserving cycles per base.
    """
    # Collect and sort sources deterministically by (base, src_cycle or 0, name)
    sources = []
    for child in src_results.iterdir():
        if not child.is_dir():
            continue
        base, cycle = parse_base_and_cycle(child.name)
        sources.append((base, cycle if cycle is not None else 0, child))
    sources.sort(key=lambda x: (x[0], x[1], x[2].name))

    # Reserve cycles per base so we never repeat in this batch
    reservations: Dict[str, int] = {}

    ops = []
    for base, _src_cycle, child in sources:
        if base not in reservations:
            reservations[base] = next_cycle_seed(base, dest_results)
        # Find a free candidate considering possible races
        while True:
            dest_name = f"{base}_cycle_{reservations[base]}"
            dst_path = dest_results / dest_name
            if not dst_path.exists():
                break
            reservations[base] += 1
        ops.append((child, dst_path))
        reservations[base] += 1  # advance for the next source of the same base
    return ops


def copy_tree(src: Path, dst: Path) -> None:
    """Copy a directory tree. Fails if dst exists (it shouldn't)."""
    shutil.copytree(src, dst)  # keep overwrite protection


def run(src_results: Path, dest_results: Path, dry_run: bool) -> None:
    """Orchestrate the copy plan and execute it."""
    if not src_results.exists() or not src_results.is_dir():
        raise SystemExit(f"Source results not found: {src_results}")
    dest_results.mkdir(parents=True, exist_ok=True)

    ops = plan_operations(src_results, dest_results)
    if not ops:
        print("Nothing to copy.")
        return

    print(f"Planned copies ({'DRY-RUN' if dry_run else 'EXECUTE'}):")
    for src, dst in ops:
        print(f"  {src.name}  ->  {dst.name}")

    if dry_run:
        print("\nDry-run only. No changes were made.")
        return

    for src, dst in ops:
        copy_tree(src, dst)

    print("\nDone. All folders copied with unique cycle aliases.")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Copy results to destination using next available _cycle_N."
    )
    p.add_argument(
        "--src",
        type=Path,
        default=Path("OlD_Results_2/results"),
        help="Source results directory.",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("results"),
        help="Destination results directory.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and show operations without copying.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = make_parser()
    if argv is None:
        # In Jupyter/IPython there are unknown args (e.g., -f <path>). Ignore them.
        args, _unknown = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)
    run(args.src, args.dst, args.dry_run)


if __name__ == "__main__":
    main()
