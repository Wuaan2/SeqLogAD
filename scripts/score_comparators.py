"""Score the four label-free P2.6 comparators for every fold: <output-root>/<fold_id>/comparators.parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqlogad.diversity.comparators import score_comparators  # noqa: E402
from seqlogad.diversity.units import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="extracted P2.2 sequence bundle (phase2_sequence)")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    for fold_id in load_config(ROOT)["folds"]:
        path = Path(args.output_root) / fold_id / "comparators.parquet"
        if path.exists():
            raise SystemExit(f"{path} exists; refusing to overwrite")
        frame = score_comparators(args.data_root, fold_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path, compression="zstd")
        counts = frame.group_by("architecture_id").len().sort("architecture_id").iter_rows()
        print(fold_id, frame.height, dict(counts), flush=True)


if __name__ == "__main__":
    main()
