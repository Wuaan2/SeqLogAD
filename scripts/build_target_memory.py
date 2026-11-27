"""Build the ADAPT-MEM-001 statistics component (P3.2, CPU part) from the approved burn-in buffers.

Usage: python scripts/build_target_memory.py --project-root <root> [--fold FOLD-TARGET-ARCH-HDFS ...]
The root holds configs/ and data/ (raw logs and the Phase-1 streams from scripts/build_streams.py). The
folds and buffers are rebuilt label-free and must match the G0 identities pinned in adapt-mem-v1.yaml.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqlogad.memory.build import build_memory_statistics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--fold", action="append", default=None)
    args = parser.parse_args()
    print(json.dumps(build_memory_statistics(Path(args.project_root).resolve(), ROOT, args.fold), indent=2))


if __name__ == "__main__":
    main()
