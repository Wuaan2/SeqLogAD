"""Verify P2-FREEZE-001: every artifact hash, inner manifest, run target ZIP and selected checkpoint.

Usage: python scripts/verify_p2_freeze.py --root <project root> [--quick]
The project root is the folder holding Ma_nguon/ and the P2.x_* folders; it may be moved or copied.
--quick checks file hashes only. Exit code 1 on any MISSING or MISMATCH.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqlogad.freeze import verify  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    results = verify(ROOT, args.root, deep=not args.quick)
    for row in results:
        if row["status"] != "OK":
            print(json.dumps(row))
    bad = sum(row["status"] != "OK" for row in results)
    print(json.dumps({"P2_FREEZE_001": "PASS" if not bad else "FAIL", "checks": len(results), "failed": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
