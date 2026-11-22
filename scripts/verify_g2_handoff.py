"""Verify the P3.1 G2 handoff (G2-HANDOFF-001): the G2 record, its frozen basis and the full P2-FREEZE-001 check.

Usage: python scripts/verify_g2_handoff.py [--root <project root>] [--quick]
Without --root only the repository-side G2 checks run. Exit code 1 on any MISSING or MISMATCH.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqlogad.handoff import verify_g2_handoff  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    results = verify_g2_handoff(ROOT, args.root, deep=not args.quick)
    for row in results:
        if row["status"] != "OK":
            print(json.dumps(row))
    bad = sum(row["status"] != "OK" for row in results)
    print(json.dumps({"G2_HANDOFF_001": "PASS" if not bad else "FAIL", "checks": len(results), "failed": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
