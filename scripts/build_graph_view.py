"""Build the P2.4 graph-view zip (GTAT-ST-001 §3.1) from the P2.1/P2.2 bundles and raw logs.

Example (local layout, where Hadoop/OpenStack sit one folder deeper than ArchitectureSpec.raw_root):
  python -B scripts/build_graph_view.py --data-root ../.. \
    --semantic-zip ../../P2.1_Semantic/uploads/phase2-final.zip \
    --sequence-zip ../../P2.2_Sequence/uploads/phase2-sequence-s42.zip \
    --raw-root ARCH-HADOOP=../../data/raw/hadoop/Hadoop \
    --raw-root ARCH-OPENSTACK=../../data/raw/openstack/OpenStack \
    --output ../../P2.4_GTAT/uploads/phase2-graph-s42.zip --jobs 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqlogad.graph.prepare import build_graph_bundle  # noqa: E402
from seqlogad.protocol.architectures import ARCHITECTURES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--data-root", default=str(ROOT), help="folder that holds data/raw/...")
    parser.add_argument("--semantic-zip", required=True)
    parser.add_argument("--sequence-zip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-root", action="append", default=[], metavar="ARCH=PATH",
                        help="override <data-root>/<ArchitectureSpec.raw_root> for one architecture")
    parser.add_argument("--work-dir", default=None, help="parent of the temporary build folder")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    roots = {arch: Path(args.data_root) / spec.raw_root for arch, spec in ARCHITECTURES.items()}
    for item in args.raw_root:
        arch, _, path = item.partition("=")
        if arch not in ARCHITECTURES or not path:
            parser.error(f"bad --raw-root {item!r}")
        roots[arch] = Path(path)
    report = build_graph_bundle(args.project_root, args.semantic_zip, args.sequence_zip, args.output,
                                roots, work_dir=args.work_dir, jobs=args.jobs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
