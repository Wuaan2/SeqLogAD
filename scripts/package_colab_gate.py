"""Build the two Colab uploads of the P3.7 synthetic scoring pass: the code ZIP and the synthetic view ZIP.

Code: same layout as scripts/package_colab_memory.py plus the adaptation package and FUS-GATE-001. View: the synthetic
view of each fold (mutations.parquet, rows.parquet) under synthetic/FUS-GATE-001/, unchanged.
Usage: python scripts/package_colab_gate.py <code.zip> <view.zip> --view-root <folder holding synthetic/FUS-GATE-001>
"""
import argparse
import hashlib
import json
from pathlib import Path

from package_colab_memory import MEMORY_FILES, MEMORY_PACKAGES, write_zip
from package_colab_s42 import FILES, PACKAGES, git

ROOT_NAME = "seqlogad_code_gate"
GATE_FILES = ["configs/protocols/fus-gate-v1.yaml", "docs/protocol/P3.7-exploratory-gate.md"]
GATE_PACKAGES = ["src/seqlogad/adaptation"]
VIEW = "synthetic/FUS-GATE-001"
FOLDS = ["FOLD-TARGET-ARCH-BGL", "FOLD-TARGET-ARCH-HDFS"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("code_zip")
    parser.add_argument("view_zip")
    parser.add_argument("--view-root", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    modules = [p.relative_to(repo).as_posix() for package in PACKAGES + MEMORY_PACKAGES + GATE_PACKAGES
               for p in (repo / package).rglob("*.py")]
    names = sorted(set(FILES + MEMORY_FILES + GATE_FILES + modules))
    blobs = {n: (repo / n).read_bytes().replace(b"\r\n", b"\n") for n in names}
    files = {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    changed = git(repo, "status", "--porcelain", "--", *names).decode().strip()
    blobs["CODE_PROVENANCE.json"] = json.dumps(
        {"files": files, "git": {"base_commit": commit, "dirty": bool(changed)}, "real_gpu_training_performed": False,
         "schema": "SEQLOGAD-CODE-PROVENANCE-2", "seed": 42, "task": "P3.7", "protocol": "FUS-GATE-001"},
        indent=2, sort_keys=True).encode() + b"\n"
    blobs["checksums.sha256"] = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n"
                                        for n, b in sorted(blobs.items())).encode()
    code_sha = write_zip(args.code_zip, blobs, f"{ROOT_NAME}/")
    view = {f"{VIEW}/{fold}/{name}": (Path(args.view_root) / VIEW / fold / name).read_bytes()
            for fold in FOLDS for name in ("mutations.parquet", "rows.parquet")}
    view_sha = write_zip(args.view_zip, view)
    print(json.dumps({"code_zip": args.code_zip, "code_sha256": code_sha, "view_zip": args.view_zip,
                      "view_sha256": view_sha, "base_commit": commit, "dirty": bool(changed), "files": len(files)}))


if __name__ == "__main__":
    main()
