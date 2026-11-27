"""Build the two Colab uploads of the P3.2 prototype pass: the code ZIP and the burn-in view ZIP.

Code: same layout as scripts/package_colab_seeds.py (working tree, LF bytes, provenance with base commit and dirty
flag) plus the memory package and the ADAPT-MEM-001 specs. View: the statistics component of each fold, unchanged,
under data/processed/memory/ so the extracted folder is a valid --view-root (every file is re-hashed on load).
Usage: python scripts/package_colab_memory.py <code.zip> <view.zip> --root <folder holding data/processed/memory>
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from package_colab_s42 import FILES, PACKAGES, STAMP, git

ROOT_NAME = "seqlogad_code_memory"
MEMORY_FILES = ["configs/protocols/adapt-mem-v1.yaml", "configs/protocols/adapt-mem-prototypes-v1.yaml",
                "configs/protocols/adapt-mem-statistics-v1.yaml", "configs/protocols/p2-freeze-v1.yaml",
                "docs/protocol/P3.2-target-memory.md"]
MEMORY_PACKAGES = ["src/seqlogad/memory", "src/seqlogad/evaluation"]  # graph.prepare imports evaluation.test_seal
VIEW = "data/processed/memory/ADAPT-MEM-001"
FOLDS = ["FOLD-TARGET-ARCH-BGL", "FOLD-TARGET-ARCH-HDFS"]


def write_zip(out, blobs, prefix=""):
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(blobs):
            info = zipfile.ZipInfo(prefix + name, STAMP)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, blobs[name])
    return hashlib.sha256(out.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("code_zip")
    parser.add_argument("view_zip")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    modules = [p.relative_to(repo).as_posix() for package in PACKAGES + MEMORY_PACKAGES
               for p in (repo / package).rglob("*.py")]
    names = sorted(set(FILES + MEMORY_FILES + modules))
    blobs = {n: (repo / n).read_bytes().replace(b"\r\n", b"\n") for n in names}
    files = {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    changed = git(repo, "status", "--porcelain", "--", *names).decode().strip()
    blobs["CODE_PROVENANCE.json"] = json.dumps(
        {"files": files, "git": {"base_commit": commit, "dirty": bool(changed)}, "real_gpu_training_performed": False,
         "schema": "SEQLOGAD-CODE-PROVENANCE-2", "seed": 42, "task": "P3.2", "protocol": "ADAPT-MEM-001"},
        indent=2, sort_keys=True).encode() + b"\n"
    blobs["checksums.sha256"] = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n"
                                        for n, b in sorted(blobs.items())).encode()
    code_sha = write_zip(args.code_zip, blobs, f"{ROOT_NAME}/")
    view = {}
    for fold in FOLDS:
        folder = Path(args.root) / VIEW / fold / "statistics"
        for path in sorted(folder.iterdir()):
            view[f"{VIEW}/{fold}/statistics/{path.name}"] = path.read_bytes()
    view_sha = write_zip(args.view_zip, view)
    print(json.dumps({"code_zip": args.code_zip, "code_sha256": code_sha, "view_zip": args.view_zip,
                      "view_sha256": view_sha, "base_commit": commit, "dirty": bool(changed), "files": len(files)}))


if __name__ == "__main__":
    main()
