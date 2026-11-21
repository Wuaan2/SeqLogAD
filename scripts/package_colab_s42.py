"""Build the Colab code ZIP for the seed-42 Phase-2 runs (P2.1-P2.4 and the P2.6 scoring pass) from a clean git commit.

Same layout as the historical upload (seqlogad_code_all_experts_s42/ + CODE_PROVENANCE.json
+ checksums.sha256). File bytes come from git blobs (LF), so the ZIP is reproducible.
Usage: python scripts/package_colab_s42.py <output.zip>
"""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT_NAME = "seqlogad_code_all_experts_s42"
FILES = ["pyproject.toml", "configs/models/base-freeze-v1.yaml", "configs/models/semantic-v1.yaml",
         "configs/models/sequence-reference-v1.yaml", "configs/models/sequence-reference-v12.yaml",
         "configs/models/sequence-reference-v13.yaml", "configs/models/sequence-reference-v13r1.yaml",
         "configs/models/sequence-lightweight-v1.yaml", "configs/models/gtat-v1.yaml",
         "configs/protocols/phase2-architecture-v1.yaml", "configs/protocols/phase2-development-s42-v1.yaml",
         "docs/protocol/P2.3-lightweight-sequence-expert.md",
         "docs/protocol/P2.4-gtat-structural-temporal-expert.md",
         "configs/protocols/c2-source-label-scope-v1.yaml", "configs/protocols/p2.6-diversity-v1.yaml",
         "docs/protocol/P2.6-expert-diversity-preregistration.md",
         "docs/protocol/P3.1-source-labeled-development-amendment-2026-09-27.md",
         "src/seqlogad/__init__.py", "src/seqlogad/common/__init__.py", "src/seqlogad/common/checksum.py"]
PACKAGES = ["src/seqlogad/diversity", "src/seqlogad/evidence", "src/seqlogad/graph", "src/seqlogad/phase2_execution", "src/seqlogad/protocol",
            "src/seqlogad/semantic", "src/seqlogad/sequence"]
STAMP = (2026, 1, 1, 0, 0, 0)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def main(out):
    repo = Path(__file__).resolve().parents[1]
    if git(repo, "status", "--porcelain").strip():
        raise SystemExit("working tree is dirty; commit first")
    listed = git(repo, "ls-tree", "-r", "--name-only", "HEAD", *PACKAGES).decode().split()
    names = sorted(FILES + [n for n in listed if n.endswith(".py")])
    blobs = {n: git(repo, "show", f"HEAD:{n}") for n in names}
    files = {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    prov = json.dumps({"files": files, "git": {"commit": commit, "diff_sha256": hashlib.sha256(b"").hexdigest(),
                       "dirty": False}, "real_gpu_training_performed": False,
                       "schema": "SEQLOGAD-CODE-PROVENANCE-2", "seed": 42}, indent=2, sort_keys=True).encode() + b"\n"
    blobs["CODE_PROVENANCE.json"] = prov
    ledger = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in sorted(blobs.items()))
    blobs["checksums.sha256"] = ledger.encode()
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sorted(blobs):
            info = zipfile.ZipInfo(f"{ROOT_NAME}/{n}", STAMP)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, blobs[n])
    print(json.dumps({"zip": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "commit": commit, "files": len(files)}))


if __name__ == "__main__":
    main(sys.argv[1])
