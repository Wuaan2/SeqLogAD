"""Build the Colab code ZIP for the SEED-CONFIRM-001 runs (seeds 3407 and 8675309).

Same file set and reproducible layout as scripts/package_colab_s42.py, plus the seed-confirmation package and
protocol, under its own root folder so it never mixes with a seed-42 runtime. Files are read from the working
tree with LF line endings (as git stores them), so the ZIP can be built before the work is committed; the
provenance then records the base commit and "dirty": true, and every file's SHA-256 identifies the code.
Usage: python scripts/package_colab_seeds.py <output.zip>
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from package_colab_s42 import FILES, PACKAGES, STAMP, git

ROOT_NAME = "seqlogad_code_seed_confirmation"
SEED_FILES = ["configs/protocols/seed-confirmation-v1.yaml", "docs/protocol/SEED-CONFIRM-001-seed-confirmation.md",
              "configs/protocols/p2-freeze-v1.yaml"]


def main(out):
    repo = Path(__file__).resolve().parents[1]
    modules = [p.relative_to(repo).as_posix() for package in PACKAGES + ["src/seqlogad/seeds"]
               for p in (repo / package).rglob("*.py")]
    names = sorted(set(FILES + SEED_FILES + modules))
    blobs = {n: (repo / n).read_bytes().replace(b"\r\n", b"\n") for n in names}
    files = {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    changed = git(repo, "status", "--porcelain", "--", *names).decode().strip()
    prov = json.dumps({"files": files, "git": {"base_commit": commit, "dirty": bool(changed)},
                       "real_gpu_training_performed": False, "schema": "SEQLOGAD-CODE-PROVENANCE-2",
                       "primary_seed": 42, "confirmation_seeds": [3407, 8675309], "protocol": "SEED-CONFIRM-001"},
                      indent=2, sort_keys=True).encode() + b"\n"
    blobs["CODE_PROVENANCE.json"] = prov
    ledger = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in sorted(blobs.items()))
    blobs["checksums.sha256"] = ledger.encode()
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sorted(blobs):
            info = zipfile.ZipInfo(f"{ROOT_NAME}/{n}", STAMP)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, blobs[n])
    print(json.dumps({"zip": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "base_commit": commit, "dirty": bool(changed), "files": len(files)}))


if __name__ == "__main__":
    main(sys.argv[1])
