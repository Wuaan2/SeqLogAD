"""Convert and validate every native evidence file inside Phase-2 FINAL output ZIPs (P2.5 EVID-CS-001).

Usage: python scripts/validate_evidence.py <seqlogad_outputs_P2.x_S42_FINAL.zip> [...]
Exit code 1 if any envelope fails.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seqlogad.evidence.convert import convert  # noqa: E402


def native_evidence(final_zip):
    with zipfile.ZipFile(final_zip) as outer:
        for name in sorted(n for n in outer.namelist() if n.endswith("_S42.zip")):
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                for member in sorted(n for n in inner.namelist() if n.endswith("/sanity-evidence.json")):
                    yield f"{Path(name).name}:{member}", json.loads(inner.read(member))


def main(paths):
    failures = 0
    for path in paths:
        for where, native in native_evidence(path):
            try:
                envelope = convert(native)
                print(json.dumps({"source": where, "result": "PASS", "expert_id": envelope["expert_id"],
                                  "fold_id": envelope["fold_id"], "target_record_id": envelope["target_record_id"],
                                  "status": envelope["status"], "score_raw": envelope["score_raw"]}))
            except ValueError as exc:
                failures += 1
                print(json.dumps({"source": where, "result": "FAIL", "error": str(exc)}))
    print(json.dumps({"EVID_CS_001": "PASS" if not failures else "FAIL", "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
