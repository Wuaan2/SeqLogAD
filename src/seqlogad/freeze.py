"""Verify the P2.7 freeze manifest (P2-FREEZE-001) against a project root that may have moved."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import yaml

from seqlogad.common.checksum import sha256_file

MANIFEST = "configs/protocols/p2-freeze-v1.yaml"


def load_manifest(repo_root):
    return yaml.safe_load((Path(repo_root) / MANIFEST).read_text(encoding="utf-8"))["p2_freeze"]


def _check(results, name, ok, detail):
    results.append({"item": name, "status": "OK" if ok else "MISMATCH", "detail": detail})


def verify(repo_root, project_root, deep=True):
    """Return one result per check; status is OK, MISSING or MISMATCH."""
    freeze, root, results = load_manifest(repo_root), Path(project_root), []
    for expert in freeze["experts"].values():
        for key in ("config", "contract", "notebook"):
            _check(results, f"repo:{expert[key]}", (Path(repo_root) / expert[key]).is_file(), "file present in repository")
    for relative, digest in freeze["repo_files"].items():
        path = Path(repo_root) / relative
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if path.is_file() else None
        _check(results, f"repo_hash:{relative}", actual == digest, "LF-normalised SHA-256")
    for name, artifact in freeze["artifacts"].items():
        path = root / artifact["path"]
        if not path.is_file():
            results.append({"item": name, "status": "MISSING", "detail": artifact["path"]})
            continue
        _check(results, name, sha256_file(path) == artifact["sha256"], artifact["path"])
        for member, digest in artifact.get("inner_manifest", {}).items():
            with zipfile.ZipFile(path) as archive:
                found = [n for n in archive.namelist() if PurePosixPath(n).name == member
                         and hashlib.sha256(archive.read(n)).hexdigest() == digest]
            _check(results, f"{name}:{member}", len(found) == 1, digest)
    if deep:
        for task, expert in freeze["experts"].items():
            final = root / freeze["artifacts"][expert["final"]]["path"]
            if not final.is_file():
                continue  # already reported as MISSING
            with zipfile.ZipFile(final) as outer:
                for target, run in expert["runs"].items():
                    payload = outer.read(f"{task}_S42/targets/{task}_{target}_S42.zip")
                    _check(results, f"{task}/{target}:target_zip",
                           hashlib.sha256(payload).hexdigest() == run["target_zip_sha256"], run["run_id"])
                    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                        weights = f"{run['run_id']}/checkpoints/{run['checkpoint']}/adapter/adapter_model.safetensors"
                        ok = weights in inner.namelist() and \
                            hashlib.sha256(inner.read(weights)).hexdigest() == run["weights_sha256"]
                        _check(results, f"{task}/{target}:weights", ok, weights)
        _verify_evidence(freeze, root, results)
    return results


def _envelopes(text, fold_id, expert_id, name, results):
    """Validate every EVID-CS-001 line; return the number of distinct target records."""
    from seqlogad.evidence.contract import EvidenceError, validate

    records, bad = set(), 0
    for line in text.splitlines():
        try:
            envelope = validate(json.loads(line))
            bad += envelope["fold_id"] != fold_id or envelope["expert_id"] != expert_id
            records.add(envelope["target_record_id"])
        except (EvidenceError, ValueError):
            bad += 1
    _check(results, f"{name}:schema", bad == 0, f"{bad} invalid or foreign envelopes")
    return len(records)


def _verify_evidence(freeze, root, results):
    """Schema compatibility and completeness: one valid envelope per evaluation window, receipts clean."""
    checks = freeze["evidence_checks"]
    archive_path = root / freeze["artifacts"][checks["zip"]]["path"]
    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            for fold_id, windows in checks["windows"].items():
                for expert_id in checks["experts_in_zip"]:
                    name = f"evidence:{fold_id}:{expert_id}"
                    payload = archive.read(f"evidence/{fold_id}/{expert_id}.jsonl")
                    receipt = json.loads(archive.read(f"evidence/{fold_id}/{expert_id}.receipt.json"))
                    _check(results, f"{name}:receipt", receipt["sanity_reproduction"]["pass"]
                           and receipt["labels_read"] is False and receipt["windows"] == windows
                           and receipt["evidence_sha256"] == hashlib.sha256(payload).hexdigest(),
                           "sanity reproduced, no label read, digest matches")
                    count = _envelopes(payload.decode("utf-8"), fold_id, expert_id, name, results)
                    _check(results, f"{name}:complete", count == windows, f"{count} of {windows} windows")
    for key, fold_id in checks["jsonl"].items():
        path = root / freeze["artifacts"][key]["path"]
        if path.is_file():
            count = _envelopes(path.read_text(encoding="utf-8"), fold_id, "GTAT_STRUCTURAL_TEMPORAL",
                               f"evidence:{fold_id}:GTAT", results)
            _check(results, f"evidence:{fold_id}:GTAT:complete", count == checks["windows"][fold_id],
                   f"{count} of {checks['windows'][fold_id]} windows")
