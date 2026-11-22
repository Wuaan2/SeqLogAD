"""P3.1 G2 handoff (G2-HANDOFF-001): the P2.7 freeze, the researcher's G2 record and the closed Phase 3 agree.

Verification only: nothing is rerun or scored, no target label is read and no gate is signed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from seqlogad.freeze import load_manifest, verify

G2_RECORD = "configs/protocols/g2-signature.yaml"


def _yaml(repo_root, relative):
    return yaml.safe_load((Path(repo_root) / relative).read_text(encoding="utf-8"))


def lf_digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def signed_digest_matches(data, digest):
    """The G2 record hashed some reports on a CRLF checkout; accept the LF or CRLF rendering of the same text."""
    lf = data.replace(b"\r\n", b"\n")
    return digest in {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}


def verify_g2_handoff(repo_root, project_root=None, deep=True):
    """Return one result per check (status OK or MISMATCH); with project_root also run the full P2-FREEZE-001 check."""
    results = []

    def check(name, ok, detail):
        results.append({"item": name, "status": "OK" if ok else "MISMATCH", "detail": detail})

    freeze, g2 = load_manifest(repo_root), _yaml(repo_root, G2_RECORD)["g2_signature"]
    state, basis = _yaml(repo_root, "configs/active-state.yaml")["active_state"], g2["basis"]
    check("g2:decision", g2["gate"] == "G2" and g2["decision"] == freeze["gates"]["G2"]["state"]
          == state["gates"]["G2"] == "PARTIAL", "record, freeze and active state all say PARTIAL")
    check("g2:signed_by", g2["signed_by"] == "RESEARCHER", g2["signed_on"])
    check("g2:record_frozen", freeze["gates"]["G2"]["record"] == G2_RECORD and G2_RECORD in freeze["repo_files"],
          "the G2 record is bound by P2-FREEZE-001")
    for key in ("preregistration", "label_scope"):
        check(f"g2:basis:{key}", basis[key] in freeze["repo_files"], basis[key])
    for key in ("report", "report_data", "post_hoc_review"):
        path, data = basis[key]["path"], (Path(repo_root) / basis[key]["path"]).read_bytes()
        check(f"g2:basis:{key}", signed_digest_matches(data, basis[key]["sha256"])
              and freeze["repo_files"].get(path) == lf_digest(data), f"{path}: signed text is the frozen text")
    evidence = freeze["artifacts"][freeze["evidence_checks"]["zip"]]
    check("g2:evidence_zip", basis["evidence_zip_sha256"] == evidence["sha256"], evidence["path"])
    check("p3:closed_at_handoff", state["next_authorized_task"] == "P3.1" and state["target_adaptation_status"]
          == "NOT_STARTED" and freeze["researcher_review"]["p3_authorized_tasks"] == ["P3.1"],
          "only P3.1 is authorized; no target adaptation has started")
    if project_root is not None:
        results += verify(repo_root, project_root, deep=deep)
    return results
