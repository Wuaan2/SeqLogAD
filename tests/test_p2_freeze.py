"""P2.7 freeze manifest: complete, consistent with the gates, and verifiable from a relocated root."""
import re
from pathlib import Path

import yaml

from seqlogad.freeze import load_manifest, verify

ROOT = Path(__file__).resolve().parents[1]
FREEZE = load_manifest(ROOT)
SHA = re.compile(r"[0-9a-f]{64}")


def test_every_expert_and_fold_is_bound_to_hashed_artifacts():
    assert list(FREEZE["experts"]) == ["P2.1", "P2.2", "P2.3", "P2.4"]
    assert all(SHA.fullmatch(a["sha256"]) for a in FREEZE["artifacts"].values())
    for expert in FREEZE["experts"].values():
        assert {expert["code"], expert["final"], *expert["data"]} <= set(FREEZE["artifacts"])
        assert set(expert["runs"]) == {"BGL", "HDFS"}
        for run in expert["runs"].values():
            assert SHA.fullmatch(run["target_zip_sha256"]) and SHA.fullmatch(run["weights_sha256"])
            assert run["checkpoint"].startswith("step-")
        assert expert["disposition"].startswith("RETAINED_")


def test_freeze_respects_gates_and_selects_no_winner():
    g2 = yaml.safe_load((ROOT / "configs/protocols/g2-signature.yaml").read_text(encoding="utf-8"))["g2_signature"]
    assert FREEZE["gates"]["G2"]["state"] == g2["decision"] == "PARTIAL"
    assert FREEZE["sequence_winner"] == "UNDECIDED"
    assert FREEZE["p3_launch"] == "MANUAL_AFTER_RESEARCHER_REVIEW"
    assert FREEZE["status"] == "FROZEN_ACCEPTED_BY_RESEARCHER"
    review = FREEZE["researcher_review"]
    assert review["decided_by"] == "RESEARCHER" and review["p3_authorized_tasks"] == ["P3.1"]
    assert review["multi_seed_stability_claims"] == "FORBIDDEN"
    state = yaml.safe_load((ROOT / "configs/active-state.yaml").read_text(encoding="utf-8"))["active_state"]
    assert state["next_authorized_task"] in review["p3_authorized_tasks"]


def test_completion_status_update_has_exact_provenance():
    import openpyxl

    from seqlogad.common.checksum import sha256_file

    p2 = yaml.safe_load((ROOT / "configs/protocols/phase2-architecture-v1.yaml").read_text(encoding="utf-8"))["phase2"]
    update = p2["tasks"]["P2.7"]["roadmap_update"]
    before_path, after_path = ROOT / update["previous_workbook"], ROOT / "Bang_ke_hoach_SeqLogAD.xlsx"
    assert sha256_file(before_path) == update["from_sha256"] == p2["tasks"]["P2.1"]["roadmap_update"]["to_sha256"]
    assert sha256_file(after_path) == update["to_sha256"]
    books = [openpyxl.load_workbook(p) for p in (before_path, after_path)]
    before, after = [{f"{s.title}!{c.coordinate}": c.value for s in b for row in s for c in row if c.value is not None}
                     for b in books]
    changed = {k: {"before": before.get(k), "after": after.get(k)}
               for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changed == update["changed_cells"] and update["gate_signature_created"] is False
    for sheet in books[0]:
        assert str(sheet.data_validations) == str(books[1][sheet.title].data_validations)
        assert str(sheet.merged_cells) == str(books[1][sheet.title].merged_cells)


def test_relocated_root_reports_missing_and_tampered_artifacts(tmp_path):
    name, artifact = next(iter(FREEZE["artifacts"].items()))
    (tmp_path / artifact["path"]).parent.mkdir(parents=True)
    (tmp_path / artifact["path"]).write_bytes(b"tampered")
    results = {row["item"]: row["status"] for row in verify(ROOT, tmp_path)}
    assert results[name] == "MISMATCH"
    assert all(status == "MISSING" for item, status in results.items()
               if item in FREEZE["artifacts"] and item != name)
    assert all(status == "OK" for item, status in results.items() if item.startswith(("repo:", "repo_hash:")))


def test_invalid_or_incomplete_evidence_is_reported(tmp_path):
    path = tmp_path / FREEZE["artifacts"]["evidence_p2_4_bgl"]["path"]
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version": "EVID-CS-001"}\n', encoding="utf-8")
    results = {row["item"]: row["status"] for row in verify(ROOT, tmp_path)}
    assert results["evidence:FOLD-TARGET-ARCH-BGL:GTAT:schema"] == "MISMATCH"
    assert results["evidence:FOLD-TARGET-ARCH-BGL:GTAT:complete"] == "MISMATCH"
