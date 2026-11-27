"""P3.1 G2 handoff: the researcher's G2 record, its frozen basis and the closed Phase 3 agree."""
import hashlib
from pathlib import Path

import yaml

from seqlogad.handoff import signed_digest_matches, verify_g2_handoff

ROOT = Path(__file__).resolve().parents[1]


def test_repository_side_g2_handoff_checks_pass():
    results = verify_g2_handoff(ROOT)
    assert [r for r in results if r["status"] != "OK"] == []
    assert len(results) == 10


def test_signed_digest_accepts_either_line_ending_but_not_other_text():
    lf = b"line one\nline two\n"
    crlf_digest = hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()
    assert signed_digest_matches(lf, crlf_digest) and signed_digest_matches(lf.replace(b"\n", b"\r\n"), crlf_digest)
    assert not signed_digest_matches(b"line one\nline 2\n", crlf_digest)


def test_receipt_records_a_passing_verification_and_signs_nothing():
    receipt = yaml.safe_load((ROOT / "configs/protocols/p3.1-g2-handoff-v1.yaml").read_text(encoding="utf-8"))
    handoff = receipt["g2_handoff"]
    assert handoff["task"] == "P3.1" and handoff["verified_by"] == "CODING_AGENT"
    assert handoff["gate_signature_created"] is False and handoff["g2_state"] == "PARTIAL"
    assert all(result["failed"] == 0 for result in handoff["result"].values())
    assert handoff["phase3"]["learned_fusion"] == "CLOSED_G2_NOT_PASSED"
    assert handoff["researcher_review"]["decided_by"] == "RESEARCHER"
    assert handoff["researcher_review"]["p3_authorized_tasks"] == []


def test_p3_1_status_update_has_exact_provenance():
    import openpyxl

    from seqlogad.common.checksum import sha256_file

    update = yaml.safe_load((ROOT / "configs/protocols/p3.1-g2-handoff-v1.yaml").read_text(encoding="utf-8"))[
        "g2_handoff"]["roadmap_update"]
    p2 = yaml.safe_load((ROOT / "configs/protocols/phase2-architecture-v1.yaml").read_text(encoding="utf-8"))["phase2"]
    following = yaml.safe_load((ROOT / "configs/protocols/adapt-mem-v1.yaml").read_text(encoding="utf-8"))[
        "adapt_mem"]["roadmap_update"]
    before_path, after_path = ROOT / update["previous_workbook"], ROOT / following["previous_workbook"]
    assert sha256_file(before_path) == update["from_sha256"] == p2["tasks"]["P2.7"]["roadmap_update"]["to_sha256"]
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
