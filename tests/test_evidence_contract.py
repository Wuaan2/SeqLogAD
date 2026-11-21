"""P2.5 EVID-CS-001: real P2.1–P2.4 evidence converts and validates; malformed envelopes are rejected."""
import copy
import json
from pathlib import Path

import pytest

from seqlogad.evidence.contract import EvidenceError, index_by_target, validate
from seqlogad.evidence.convert import convert

FIXTURES = Path(__file__).parent / "fixtures/evidence"
NATIVE = {task: json.loads((FIXTURES / f"native-{task}.json").read_text()) for task in ("P2.1", "P2.2", "P2.3", "P2.4")}


def converted(task):
    return convert(copy.deepcopy(NATIVE[task]))


@pytest.mark.parametrize("task", sorted(NATIVE))
def test_every_expert_converts_to_a_valid_envelope(task):
    envelope = converted(task)
    assert envelope["schema_version"] == "EVID-CS-001"
    assert envelope["expert_metadata"]["native_schema_version"] == NATIVE[task]["schema_version"]
    assert envelope["score_raw"] == NATIVE[task]["score_raw"]


def test_window_experts_join_on_the_same_target_record():
    targets = {task: converted(task)["target_record_id"] for task in ("P2.2", "P2.3", "P2.4")}
    assert set(targets.values()) == {"ARCH-HADOOP:253148"}
    assert converted("P2.1")["unit_kind"] == "LOG_RECORD"


def test_gtat_draft_is_normalised():
    envelope = converted("P2.4")
    assert envelope["status"] == "VALID" and envelope["reason"] is None
    assert envelope["uncertainty"] is None and envelope["uncertainty_method"] == "UNCERTAINTY_NOT_ESTIMATED_IN_PHASE2"
    assert [item["kind"] for item in envelope["evidence"]][:3] == ["temporal_nll", "structural_nll", "content_nll"]


def test_abstain_is_valid_without_a_score():
    native = copy.deepcopy(NATIVE["P2.1"])
    native.update(score_raw=None, uncertainty=None, coverage=0.0, evidence=[], localisation=[],
                  status="ABSTAIN", reason="empty message after normalisation")
    assert convert(native)["status"] == "ABSTAIN"


def test_one_expert_is_consumable_alone_and_identities_are_unique():
    envelope = converted("P2.2")
    assert list(index_by_target([envelope])) == [("FOLD-TARGET-ARCH-BGL", "ARCH-HADOOP:253148")]
    with pytest.raises(EvidenceError, match="duplicate"):
        index_by_target([envelope, copy.deepcopy(envelope)])
    with pytest.raises(EvidenceError, match="one expert"):
        index_by_target([envelope, converted("P2.4")])


def test_unknown_native_schema_is_refused():
    with pytest.raises(ValueError, match="no EVID-CS-001 converter"):
        convert(dict(NATIVE["P2.2"], schema_version="SEQUENCE-EVIDENCE-0"))


def _set(key, value):
    return lambda e: e.__setitem__(key, value)


INVALID = {
    "missing_field": lambda e: e.pop("coverage"),
    "extra_field": _set("label", True),
    "calibrated_score": _set("score_calibrated", 0.5),
    "nonfinite_score": _set("score_raw", float("nan")),
    "abstain_with_score": lambda e: e.update(status="ABSTAIN", reason="x"),
    "valid_with_reason": _set("reason", "unexpected"),
    "low_coverage_without_reason": _set("status", "LOW_COVERAGE"),
    "coverage_above_one": _set("coverage", 1.5),
    "negative_uncertainty": _set("uncertainty", -0.1),
    "missing_uncertainty_method": _set("uncertainty_method", ""),
    "wrong_role": _set("expert_role", "SEMANTIC"),
    "bad_checkpoint_hash": _set("checkpoint_sha256", "abc"),
    "records_out_of_order": lambda e: e.__setitem__("record_ids", list(reversed(e["record_ids"]))),
    "target_mismatch": _set("target_record_id", "ARCH-HADOOP:1"),
    "foreign_architecture_record": lambda e: e["record_ids"].__setitem__(0, "ARCH-BGL:1"),
    "negative_position": lambda e: e["localisation"][0].__setitem__("target_token_position_in_model_input", -1),
    "evidence_without_kind": lambda e: e["evidence"][0].pop("kind"),
    "evidence_nonfinite": lambda e: e["evidence"][0].__setitem__("value", float("inf")),
    "unknown_status": _set("status", "SCORED"),
    "metadata_without_native_schema": _set("expert_metadata", {}),
}


@pytest.mark.parametrize("case", sorted(INVALID))
def test_malformed_envelopes_are_rejected(case):
    envelope = converted("P2.2")
    INVALID[case](envelope)
    with pytest.raises(EvidenceError):
        validate(envelope)


def test_log_record_must_score_one_record_and_gtat_rank_must_match():
    semantic = converted("P2.1")
    semantic["record_ids"] = ["ARCH-HADOOP:1", semantic["record_ids"][0]]
    with pytest.raises(EvidenceError, match="exactly one record"):
        validate(semantic)
    gtat = converted("P2.4")
    gtat["localisation"][0]["target_rank"] += 1
    with pytest.raises(EvidenceError, match="target_rank"):
        validate(gtat)
