"""EVID-CS-001: the strict common evidence envelope shared by the four Phase-2 experts (P2.5)."""
from __future__ import annotations

import math
import re

SCHEMA = "EVID-CS-001"
EXPERT_ROLES = {
    "SEMANTIC_LLAMA": "SEMANTIC",
    "SEQUENCE_LLAMA_REFERENCE": "SEQUENCE",
    "SEQUENCE_LIGHTWEIGHT": "SEQUENCE",
    "GTAT_STRUCTURAL_TEMPORAL": "STRUCTURAL_TEMPORAL",
}
UNIT_KINDS = {"LOG_RECORD", "TARGET_WINDOW"}
STATUSES = {"VALID", "LOW_COVERAGE", "ABSTAIN"}
FIELDS = {
    "schema_version", "expert_id", "expert_role", "fold_id", "architecture_id", "unit_id", "unit_kind",
    "record_ids", "target_record_id", "score_raw", "score_direction", "score_calibrated",
    "calibration_status", "uncertainty", "uncertainty_method", "coverage", "evidence", "localisation",
    "status", "reason", "model_revision", "checkpoint_sha256", "input_manifest_sha256",
    "context_config_sha256", "expert_metadata",
}
ARCHITECTURE = re.compile(r"ARCH-[A-Z0-9]+")
FOLD = re.compile(r"FOLD-TARGET-ARCH-[A-Z0-9]+")
RECORD = re.compile(r"(ARCH-[A-Z0-9]+):(\d+)")
SHA256 = re.compile(r"[0-9a-f]{64}")


class EvidenceError(ValueError):
    """An envelope that violates EVID-CS-001."""


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _scalar(value):
    return value is None or isinstance(value, (str, bool)) or _finite(value)


def validate(envelope: dict) -> dict:
    """Raise EvidenceError unless the envelope is a complete, consistent EVID-CS-001 record."""
    def require(condition, message):
        if not condition:
            raise EvidenceError(f"{envelope.get('expert_id')}/{envelope.get('unit_id')}: {message}")

    require(isinstance(envelope, dict), "envelope must be an object")
    require(set(envelope) == FIELDS, f"field set mismatch: missing {sorted(FIELDS - set(envelope))}, "
                                     f"unexpected {sorted(set(envelope) - FIELDS)}")
    require(envelope["schema_version"] == SCHEMA, "schema_version must be EVID-CS-001")
    require(EXPERT_ROLES.get(envelope["expert_id"]) == envelope["expert_role"], "unknown expert or wrong role")
    require(isinstance(envelope["fold_id"], str) and FOLD.fullmatch(envelope["fold_id"]), "bad fold_id")
    architecture = envelope["architecture_id"]
    require(isinstance(architecture, str) and ARCHITECTURE.fullmatch(architecture), "bad architecture_id")
    require(isinstance(envelope["unit_id"], str) and envelope["unit_id"], "unit_id must be a non-empty string")
    require(envelope["unit_kind"] in UNIT_KINDS, "unknown unit_kind")

    records = envelope["record_ids"]
    require(isinstance(records, list) and records, "record_ids must be a non-empty list")
    ranks = []
    for record in records:
        match = RECORD.fullmatch(record) if isinstance(record, str) else None
        require(match and match.group(1) == architecture, f"bad record id {record!r}")
        ranks.append(int(match.group(2)))
    require(all(a < b for a, b in zip(ranks, ranks[1:])), "record_ids must be unique and chronological")
    require(envelope["target_record_id"] == records[-1], "target_record_id must be the last record id")
    if envelope["unit_kind"] == "LOG_RECORD":
        require(len(records) == 1, "a LOG_RECORD unit scores exactly one record")

    status = envelope["status"]
    require(status in STATUSES, "unknown status")
    if status == "ABSTAIN":
        require(envelope["score_raw"] is None, "ABSTAIN must not carry a score")
    else:
        require(_finite(envelope["score_raw"]), "score_raw must be finite unless ABSTAIN")
    require((envelope["reason"] is None) == (status == "VALID"), "reason is required exactly when status is not VALID")
    require(envelope["reason"] is None or (isinstance(envelope["reason"], str) and envelope["reason"]), "empty reason")
    require(envelope["score_direction"] == "HIGHER_IS_MORE_ANOMALOUS", "score_direction is fixed")
    require(envelope["score_calibrated"] is None and envelope["calibration_status"] == "NOT_CALIBRATED",
            "Phase-2 evidence is never calibrated")

    uncertainty = envelope["uncertainty"]
    require(uncertainty is None or (_finite(uncertainty) and uncertainty >= 0), "uncertainty must be null or >= 0")
    require(isinstance(envelope["uncertainty_method"], str) and envelope["uncertainty_method"],
            "uncertainty_method must name the method or why uncertainty is unavailable")
    require(_finite(envelope["coverage"]) and 0 <= envelope["coverage"] <= 1, "coverage must be in [0, 1]")

    evidence = envelope["evidence"]
    require(isinstance(evidence, list), "evidence must be a list")
    require(evidence or status == "ABSTAIN", "scored units need at least one evidence item")
    for item in evidence:
        require(isinstance(item, dict) and isinstance(item.get("kind"), str) and item["kind"],
                "evidence items need a kind")
        require("value" in item and (item["value"] is None or _finite(item["value"])),
                "evidence value must be finite or null")
        require(all(_scalar(value) for value in item.values()), "evidence items hold scalars only")
    localisation = envelope["localisation"]
    require(isinstance(localisation, list), "localisation must be a list")
    for item in localisation:
        require(isinstance(item, dict) and item and all(_scalar(value) for value in item.values()),
                "localisation items are non-empty objects of scalars")
        for key, value in item.items():
            if "position" in key or key.endswith("_rank"):
                require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                        f"localisation {key} must be a non-negative integer")
            if key == "target_rank":
                require(value == ranks[-1], "localisation target_rank must match target_record_id")

    require(isinstance(envelope["model_revision"], str) and envelope["model_revision"], "model_revision missing")
    for key in ("checkpoint_sha256", "input_manifest_sha256", "context_config_sha256"):
        require(isinstance(envelope[key], str) and SHA256.fullmatch(envelope[key]), f"{key} must be SHA-256 hex")
    metadata = envelope["expert_metadata"]
    require(isinstance(metadata, dict) and isinstance(metadata.get("native_schema_version"), str),
            "expert_metadata needs native_schema_version")
    require(all(_scalar(value) for value in metadata.values()), "expert_metadata holds scalars only")
    return envelope


def index_by_target(envelopes) -> dict:
    """Validate one expert's envelopes and key them by (fold, target record): consumable on its own."""
    index = {}
    experts = set()
    for envelope in envelopes:
        validate(envelope)
        experts.add(envelope["expert_id"])
        if len(experts) > 1:
            raise EvidenceError("index_by_target takes the envelopes of one expert at a time")
        key = (envelope["fold_id"], envelope["target_record_id"])
        if key in index:
            raise EvidenceError(f"duplicate evidence for {key} from {envelope['expert_id']}")
        index[key] = envelope
    return index
