"""Convert each expert's native Phase-2 evidence into EVID-CS-001, then validate it."""
from __future__ import annotations

from .contract import SCHEMA, validate

COMMON = ("expert_id", "expert_role", "fold_id", "architecture_id", "unit_id", "record_ids", "score_raw",
          "score_direction", "score_calibrated", "calibration_status", "uncertainty", "uncertainty_method",
          "coverage", "status", "reason", "model_revision", "checkpoint_sha256", "input_manifest_sha256",
          "context_config_sha256")
SEQUENCE_EXTRAS = ("adaptation_variant", "encoding_variant", "training_context_policy")
GTAT_COMPONENTS = ("temporal_nll", "structural_nll", "content_nll")


def _base(native, unit_kind, **metadata):
    envelope = {key: native[key] for key in COMMON}
    envelope.update(schema_version=SCHEMA, unit_kind=unit_kind, target_record_id=native["record_ids"][-1],
                    evidence=native["evidence"], localisation=native["localisation"],
                    expert_metadata={"native_schema_version": native["schema_version"], **metadata})
    return envelope


def from_semantic(native):
    return _base(native, "LOG_RECORD")


def from_sequence(native):
    return _base(native, "TARGET_WINDOW", **{key: native[key] for key in SEQUENCE_EXTRAS})


def from_gtat(native):
    envelope = _base(native, "TARGET_WINDOW")
    parts = native["evidence"]
    envelope["evidence"] = ([{"kind": name, "value": parts[name]} for name in GTAT_COMPONENTS]
                            + [{"kind": "context_records", "value": parts["context_records"]}]
                            if parts else [])
    envelope["localisation"] = [native["localisation"]] if native["localisation"] else []
    # The P2.4 draft parked "uncertainty not estimated" in reason and used SCORED for a valid score.
    if native["uncertainty"] is None and native["uncertainty_method"] is None:
        envelope["uncertainty_method"] = native["reason"] or "NOT_ESTIMATED_IN_PHASE2"
    if native["status"] == "SCORED":
        envelope.update(status="VALID", reason=None)
    return envelope


CONVERTERS = {
    "SEMANTIC-EVIDENCE-1": from_semantic,
    "SEQUENCE-EVIDENCE-1": from_sequence,
    "SEQLOGAD-EVIDENCE-DRAFT-P2.4": from_gtat,
}


def convert(native: dict) -> dict:
    converter = CONVERTERS.get(native.get("schema_version"))
    if converter is None:
        raise ValueError(f"no EVID-CS-001 converter for {native.get('schema_version')!r}")
    return validate(converter(native))
