"""UNC-CS-001 (P3.5): per-expert uncertainty and the UNKNOWN_REVIEW (abstain) rule.

Spec: configs/protocols/unc-cs-v1.yaml. An expert abstains only on insufficient evidence (ABSTAIN, LOW_COVERAGE, no
calibrated scale), never on novelty. The expert's native uncertainty is put on one scale with a target-normal ECDF
from the burn-in buffer and reported; it never decides an abstention. A window needs at least two SCORED experts.
Labels are read only by risk_coverage, in the locked final evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from seqlogad.adaptation.calibration import SCORED
from seqlogad.adaptation.store import (ADAPT_ROOT, AdaptationError, load_artifact, load_frozen_memory, memory_table,
                                       write_artifact)
from seqlogad.adaptation.store import load_spec as _load_spec
from seqlogad.common.checksum import sha256_file
from seqlogad.diversity.analysis import average_precision

SPEC = "configs/protocols/unc-cs-v1.yaml"
ARTIFACT = "UNC-CS-001"
REVIEW = "UNKNOWN_REVIEW"


def load_spec(repo_root):
    spec = _load_spec(repo_root, SPEC, "unc_cs")
    if (spec["window"]["quorum"] != 2 or spec["expert"]["native_uncertainty"]["minimum"] != 1000
            or list(spec["expert"]["review_when"]) != ["ABSTAIN", "LOW_COVERAGE", "UNCALIBRATED"]):
        raise AdaptationError("UNC-CS-001 rule changed")
    return spec


def expert_state(status, calibration_status):
    """(state, reason) of one scored unit: the first insufficient-evidence condition is the reason."""
    if status in ("ABSTAIN", "LOW_COVERAGE"):
        return REVIEW, status
    if calibration_status != "CALIBRATED":
        return REVIEW, "UNCALIBRATED"
    return "SCORED", None


def normalise(reference_sorted, values):
    """Target-normal ECDF: the share of reference values <= each value; None without a reference or a finite value."""
    if reference_sorted is None:
        return [None] * len(values)
    reference = np.asarray(reference_sorted, dtype=np.float64)
    return [None if v is None or not math.isfinite(v) else float(np.searchsorted(reference, v, side="right") / reference.size)
            for v in values]


def window_states(states, quorum):
    """One row per target window: SCORED experts, state, reason and the mean normalised native uncertainty."""
    scored = pl.col("expert_state") == "SCORED"
    short = pl.col("scored_experts") < quorum
    return (states.group_by("target_rank")
            .agg(scored.sum().alias("scored_experts"), pl.col("native_uncertainty").filter(scored).mean().alias("uncertainty"))
            .with_columns(pl.when(short).then(pl.lit(REVIEW)).otherwise(pl.lit("ASSESSABLE")).alias("window_state"),
                          pl.when(short).then(pl.lit("QUORUM")).otherwise(pl.lit(None, pl.String)).alias("reason"))
            .sort("target_rank"))


def risk_coverage(labels, scores, uncertainty, levels):
    """Final evaluation only: AP of the fused score on the windows kept at each coverage level, lowest uncertainty first."""
    order = np.argsort([math.inf if u is None else u for u in uncertainty], kind="stable")
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    return {str(level): average_precision(scores[order[: round(level * order.size)]], labels[order[: round(level * order.size)]])
            for level in levels}


def _pass_uncertainty(pass_root, fold_id, expert, digest):
    """The expert's buffer-pass envelopes, accepted only if they are the evidence the frozen memory binds."""
    path = Path(pass_root) / fold_id / f"{expert}.jsonl"
    if sha256_file(path) != digest:
        raise AdaptationError(f"{path} is not the buffer pass the frozen memory binds")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return pl.DataFrame({"unit_id": [r["unit_id"] for r in rows], "native_raw": [r["uncertainty"] for r in rows],
                         "envelope_reason": [r["reason"] for r in rows],
                         "uncertainty_method": [r["uncertainty_method"] for r in rows]},
                        schema={"unit_id": pl.String, "native_raw": pl.Float64, "envelope_reason": pl.String,
                                "uncertainty_method": pl.String})


def build_uncertainty(root, repo_root, pass_root, folds=None):
    """Native-uncertainty references and the example buffer states of UNC-CS-001; frozen per fold."""
    spec = load_spec(repo_root)
    minimum, quorum = spec["expert"]["native_uncertainty"]["minimum"], spec["window"]["quorum"]
    report = {}
    for fold_id in folds or spec["folds"]:
        memory, manifests = load_frozen_memory(root, fold_id)
        calibration = load_artifact(root, "CAL-CS-001", fold_id)
        if calibration["memory_sha256"] != memory["sha256"]:
            raise AdaptationError(f"{fold_id}: CAL-CS-001 was built on another memory")
        receipts = manifests["prototypes"]["pass_receipts"]
        scores = memory_table(root, fold_id, "buffer_scores", "scores")
        states, references, experts = [], [], {}
        for expert in spec["experts"]:
            rows = scores.filter(pl.col("expert_id") == expert).join(
                _pass_uncertainty(pass_root, fold_id, expert, receipts[expert]["evidence_sha256"]),
                on="unit_id", how="left", validate="1:1")
            ok = rows.filter(pl.col("status").is_in(SCORED) & pl.col("native_raw").is_finite())
            reference = np.sort(ok["native_raw"].to_numpy()) if ok.height >= minimum else None
            pairs = [expert_state(s, calibration["experts"][expert]["status"]) for s in rows["status"]]
            rows = rows.with_columns(
                pl.col("target_record_id").str.split(":").list.last().cast(pl.Int64).alias("target_rank"),
                pl.Series("expert_state", [p[0] for p in pairs]),
                pl.Series("reason", [p[1] for p in pairs], dtype=pl.String),
                (1 - pl.col("coverage")).alias("input_uncertainty"),
                pl.Series("native_uncertainty", normalise(reference, rows["native_raw"].to_list()), dtype=pl.Float64))
            states.append(rows.select("expert_id", "unit_id", "target_rank", "status", "envelope_reason", "coverage",
                                      "expert_state", "reason", "input_uncertainty", "native_uncertainty"))
            if reference is not None:
                references.append(pl.DataFrame({"expert_id": [expert] * reference.size, "native_raw": reference}))
            experts[expert] = {
                "native_uncertainty": "AVAILABLE" if reference is not None else "NOT_AVAILABLE",
                "n_reference": int(ok.height), "methods": sorted(rows["uncertainty_method"].unique().to_list()),
                "quantiles": {q: float(np.quantile(reference, float(q))) for q in ("0.5", "0.9", "0.99")}
                if reference is not None else None,
                "units": rows.height, "unknown_review": dict(rows.filter(pl.col("reason").is_not_null())
                                                             .group_by("reason").len().iter_rows())}
        windows = window_states(pl.concat(states), quorum)
        summary = {"windows": windows.height, "unknown_review": int((windows["window_state"] == REVIEW).sum())}
        manifest = {"artifact_id": ARTIFACT, "fold_id": fold_id, "memory_key": memory["memory_key"],
                    "memory_sha256": memory["sha256"],
                    "calibration_sha256": sha256_file(Path(root) / ADAPT_ROOT / "CAL-CS-001" / fold_id / "artifact.json"),
                    "spec_sha256": spec["sha256"], "builder_sha256": sha256_file(Path(__file__)), "quorum": quorum,
                    "experts": experts, "buffer_windows": summary, "target_labels_read": False}
        tables = {"buffer_states": pl.concat(states), "buffer_windows": windows}
        if references:
            tables["reference"] = pl.concat(references)
        write_artifact(root, ARTIFACT, fold_id, tables, manifest)
        report[fold_id] = {"experts": {e: {k: v[k] for k in ("native_uncertainty", "unknown_review")}
                                       for e, v in experts.items()}, "buffer_windows": summary}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="freeze UNC-CS-001 from the frozen fold memories and CAL-CS-001")
    build.add_argument("--root", required=True, help="folder holding data/processed/memory")
    build.add_argument("--repo-root", required=True)
    build.add_argument("--pass-root", required=True, help="P3.2 buffer_pass folder holding <fold>/<EXPERT>.jsonl")
    args = parser.parse_args()
    print(json.dumps(build_uncertainty(args.root, args.repo_root, args.pass_root), indent=2))


if __name__ == "__main__":
    main()
