"""CAL-CS-001 (P3.3): per-fold target-normal quantile scaling of each expert's raw score.

Spec: configs/protocols/cal-cs-v1.yaml, fixed before any buffer score was summarised. An expert's reference is its
buffer_scores rows in the frozen ADAPT-MEM-001 fold memory; nothing else is read. The scaling is strictly increasing,
so each expert keeps its ranking; it only puts the experts on one scale (0 = buffer median, 1 = buffer 99th
percentile). Envelopes are not rewritten here: consumers apply the frozen table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from seqlogad.adaptation.store import AdaptationError, load_frozen_memory, memory_table, write_artifact
from seqlogad.adaptation.store import load_spec as _load_spec
from seqlogad.common.checksum import sha256_file

SPEC = "configs/protocols/cal-cs-v1.yaml"
ARTIFACT = "CAL-CS-001"
SCORED = ("VALID", "LOW_COVERAGE")


def load_spec(repo_root):
    spec = _load_spec(repo_root, SPEC, "cal_cs")
    if spec["rule"]["anchors"] != {"low": 0.5, "high": 0.99} or spec["reference"]["minimum"] != 1000:
        raise AdaptationError("CAL-CS-001 rule changed")
    return spec


def reference_rows(frame):
    """One expert's reference scores and target ranks in chronological order, and its ABSTAIN count."""
    ok = frame.filter(pl.col("status").is_in(SCORED) & pl.col("score_raw").is_not_null() & pl.col("score_raw").is_finite())
    ok = ok.with_columns(pl.col("target_record_id").str.split(":").list.last().cast(pl.Int64).alias("target_rank"))
    ok = ok.sort("target_rank")
    return ok["score_raw"].to_numpy(), ok["target_rank"].to_numpy(), int((frame["status"] == "ABSTAIN").sum())


def fit(scores, spec):
    """Anchors of one (fold, expert) reference, or the reason it cannot be calibrated."""
    n = int(scores.size)
    entry = {"n_reference": n, "low": None, "high": None,
             "quantiles": {str(q): float(np.quantile(scores, q)) for q in spec["quantile_grid"]} if n else {}}
    if n < spec["reference"]["minimum"]:
        return dict(entry, status="INSUFFICIENT_REFERENCE")
    anchors = spec["rule"]["anchors"]
    low, high = (float(v) for v in np.quantile(scores, [anchors["low"], anchors["high"]]))
    return dict(entry, low=low, high=high, status="CALIBRATED" if high > low else "DEGENERATE_SCALE")


def calibrate(entry, scores):
    """score_calibrated for each raw score; None for an ABSTAIN (None) or an expert that is not CALIBRATED."""
    if entry["status"] != "CALIBRATED":
        return [None] * len(scores)
    return [None if s is None else (s - entry["low"]) / (entry["high"] - entry["low"]) for s in scores]


def tail_p(reference_sorted, scores):
    """Conservative tail p-value (1 + #{reference >= s}) / (n + 1) for each score."""
    reference = np.asarray(reference_sorted, dtype=np.float64)
    at_least = reference.size - np.searchsorted(reference, np.asarray(scores, dtype=np.float64), side="left")
    return (1 + at_least) / (reference.size + 1)


def sanity(scores, entry, spec):
    """The preregistered label-free checks; they are reported and never change the rule."""
    if entry["status"] != "CALIBRATED":
        return None
    anchors, checks = spec["rule"]["anchors"], spec["sanity"]
    calibrated = np.asarray(calibrate(entry, scores.tolist()), dtype=np.float64)
    first, second = scores[: scores.size // 2], scores[scores.size // 2:]
    exceedance = float((second > np.quantile(first, anchors["high"])).mean())
    low_band, high_band = checks["temporal_split_band"]
    return {
        "anchors_reproduced": bool(abs(np.quantile(calibrated, anchors["low"])) < 1e-9
                                   and abs(np.quantile(calibrated, anchors["high"]) - 1) < 1e-9),
        "rank_preserved": bool(np.array_equal(np.argsort(scores, kind="stable"), np.argsort(calibrated, kind="stable"))),
        "temporal_split": {"first_half": int(first.size), "second_half": int(second.size),
                           "second_half_exceedance": exceedance,
                           "state": "STABLE" if low_band <= exceedance <= high_band else "UNSTABLE"},
        "histogram": {"edges": checks["histogram_edges"],
                      "counts": np.histogram(calibrated, [-np.inf, *checks["histogram_edges"], np.inf])[0].tolist()},
    }


def build_calibration(root, repo_root, folds=None):
    """Fit, check and freeze CAL-CS-001 for each fold from its frozen memory."""
    spec = load_spec(repo_root)
    report = {}
    for fold_id in folds or spec["folds"]:
        memory, manifests = load_frozen_memory(root, fold_id)
        if manifests["buffer_scores"].get("calibrated") is not False:
            raise AdaptationError(f"{fold_id}: the buffer scores are not the raw ones")
        scores = memory_table(root, fold_id, "buffer_scores", "scores")
        if set(scores["expert_id"].unique()) != set(spec["experts"]):
            raise AdaptationError(f"{fold_id}: buffer scores do not cover exactly {spec['experts']}")
        experts, reference = {}, []
        for expert in spec["experts"]:
            frame = scores.filter(pl.col("expert_id") == expert)
            values, ranks, abstain = reference_rows(frame)
            entry = fit(values, spec)
            experts[expert] = dict(entry, n_rows=frame.height, n_abstain=abstain, sanity=sanity(values, entry, spec))
            reference.append(pl.DataFrame({"expert_id": [expert] * values.size, "target_rank": ranks, "score_raw": values},
                                          schema={"expert_id": pl.String, "target_rank": pl.Int64,
                                                  "score_raw": pl.Float64}).sort("score_raw"))
        manifest = {"artifact_id": ARTIFACT, "fold_id": fold_id, "memory_key": memory["memory_key"],
                    "memory_sha256": memory["sha256"], "spec_sha256": spec["sha256"],
                    "builder_sha256": sha256_file(Path(__file__)), "rule": spec["rule"]["name"],
                    "anchors": spec["rule"]["anchors"], "experts": experts, "target_labels_read": False}
        write_artifact(root, ARTIFACT, fold_id, {"reference": pl.concat(reference)}, manifest)
        report[fold_id] = {e: {k: v[k] for k in ("status", "n_reference", "low", "high")} for e, v in experts.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="fit and freeze CAL-CS-001 from the frozen fold memories")
    build.add_argument("--root", required=True, help="folder holding data/processed/memory")
    build.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    print(json.dumps(build_calibration(args.root, args.repo_root), indent=2))


if __name__ == "__main__":
    main()
