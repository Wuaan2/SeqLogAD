"""CAL-CS-001 (P3.3) and DRIFT-CS-001 (P3.4): preregistered rules, label-free reference, frozen-memory input."""
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from seqlogad.adaptation import calibration, drift
from seqlogad.adaptation.store import ADAPT_ROOT, AdaptationError, load_artifact
from seqlogad.memory.build import memory_key, write_component
from seqlogad.memory.prototypes import freeze_memory
from seqlogad.semantic.prepare import hash64

ROOT = Path(__file__).resolve().parents[1]
FOLD, ARCH = "FOLD-TARGET-ARCH-BGL", "ARCH-BGL"
EXPERTS = ["SEMANTIC_LLAMA", "SEQUENCE_LLAMA_REFERENCE", "SEQUENCE_LIGHTWEIGHT", "GTAT_STRUCTURAL_TEMPORAL"]
CAL_SPEC, DRIFT_SPEC = calibration.load_spec(ROOT), drift.load_spec(ROOT)


def _manifest():
    return {"artifact_id": "ADAPT-MEM-001", "fold_id": FOLD, "target_architecture": ARCH, "buffer_id": "BUFFER-ARCH-BGL",
            "buffer_membership_sha256": "ab", "memory_key": memory_key(ARCH, "ab"), "partition": "TARGET_BURN_IN"}


def _memory(root, windows=64, scores_per_expert=1200, freeze=True):
    """A tiny fold memory in the ADAPT-MEM-001 layout: 8-record windows over messages m0..m9."""
    ranks = list(range(windows + 8))
    message = {r: f"m{r % 10}" for r in ranks}
    events = pl.DataFrame({"architecture_id": ARCH, "partition": "TARGET_BURN_IN", "rank": ranks,
                           "normalized_message": [message[r] for r in ranks]})
    unigrams = (pl.DataFrame({"norm_h": [hash64(message[r].encode()) for r in ranks]})
                .group_by("norm_h").agg(pl.len().cast(pl.UInt32).alias("count")))
    targets = ranks[8:]
    window_ids = [f"w{t}" for t in targets]
    context = [list(range(t - 8, t)) for t in targets]
    frame = {"window_id": window_ids, "target_rank": targets, "context_ranks": context}
    nodes = pl.DataFrame({"rank": ranks, "component_text": ["kernel"] * len(ranks), "severity": [0] * len(ranks),
                          "location": [f"R0{r % 3}" for r in ranks], "request_group": [None] * len(ranks)},
                         schema_overrides={"severity": pl.Int8, "request_group": pl.String})
    attributes = pl.concat([nodes.group_by(pl.col(f).cast(pl.String).alias("value")).agg(pl.len().alias("count"))
                            .with_columns(pl.lit(f).alias("field")) for f in ("component_text", "severity", "location",
                                                                               "request_group")]).select("field", "value", "count")
    write_component(root, FOLD, "statistics", {
        "unigrams": unigrams, "events": events, "windows": pl.DataFrame(frame),
        "graphs": pl.DataFrame(dict(frame, status=["SCORED"] * windows)), "nodes": nodes, "node_attributes": attributes},
        _manifest())
    rng = np.random.default_rng(0)
    n = scores_per_expert
    rows = pl.DataFrame({
        "expert_id": [e for e in EXPERTS for _ in range(n)],
        "target_record_id": [f"{ARCH}:{r}" for _ in EXPERTS for r in range(n)],
        "score_raw": rng.normal(size=len(EXPERTS) * n).tolist(),
        "status": ["VALID"] * (len(EXPERTS) * n)})
    rows = rows.with_columns(  # two ABSTAIN rows per expert, with no score
        pl.when(pl.col("target_record_id").is_in([f"{ARCH}:0", f"{ARCH}:1"])).then(pl.lit("ABSTAIN"))
        .otherwise(pl.col("status")).alias("status"),
        pl.when(pl.col("target_record_id").is_in([f"{ARCH}:0", f"{ARCH}:1"])).then(None)
        .otherwise(pl.col("score_raw")).alias("score_raw"))
    write_component(root, FOLD, "buffer_scores", {"scores": rows}, dict(_manifest(), calibrated=False))
    write_component(root, FOLD, "prototypes", {
        "centroids": pl.DataFrame({"expert_id": EXPERTS, "cluster": [0] * 4}),
        "assignments": pl.DataFrame({"expert_id": [e for e in EXPERTS for _ in range(10)],
                                     "cosine_distance": [i / 10 for _ in EXPERTS for i in range(10)]})}, _manifest())
    if freeze:
        freeze_memory(root, FOLD)


# ---------------------------------------------------------------- P3.3 / P3.4 opening

def test_specs_are_preregistered_on_the_four_experts_and_both_target_folds():
    assert CAL_SPEC["experts"] == EXPERTS and CAL_SPEC["folds"] == DRIFT_SPEC["folds"]
    assert set(CAL_SPEC["folds"]) == {"FOLD-TARGET-ARCH-HDFS", "FOLD-TARGET-ARCH-BGL"}  # EXC-007
    assert CAL_SPEC["rule"]["anchors"] == {"low": 0.5, "high": 0.99} and DRIFT_SPEC["block_drift"]["block"] == 512
    assert DRIFT_SPEC["roadmap_update"] == "configs/protocols/cal-cs-v1.yaml#cal_cs.roadmap_update"
    for spec in (CAL_SPEC, DRIFT_SPEC):
        assert spec["authorized_by"] == "RESEARCHER" and any("label" in rule for rule in spec["forbidden"])


def test_p3_3_p3_4_status_update_has_exact_provenance():
    import openpyxl
    import yaml

    from seqlogad.common.checksum import sha256_file

    update = CAL_SPEC["roadmap_update"]
    p32 = yaml.safe_load((ROOT / "configs/protocols/adapt-mem-v1.yaml").read_text(encoding="utf-8"))["adapt_mem"]
    before_path, after_path = ROOT / update["previous_workbook"], ROOT / "Bang_ke_hoach_SeqLogAD.xlsx"
    assert sha256_file(before_path) == update["from_sha256"] == p32["roadmap_update"]["to_sha256"]
    assert sha256_file(after_path) == update["to_sha256"]
    books = [openpyxl.load_workbook(p) for p in (before_path, after_path)]
    before, after = [{f"{s.title}!{c.coordinate}": c.value for s in b for row in s for c in row if c.value is not None}
                     for b in books]
    changed = {k: {"before": before.get(k), "after": after.get(k)}
               for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changed == update["changed_cells"] and update["gate_signature_created"] is False
    for sheet in books[0]:
        assert str(sheet.data_validations) == str(books[1][sheet.title].data_validations)


# ---------------------------------------------------------------- CAL-CS-001

def test_scaling_puts_median_at_zero_and_q99_at_one_and_keeps_the_ranking():
    scores = np.random.default_rng(1).gamma(2.0, 3.0, size=8000)
    entry = calibration.fit(scores, CAL_SPEC)
    assert entry["status"] == "CALIBRATED" and entry["n_reference"] == 8000
    calibrated = np.asarray(calibration.calibrate(entry, scores.tolist()))
    assert abs(np.quantile(calibrated, 0.5)) < 1e-9 and abs(np.quantile(calibrated, 0.99) - 1) < 1e-9
    checks = calibration.sanity(scores, entry, CAL_SPEC)
    assert checks["anchors_reproduced"] and checks["rank_preserved"] and checks["temporal_split"]["state"] == "STABLE"
    assert sum(checks["histogram"]["counts"]) == 8000
    assert calibration.calibrate(entry, [None, entry["high"]]) == [None, 1.0]


def test_uncalibratable_experts_get_no_score_and_no_fallback():
    assert calibration.fit(np.ones(2000), CAL_SPEC)["status"] == "DEGENERATE_SCALE"
    small = calibration.fit(np.arange(999.0), CAL_SPEC)
    assert small["status"] == "INSUFFICIENT_REFERENCE" and small["low"] is None
    assert calibration.calibrate(small, [1.0]) == [None] and calibration.sanity(np.arange(999.0), small, CAL_SPEC) is None


def test_tail_p_counts_ties_as_at_least_as_extreme():
    assert calibration.tail_p([1, 2, 2, 3], [2, 3.5, 0]).tolist() == [0.8, 0.2, 1.0]


def test_temporal_split_flags_a_second_half_that_moved():
    rng = np.random.default_rng(2)
    moved = np.concatenate([rng.normal(0, 1, 4000), rng.normal(3, 1, 4000)])
    assert calibration.sanity(moved, calibration.fit(moved, CAL_SPEC), CAL_SPEC)["temporal_split"]["state"] == "UNSTABLE"


def test_calibration_reads_only_a_frozen_memory_and_is_written_once(tmp_path):
    _memory(tmp_path, freeze=False)
    with pytest.raises(AdaptationError, match="not frozen"):
        calibration.build_calibration(tmp_path, ROOT, [FOLD])
    freeze_memory(tmp_path, FOLD)
    report = calibration.build_calibration(tmp_path, ROOT, [FOLD])
    assert {e["status"] for e in report[FOLD].values()} == {"CALIBRATED"}
    manifest = load_artifact(tmp_path, "CAL-CS-001", FOLD)
    assert manifest["memory_key"] == memory_key(ARCH, "ab") and manifest["target_labels_read"] is False
    assert all(e["n_abstain"] == 2 and e["n_reference"] == 1198 for e in manifest["experts"].values())
    out = tmp_path / ADAPT_ROOT / "CAL-CS-001" / FOLD
    assert not os.access(out / "reference.parquet", os.W_OK)
    with pytest.raises(AdaptationError, match="written once"):
        calibration.build_calibration(tmp_path, ROOT, [FOLD])


# ---------------------------------------------------------------- DRIFT-CS-001

def test_jsd_is_zero_on_the_reference_and_one_on_unseen_events():
    reference = {1: 30, 2: 10}
    assert abs(drift.jsd({1: 3, 2: 1}, reference)) < 1e-12
    assert abs(drift.jsd({7: 5}, reference) - 1.0) < 1e-12
    assert abs(drift.jsd({1: 5}, {1: 0, 2: 10}) - 1.0) < 1e-12  # a left-out count of 0 is absent


def test_window_coverage_measures():
    unigrams, transitions = {1: 4, 2: 1, 3: 0}, {(1, 2)}
    assert drift.event_coverage([1, 2, 3, 9], unigrams) == 0.5
    assert drift.transition_coverage([1, 2, 1], transitions) == 0.5 and drift.transition_coverage([1], transitions) is None
    nodes = [{"component_text": "kernel", "severity": 0, "location": "R00"}, {"component_text": "app", "severity": 0, "location": None}]
    attributes = {("component_text", "kernel"): 3, ("severity", "0"): 3, ("location", "R00"): 1, ("location", None): 1}
    assert drift.topology_coverage(nodes, attributes, DRIFT_SPEC["topology_fields"]) == 5 / 6


def test_block_state_is_review_only_above_a_reference_threshold():
    thresholds = {"event_jsd": 0.3, "unseen_event_rate": 0.0, "unseen_topology_rate": 0.1}
    assert drift.state({"event_jsd": 0.3, "unseen_event_rate": 0.0, "unseen_topology_rate": 0.1}, thresholds) == "IN_REFERENCE"
    assert drift.state({"event_jsd": 0.1, "unseen_event_rate": 0.01, "unseen_topology_rate": 0.0}, thresholds) == "DRIFT_REVIEW"
    unseen = drift.block_statistics([99, 1], [99, 1], [("location", "R99")], {1: 5}, {("location", "R00"): 5})
    assert unseen["unseen_event_rate"] == 0.5 and unseen["unseen_topology_rate"] == 1.0


def test_prototype_distance_p_uses_the_buffer_distances():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert abs(drift.nearest_distance(centroids, [2.0, 0.0])) < 1e-12
    assert drift.prototype_p([0.1, 0.2, 0.3], [0.25]).tolist() == [0.5]


def test_drift_reference_leaves_each_block_out_and_is_frozen(tmp_path, monkeypatch):
    _memory(tmp_path)
    spec = drift.load_spec(ROOT)
    monkeypatch.setattr(drift, "load_spec", lambda _: dict(spec, block_drift=dict(spec["block_drift"], block=16)))
    source = tmp_path / "evidence" / FOLD
    source.mkdir(parents=True)
    for expert in EXPERTS:
        lines = [json.dumps({"score_raw": float(i), "status": "VALID"}) for i in range(50)]
        lines.append(json.dumps({"score_raw": None, "status": "ABSTAIN"}))
        (source / f"{expert}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = drift.build_drift_reference(tmp_path, ROOT, tmp_path / "evidence", [FOLD])
    assert report[FOLD]["reference_blocks"] == 4  # 64 windows / 16
    manifest = load_artifact(tmp_path, "DRIFT-CS-001", FOLD)
    assert set(manifest["thresholds"]) == set(drift.STATISTICS) and manifest["target_labels_read"] is False
    assert manifest["thresholds"]["unseen_event_rate"] == 0.0  # every message recurs outside each block
    assert set(manifest["source_to_target_gap"]) == set(EXPERTS)
    assert manifest["source_to_target_gap"][EXPERTS[0]]["n_source"] == 50
    assert manifest["prototype_distance"][EXPERTS[0]]["n"] == 10
    with pytest.raises(AdaptationError, match="written once"):
        drift.build_drift_reference(tmp_path, ROOT, tmp_path / "evidence", [FOLD])
