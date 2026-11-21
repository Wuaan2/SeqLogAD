"""P2.6 diversity analysis and comparators on synthetic envelopes, labels and windows."""
import copy
import inspect
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from seqlogad.diversity import analysis, comparators
from seqlogad.diversity.units import load_config
from seqlogad.evidence.contract import EXPERT_ROLES, EvidenceError
from seqlogad.sequence.source_development_metrics import _binary_metrics

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT)
FOLD = "FOLD-TARGET-ARCH-BGL"
EXPERTS = list(CFG["experts"])


def _envelope(expert, arch, rank, score, fold=FOLD):
    record = f"{arch}:{rank}"
    abstain = score is None
    return {
        "schema_version": "EVID-CS-001", "expert_id": expert, "expert_role": EXPERT_ROLES[expert], "fold_id": fold,
        "architecture_id": arch, "unit_id": f"unit-{rank}", "unit_kind": "LOG_RECORD", "record_ids": [record],
        "target_record_id": record, "score_raw": score, "score_direction": "HIGHER_IS_MORE_ANOMALOUS",
        "score_calibrated": None, "calibration_status": "NOT_CALIBRATED", "uncertainty": None,
        "uncertainty_method": "NOT_ESTIMATED_IN_PHASE2", "coverage": 0.0 if abstain else 1.0,
        "evidence": [] if abstain else [{"kind": "nll", "value": score}], "localisation": [],
        "status": "ABSTAIN" if abstain else "VALID", "reason": "NO_CONTEXT" if abstain else None,
        "model_revision": "test", "checkpoint_sha256": "0" * 64, "input_manifest_sha256": "1" * 64,
        "context_config_sha256": "2" * 64, "expert_metadata": {"native_schema_version": "TEST"},
    }


def _windows(rows):
    """rows: (window_id, architecture_id, unit_id, target_rank)."""
    return pl.DataFrame(rows, orient="row", schema=["window_id", "architecture_id", "unit_id", "target_rank"]).with_columns(
        pl.concat_str([pl.col("architecture_id"), pl.lit(":"), pl.col("target_rank").cast(pl.Utf8)]).alias("target_record_id"))


def test_ap_auroc_far_on_a_known_case_and_tie_rule_matches_source_development():
    scores, labels = [0.9, 0.8, 0.7, 0.6], [True, False, True, False]
    assert analysis.average_precision(scores, labels) == pytest.approx((1 + 2 / 3) / 2)
    assert analysis.auroc(scores, labels) == pytest.approx(0.75)
    assert analysis.far_at_recall(scores, labels, 0.5) == 0.0
    assert analysis.far_at_recall(scores, labels, 0.8) == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    tied = rng.integers(0, 6, 200).astype(float)
    truth = rng.random(200) < 0.3
    reference = _binary_metrics(list(zip(tied.tolist(), truth.tolist())))
    assert analysis.average_precision(tied, truth) == pytest.approx(reference["average_precision"])
    assert analysis.auroc(tied, truth) == pytest.approx(reference["auroc"])
    assert math.isnan(analysis.average_precision([1.0, 2.0], [False, False]))


def test_block_aggregation_excludes_abstain_and_never_zero_fills():
    windows = _windows([("w1", "ARCH-HDFS", "b1", 1), ("w2", "ARCH-HDFS", "b1", 2), ("w3", "ARCH-HDFS", "b2", 3),
                        ("w4", "ARCH-HDFS", "b2", 4), ("w5", "ARCH-HDFS", "b3", 5), ("w6", "ARCH-HDFS", "b3", 6)])
    scores = {"w1": 1.0, "w2": None, "w3": None, "w4": None, "w5": 2.0, "w6": 4.0}
    assert analysis.unit_scores(windows, scores) == {"b1": 1.0, "b3": 3.0}  # b2 fully abstained: absent, not 0
    assert analysis.combine([{"b1": 0.2, "b3": 0.8}, {"b3": 0.4}]) == {"b1": 0.2, "b3": pytest.approx(0.6)}
    assert analysis.mid_rank_percentile({"a": 1.0, "b": 1.0, "c": 3.0}) == {"a": 1 / 3, "b": 1 / 3, "c": 5 / 6}


def test_evidence_loading_validates_envelopes_and_keeps_abstain(tmp_path):
    windows = _windows([("w1", "ARCH-BGL", "w1", 10), ("w2", "ARCH-BGL", "w2", 11)])
    path = tmp_path / "SEMANTIC_LLAMA.jsonl"
    path.write_text("\n".join(json.dumps(_envelope("SEMANTIC_LLAMA", "ARCH-BGL", r, s)) for r, s in ((10, 2.5), (11, None))))
    scores, stats = analysis.load_expert_scores(path, FOLD, "SEMANTIC_LLAMA", windows)
    assert scores == {"w1": 2.5, "w2": None} and stats["ABSTAIN"] == 1 and stats["VALID"] == 1
    with pytest.raises(EvidenceError, match="another expert"):
        analysis.load_expert_scores(path, FOLD, "GTAT_STRUCTURAL_TEMPORAL", windows)
    path.write_text(json.dumps(_envelope("SEMANTIC_LLAMA", "ARCH-BGL", 10, 2.5)))
    with pytest.raises(EvidenceError, match="no evidence"):
        analysis.load_expert_scores(path, FOLD, "SEMANTIC_LLAMA", windows)
    broken = _envelope("SEMANTIC_LLAMA", "ARCH-BGL", 10, None) | {"score_raw": 0.0}
    path.write_text(json.dumps(broken))
    with pytest.raises(EvidenceError, match="ABSTAIN must not carry a score"):
        analysis.load_expert_scores(path, FOLD, "SEMANTIC_LLAMA", windows)


def test_bootstrap_is_deterministic_and_paired():
    units = [f"u{i}" for i in range(40)]
    labels = {u: i % 3 == 0 for i, u in enumerate(units)}
    methods = {"a": {u: float(i % 7) for i, u in enumerate(units)}, "b": {u: float(i % 5) for i, u in enumerate(units[:30])}}
    first = analysis.bootstrap_ap(units, labels, methods, 300, 42)
    again = analysis.bootstrap_ap(units, labels, methods, 300, 42)
    other = analysis.bootstrap_ap(units, labels, methods, 300, 7)
    assert all(np.array_equal(first[m], again[m], equal_nan=True) for m in methods)
    assert not np.array_equal(first["a"], other["a"], equal_nan=True)
    same = analysis.bootstrap_ap(units, labels, {"a": methods["a"], "copy": dict(methods["a"])}, 100, 42)
    assert np.array_equal(same["a"], same["copy"])  # one resample shared by every method


def test_verdict_and_g2_rules():
    assert analysis.verdict(0.2, 0.01, 0.5, 19, CFG) == "INSUFFICIENT_POWER"
    assert analysis.verdict(0.05, 0.01, 0.5, 20, CFG) == "ADDS_EVIDENCE"  # margin = 10% of 0.5
    assert analysis.verdict(0.049, 0.01, 0.5, 20, CFG) == "NOT_SHOWN"
    assert analysis.verdict(0.2, 0.0, 0.5, 20, CFG) == "NOT_SHOWN"  # lower bound must be strictly above 0
    adds, no = "ADDS_EVIDENCE", "NOT_SHOWN"
    assert analysis.g2_recommendation({"F1": {"A": adds, "B": no}, "F2": {"A": adds}}) == "PASS"
    assert analysis.g2_recommendation({"F1": {"A": adds}, "F2": {"A": no, "B": adds}}) == "PARTIAL"
    assert analysis.g2_recommendation({"F1": {"A": no}, "F2": {"B": "INSUFFICIENT_POWER"}}) == "NOT_SHOWN"


def test_fold_analysis_end_to_end_on_synthetic_scores():
    rng = np.random.default_rng(1)
    rows, labels = [], {}
    for block in range(80):
        positive = block < 25
        for k in range(2):
            window = f"h{block}-{k}"
            rows.append((window, "ARCH-HDFS", f"blk{block}", 2 * block + k))
            labels[window] = {"architecture_id": "ARCH-HDFS", "window_id": window, "unit_id": f"blk{block}",
                              "anomaly": positive, "target_rank": 2 * block + k}
    rows += [(f"o{i}", "ARCH-OPENSTACK", f"o{i}", i) for i in range(30)]
    windows = _windows(rows)
    truth = np.array([labels[w]["anomaly"] if w in labels else False for w in windows["window_id"]], float)
    signal = {"SEMANTIC_LLAMA": 2.0, "SEQUENCE_LLAMA_REFERENCE": 0.0, "SEQUENCE_LIGHTWEIGHT": 0.3, "GTAT_STRUCTURAL_TEMPORAL": 1.5}
    scores = {e: dict(zip(windows["window_id"], (s * truth + rng.normal(size=len(truth))).tolist())) for e, s in signal.items()}
    scores["GTAT_STRUCTURAL_TEMPORAL"]["h0-0"] = None  # ABSTAIN: block blk0 keeps its other window
    comparator_scores = {c: dict(zip(windows["window_id"], rng.normal(size=len(truth)).tolist())) for c in comparators.COMPARATORS}
    cfg = copy.deepcopy(CFG)
    cfg["uncertainty"]["replicates"] = 200
    result = analysis.fold_analysis(cfg, windows, labels, scores, comparator_scores)
    block = result["labeled"]["ARCH-HDFS"]
    assert set(result["labeled"]) == {"ARCH-HDFS"} and set(result["label_free"]) == {"ARCH-HDFS", "ARCH-OPENSTACK"}
    assert block["units"] == 80 and block["positive_units"] == 25 and block["best_single"] == "SEMANTIC_LLAMA"
    assert block["verdicts"]["SEMANTIC_LLAMA"] == "BEST_SINGLE"
    assert set(block["verdicts"].values()) <= {"BEST_SINGLE", "ADDS_EVIDENCE", "NOT_SHOWN"}
    assert set(block["combinations"]) == {f"SEMANTIC_LLAMA+{e}" for e in EXPERTS[1:]} | {"ALL_FOUR"}
    assert result["label_free"]["ARCH-HDFS"]["coverage"]["GTAT_STRUCTURAL_TEMPORAL"]["abstain_windows"] == 1
    assert block["singles"]["GTAT_STRUCTURAL_TEMPORAL"]["coverage"] == 1.0
    assert block["p2_3_minus_p2_2"]["delta_ci"]["valid_replicates"] == 200
    assert set(block["comparators"]) == set(comparators.COMPARATORS)
    only = {"GTAT_STRUCTURAL_TEMPORAL": scores["GTAT_STRUCTURAL_TEMPORAL"]}  # partial check: one expert
    single = analysis.fold_analysis(cfg, windows, labels, only, comparator_scores)["labeled"]["ARCH-HDFS"]
    assert single["best_single"] == "GTAT_STRUCTURAL_TEMPORAL" and single["combinations"] == {}
    assert single["p2_3_minus_p2_2"] is None and single["pairs"] == {}


def test_comparators_are_finite_label_free_and_rank_rare_messages_higher():
    def frame(rows, partition):
        return pl.DataFrame(rows, orient="row", schema={"rank": pl.Int64, "normalized_message": pl.String}).with_columns(
            pl.lit(partition).alias("partition"))
    train_events = frame([(i, "common" if i % 4 else "rare") for i in range(40)], "SOURCE_TRAIN")
    train_windows = pl.DataFrame({"window_id": [f"t{i}" for i in range(2, 40)], "architecture_id": "ARCH-HDFS",
                                  "target_rank": list(range(2, 40)), "context_ranks": [[i - 2, i - 1] for i in range(2, 40)]})
    eval_events = frame([(100, "common"), (101, "common"), (102, "never seen"), (103, "common")], "SOURCE_VALIDATION")
    eval_windows = pl.DataFrame({"window_id": ["e1", "e2"], "architecture_id": "ARCH-HDFS",
                                 "target_rank": [101, 102], "context_ranks": [[100], [103]]})
    isolation = CFG["comparators"]["isolation_forest"]
    out = comparators.comparator_scores(train_windows, train_events, eval_windows, eval_events, isolation)
    assert list(out.columns) == comparators.COLUMNS and out.height == 2 * len(comparators.COMPARATORS)
    assert out["score"].is_finite().all()
    by = {(w, c): s for w, _, c, s in out.iter_rows()}
    assert by[("e2", "unseen_rarity")] > by[("e1", "unseen_rarity")]
    assert by[("e2", "ngram_markov")] > by[("e1", "ngram_markov")]
    again = comparators.comparator_scores(train_windows, train_events, eval_windows, eval_events, isolation)
    assert out.equals(again)  # random_state fixed
    assert list(inspect.signature(comparators.score_comparators).parameters) == ["data_root", "fold_id"]
    assert not any("label" in p for p in inspect.signature(comparators.comparator_scores).parameters)
