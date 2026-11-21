"""P2.6 diversity analysis DIV-P2-001: per-expert metrics, pairs, equal-weight combinations, bootstrap, verdicts.

Units: HDFS block (mean of its scored windows), every other source one window. An ABSTAIN window is
excluded for that expert and counted in coverage; it is never zero-filled. Labels come only from the
source-development sidecar and only for the fold's labeled source (C2 amendment, option A).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

from seqlogad.common.checksum import sha256_file
from seqlogad.evidence.contract import EvidenceError, index_by_target
from seqlogad.sequence.source_development import load_source_development

from .comparators import COLUMNS as COMPARATOR_COLUMNS, COMPARATORS
from .units import CONFIG, evaluation_windows, load_config

RECALLS = (0.5, 0.8)
P2_3, P2_2 = "SEQUENCE_LIGHTWEIGHT", "SEQUENCE_LLAMA_REFERENCE"

# ----------------------------------------------------------------- metrics


def _groups(scores, labels):
    """Cumulative (tp, n) at the end of each tie group, scores descending."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if not len(s):
        return np.zeros(0), np.zeros(0), 0, 0
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last = np.r_[s[1:] != s[:-1], True]
    return np.cumsum(y)[last], np.arange(1, len(s) + 1)[last], int(y.sum()), len(y)


def average_precision(scores, labels) -> float:
    """Non-interpolated AP with tied scores taken as one threshold (as source_development_metrics)."""
    tp, n, positives, total = _groups(scores, labels)
    if positives == 0 or positives == total:
        return math.nan
    return float(np.sum(tp / n * np.diff(np.r_[0, tp])) / positives)


def auroc(scores, labels) -> float:
    y = np.asarray(labels, bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        return math.nan
    ranks = rankdata(np.asarray(scores, float))  # mid-ranks = half credit for ties
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def far_at_recall(scores, labels, recall: float) -> float:
    tp, n, positives, total = _groups(scores, labels)
    if positives == 0 or positives == total:
        return math.nan
    g = int(np.argmax(tp / positives >= recall))
    return float((n[g] - tp[g]) / (total - positives))


def binary_metrics(scores, labels) -> dict:
    y = np.asarray(labels, bool)
    return {"average_precision": average_precision(scores, labels), "auroc": auroc(scores, labels),
            **{f"far_at_recall_{r}": far_at_recall(scores, labels, r) for r in RECALLS},
            "units": int(len(y)), "positives": int(y.sum())}


# ----------------------------------------------------------------- units and combinations


def unit_scores(windows: pl.DataFrame, window_scores: dict) -> dict[str, float]:
    """Mean of the scored windows of each unit; a unit with no scored window is absent."""
    grouped = defaultdict(list)
    for window_id, unit in windows.select("window_id", "unit_id").iter_rows():
        value = window_scores[window_id]
        if value is not None:
            grouped[unit].append(value)
    return {unit: float(np.mean(values)) for unit, values in grouped.items()}


def mid_rank_percentile(scores: dict[str, float]) -> dict[str, float]:
    keys = sorted(scores)
    if not keys:
        return {}
    ranks = rankdata([scores[k] for k in keys])
    return {k: float((r - 0.5) / len(keys)) for k, r in zip(keys, ranks)}


def combine(percentiles: list[dict[str, float]]) -> dict[str, float]:
    """Equal-weight mean over the experts that scored the unit; no learned weights."""
    pooled = defaultdict(list)
    for table in percentiles:
        for unit, value in table.items():
            pooled[unit].append(value)
    return {unit: float(np.mean(values)) for unit, values in pooled.items()}


def _aligned(units: list[str], labels: dict[str, bool], scores: dict[str, float]):
    kept = [u for u in units if u in scores]
    return [scores[u] for u in kept], [labels[u] for u in kept]


def bootstrap_ap(units: list[str], labels: dict[str, bool], methods: dict[str, dict], replicates: int,
                 seed: int) -> dict[str, np.ndarray]:
    """Paired bootstrap over labeled units: the same resample for every method; NaN if a class is absent."""
    rng = np.random.default_rng(seed)
    y = np.array([labels[u] for u in units], bool)
    arrays = {m: np.array([s.get(u, math.nan) for u in units], float) for m, s in methods.items()}
    out = {m: np.empty(replicates) for m in methods}
    for b in range(replicates):
        idx = rng.integers(0, len(units), len(units))
        for m, values in arrays.items():
            v = values[idx]
            keep = ~np.isnan(v)
            out[m][b] = average_precision(v[keep], y[idx][keep])
    return out


def interval(samples: np.ndarray) -> dict:
    valid = samples[~np.isnan(samples)]
    if not len(valid):
        return {"lower": math.nan, "upper": math.nan, "valid_replicates": 0}
    lower, upper = np.percentile(valid, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper), "valid_replicates": int(len(valid))}


def verdict(delta: float, lower: float, best_ap: float, positives: int, cfg: dict) -> str:
    if positives < int(cfg["evaluation_set"]["minimum_positive_units"]):
        return "INSUFFICIENT_POWER"
    margin = float(cfg["practical_margin"]["delta_ap_relative_to_best_single"]) * best_ap
    return "ADDS_EVIDENCE" if delta >= margin and lower > 0 else "NOT_SHOWN"


def g2_recommendation(verdicts: dict[str, dict[str, str]]) -> str:
    """PASS: one non-best expert adds evidence in every fold; PARTIAL: some expert in some fold."""
    adds = [{e for e, v in per_fold.items() if v == "ADDS_EVIDENCE"} for per_fold in verdicts.values()]
    if adds and set.intersection(*adds):
        return "PASS"
    return "PARTIAL" if any(adds) else "NOT_SHOWN"


def _alarms(scores: dict[str, float], fraction: float) -> set[str]:
    k = math.ceil(fraction * len(scores))
    return {u for u, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]}


def _spearman(a: dict, b: dict) -> float | None:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    rho = spearmanr([a[u] for u in common], [b[u] for u in common]).statistic
    return None if math.isnan(rho) else float(rho)


# ----------------------------------------------------------------- fold analysis


def fold_analysis(cfg: dict, windows: pl.DataFrame, labels: dict[str, dict], expert_scores: dict[str, dict],
                  comparator_scores: dict[str, dict]) -> dict:
    """All DIV-P2-001 numbers for one fold. ``labels`` maps window_id to its sidecar entry."""
    budget = float(cfg["metrics"]["alarm_budget_fraction"])
    boot = cfg["uncertainty"]
    experts = [e for e in cfg["experts"] if e in expert_scores]
    result = {"label_free": {}, "labeled": {}}
    for arch in sorted(windows["architecture_id"].unique()):
        source = windows.filter(pl.col("architecture_id") == arch)
        units = {e: unit_scores(source, s) for e, s in expert_scores.items()}
        n_units = source["unit_id"].n_unique()
        result["label_free"][arch] = {
            "windows": source.height, "units": n_units,
            "coverage": {e: {"scored_windows": sum(s[w] is not None for w in source["window_id"]),
                             "abstain_windows": sum(s[w] is None for w in source["window_id"]),
                             "unit_coverage": len(units[e]) / n_units} for e, s in expert_scores.items()},
            "pairs": {f"{a}|{b}": {"spearman": _spearman(units[a], units[b]),
                                   "top5_overlap": _overlap(units[a], units[b], budget)}
                      for a, b in combinations(experts, 2)},
        }
        labeled = [w for w in source["window_id"] if w in labels]
        if not labeled:
            continue
        if len(labeled) != source.height:
            raise ValueError(f"{arch}: labeled source has unlabeled evaluation windows")
        unit_label = {}
        for window_id, unit in source.select("window_id", "unit_id").iter_rows():
            entry = labels[window_id]
            if entry["unit_id"] != unit or unit_label.setdefault(unit, entry["anomaly"]) != entry["anomaly"]:
                raise ValueError(f"{arch}: sidecar unit/label mismatch")
        result["labeled"][arch] = _labeled(cfg, experts, units, unit_label, comparator_scores, source, budget, boot)
    return result


def _overlap(a: dict, b: dict, fraction: float) -> float | None:
    if not a or not b:
        return None
    top_a, top_b = _alarms(a, fraction), _alarms(b, fraction)
    return len(top_a & top_b) / min(len(top_a), len(top_b))


def _labeled(cfg, experts, units, unit_label, comparator_scores, source, budget, boot) -> dict:
    order = sorted(unit_label)
    positives = sum(unit_label.values())
    singles = {e: dict(binary_metrics(*_aligned(order, unit_label, units[e])), coverage=len(units[e]) / len(order))
               for e in experts}
    best = max(experts, key=lambda e: (np.nan_to_num(singles[e]["average_precision"], nan=-1.0), -experts.index(e)))
    pct = {e: mid_rank_percentile(units[e]) for e in experts}
    methods = {e: units[e] for e in experts}
    methods.update({f"{best}+{e}": combine([pct[best], pct[e]]) for e in experts if e != best})
    if len(experts) > 1:
        methods["ALL_FOUR" if len(experts) == len(cfg["experts"]) else "ALL_AVAILABLE"] = combine([pct[e] for e in experts])
    reps = bootstrap_ap(order, unit_label, methods, int(boot["replicates"]), int(boot["seed"]))
    best_ap = singles[best]["average_precision"]
    combos, verdicts = {}, {}
    for name, scores in methods.items():
        if name in experts:
            continue
        point = binary_metrics(*_aligned(order, unit_label, scores))
        delta = point["average_precision"] - best_ap
        ci = interval(reps[name] - reps[best])
        combos[name] = dict(point, coverage=len(scores) / len(order), delta_ap_vs_best=delta, delta_ci=ci)
        if "+" in name:
            expert = name.split("+", 1)[1]
            verdicts[expert] = verdict(delta, ci["lower"], best_ap, positives, cfg)
            combos[name]["verdict"] = verdicts[expert]
    verdicts[best] = "BEST_SINGLE"
    p23 = ({"delta_ap": singles[P2_3]["average_precision"] - singles[P2_2]["average_precision"],
            "delta_ci": interval(reps[P2_3] - reps[P2_2])} if {P2_3, P2_2} <= set(experts) else None)
    alarms = {e: _alarms(units[e], budget) for e in experts}
    pairs = {}
    for a, b in combinations(experts, 2):
        both = set(units[a]) & set(units[b])
        pos = [u for u in both if unit_label[u]]
        pairs[f"{a}|{b}"] = {
            "spearman": _spearman(units[a], units[b]), "units_scored_by_both": len(both),
            "double_fault_rate": (sum(u not in alarms[a] and u not in alarms[b] for u in pos) / len(pos)) if pos else None,
            "disagreement": (sum((u in alarms[a]) != (u in alarms[b]) for u in both) / len(both)) if both else None,
        }
    comparators = {}
    for name in COMPARATORS:
        scores = unit_scores(source, comparator_scores[name])
        comparators[name] = binary_metrics(*_aligned(order, unit_label, scores))
    beats = {e: {c: bool(singles[e]["average_precision"] > comparators[c]["average_precision"]) for c in COMPARATORS}
             for e in experts}
    return {"units": len(order), "positive_units": positives, "best_single": best,
            "insufficient_power": positives < int(cfg["evaluation_set"]["minimum_positive_units"]),
            "singles": singles, "combinations": combos, "verdicts": verdicts, "p2_3_minus_p2_2": p23,
            "pairs": pairs, "comparators": comparators, "expert_beats_comparator": beats}


# ----------------------------------------------------------------- inputs


def load_expert_scores(path: Path, fold_id: str, expert_id: str, windows: pl.DataFrame) -> tuple[dict, dict]:
    """Validated EVID-CS-001 envelopes joined to windows on (fold_id, target_record_id); None = ABSTAIN."""
    envelopes = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    index = index_by_target(envelopes)
    if any(e["expert_id"] != expert_id or e["fold_id"] != fold_id for e in index.values()):
        raise EvidenceError(f"{path}: envelope from another expert or fold")
    scores, statuses = {}, defaultdict(int)
    for window_id, arch, record in windows.select("window_id", "architecture_id", "target_record_id").iter_rows():
        envelope = index.get((fold_id, record))
        if envelope is None or envelope["architecture_id"] != arch:
            raise EvidenceError(f"{expert_id}: no evidence for evaluation window {window_id} ({record})")
        statuses[envelope["status"]] += 1
        scores[window_id] = None if envelope["status"] == "ABSTAIN" else float(envelope["score_raw"])
    return scores, {"envelopes": len(index), "unmatched_envelopes": len(index) - windows.height, **statuses}


def load_comparator_scores(path: Path, windows: pl.DataFrame) -> dict[str, dict]:
    frame = pl.read_parquet(path)
    if list(frame.columns) != COMPARATOR_COLUMNS or not frame["score"].is_finite().all():
        raise ValueError("comparator table contract violation")
    out = {}
    for name in COMPARATORS:
        part = frame.filter(pl.col("comparator") == name)
        out[name] = dict(part.select("window_id", "score").iter_rows())
        if set(out[name]) != set(windows["window_id"]) or part.height != windows.height:
            raise ValueError(f"comparator {name} does not cover the evaluation windows exactly")
    return out


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def analyze(repo_root, data_root, sidecar_root, evidence_root, experts=None) -> dict:
    """DIV-P2-001 report. ``experts`` (opt-in subset, for partial checks) yields no G2 recommendation."""
    cfg = load_config(repo_root)
    evidence = Path(evidence_root)
    chosen = list(cfg["experts"]) if experts is None else [e for e in cfg["experts"] if e in experts]
    if not chosen or (experts is not None and set(experts) - set(cfg["experts"])):
        raise ValueError(f"unknown or empty expert subset: {experts}")
    complete = chosen == list(cfg["experts"])
    report = {"id": cfg["id"], "version": cfg["version"], "config_sha256": sha256_file(Path(repo_root) / CONFIG),
              "sequence_bundle_sha256": sha256_file(Path(data_root) / "manifests/bundle.json"),
              "sidecar_manifest_sha256": sha256_file(Path(sidecar_root) / "manifest.json"),
              "label_scope": {"authority": cfg["label_scope"], "labeled_partition": "SOURCE_VALIDATION",
                              "labeled_sources_per_fold": {}, "target_labels_read": False},
              "inputs": {}, "folds": {}}
    verdicts = {}
    for fold_id in cfg["folds"]:
        windows = evaluation_windows(data_root, fold_id)
        labels = load_source_development(sidecar_root, data_root, fold_id, windows.to_dicts())
        scores, inputs = {}, {}
        for expert in chosen:
            path = evidence / fold_id / f"{expert}.jsonl"
            scores[expert], inputs[expert] = load_expert_scores(path, fold_id, expert, windows)
            inputs[expert]["sha256"] = sha256_file(path)
        comparator_path = evidence / fold_id / "comparators.parquet"
        comparators = load_comparator_scores(comparator_path, windows)
        inputs["comparators.parquet"] = {"sha256": sha256_file(comparator_path)}
        fold = fold_analysis(cfg, windows, labels, scores, comparators)
        report["inputs"][fold_id] = inputs
        report["folds"][fold_id] = fold
        report["label_scope"]["labeled_sources_per_fold"][fold_id] = sorted(fold["labeled"])
        verdicts[fold_id] = {e: v for arch in fold["labeled"].values() for e, v in arch["verdicts"].items()}
    report["experts_analyzed"], report["complete"] = chosen, complete
    report["g2_recommendation"] = {"verdict": g2_recommendation(verdicts) if complete else "INCOMPLETE_EVIDENCE",
                                   "signed_by": cfg["g2_recommendation"]["signed_by"],
                                   "status": "RECOMMENDED_NOT_SIGNED", "per_fold_verdicts": verdicts}
    return _clean(report)
