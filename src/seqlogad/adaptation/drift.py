"""DRIFT-CS-001 (P3.4): domain drift, buffer coverage and GTAT topology coverage against the frozen target memory.

Spec: configs/protocols/drift-cs-v1.yaml. The reference is the frozen ADAPT-MEM-001 fold memory. The block
thresholds come from the memory's own 32,768 statistics windows: 64 chronological blocks, each scored against the
buffer with its own records left out. DRIFT_REVIEW is an output for review; nothing is recalibrated or refitted.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from seqlogad.adaptation.calibration import SCORED, tail_p
from seqlogad.adaptation.store import AdaptationError, load_frozen_memory, memory_table, write_artifact
from seqlogad.adaptation.store import load_spec as _load_spec
from seqlogad.common.checksum import sha256_file
from seqlogad.semantic.prepare import hash64

SPEC = "configs/protocols/drift-cs-v1.yaml"
ARTIFACT = "DRIFT-CS-001"
STATISTICS = ("event_jsd", "unseen_event_rate", "unseen_topology_rate")


def load_spec(repo_root):
    spec = _load_spec(repo_root, SPEC, "drift_cs")
    blocks = spec["block_drift"]
    if blocks["block"] != 512 or blocks["threshold"] != "MAX_OF_REFERENCE_BLOCKS" or tuple(blocks["statistics"]) != STATISTICS:
        raise AdaptationError("DRIFT-CS-001 rule changed")
    return spec


def _value(value):
    return None if value is None else str(value)  # node_attributes stores every value as a string (or null)


def jsd(block_counts, reference_counts):
    """Jensen-Shannon divergence in bits; block events absent from the reference (count <= 0) share one UNSEEN bin."""
    keys = [e for e, c in reference_counts.items() if c > 0]
    q = np.array([reference_counts[e] for e in keys], dtype=np.float64)
    n = sum(block_counts.values())
    p = np.array([block_counts.get(e, 0) for e in keys], dtype=np.float64)
    p, q = np.append(p, n - p.sum()) / n, np.append(q / q.sum(), 0.0)

    def entropy(x):
        x = x[x > 0]
        return float(-(x * np.log2(x)).sum())

    return entropy((p + q) / 2) - (entropy(p) + entropy(q)) / 2


def event_coverage(norm_hs, unigram_counts):
    return sum(unigram_counts.get(h, 0) > 0 for h in norm_hs) / len(norm_hs)


def transition_coverage(norm_hs, transitions):
    pairs = list(zip(norm_hs, norm_hs[1:]))
    return sum(pair in transitions for pair in pairs) / len(pairs) if pairs else None


def topology_coverage(nodes, attribute_counts, fields):
    values = [(field, _value(node[field])) for node in nodes for field in fields]
    return sum(attribute_counts.get(v, 0) > 0 for v in values) / len(values)


def nearest_distance(centroids, vector):
    """Cosine distance of an embedding to its nearest L2-normalised prototype."""
    x = np.asarray(vector, dtype=np.float64)
    return float(1.0 - (np.asarray(centroids, dtype=np.float64) @ (x / np.linalg.norm(x))).max())


def prototype_p(reference_distances_sorted, distances):
    return tail_p(reference_distances_sorted, distances)


def block_statistics(target_h, record_h, node_values, unigram_counts, attribute_counts):
    """The three DRIFT-CS-001 statistics of one block against a reference (a count <= 0 means absent)."""
    return {"event_jsd": jsd(Counter(target_h), unigram_counts),
            "unseen_event_rate": sum(unigram_counts.get(h, 0) <= 0 for h in record_h) / len(record_h),
            "unseen_topology_rate": sum(attribute_counts.get(v, 0) <= 0 for v in node_values) / len(node_values)}


def state(statistics, thresholds):
    return "DRIFT_REVIEW" if any(statistics[s] > thresholds[s] for s in STATISTICS) else "IN_REFERENCE"


def reference_counts(stats, fields):
    """Unigram counts by norm_h and node-attribute counts by (field, value) of the buffer."""
    unigrams = dict(stats["unigrams"].select("norm_h", "count").iter_rows())
    attributes = {(f, v): c for f, v, c in stats["node_attributes"].select("field", "value", "count").iter_rows()
                  if f in fields}
    return unigrams, attributes


def reference_blocks(stats, block, fields):
    """Leave-block-out statistics of the chronological blocks of the statistics windows."""
    unigrams, attributes = reference_counts(stats, fields)
    text_h = {r: hash64(m.encode()) for r, m in stats["events"].select("rank", "normalized_message").iter_rows()}
    node_values = {row[0]: [(f, _value(v)) for f, v in zip(fields, row[1:])]
                   for row in stats["nodes"].select("rank", *fields).iter_rows()}
    graph_ranks = {w: [*c, t] for w, c, t in stats["graphs"].select("window_id", "context_ranks", "target_rank").iter_rows()}
    windows = stats["windows"].sort("target_rank")
    rows = []
    for start in range(0, windows.height - block + 1, block):
        part = windows.slice(start, block)
        records = sorted({r for c, t in part.select("context_ranks", "target_rank").iter_rows() for r in (*c, t)})
        nodes = sorted({r for w in part["window_id"] for r in graph_ranks[w]})
        record_h = [text_h[r] for r in records]
        values = [v for r in nodes for v in node_values[r]]
        left_u, left_a = Counter(record_h), Counter(values)
        rows.append({"block": start // block, "first_target_rank": part["target_rank"][0],
                     "last_target_rank": part["target_rank"][-1], "records": len(records), "nodes": len(nodes),
                     **block_statistics([text_h[t] for t in part["target_rank"]], record_h, values,
                                        {h: c - left_u.get(h, 0) for h, c in unigrams.items()},
                                        {k: c - left_a.get(k, 0) for k, c in attributes.items()})})
    return pl.DataFrame(rows)


def _source_scores(folder, expert):
    path = Path(folder) / f"{expert}.jsonl"
    envelopes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [e["score_raw"] for e in envelopes if e["status"] in SCORED and e["score_raw"] is not None], sha256_file(path)


def source_gap(source, target):
    """Descriptive source-to-target shift of one expert's raw scores."""
    from scipy.stats import ks_2samp

    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    median, q99 = np.quantile(source, [0.5, 0.99])
    return {"n_source": int(source.size), "n_target": int(target.size),
            "ks_distance": float(ks_2samp(source, target).statistic),
            "median_shift": float((np.median(target) - median) / (q99 - median)) if q99 > median else None}


def build_drift_reference(root, repo_root, source_evidence, folds=None):
    """Reference blocks, thresholds, prototype-distance references and the source gap; frozen per fold."""
    spec = load_spec(repo_root)
    fields, block = spec["topology_fields"], spec["block_drift"]["block"]
    report = {}
    for fold_id in folds or spec["folds"]:
        memory, _ = load_frozen_memory(root, fold_id)
        stats = {n: memory_table(root, fold_id, "statistics", n)
                 for n in ("unigrams", "node_attributes", "events", "nodes", "graphs", "windows")}
        blocks = reference_blocks(stats, block, fields)
        if blocks.height < 2:
            raise AdaptationError(f"{fold_id}: too few reference blocks ({blocks.height})")
        thresholds = {s: float(blocks[s].max()) for s in STATISTICS}
        assignments = memory_table(root, fold_id, "prototypes", "assignments")
        scores = memory_table(root, fold_id, "buffer_scores", "scores")
        distances, gap = {}, {}
        for expert in sorted(scores["expert_id"].unique()):
            d = assignments.filter((pl.col("expert_id") == expert) & pl.col("cosine_distance").is_not_nan())["cosine_distance"]
            distances[expert] = {"n": d.len(), "q50": float(d.quantile(0.5, "linear")), "q99": float(d.quantile(0.99, "linear"))}
            target = scores.filter((pl.col("expert_id") == expert) & pl.col("status").is_in(SCORED))["score_raw"].drop_nulls()
            source, digest = _source_scores(Path(source_evidence) / fold_id, expert)
            gap[expert] = dict(source_gap(source, target.to_numpy()), source_evidence_sha256=digest)
        manifest = {"artifact_id": ARTIFACT, "fold_id": fold_id, "memory_key": memory["memory_key"],
                    "memory_sha256": memory["sha256"], "spec_sha256": spec["sha256"],
                    "builder_sha256": sha256_file(Path(__file__)), "block": block, "reference_blocks": blocks.height,
                    "thresholds": thresholds, "topology_fields": fields, "prototype_distance": distances,
                    "source_to_target_gap": gap, "target_labels_read": False}
        write_artifact(root, ARTIFACT, fold_id, {"blocks": blocks}, manifest)
        report[fold_id] = {"thresholds": thresholds, "reference_blocks": blocks.height}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="freeze the DRIFT-CS-001 reference from the frozen fold memories")
    build.add_argument("--root", required=True, help="folder holding data/processed/memory")
    build.add_argument("--repo-root", required=True)
    build.add_argument("--source-evidence", required=True, help="P2.6 evidence folder holding <fold>/<EXPERT>.jsonl")
    args = parser.parse_args()
    print(json.dumps(build_drift_reference(args.root, args.repo_root, args.source_evidence), indent=2))


if __name__ == "__main__":
    main()
