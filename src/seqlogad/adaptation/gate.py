"""FUS-GATE-001 (P3.7, exploratory): the learned gate, trained self-supervised on the synthetic copies.

Spec: configs/protocols/fus-gate-v1.yaml. The features and family scores of a copy are computed against the normal
reference of its system: a source episode uses the source's training windows and its own CLEAN copies; the target
uses the frozen ADAPT-MEM-001 memory, CAL-CS-001 and UNC-CS-001. The gate learns only the paired ranking of CLEAN
against mutated copies, so no label is read. G2 stays PARTIAL and every output is EXPLORATORY.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from seqlogad.adaptation import baselines, drift, synthetic
from seqlogad.adaptation.store import (ADAPT_ROOT, AdaptationError, load_artifact, load_frozen_memory, memory_table,
                                       write_artifact)
from seqlogad.adaptation.uncertainty import normalise
from seqlogad.common.checksum import sha256_file
from seqlogad.diversity.analysis import auroc
from seqlogad.semantic.prepare import hash64

ARTIFACT = "FUS-GATE-001"
FAMILIES = ("SEMANTIC", "SEQUENCE", "STRUCTURAL")
TASK_OF = {"SEMANTIC_LLAMA": "P2.1", "SEQUENCE_LLAMA_REFERENCE": "P2.2", "SEQUENCE_LIGHTWEIGHT": "P2.3",
           "GTAT_STRUCTURAL_TEMPORAL": "P2.4"}
FAMILY_OF = {"SEMANTIC_LLAMA": "SEMANTIC", "SEQUENCE_LLAMA_REFERENCE": "SEQUENCE", "SEQUENCE_LIGHTWEIGHT": "SEQUENCE",
             "GTAT_STRUCTURAL_TEMPORAL": "STRUCTURAL"}
FEATURES = ("semantic_native_uncertainty", "sequence_native_uncertainty", "input_coverage", "event_coverage",
            "transition_coverage", "topology_coverage", "context_length", "scored_share")
TOPOLOGY = ("component_text", "severity", "location")
MISSING = 0.5  # spec: a feature that cannot be computed


def load_spec(repo_root):
    spec = synthetic.load_spec(repo_root)
    names = [next(iter(item)) for item in spec["features"]]
    if tuple(names) != FEATURES or tuple(spec["families"]) != FAMILIES or spec["gate"]["training"]["seed"] != 42:
        raise AdaptationError("FUS-GATE-001 gate design changed")
    return spec


# ---------------------------------------------------------------- shared assembly

def scale_of(reference):
    """CAL-CS-001 anchors (q50, q99) of a reference, or None when the scale is degenerate or empty."""
    reference = np.asarray([v for v in reference if v is not None and np.isfinite(v)], dtype=np.float64)
    if reference.size == 0:
        return None
    low, high = np.quantile(reference, [0.5, 0.99])
    return (float(low), float(high)) if high > low else None


def assemble(experts, windows):
    """Feature matrix X, family scores S and family mask M, one row per unit of `windows`.

    experts: unit, expert_id, scored (bool), score (scaled), native (ECDF or None), coverage.
    windows: unit, event_coverage, transition_coverage, topology_coverage, context_length."""
    units = windows["unit"].to_list()
    at = {u: i for i, u in enumerate(units)}
    n = len(units)
    family = {f: [[] for _ in range(n)] for f in FAMILIES}
    native = {"SEMANTIC": [[] for _ in range(n)], "SEQUENCE": [[] for _ in range(n)]}
    coverage, scored = [[] for _ in range(n)], np.zeros(n)
    for unit, expert, ok, score, value, cover in experts.select(
            "unit", "expert_id", "scored", "score", "native", "coverage").iter_rows():
        i, fam = at.get(unit), FAMILY_OF[expert]
        if i is None or not ok:
            continue
        family[fam][i].append(score)
        if fam in native and value is not None:
            native[fam][i].append(value)
        coverage[i].append(cover)
        scored[i] += 1
    x = np.full((n, len(FEATURES)), MISSING)
    x[:, 0] = [np.mean(v) if v else MISSING for v in native["SEMANTIC"]]
    x[:, 1] = [np.mean(v) if v else MISSING for v in native["SEQUENCE"]]
    x[:, 2] = [min(v) if v else 0.0 for v in coverage]
    for k, name in enumerate(("event_coverage", "transition_coverage", "topology_coverage", "context_length"), start=3):
        x[:, k] = [MISSING if v is None else v for v in windows[name].to_list()]
    x[:, 7] = scored / len(TASK_OF)
    s = np.array([[np.mean(family[f][i]) if family[f][i] else np.nan for f in FAMILIES] for i in range(n)])
    return x, s, ~np.isnan(s), units


def window_coverage(messages, nodes, unigrams, transitions, attributes):
    """DRIFT-CS-001 event, transition and topology coverage of one window (transition 1.0 for one record)."""
    hashes = [hash64(m.encode()) for m in messages]
    transition = drift.transition_coverage(hashes, transitions)
    return (drift.event_coverage(hashes, unigrams), 1.0 if transition is None else transition,
            drift.topology_coverage(nodes, attributes, TOPOLOGY) if nodes else MISSING)


# ---------------------------------------------------------------- source episodes

def copy_scores(scores_root, view_root, fold_id):
    """One row per (copy, expert): the score of the copy, or of its CLEAN copy when the copy does not change the
    expert's input. Each score file must match its receipt and the view it was scored on."""
    view = Path(view_root) / synthetic.VIEW / fold_id
    table = pl.read_parquet(view / "mutations.parquet")
    frames, digests = [], {}
    for expert, task in TASK_OF.items():
        path = Path(scores_root) / fold_id / f"{expert}.jsonl"
        receipt = json.loads(path.with_suffix(".receipt.json").read_text(encoding="utf-8"))
        if (receipt["scores_sha256"] != sha256_file(path) or receipt["labels_read"] is not False
                or not receipt["sanity_reproduction"]["pass"]
                or receipt["mutations_sha256"] != sha256_file(view / "mutations.parquet")):
            raise AdaptationError(f"{fold_id} {expert}: synthetic scores do not match their receipt or view")
        digests[expert] = receipt["scores_sha256"]
        scores = pl.read_ndjson(path).select(
            pl.col("synthetic_id").cast(pl.String), pl.col("score_raw").cast(pl.Float64), pl.col("status").cast(pl.String),
            pl.col("coverage").cast(pl.Float64), pl.col("uncertainty").cast(pl.Float64))
        reads = list(synthetic.READS[task])
        source = pl.when(pl.col("mutation").is_in(reads)).then(pl.col("synthetic_id")) \
            .otherwise(pl.lit("CLEAN:") + pl.col("base_window_id"))
        frames.append(table.select("synthetic_id", "mutation", "base_window_id", "architecture_id",
                                   source.alias("scored_as"))
                      .join(scores.rename({"synthetic_id": "scored_as"}), on="scored_as", how="left", validate="m:1")
                      .with_columns(pl.lit(expert).alias("expert_id")))
    out = pl.concat(frames)
    if out["status"].null_count():
        raise AdaptationError(f"{fold_id}: a copy has no score for an expert that reads it")
    return out, digests


def source_reference(data_root, graph_root, fold_id, architecture):
    """Unigram and transition counts of the source's training windows and its training node attributes."""
    from seqlogad.graph import contracts

    root = Path(data_root) / "folds" / fold_id / "sequence"
    events = pl.read_parquet(root / "metadata/events.parquet").filter(
        (pl.col("architecture_id") == architecture) & (pl.col("partition") == "SOURCE_TRAIN"))
    message = dict(events.select("rank", "normalized_message").iter_rows())
    unigrams = Counter(hash64(m.encode()) for m in message.values())
    windows = pl.read_parquet(root / "train/windows.parquet").filter(pl.col("architecture_id") == architecture)
    transitions = set()
    for context, target in windows.select("context_ranks", "target_rank").iter_rows():
        chain = [hash64(message[r].encode()) for r in [*context, target]]
        transitions.update(zip(chain, chain[1:]))
    nodes = contracts.load_graph_nodes(graph_root, fold_id).filter(
        (pl.col("architecture_id") == architecture) & (pl.col("partition") == "SOURCE_TRAIN"))
    attributes = Counter((f, drift._value(v)) for f in TOPOLOGY for v in nodes[f].to_list())
    return unigrams, transitions, attributes


def episode_inputs(root_paths, fold_id, spec):
    """X, S, M, units and episode/mutation/base of every synthetic copy of the fold."""
    view_root, scores_root, data_root, graph_root = root_paths
    from seqlogad.graph import contracts

    scores, digests = copy_scores(scores_root, view_root, fold_id)
    view = Path(view_root) / synthetic.VIEW / fold_id
    table = pl.read_parquet(view / "mutations.parquet")
    rows = {r["synthetic_id"]: r for r in pl.read_parquet(view / "rows.parquet").iter_rows(named=True)}
    shuffled = {k: v["context_messages"] for k, v in rows.items() if v["mutation"] == "SHUFFLED_CONTEXT"}
    nodes = contracts.load_graph_nodes(graph_root, fold_id)
    graphs = contracts.load_graph_windows(graph_root, fold_id, "val")
    extra_nodes, extra_graphs = synthetic.graph_copies(nodes, graphs, table, shuffled)
    graph_nodes = {}
    columns = ["architecture_id", "partition", "rank", *TOPOLOGY]
    for part, frame in pl.concat([nodes.filter(pl.col("partition") == "SOURCE_VALIDATION").select(columns),
                                  extra_nodes.select(columns)]).partition_by("architecture_id", "partition",
                                                                             as_dict=True).items():
        graph_nodes[part] = {r["rank"]: r for r in frame.select("rank", *TOPOLOGY).iter_rows(named=True)}
    graph = {r["window_id"]: r for r in pl.concat([graphs, extra_graphs]).iter_rows(named=True)}
    expert_rows, window_rows = [], []
    for architecture in sorted(table["architecture_id"].unique()):
        unigrams, transitions, attributes = source_reference(data_root, graph_root, fold_id, architecture)
        part = scores.filter(pl.col("architecture_id") == architecture)
        clean = part.filter((pl.col("mutation") == "CLEAN") & (pl.col("status") == "VALID"))
        for expert in TASK_OF:
            reference = clean.filter(pl.col("expert_id") == expert)
            anchors = scale_of(reference["score_raw"].to_list())
            native_ref = np.sort([v for v in reference["uncertainty"].to_list() if v is not None and np.isfinite(v)])
            mine = part.filter(pl.col("expert_id") == expert)
            natives = normalise(native_ref if native_ref.size else None, mine["uncertainty"].to_list())
            for (unit, status, raw, cover), value in zip(
                    mine.select("synthetic_id", "status", "score_raw", "coverage").iter_rows(), natives):
                ok = status == "VALID" and raw is not None and np.isfinite(raw) and anchors is not None
                expert_rows.append({"unit": unit, "expert_id": expert, "scored": ok, "native": value,
                                    "coverage": cover, "score": (raw - anchors[0]) / (anchors[1] - anchors[0]) if ok else None})
        for item in table.filter(pl.col("architecture_id") == architecture).iter_rows(named=True):
            row = rows.get(item["synthetic_id"]) or rows["CLEAN:" + item["base_window_id"]]  # FOREIGN_TOPOLOGY keeps text
            g = graph[item["base_window_id"] if item["mutation"] == "CLEAN" else item["synthetic_id"]]
            store = graph_nodes[(architecture, g["partition"])]
            gnodes = [store[r] for r in [*(g["context_ranks"] or []), g["target_rank"]]]
            event, transition, topology = window_coverage([*row["context_messages"], row["target_message"]], gnodes,
                                                          unigrams, transitions, attributes)
            window_rows.append({"unit": item["synthetic_id"], "event_coverage": event, "transition_coverage": transition,
                                "topology_coverage": topology, "context_length": (len(row["context_ranks"]) + 1) / 9,
                                "episode": architecture, "mutation": item["mutation"], "base": item["base_window_id"]})
    windows = pl.DataFrame(window_rows)
    x, s, m, units = assemble(pl.DataFrame(expert_rows, schema_overrides={"native": pl.Float64, "score": pl.Float64}),
                              windows)
    return x, s, m, windows, digests


# ---------------------------------------------------------------- the gate

def _gate():
    import torch

    return torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(),
                               torch.nn.Linear(16, 3))


def fused(model, x, s, m):
    """Gate weights (softmax over the families with a score) and the fused score."""
    import torch

    logits = model(torch.as_tensor(x, dtype=torch.float32)).masked_fill(~torch.as_tensor(m), float("-inf"))
    weights = torch.softmax(logits, dim=1)
    return weights, (weights * torch.as_tensor(np.nan_to_num(s), dtype=torch.float32)).sum(dim=1)


def pairs_of(windows, m, episodes):
    """(CLEAN index, mutated index) of every pair whose two copies have at least 2 families (the quorum)."""
    index = {u: i for i, u in enumerate(windows["unit"].to_list())}
    ok = m.sum(axis=1) >= 2
    out = []
    for unit, episode, mutation, base in windows.select("unit", "episode", "mutation", "base").iter_rows():
        clean = index["CLEAN:" + base]
        if mutation != "CLEAN" and episode in episodes and ok[index[unit]] and ok[clean]:
            out.append((clean, index[unit]))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def train(x, s, m, pairs, epochs, cfg, held_out=None):
    """Full-batch Adam on the paired ranking loss; returns the model and the held-out loss after each epoch."""
    import torch

    torch.manual_seed(cfg["seed"])
    model = _gate()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    curve = []
    for _ in range(epochs):
        _, f = fused(model, x, s, m)
        loss = torch.nn.functional.softplus(f[pairs[:, 0]] - f[pairs[:, 1]]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if held_out is not None:
            with torch.no_grad():
                _, f = fused(model, x, s, m)
                curve.append(float(torch.nn.functional.softplus(f[held_out[:, 0]] - f[held_out[:, 1]]).mean()))
    return model, curve


def held_out_report(x, s, m, windows, episode, model):
    """AUROC of mutated vs CLEAN copies of one held-out episode, per mutation, for the gate, equal weight and families."""
    import torch

    with torch.no_grad():
        _, f = fused(model, x, s, m)
    gate = f.numpy()
    equal = np.where(m.any(axis=1), np.nanmean(np.where(m, s, np.nan), axis=1), np.nan)
    frame = windows.with_row_index("i").filter(pl.col("episode") == episode)
    out = []
    for mutation in synthetic.MUTATIONS:
        part = frame.filter(pl.col("mutation").is_in(["CLEAN", mutation]))
        idx = part["i"].to_numpy()
        keep = m[idx].sum(axis=1) >= 2
        idx, labels = idx[keep], (part["mutation"].to_numpy()[keep] == mutation)
        row = {"episode": episode, "mutation": mutation, "copies": int(labels.sum()),
               "GATE": auroc(gate[idx], labels), "EQUAL_WEIGHT": auroc(equal[idx], labels)}
        for k, family in enumerate(FAMILIES):
            have = m[idx, k]
            row[family] = auroc(s[idx[have], k], labels[have]) if have.any() else None
        out.append(row)
    return out


def fit_gate(x, s, m, windows, cfg):
    """Choose the epochs by leave-one-episode-out, report each held-out episode, then train on every episode."""
    episodes = sorted(windows["episode"].unique().to_list())
    curves = []
    for held in episodes:
        rest = [e for e in episodes if e != held]
        _, curve = train(x, s, m, pairs_of(windows, m, rest), cfg["max_epochs"], cfg, pairs_of(windows, m, [held]))
        curves.append(curve)
    epochs = int(np.argmin(np.mean(curves, axis=0))) + 1
    report = []
    for held in episodes:
        model, _ = train(x, s, m, pairs_of(windows, m, [e for e in episodes if e != held]), epochs, cfg)
        report.extend(held_out_report(x, s, m, windows, held, model))
    model, _ = train(x, s, m, pairs_of(windows, m, episodes), epochs, cfg)
    return model, epochs, np.mean(curves, axis=0).tolist(), report


# ---------------------------------------------------------------- target buffer (example outputs)

def buffer_inputs(root, fold_id):
    """X, S, M of the UNC-CS-001 buffer windows against the frozen memory, CAL-CS-001 and UNC-CS-001."""
    memory, _ = load_frozen_memory(root, fold_id)
    cal, unc = load_artifact(root, "CAL-CS-001", fold_id), load_artifact(root, "UNC-CS-001", fold_id)
    states = baselines._buffer_states(root, fold_id, unc, cal)
    experts = states.select(pl.col("target_rank").alias("unit"), "expert_id",
                            (pl.col("expert_state") == "SCORED").alias("scored"),
                            pl.col("score_calibrated").alias("score"), pl.col("native_uncertainty").alias("native"),
                            "coverage")
    stats = {n: memory_table(root, fold_id, "statistics", n)
             for n in ("unigrams", "transitions", "node_attributes", "events", "windows", "graphs", "nodes")}
    unigrams = dict(stats["unigrams"].select("norm_h", "count").iter_rows())
    transitions = set(stats["transitions"].select("prev_h", "next_h").iter_rows())
    attributes = {(f, v): c for f, v, c in stats["node_attributes"].select("field", "value", "count").iter_rows()
                  if f in TOPOLOGY}
    message = dict(stats["events"].select("rank", "normalized_message").iter_rows())
    node = {r["rank"]: r for r in stats["nodes"].iter_rows(named=True)}
    graph = {t: c for t, c in stats["graphs"].select("target_rank", "context_ranks").iter_rows()}
    units = set(experts["unit"].to_list())
    rows = []
    for context, target in stats["windows"].select("context_ranks", "target_rank").iter_rows():
        if target not in units:
            continue
        gnodes = [node[r] for r in [*(graph.get(target) or []), target] if r in node]
        event, transition, topology = window_coverage([message[r] for r in [*context, target]], gnodes,
                                                      unigrams, transitions, attributes)
        rows.append({"unit": target, "event_coverage": event, "transition_coverage": transition,
                     "topology_coverage": topology, "context_length": (len(context) + 1) / 9})
    windows = pl.DataFrame(rows).sort("unit")
    x, s, m, ranks = assemble(experts, windows)
    return x, s, m, ranks, memory, cal, unc


def build_gate(root, repo_root, view_root, scores_root, data_root, graph_root, folds=None):
    """Train, report and freeze the exploratory gate per fold; apply it to the buffer windows as example outputs."""
    import torch

    spec = load_spec(repo_root)
    cfg = spec["gate"]["training"]
    report = {}
    for fold_id in folds or spec["folds"]:
        x, s, m, windows, digests = episode_inputs((view_root, scores_root, data_root, graph_root), fold_id, spec)
        model, epochs, curve, held = fit_gate(x, s, m, windows, cfg)
        with torch.no_grad():
            weights, f = fused(model, x, s, m)
        by_mutation = (windows.with_columns(*(pl.Series(fam, weights[:, k].numpy()) for k, fam in enumerate(FAMILIES)))
                       .group_by("mutation").agg(*(pl.col(fam).mean() for fam in FAMILIES)).sort("mutation"))
        bx, bs, bm, ranks, memory, cal, unc = buffer_inputs(root, fold_id)
        with torch.no_grad():
            bw, bf = fused(model, bx, bs, bm)
        quorum = bm.sum(axis=1) >= 2
        predictions = pl.DataFrame({"target_rank": ranks, **{f"weight_{fam}": bw[:, k].numpy() for k, fam in enumerate(FAMILIES)},
                                    "score": np.where(quorum, bf.numpy(), np.nan),
                                    "status": np.where(quorum, "SCORED", "UNKNOWN_REVIEW")})
        weights_file = io.BytesIO()
        torch.save(model.state_dict(), weights_file)
        view = Path(view_root) / synthetic.VIEW / fold_id
        manifest = {"artifact_id": ARTIFACT, "fold_id": fold_id, "mode": "EXPLORATORY", "g2": "PARTIAL",
                    "memory_key": memory["memory_key"], "memory_sha256": memory["sha256"],
                    "calibration_sha256": sha256_file(Path(root) / ADAPT_ROOT / "CAL-CS-001" / fold_id / "artifact.json"),
                    "uncertainty_sha256": sha256_file(Path(root) / ADAPT_ROOT / "UNC-CS-001" / fold_id / "artifact.json"),
                    "synthetic_scores_sha256": digests, "mutations_sha256": sha256_file(view / "mutations.parquet"),
                    "rows_sha256": sha256_file(view / "rows.parquet"), "spec_sha256": spec["sha256"],
                    "builder_sha256": sha256_file(Path(__file__)), "torch_version": torch.__version__,
                    "epochs": epochs, "pairs": int(len(pairs_of(windows, m, windows["episode"].unique().to_list()))),
                    "held_out": held, "weights_by_mutation": by_mutation.to_dicts(),
                    "buffer_mean_weights": {fam: float(bw[:, k].mean()) for k, fam in enumerate(FAMILIES)},
                    "buffer_unknown_review": int((~quorum).sum()), "target_labels_read": False,
                    "source_labels_read": False}
        write_artifact(root, ARTIFACT, fold_id, {"buffer_predictions": predictions, "held_out": pl.DataFrame(held),
                                                 "epoch_curve": pl.DataFrame({"mean_held_out_loss": curve}),
                                                 "gate.pt": weights_file.getvalue()}, manifest)
        report[fold_id] = {"epochs": epochs, "held_out": held, "weights_by_mutation": by_mutation.to_dicts(),
                           "buffer_mean_weights": manifest["buffer_mean_weights"]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="train and freeze the exploratory FUS-GATE-001 gate")
    for name in ("--root", "--repo-root", "--view-root", "--scores-root", "--data-root", "--graph-root"):
        build.add_argument(name, required=True)
    args = parser.parse_args()
    print(json.dumps(build_gate(args.root, args.repo_root, args.view_root, args.scores_root, args.data_root,
                                args.graph_root), indent=2, default=str))


if __name__ == "__main__":
    main()
