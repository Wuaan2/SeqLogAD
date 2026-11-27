"""ADAPT-MEM-001 prototypes and buffer scores (P3.2): the frozen experts read the burn-in windows.

Spec: configs/protocols/adapt-mem-prototypes-v1.yaml. The pass reuses the P2.6 scorers unchanged, and every expert
must reproduce its stored sanity score first. Each chunk writes its embeddings (.npy) and then its EVID-CS-001
envelopes (.jsonl, moved into place last), so a disconnect resumes at the first chunk without envelopes. The
prototypes (k-means) and the buffer-score table are built afterwards on a CPU, and the fold memory is frozen.
Label-free: only the statistics component of the approved buffer is read.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from seqlogad.common.checksum import sha256_file
from seqlogad.diversity import score as p26
from seqlogad.evidence.contract import validate
from seqlogad.evidence.convert import convert
from seqlogad.memory.build import MEMORY_ROOT, PARTITION, AdaptMemoryError, load_component, write_component
from seqlogad.phase2_execution.specs import FOLDS
from seqlogad.semantic.contracts import dump, read
from seqlogad.sequence.prepare import _candidate_score

SPEC = "configs/protocols/adapt-mem-prototypes-v1.yaml"
CHUNK = 256


def load_spec(repo_root):
    path = Path(repo_root) / SPEC
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))["adapt_mem_prototypes"]
    if spec["status"] != "PREREGISTERED" or spec["prototypes"]["seed"] != 42 or spec["windows"]["limit"] != 8192:
        raise AdaptMemoryError("ADAPT-MEM-001 prototype spec changed")
    return dict(spec, sha256=sha256_file(path))


def buffer_rows(view_root, fold_id, limit):
    """The `limit` statistics windows with the lowest P2.2 candidate score, in rank order, with their messages."""
    manifest = load_component(view_root, fold_id, "statistics")  # re-hashes every file of the component
    stats = Path(view_root) / MEMORY_ROOT / fold_id / "statistics"
    windows = pl.read_parquet(stats / "windows.parquet").to_dicts()
    chosen = sorted(windows, key=lambda r: _candidate_score(r["architecture_id"], PARTITION, r["target_rank"]))[:limit]
    events = pl.read_parquet(stats / "events.parquet")
    messages = {(a, r): m for a, r, m in events.select("architecture_id", "rank", "normalized_message").iter_rows()}
    return p26.hydrate(sorted(chosen, key=lambda r: r["target_rank"]), messages), manifest


@contextmanager
def graphs_with_buffer(view_root, fold_id):
    """Let the frozen P2.4 scorer see the buffer graphs next to its own validation graphs (for the sanity check)."""
    from seqlogad.graph import contracts

    nodes, windows = contracts.load_graph_nodes, contracts.load_graph_windows
    stats = Path(view_root) / MEMORY_ROOT / fold_id / "statistics"

    def with_nodes(root, fold):
        frame = nodes(root, fold)
        return pl.concat([frame, pl.read_parquet(stats / "nodes.parquet").select(frame.columns)], how="vertical_relaxed")

    def with_windows(root, fold, partition):
        frame = windows(root, fold, partition)
        return pl.concat([frame, pl.read_parquet(stats / "graphs.parquet").select(frame.columns)], how="vertical_relaxed")

    contracts.load_graph_nodes, contracts.load_graph_windows = with_nodes, with_windows
    try:
        yield
    finally:
        contracts.load_graph_nodes, contracts.load_graph_windows = nodes, windows


def _with_embeddings(task, score, one, model):
    """Score a chunk and return [(native, vector or None)]; the vector is read by a hook, the scorer is untouched."""
    captured = []
    if task == "P2.4":
        hook = model.readout.register_forward_hook(
            lambda _m, _i, out: captured.extend(out.detach().float().cpu().numpy()))
        dim = model.readout[1].out_features

        def run(chunk):
            captured.clear()
            natives = score(chunk)
            if len(captured) != sum(n["status"] == "SCORED" for n in natives):
                raise AdaptMemoryError("GTAT readouts do not match the scored windows")
            vectors = iter(captured)
            return [(n, next(vectors) if n["status"] == "SCORED" else None) for n in natives]
    else:
        head = (model.get_base_model() if hasattr(model, "get_base_model") else model).get_output_embeddings()
        hook = head.register_forward_pre_hook(
            lambda _m, args: captured.append(args[0][0, -1].detach().float().cpu().numpy()))
        dim = head.in_features

        def run(chunk):
            out = []
            for row in chunk:
                captured.clear()
                native = score([row])[0] if task == "P2.1" else one(row)
                if len(captured) > 1:
                    raise AdaptMemoryError("more than one forward pass for one window")
                out.append((native, captured[0] if captured else None))
            return out
    return run, dim, hook


def score_chunks(rows, run, dim, out_dir, progress=print):
    """Resumable chunk loop; returns the envelopes and the float16 embedding matrix in row order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = math.ceil(len(rows) / CHUNK)
    for index in range(count):
        base = out_dir / f"chunk-{index:05d}"
        if base.with_suffix(".jsonl").is_file() and base.with_suffix(".npy").is_file():
            continue
        chunk = rows[index * CHUNK:(index + 1) * CHUNK]
        matrix = np.full((len(chunk), dim), np.nan, dtype=np.float16)
        lines = []
        for position, (row, (native, vector)) in enumerate(zip(chunk, run(chunk), strict=True)):
            envelope = convert(native)
            if envelope["target_record_id"] != row["target_record_id"]:
                raise AdaptMemoryError("scored envelope does not match its burn-in window")
            if vector is not None:
                matrix[position] = vector
                if not np.isfinite(matrix[position]).all():
                    raise AdaptMemoryError("embedding does not fit float16")
            lines.append(json.dumps(envelope, sort_keys=True))
        with open(base.with_suffix(".npy.tmp"), "wb") as handle:
            np.save(handle, matrix)
        os.replace(base.with_suffix(".npy.tmp"), base.with_suffix(".npy"))
        base.with_suffix(".jsonl.tmp").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(base.with_suffix(".jsonl.tmp"), base.with_suffix(".jsonl"))  # marks the chunk complete
        progress(json.dumps({"chunk": index + 1, "of": count}))
    envelopes = [json.loads(line) for index in range(count)
                 for line in (out_dir / f"chunk-{index:05d}.jsonl").read_text(encoding="utf-8").splitlines()]
    vectors = np.concatenate([np.load(out_dir / f"chunk-{index:05d}.npy") for index in range(count)])
    if [e["target_record_id"] for e in envelopes] != [r["target_record_id"] for r in rows] or len(vectors) != len(rows):
        raise AdaptMemoryError("chunks do not cover exactly the selected burn-in windows")
    return envelopes, vectors


def run_task(task, target, *, repo_root, view_root, data_root, semantic_root, graph_root, final_zip, output_root,
             work_root, device="cuda"):
    """Scores and embeddings of one frozen expert on the selected burn-in windows of one target fold."""
    spec, fold_id, expert = load_spec(repo_root), FOLDS[target], p26.TASKS[task]
    folder = Path(output_root) / fold_id
    receipt_path = folder / f"{expert}.receipt.json"
    if receipt_path.is_file():
        print(f"{task}_{target}=SKIP_COMPLETED", flush=True)
        return read(receipt_path)
    rows, statistics = buffer_rows(view_root, fold_id, spec["windows"]["limit"])
    run = p26.extract_run(final_zip, task, target, work_root)
    sanity = read(run / "sanity-evidence.json")
    with graphs_with_buffer(view_root, fold_id) if task == "P2.4" else nullcontext():
        if task == "P2.1":
            score, reproduce, model = p26.semantic_scorer(repo_root, run, semantic_root, fold_id)
            one = None
        elif task == "P2.4":
            score, one, model = p26.gtat_scorer(repo_root, run, graph_root, fold_id, device)
        else:
            score, one, model = p26.sequence_scorer(repo_root, run, lightweight=task == "P2.3")
    if task != "P2.1":  # same sanity rule as the P2.6 pass: the run's own stored validation window
        val = pl.read_parquet(Path(data_root) / "folds" / fold_id / "sequence/val/windows.parquet")
        sample = p26.hydrate(val.filter(pl.col("window_id") == sanity["unit_id"]).to_dicts(),
                             p26._messages(data_root, fold_id))
        if len(sample) != 1:
            raise AdaptMemoryError(f"{task} sanity window not found in the sequence bundle")
        reproduce = lambda _: one(sample[0])["score_raw"]  # noqa: E731
    reproduced, expected = reproduce(sanity), sanity["score_raw"]
    check = {"expected": expected, "reproduced": reproduced, "abs_diff": abs(reproduced - expected),
             "pass": math.isclose(reproduced, expected, rel_tol=p26.REL_TOL, abs_tol=p26.ABS_TOL)}
    print(json.dumps({"task": task, "target": target, "sanity_reproduction": check}), flush=True)
    if not check["pass"]:
        raise AdaptMemoryError(f"{task} {target} cannot reproduce its sanity evidence; refusing to score")
    scorer, dim, hook = _with_embeddings(task, score, one, model)
    start = time.monotonic()
    try:
        envelopes, vectors = score_chunks(rows, scorer, dim, folder / expert,
                                          lambda line: print(f"{task} {target} {line}", flush=True))
    finally:
        hook.remove()
    for envelope in envelopes:
        validate(envelope)
    evidence, matrix = folder / f"{expert}.jsonl", folder / f"{expert}.embeddings.npy"
    evidence.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in envelopes), encoding="utf-8")
    with open(matrix, "wb") as handle:
        np.save(handle, vectors)
    ids = "\n".join(r["window_id"] for r in rows).encode()
    receipt = {"schema": "SEQLOGAD-P3.2-BUFFER-PASS-RECEIPT-1", "task": task, "expert_id": expert, "fold_id": fold_id,
               "memory_key": statistics["memory_key"], "spec_sha256": spec["sha256"],
               "statistics_component_sha256": sha256_file(Path(view_root) / MEMORY_ROOT / fold_id
                                                           / "statistics/component.json"),
               "final_zip_sha256": p26.FINAL_SHA256[task], "run_id": run.name, "sanity_reproduction": check,
               "windows": len(rows), "window_ids_sha256": hashlib.sha256(ids).hexdigest(),
               "status_counts": {s: sum(e["status"] == s for e in envelopes) for s in ("VALID", "LOW_COVERAGE", "ABSTAIN")},
               "embedding_dim": dim, "embeddings_with_vector": int((~np.isnan(vectors).any(axis=1)).sum()),
               "evidence_sha256": sha256_file(evidence), "embeddings_sha256": sha256_file(matrix),
               "seconds_this_session": time.monotonic() - start, "labels_read": False,
               "records_outside_buffer_read": False}
    dump(receipt_path, receipt)
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(f"{task}_{target}=BUFFER_PASS_DONE", flush=True)
    return receipt


def kmeans_prototypes(vectors, k, seed, n_init):
    """L2-normalised k-means over the windows that have a vector; returns centroids, assignment and cosine distance."""
    from sklearn.cluster import KMeans

    present = ~np.isnan(vectors).any(axis=1)
    x = vectors[present].astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    fitted = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)
    centroids = fitted.cluster_centers_ / np.linalg.norm(fitted.cluster_centers_, axis=1, keepdims=True)
    cluster = np.full(len(vectors), -1, dtype=np.int32)
    distance = np.full(len(vectors), np.nan, dtype=np.float32)
    cluster[present] = fitted.labels_
    distance[present] = 1.0 - (x * centroids[fitted.labels_]).sum(axis=1)
    return centroids.astype(np.float32), cluster, distance


def _pass_outputs(pass_root, fold_id, expert, statistics, spec):
    folder = Path(pass_root) / fold_id
    receipt = read(folder / f"{expert}.receipt.json")
    evidence, matrix = folder / f"{expert}.jsonl", folder / f"{expert}.embeddings.npy"
    bad = {"evidence": sha256_file(evidence) != receipt["evidence_sha256"],
           "embeddings": sha256_file(matrix) != receipt["embeddings_sha256"],
           "memory_key": receipt["memory_key"] != statistics["memory_key"],
           "spec": receipt["spec_sha256"] != spec["sha256"], "sanity": not receipt["sanity_reproduction"]["pass"],
           "labels": receipt["labels_read"] is not False}
    if any(bad.values()):
        raise AdaptMemoryError(f"{fold_id} {expert} buffer pass rejected: {[k for k, v in bad.items() if v]}")
    envelopes = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines()]
    return receipt, envelopes, np.load(matrix)


def build_prototypes(root, repo_root, pass_root, folds=None):
    """CPU step: k-means prototypes and the buffer-score table per fold, then freeze the fold memory."""
    spec = load_spec(repo_root)
    proto = spec["prototypes"]
    report = {}
    for fold_id in folds or spec["folds"]:
        statistics = load_component(root, fold_id, "statistics")
        centroids, assignments, scores, receipts = [], [], [], {}
        for task, expert in ((t, e["expert_id"]) for t, e in spec["experts"].items()):
            receipt, envelopes, vectors = _pass_outputs(pass_root, fold_id, expert, statistics, spec)
            receipts[expert] = receipt
            k = min(proto["k"], int((~np.isnan(vectors).any(axis=1)).sum()))
            center, cluster, distance = kmeans_prototypes(vectors, k, proto["seed"], proto["n_init"])
            sizes = np.bincount(cluster[cluster >= 0], minlength=k)
            centroids.append(pl.DataFrame({"expert_id": [expert] * k, "cluster": list(range(k)), "size": sizes.tolist(),
                                           "centroid": center.tolist()}))
            assignments.append(pl.DataFrame({
                "expert_id": [expert] * len(envelopes), "unit_id": [e["unit_id"] for e in envelopes],
                "target_record_id": [e["target_record_id"] for e in envelopes],
                "cluster": cluster.tolist(), "cosine_distance": distance.tolist()}))
            scores.append(pl.DataFrame({
                "expert_id": [expert] * len(envelopes), "unit_id": [e["unit_id"] for e in envelopes],
                "target_record_id": [e["target_record_id"] for e in envelopes],
                "score_raw": [e["score_raw"] for e in envelopes], "status": [e["status"] for e in envelopes],
                "coverage": [float(e["coverage"]) for e in envelopes]}, schema_overrides={"score_raw": pl.Float64}))
        if len({r["window_ids_sha256"] for r in receipts.values()}) != 1:
            raise AdaptMemoryError(f"{fold_id}: experts scored different burn-in windows")
        common = {key: statistics[key] for key in ("artifact_id", "fold_id", "target_architecture", "buffer_id",
                                                   "buffer_membership_sha256", "memory_key", "partition")}
        provenance = dict(common, spec_sha256=spec["sha256"], builder_sha256=sha256_file(Path(__file__)),
                          pass_receipts={e: {k: r[k] for k in ("evidence_sha256", "embeddings_sha256", "run_id",
                                                                "window_ids_sha256", "windows", "final_zip_sha256")}
                                         for e, r in receipts.items()},
                          target_labels_read=False, records_outside_buffer_read=False)
        write_component(root, fold_id, "prototypes",
                        {"centroids": pl.concat(centroids), "assignments": pl.concat(assignments)},
                        dict(provenance, component="prototypes", method=proto))
        write_component(root, fold_id, "buffer_scores", {"scores": pl.concat(scores)},
                        dict(provenance, component="buffer_scores", calibrated=False))
        report[fold_id] = freeze_memory(root, fold_id)
    return report


def freeze_memory(root, fold_id, components=("statistics", "prototypes", "buffer_scores")):
    """Write memory.json once: it binds every component manifest; the fold memory is then frozen."""
    folder = Path(root) / MEMORY_ROOT / fold_id
    path = folder / "memory.json"
    if path.exists():
        raise AdaptMemoryError(f"{path} exists; the fold memory is frozen")
    manifests = {name: load_component(root, fold_id, name) for name in components}
    keys = {m["memory_key"] for m in manifests.values()}
    if len(keys) != 1:
        raise AdaptMemoryError(f"{fold_id}: components carry different memory keys")
    memory = {"artifact_id": "ADAPT-MEM-001", "fold_id": fold_id, "memory_key": keys.pop(), "status": "FROZEN",
              "components": {name: sha256_file(folder / name / "component.json") for name in components},
              "target_labels_read": False}
    path.write_text(json.dumps(memory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o444)
    return memory


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gpu = sub.add_parser("pass", help="score and embed the burn-in windows with the frozen experts")
    for name in ("--repo-root", "--view-root", "--data-root", "--output-root", "--work-root"):
        gpu.add_argument(name, required=True)
    gpu.add_argument("--semantic-root")
    gpu.add_argument("--graph-root")
    gpu.add_argument("--final", action="append", required=True, help="TASK=path to its seed-42 FINAL ZIP")
    gpu.add_argument("--tasks", default="P2.3,P2.2,P2.1")
    gpu.add_argument("--targets", default="BGL,HDFS")
    gpu.add_argument("--device", default="cuda")
    cpu = sub.add_parser("build", help="k-means prototypes, buffer-score table and memory freeze")
    cpu.add_argument("--root", required=True, help="folder holding data/processed/memory")
    cpu.add_argument("--repo-root", required=True)
    cpu.add_argument("--pass-root", required=True, help="folder holding <fold>/<EXPERT>.jsonl from the pass")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_prototypes(args.root, args.repo_root, args.pass_root), indent=2))
        return
    finals = dict(item.split("=", 1) for item in args.final)
    for task in args.tasks.split(","):
        for target in args.targets.split(","):
            run_task(task, target, repo_root=args.repo_root, view_root=args.view_root, data_root=args.data_root,
                     semantic_root=args.semantic_root, graph_root=args.graph_root, final_zip=finals[task],
                     output_root=args.output_root, work_root=args.work_root, device=args.device)
    print("P3.2_BUFFER_PASS=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
