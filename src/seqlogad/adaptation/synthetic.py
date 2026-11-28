"""FUS-GATE-001 (P3.7, exploratory): the synthetic windows of the gate and their expert scores.

Spec: configs/protocols/fus-gate-v1.yaml. The base windows are the 512 checkpoint-selection windows of each source of a
target fold (the windows DIV-P2-001 excludes). Each gives a CLEAN copy and up to three mutated copies, drawn with seed
42. No label is read. `build` writes the mutation table and the message rows (CPU); `score` runs one frozen expert
over the copies whose input it reads, resumably (P2.1-P2.3 on a GPU, P2.4 on a CPU).
"""
from __future__ import annotations

import argparse
import gc
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
from seqlogad.sequence.objective import select_windows

SPEC = "configs/protocols/fus-gate-v1.yaml"
VIEW = "synthetic/FUS-GATE-001"
MUTATIONS = ("FOREIGN_TARGET", "SHUFFLED_CONTEXT", "FOREIGN_TOPOLOGY")
READS = {  # the copies whose input differs from CLEAN for each expert; the others reuse the CLEAN score
    "P2.1": ("CLEAN", "FOREIGN_TARGET"),
    "P2.2": ("CLEAN", "FOREIGN_TARGET", "SHUFFLED_CONTEXT"),
    "P2.3": ("CLEAN", "FOREIGN_TARGET", "SHUFFLED_CONTEXT"),
    "P2.4": ("CLEAN", *MUTATIONS),
}
SEED, PER_SOURCE, CHUNK = 42, 512, 256
FOLDS = ("FOLD-TARGET-ARCH-BGL", "FOLD-TARGET-ARCH-HDFS")  # EXC-007


class SyntheticError(RuntimeError):
    pass


def load_spec(repo_root):
    path = Path(repo_root) / SPEC
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))["fus_gate"]
    if spec["status"] != "PREREGISTERED" or spec["mode"] != "EXPLORATORY" or tuple(spec["synthetic"])[:3] != MUTATIONS:
        raise SyntheticError("FUS-GATE-001 synthetic design changed")
    return dict(spec, sha256=sha256_file(path))


# ---------------------------------------------------------------- build (CPU)

def base_windows(data_root, fold_id):
    """The checkpoint-selection windows of each source, with their messages, in (source, rank) order."""
    rows = select_windows(data_root, fold_id, "val", PER_SOURCE, SEED, replacement=False)
    for row in rows:
        row["target_record_id"] = f"{row['architecture_id']}:{row['target_rank']}"
    return sorted(rows, key=lambda r: (r["architecture_id"], r["target_rank"]))


def mutations(base, seed=SEED):
    """The CLEAN and mutated copies of each base window; donors come from another source of the same fold."""
    rng = np.random.default_rng(seed)
    by_source = {}
    for i, row in enumerate(base):
        by_source.setdefault(row["architecture_id"], []).append(i)
    table = []
    for row in base:
        others = [j for source, idx in by_source.items() if source != row["architecture_id"] for j in idx]
        target_donor, topology_donor = (base[others[int(rng.integers(len(others)))]]["window_id"] for _ in range(2))
        context, order = list(row["context_messages"]), None
        if len(set(context)) >= 2:
            for _ in range(100):  # a permutation that changes the message order
                candidate = rng.permutation(len(context)).tolist()
                if [context[k] for k in candidate] != context:
                    order = candidate
                    break
        common = {"base_window_id": row["window_id"], "architecture_id": row["architecture_id"],
                  "target_rank": row["target_rank"], "context_ranks": list(row["context_ranks"])}
        table.append(dict(common, synthetic_id=f"CLEAN:{row['window_id']}", mutation="CLEAN", donor_window_id=None,
                          permutation=None))
        table.append(dict(common, synthetic_id=f"FOREIGN_TARGET:{row['window_id']}", mutation="FOREIGN_TARGET",
                          donor_window_id=target_donor, permutation=None))
        if order is not None:
            table.append(dict(common, synthetic_id=f"SHUFFLED_CONTEXT:{row['window_id']}", mutation="SHUFFLED_CONTEXT",
                              donor_window_id=None, permutation=order))
        table.append(dict(common, synthetic_id=f"FOREIGN_TOPOLOGY:{row['window_id']}", mutation="FOREIGN_TOPOLOGY",
                          donor_window_id=topology_donor, permutation=None))
    return pl.DataFrame(table, schema={"synthetic_id": pl.String, "mutation": pl.String, "base_window_id": pl.String,
                                       "architecture_id": pl.String, "target_rank": pl.Int64,
                                       "context_ranks": pl.List(pl.Int64), "donor_window_id": pl.String,
                                       "permutation": pl.List(pl.Int64)})


def message_rows(base, table):
    """Hydrated rows of the copies whose message text differs from CLEAN (and CLEAN itself), for P2.1-P2.3."""
    by_id = {row["window_id"]: row for row in base}
    rows = []
    for item in table.filter(pl.col("mutation") != "FOREIGN_TOPOLOGY").iter_rows(named=True):
        row = by_id[item["base_window_id"]]
        context, target = list(row["context_messages"]), row["target_message"]
        if item["mutation"] == "FOREIGN_TARGET":
            target = by_id[item["donor_window_id"]]["target_message"]
        elif item["mutation"] == "SHUFFLED_CONTEXT":
            context = [context[k] for k in item["permutation"]]
        rows.append({"synthetic_id": item["synthetic_id"], "mutation": item["mutation"], "window_id": item["synthetic_id"],
                     "architecture_id": row["architecture_id"], "target_rank": row["target_rank"],
                     "context_ranks": list(row["context_ranks"]), "context_messages": context, "target_message": target,
                     "target_record_id": row["target_record_id"]})
    return pl.DataFrame(rows)


def build(data_root, out_root, folds=FOLDS):
    """Write mutations.parquet and rows.parquet per fold under <out_root>/synthetic/FUS-GATE-001/<fold>/."""
    report = {}
    for fold_id in folds:
        base = base_windows(data_root, fold_id)
        table = mutations(base)
        rows = message_rows(base, table)
        folder = Path(out_root) / VIEW / fold_id
        if folder.exists():
            raise SyntheticError(f"{folder} exists; the synthetic view is written once")
        folder.mkdir(parents=True)
        table.write_parquet(folder / "mutations.parquet")
        rows.write_parquet(folder / "rows.parquet")
        report[fold_id] = {"base_windows": len(base), "copies": dict(table.group_by("mutation").len().iter_rows()),
                           "mutations_sha256": sha256_file(folder / "mutations.parquet"),
                           "rows_sha256": sha256_file(folder / "rows.parquet")}
    return report


# ---------------------------------------------------------------- the GTAT copies

def graph_copies(nodes, graphs, table, shuffled):
    """One extra graph partition per mutated copy: the base window's graph nodes, changed as the mutation says.

    `shuffled` maps a SHUFFLED_CONTEXT copy to the context messages the sequence experts see; each of those records
    that the graph contains takes the same message (records outside the graph are not part of its input)."""
    node = {(r["architecture_id"], r["rank"]): r
            for r in nodes.filter(pl.col("partition") == "SOURCE_VALIDATION").iter_rows(named=True)}
    graph = {r["window_id"]: r for r in graphs.iter_rows(named=True)}
    extra_nodes, extra_graphs = [], []
    for item in table.filter(pl.col("mutation") != "CLEAN").iter_rows(named=True):
        g = graph[item["base_window_id"]]
        part, ranks = f"SYN:{item['synthetic_id']}", list(g["context_ranks"] or []) + [g["target_rank"]]
        copy = [dict(node[(g["architecture_id"], r)], partition=part) for r in ranks]
        if item["mutation"] == "FOREIGN_TARGET":
            donor = graph[item["donor_window_id"]]
            copy[-1]["event_text"] = node[(donor["architecture_id"], donor["target_rank"])]["event_text"]
        elif item["mutation"] == "SHUFFLED_CONTEXT":  # the same record-level permutation as the sequence experts see
            at = {r: k for k, r in enumerate(ranks)}
            for r, message in zip(item["context_ranks"], shuffled[item["synthetic_id"]]):
                if r in at:
                    copy[at[r]]["event_text"] = message
        else:  # FOREIGN_TOPOLOGY: component and location of the donor window's records, cycled
            donor = graph[item["donor_window_id"]]
            donors = [node[(donor["architecture_id"], r)] for r in list(donor["context_ranks"] or []) + [donor["target_rank"]]]
            for k, record in enumerate(copy):
                record["component_text"] = donors[k % len(donors)]["component_text"]
                record["location"] = donors[k % len(donors)]["location"]
        extra_nodes.extend(copy)
        extra_graphs.append(dict(g, window_id=item["synthetic_id"], partition=part))
    return (pl.DataFrame(extra_nodes, schema=nodes.schema), pl.DataFrame(extra_graphs, schema=graphs.schema))


@contextmanager
def synthetic_graphs(view_root, fold_id):
    """Let the frozen P2.4 scorer see the mutated copies next to its own validation graphs."""
    from seqlogad.graph import contracts

    load_nodes, load_windows = contracts.load_graph_nodes, contracts.load_graph_windows
    table = pl.read_parquet(Path(view_root) / VIEW / fold_id / "mutations.parquet")
    rows = pl.read_parquet(Path(view_root) / VIEW / fold_id / "rows.parquet").filter(pl.col("mutation") == "SHUFFLED_CONTEXT")
    shuffled = dict(rows.select("synthetic_id", "context_messages").iter_rows())
    cache = {}

    def extra(root, fold):
        if fold not in cache:
            cache[fold] = graph_copies(load_nodes(root, fold), load_windows(root, fold, "val"), table, shuffled)
        return cache[fold]

    def with_nodes(root, fold):
        return pl.concat([load_nodes(root, fold), extra(root, fold)[0]])

    def with_windows(root, fold, partition):
        frame = load_windows(root, fold, partition)
        return pl.concat([frame, extra(root, fold)[1]]) if partition == "val" else frame

    contracts.load_graph_nodes, contracts.load_graph_windows = with_nodes, with_windows
    try:
        yield
    finally:
        contracts.load_graph_nodes, contracts.load_graph_windows = load_nodes, load_windows


# ---------------------------------------------------------------- score (one expert, one fold, resumable)

def _done(path):
    """Synthetic ids already written; drop a trailing partial line left by a disconnect."""
    if not path.exists():
        return set()
    lines, kept, done = path.read_bytes().splitlines(), [], set()
    for line in lines:
        try:
            done.add(json.loads(line)["synthetic_id"])
            kept.append(line)
        except (ValueError, KeyError):
            break
    if len(kept) != len(lines):
        path.write_bytes(b"".join(line + b"\n" for line in kept))
    return done


def task_rows(view_root, fold_id, task):
    folder = Path(view_root) / VIEW / fold_id
    if task == "P2.4":
        table = pl.read_parquet(folder / "mutations.parquet")
        return [{"synthetic_id": r["synthetic_id"], "mutation": r["mutation"],
                 "window_id": r["base_window_id"] if r["mutation"] == "CLEAN" else r["synthetic_id"]}
                for r in table.iter_rows(named=True)]
    rows = pl.read_parquet(folder / "rows.parquet").filter(pl.col("mutation").is_in(READS[task]))
    return rows.to_dicts()


def score_task(task, target, *, repo_root, view_root, data_root, semantic_root, graph_root, final_zip, output_root,
               work_root, device="cuda"):
    """Score the copies one frozen expert reads; sanity-check the checkpoint first, as the P2.6 pass does."""
    load_spec(repo_root)
    fold_id, expert = p26.FOLDS[target], p26.TASKS[task]
    folder = Path(output_root) / fold_id
    folder.mkdir(parents=True, exist_ok=True)
    out, receipt_path = folder / f"{expert}.jsonl", folder / f"{expert}.receipt.json"
    if receipt_path.is_file():
        print(f"{task}_{target}=SKIP_COMPLETED", flush=True)
        return p26.read(receipt_path)
    rows = task_rows(view_root, fold_id, task)
    done = _done(out)
    remaining = [row for row in rows if row["synthetic_id"] not in done]
    run = p26.extract_run(final_zip, task, target, work_root)
    sanity = p26.read(run / "sanity-evidence.json")
    with synthetic_graphs(view_root, fold_id) if task == "P2.4" else nullcontext():
        if task == "P2.1":
            score, reproduce, model = p26.semantic_scorer(repo_root, run, semantic_root, fold_id)
        elif task == "P2.4":
            score, one, model = p26.gtat_scorer(repo_root, run, graph_root, fold_id, device)
        else:
            score, one, model = p26.sequence_scorer(repo_root, run, lightweight=task == "P2.3")
    if task != "P2.1":  # the run's own stored validation window, as in the P2.6 and P3.2 passes
        val = pl.read_parquet(Path(data_root) / "folds" / fold_id / "sequence/val/windows.parquet")
        sample = p26.hydrate(val.filter(pl.col("window_id") == sanity["unit_id"]).to_dicts(),
                             p26._messages(data_root, fold_id))
        if len(sample) != 1:
            raise SyntheticError(f"{task} sanity window not found in the sequence bundle")
        reproduce = lambda _: one(sample[0])["score_raw"]  # noqa: E731
    reproduced, expected = reproduce(sanity), sanity["score_raw"]
    check = {"expected": expected, "reproduced": reproduced, "abs_diff": abs(reproduced - expected),
             "pass": math.isclose(reproduced, expected, rel_tol=p26.REL_TOL, abs_tol=p26.ABS_TOL)}
    print(json.dumps({"task": task, "target": target, "sanity_reproduction": check}), flush=True)
    if not check["pass"]:
        raise SyntheticError(f"{task} {target} cannot reproduce its sanity evidence; refusing to score")
    start = time.monotonic()
    with out.open("a", encoding="utf-8") as handle:
        for begin in range(0, len(remaining), CHUNK):
            chunk = remaining[begin:begin + CHUNK]
            for row, native in zip(chunk, score(chunk)):
                envelope = p26.convert(native)
                handle.write(json.dumps({"synthetic_id": row["synthetic_id"], "mutation": row["mutation"],
                                         **{k: envelope[k] for k in ("score_raw", "status", "reason", "coverage",
                                                                     "uncertainty", "uncertainty_method")}},
                                        sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps({"task": task, "target": target, "scored": len(rows) - len(remaining) + begin + len(chunk),
                              "of": len(rows)}), flush=True)
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    if sorted(r["synthetic_id"] for r in written) != sorted(r["synthetic_id"] for r in rows):
        raise SyntheticError(f"{task} {target}: the scores do not cover exactly the copies it reads")
    view = Path(view_root) / VIEW / fold_id
    receipt = {"schema": "SEQLOGAD-P3.7-SYNTHETIC-SCORING-RECEIPT-1", "task": task, "expert_id": expert,
               "fold_id": fold_id, "final_zip_sha256": p26.FINAL_SHA256[task], "run_id": run.name,
               "sanity_reproduction": check, "copies": len(rows), "reads": list(READS[task]),
               "mutations_sha256": sha256_file(view / "mutations.parquet"), "rows_sha256": sha256_file(view / "rows.parquet"),
               "status_counts": {s: sum(r["status"] == s for r in written) for s in ("VALID", "LOW_COVERAGE", "ABSTAIN")},
               "scores_sha256": sha256_file(out), "seconds_this_session": time.monotonic() - start, "labels_read": False}
    p26.dump(receipt_path, receipt)
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(f"{task}_{target}=SYNTHETIC_SCORED", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build", help="write the synthetic view (mutations and message rows) on a CPU")
    make.add_argument("--data-root", required=True, help="extracted phase2_sequence bundle")
    make.add_argument("--out-root", required=True)
    run = sub.add_parser("score", help="score the synthetic copies with frozen experts")
    for name in ("--repo-root", "--view-root", "--data-root", "--output-root", "--work-root"):
        run.add_argument(name, required=True)
    run.add_argument("--semantic-root")
    run.add_argument("--graph-root")
    run.add_argument("--final", action="append", required=True, help="TASK=path to its seed-42 FINAL ZIP")
    run.add_argument("--tasks", default="P2.3,P2.2,P2.1")
    run.add_argument("--targets", default="BGL,HDFS")
    run.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.data_root, args.out_root), indent=2))
        return
    finals = dict(item.split("=", 1) for item in args.final)
    for task in args.tasks.split(","):
        for target in args.targets.split(","):
            score_task(task, target, repo_root=args.repo_root, view_root=args.view_root, data_root=args.data_root,
                       semantic_root=args.semantic_root, graph_root=args.graph_root, final_zip=finals[task],
                       output_root=args.output_root, work_root=args.work_root, device=args.device)
    print("P3.7_SYNTHETIC_SCORING=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
