"""P2.6 scoring pass (DIV-P2-001 §7.1): each expert's selected checkpoint scores the held-out windows.

Label-free. Shared helpers were refactored after P2.1/P2.2 trained, so the training-time implementation
digest no longer matches; instead every expert must first reproduce the score stored in its own
`sanity-evidence.json` (behavioural equivalence) or the run stops. Output: one EVID-CS-001 envelope per
line in `<output_root>/<fold_id>/<EXPERT_ID>.jsonl`, resumable, plus a receipt per expert and fold.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

import polars as pl

from seqlogad.common.checksum import sha256_file
from seqlogad.evidence.convert import convert
from seqlogad.phase2_execution.specs import FOLDS
from seqlogad.semantic.checkpoint import verify_checkpoint
from seqlogad.semantic.contracts import dump, read
from seqlogad.sequence.contracts import PARTITIONS

from .units import evaluation_windows

TASKS = {"P2.1": "SEMANTIC_LLAMA", "P2.2": "SEQUENCE_LLAMA_REFERENCE",
         "P2.3": "SEQUENCE_LIGHTWEIGHT", "P2.4": "GTAT_STRUCTURAL_TEMPORAL"}
FINAL_SHA256 = {
    "P2.1": "8308a1b28875c1012b8a1942ca90898a4ca39e6c68f52e46bdd3e9601c7e603f",
    "P2.2": "6c009f2cd91c37899e340d877e4902197f9f5f96b4e164dbf86cb749583265c6",
    "P2.3": "eb8fe68141fde1f32516b6330b108d9965c4d7d9d130736912663ef52a68e2c4",
    "P2.4": "ff82c65343229183fd6caa7c0518961e58301ee5c7f71eb2d5fdce5eabbcfde3",
}
REL_TOL, ABS_TOL = 1e-3, 1e-5
CHUNK = 64


def extract_run(final_zip, task, target, work_root):
    """Extract the fold's run directory from a verified FINAL ZIP."""
    if sha256_file(final_zip) != FINAL_SHA256[task]:
        raise ValueError(f"{task} FINAL ZIP SHA-256 mismatch")
    destination = Path(work_root) / task / target
    if not (destination / "EXTRACTED").is_file():
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True)
        with zipfile.ZipFile(final_zip) as outer:
            payload = outer.read(f"{task}_S42/targets/{task}_{target}_S42.zip")
        with zipfile.ZipFile(io.BytesIO(payload)) as inner:
            for name in inner.namelist():
                member = PurePosixPath(name)
                if member.is_absolute() or ".." in member.parts or "\\" in name:
                    raise ValueError("unsafe member in target ZIP")
            inner.extractall(destination)
        (destination / "EXTRACTED").write_text("ok\n")
    runs = [path for path in destination.iterdir() if path.is_dir()]
    if len(runs) != 1:
        raise ValueError("target ZIP must hold exactly one run directory")
    return runs[0]


def _messages(data_root, fold_id):
    events = pl.read_parquet(Path(data_root) / "folds" / fold_id / "sequence/metadata/events.parquet")
    events = events.filter(pl.col("partition") == PARTITIONS["val"])
    return {(arch, rank): message for arch, rank, message in
            events.select("architecture_id", "rank", "normalized_message").iter_rows()}


def hydrate(rows, messages):
    for row in rows:
        row["context_messages"] = [messages[(row["architecture_id"], rank)] for rank in row["context_ranks"]]
        row["target_message"] = messages[(row["architecture_id"], row["target_rank"])]
        row["target_record_id"] = f"{row['architecture_id']}:{row['target_rank']}"
    return rows


def _checkpoint(run):
    manifest = read(run / "manifest.json")
    checkpoint = run / "checkpoints" / manifest["best_checkpoint"]
    if not checkpoint.resolve().is_relative_to((run / "checkpoints").resolve()):
        raise ValueError("checkpoint path outside run")
    return manifest, checkpoint


def _record_ids(row):
    return [f"{row['architecture_id']}:{rank}" for rank in list(row["context_ranks"]) + [row["target_rank"]]]


def semantic_scorer(repo_root, run, semantic_root, fold_id):
    from seqlogad.semantic.config import identity, load_config
    from seqlogad.semantic.model import load_model
    from seqlogad.semantic.scoring import score_message

    cfg = read(run / "config.yaml")
    manifest, checkpoint = _checkpoint(run)
    defaults, base = load_config(repo_root)
    if any(cfg[key] != defaults[key] for key in ("model_id", "model_revision", "tokenizer_id",
                                                 "tokenizer_revision", "base_config_sha256", "expert_id")):
        raise ValueError("P2.1 checkpoint incompatible with the frozen base")
    # config.yaml gains real_training_completed=True after training; the run identity was taken before
    # that, so SemanticExpert.from_checkpoint rejects every finished run. Check the pre-training digest.
    if identity(dict(cfg, real_training_completed=False)) != manifest["identity"]["config_sha256"]:
        raise ValueError("P2.1 config differs from the trained run")
    verify_checkpoint(checkpoint, manifest["identity"])
    model, tokenizer, dtype = load_model(cfg, base, checkpoint / "adapter")
    if str(dtype) != manifest["precision"]:
        raise ValueError("P2.1 inference precision differs from the trained run")
    model.eval()
    metadata = {"fold_id": manifest["fold_id"],
                "checkpoint_sha256": sha256_file(checkpoint / "adapter/adapter_model.safetensors"),
                "input_manifest_sha256": manifest["data_manifest_sha256"],
                "context_config_sha256": manifest["identity"]["config_sha256"]}

    def one(message, architecture, record):
        return score_message(model, tokenizer, cfg, message,
                             dict(metadata, architecture_id=architecture, unit_id=record, record_ids=[record]))

    def score(rows):
        return [one(row["target_message"], row["architecture_id"], row["target_record_id"]) for row in rows]

    def reproduce(sanity):
        record = sanity["record_ids"][0]
        architecture, rank = record.split(":")
        records = pl.read_parquet(Path(semantic_root) / "folds" / fold_id / "semantic/val/records.parquet")
        match = records.filter((pl.col("architecture_id") == architecture) & (pl.col("rank") == int(rank)))
        if match.height != 1:
            raise ValueError("P2.1 sanity record not found in the semantic bundle")
        return one(match["normalized_message"][0], architecture, record)["score_raw"]

    return score, reproduce, model


def sequence_scorer(repo_root, run, lightweight):
    from seqlogad.sequence.scoring import score_next_log

    cfg = read(run / "config.yaml")
    manifest, checkpoint = _checkpoint(run)
    verify_checkpoint(checkpoint, manifest["identity"])
    if lightweight:
        from seqlogad.sequence import lightweight as light
        from seqlogad.sequence.model import load_tokenizer, verify_access

        _, base = light.load_config(repo_root)
        verify_access(base)
        tokenizer = load_tokenizer(cfg)
        model = light.build_model(cfg)
        light.load_weights(model, checkpoint)
        model.cuda()
    else:
        from seqlogad.sequence.config import load_config
        from seqlogad.sequence.expert import assert_encoding_compatible
        from seqlogad.sequence.model import load_model

        defaults, base = load_config(repo_root, candidate="v13r1")
        assert_encoding_compatible(cfg, defaults)
        if any(cfg[key] != defaults[key] for key in ("model_id", "model_revision", "tokenizer_id",
                                                     "tokenizer_revision", "expert_id", "adapter_name")):
            raise ValueError("P2.2 checkpoint incompatible with the frozen base/config")
        model, tokenizer, dtype = load_model(cfg, base, checkpoint / "adapter")
        if str(dtype) != manifest["precision"]:
            raise ValueError("P2.2 inference precision differs from the trained run")
    model.eval()
    metadata = {"fold_id": manifest["fold_id"],
                "checkpoint_sha256": sha256_file(checkpoint / "adapter/adapter_model.safetensors"),
                "input_manifest_sha256": manifest["data_manifest_sha256"],
                "context_config_sha256": manifest["identity"]["config_sha256"]}

    def one(row):
        return score_next_log(model, tokenizer, cfg, row["context_messages"], row["target_message"],
                              dict(metadata, architecture_id=row["architecture_id"], unit_id=row["window_id"],
                                   record_ids=_record_ids(row)))

    return (lambda rows: [one(row) for row in rows]), one, model


def gtat_scorer(repo_root, run, graph_root, fold_id, device):
    import torch

    from seqlogad.graph import config as gconfig
    from seqlogad.graph import train as gtrain
    from seqlogad.graph.contracts import load_graph_nodes, load_graph_windows
    from seqlogad.graph.features import NodeStore, examples

    cfg = gconfig.load_config(repo_root)
    manifest, checkpoint = _checkpoint(run)
    if gconfig.identity(cfg) != manifest["identity"]["config_sha256"]:
        raise ValueError("P2.4 config differs from the trained run")
    verify_checkpoint(checkpoint, manifest["identity"])
    store = NodeStore(load_graph_nodes(graph_root, fold_id), cfg)
    items = {item["window_id"]: item for item in examples(store, load_graph_windows(graph_root, fold_id, "val"))}
    model = gtrain.build_model(cfg)
    gtrain.load_weights(model, checkpoint)
    model.to(torch.device(device)).eval()
    common = {"schema_version": "SEQLOGAD-EVIDENCE-DRAFT-P2.4", "expert_id": cfg["expert_id"],
              "expert_role": cfg["role"], "fold_id": fold_id, "score_direction": cfg["score"]["direction"],
              "score_calibrated": None, "calibration_status": "NOT_CALIBRATED", "uncertainty": None,
              "uncertainty_method": "UNCERTAINTY_NOT_ESTIMATED_IN_PHASE2", "model_revision": cfg["id"],
              "checkpoint_sha256": sha256_file(checkpoint / gtrain.WEIGHTS),
              "input_manifest_sha256": manifest["graph_manifest_sha256"],
              "context_config_sha256": gconfig.identity(cfg["graph_view"])}

    def score(rows):
        chosen = [items[row["window_id"]] for row in rows]
        results = {result["window_id"]: result
                   for result in gtrain.score_examples(model, store, chosen, torch.device(device))}
        natives = []
        for item in chosen:
            result, scored = results[item["window_id"]], results[item["window_id"]]["status"] == "SCORED"
            natives.append(dict(
                common, architecture_id=item["architecture_id"], unit_id=item["window_id"],
                record_ids=[f"{item['architecture_id']}:{item['target_rank']}"],
                score_raw=result["nll"], coverage=result.get("coverage", 0.0),
                evidence=({name: result[name] for name in gtrain.COMPONENTS}
                          | {"context_records": result["context_records"]}) if scored else {},
                localisation={"target_rank": item["target_rank"]},
                status="SCORED" if scored else "ABSTAIN", reason=None if scored else "NO_CONTEXT_RECORD"))
        return natives

    return score, (lambda row: score([row])[0]), model


def _done(path):
    """Target records already written; drop a trailing partial line left by a disconnect."""
    if not path.exists():
        return set()
    lines = path.read_bytes().splitlines()
    kept, done = [], set()
    for line in lines:
        try:
            done.add(json.loads(line)["target_record_id"])
            kept.append(line)
        except (ValueError, KeyError):
            break
    if len(kept) != len(lines):
        path.write_bytes(b"".join(line + b"\n" for line in kept))
    return done


def score_task(task, target, *, repo_root, data_root, final_zip, output_root, work_root,
               semantic_root=None, graph_root=None, device="cuda"):
    fold_id = FOLDS[target]
    out = Path(output_root) / fold_id / f"{TASKS[task]}.jsonl"
    receipt_path = out.with_suffix(".receipt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = hydrate(evaluation_windows(data_root, fold_id).to_dicts(), _messages(data_root, fold_id))
    done = _done(out)
    remaining = [row for row in rows if row["target_record_id"] not in done]
    if not remaining and receipt_path.is_file():
        print(f"{task}_{target}=SKIP_COMPLETED", flush=True)
        return read(receipt_path)
    run = extract_run(final_zip, task, target, work_root)
    sanity = read(run / "sanity-evidence.json")
    if task == "P2.1":
        score, reproduce, model = semantic_scorer(repo_root, run, semantic_root, fold_id)
    elif task == "P2.4":
        score, one, model = gtat_scorer(repo_root, run, graph_root, fold_id, device)
    else:
        score, one, model = sequence_scorer(repo_root, run, lightweight=task == "P2.3")
    if task != "P2.1":
        val = pl.read_parquet(Path(data_root) / "folds" / fold_id / "sequence/val/windows.parquet")
        sample = hydrate(val.filter(pl.col("window_id") == sanity["unit_id"]).to_dicts(),
                         _messages(data_root, fold_id))
        if len(sample) != 1:
            raise ValueError(f"{task} sanity window not found in the sequence bundle")
        reproduce = lambda _: one(sample[0])["score_raw"]  # noqa: E731
    reproduced = reproduce(sanity)
    expected = sanity["score_raw"]
    passed = math.isclose(reproduced, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    check = {"expected": expected, "reproduced": reproduced, "abs_diff": abs(reproduced - expected), "pass": passed}
    print(json.dumps({"task": task, "target": target, "sanity_reproduction": check}), flush=True)
    if not passed:
        raise ValueError(f"{task} {target} cannot reproduce its sanity evidence; refusing to score")
    start, written = time.monotonic(), 0
    with out.open("a") as handle:
        for begin in range(0, len(remaining), CHUNK):
            chunk = remaining[begin:begin + CHUNK]
            for row, native in zip(chunk, score(chunk)):
                envelope = convert(native)
                if envelope["target_record_id"] != row["target_record_id"]:
                    raise ValueError("scored envelope does not match its evaluation window")
                handle.write(json.dumps(envelope, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += len(chunk)
            print(json.dumps({"task": task, "target": target, "scored": len(done) + written,
                              "of": len(rows)}), flush=True)
    envelopes = [json.loads(line) for line in out.read_text().splitlines()]
    if {envelope["target_record_id"] for envelope in envelopes} != {row["target_record_id"] for row in rows}:
        raise ValueError("evidence file does not cover exactly the evaluation windows")
    receipt = {"schema": "SEQLOGAD-P2.6-SCORING-RECEIPT-1", "task": task, "expert_id": TASKS[task],
               "fold_id": fold_id, "final_zip_sha256": FINAL_SHA256[task], "run_id": run.name,
               "sanity_reproduction": check, "windows": len(rows),
               "status_counts": {status: sum(e["status"] == status for e in envelopes)
                                 for status in ("VALID", "LOW_COVERAGE", "ABSTAIN")},
               "seconds_this_session": time.monotonic() - start, "evidence_sha256": sha256_file(out),
               "labels_read": False}
    dump(receipt_path, receipt)
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(f"{task}_{target}=SCORED", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--semantic-root", help="semantic bundle root; needed by P2.1")
    parser.add_argument("--graph-root", help="graph view root; needed by P2.4")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--final", action="append", required=True, help="TASK=path to its FINAL ZIP")
    parser.add_argument("--tasks", default="P2.4,P2.3,P2.2,P2.1")
    parser.add_argument("--targets", default="BGL,HDFS")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    finals = dict(item.split("=", 1) for item in args.final)
    for task in args.tasks.split(","):
        for target in args.targets.split(","):
            score_task(task, target, repo_root=args.repo_root, data_root=args.data_root,
                       final_zip=finals[task], output_root=args.output_root, work_root=args.work_root,
                       semantic_root=args.semantic_root, graph_root=args.graph_root, device=args.device)
    print("P2.6_SCORING=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
