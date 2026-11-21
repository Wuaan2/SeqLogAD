"""P2.4 GTAT-ST-001 trainer: label-free masked-target training, source-development checkpoint selection."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from seqlogad.common.checksum import sha256_file
from seqlogad.semantic.checkpoint import save_checkpoint, seal, verify_checkpoint, verify_seal
from seqlogad.semantic.contracts import dump
from seqlogad.semantic.prepare import git_state
from seqlogad.semantic.runtime import export_environment, seed_all
from seqlogad.sequence.contracts import validate_bundle
from seqlogad.sequence.objective import select_windows
from seqlogad.sequence.recovery_state import terminal_metrics
from seqlogad.sequence.source_development import load_source_development
from seqlogad.sequence.source_development_metrics import source_development_metrics
from seqlogad.sequence.train import validate_run_id

from .config import development_run_id, identity, implementation_digest, load_config
from .features import NodeStore, build_graph, collate, examples
from .model import GTATModel, head_losses

# Same file name as the P2.2/P2.3 adapter so checkpoint verification and Drive recovery stay shared.
WEIGHTS = "adapter/adapter_model.safetensors"
COMPONENTS = ("temporal_nll", "structural_nll", "content_nll")


def build_model(cfg):
    import torch

    torch.manual_seed(cfg["seed"])
    model = GTATModel(cfg)
    if sum(parameter.numel() for parameter in model.parameters()) > cfg["model"]["max_total_parameters"]:
        raise ValueError("P2.4 model exceeds the preregistered parameter budget")
    return model


def _write_weights(model, directory):
    from safetensors.torch import save_model

    Path(directory).mkdir(parents=True)
    save_model(model, str(Path(directory) / "adapter_model.safetensors"))


def save_gtat_checkpoint(*args, **kwargs):
    kwargs.update(adapter_name="none", write_weights=_write_weights)
    return save_checkpoint(*args, **kwargs)


def load_weights(model, checkpoint):
    from safetensors.torch import load_model

    load_model(model, str(Path(checkpoint) / WEIGHTS))


def _to(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def score_examples(model, store, items, device, batch_size=128):
    """Per-window score and components; ABSTAIN windows get null scores."""
    import torch

    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            scored = [item for item in items if item["status"] == "SCORED"]
            for start in range(0, len(scored), batch_size):
                chunk = scored[start:start + batch_size]
                graphs = [build_graph(store, item["context"], item["target"]) for item in chunk]
                batch = _to(collate(graphs), device)
                losses = head_losses(model(batch), batch)
                for index, (item, graph) in enumerate(zip(chunk, graphs)):
                    parts = {name: float(losses[name][index]) for name in COMPONENTS}
                    total = sum(parts.values())
                    if not math.isfinite(total):
                        raise ValueError("nonfinite P2.4 validation score")
                    rows.append({"architecture_id": item["architecture_id"], "window_id": item["window_id"],
                                 "nll": total, **parts, "coverage": graph["coverage"],
                                 "context_records": graph["context_records"], "status": "SCORED"})
            rows.extend({"architecture_id": item["architecture_id"], "window_id": item["window_id"],
                         "nll": None, "status": "ABSTAIN"} for item in items if item["status"] != "SCORED")
    finally:
        model.train(was_training)
    return rows


def summarize(rows, labels):
    per_source = defaultdict(lambda: defaultdict(list))
    abstained = defaultdict(int)
    for row in rows:
        if row["status"] != "SCORED":
            abstained[row["architecture_id"]] += 1
            continue
        for name in ("nll",) + COMPONENTS:
            per_source[row["architecture_id"]][name].append(row[name])
    means = {arch: {name: sum(values) / len(values) for name, values in names.items()}
             for arch, names in sorted(per_source.items())}
    scored = {row["window_id"] for row in rows if row["status"] == "SCORED"}
    usable = {window: entry for window, entry in labels.items() if window in scored}
    return {
        "aggregate_score": sum(item["nll"] for item in means.values()) / len(means),
        "per_source": means,
        "abstained_windows": dict(abstained),
        "source_development": source_development_metrics(
            [row for row in rows if row["status"] == "SCORED"], usable),
        "source_development_abstained_labeled_windows": len(labels) - len(usable),
    }


def _lr_factor(step, cfg):
    warmup, total = cfg["training"]["warmup_steps"], cfg["training"]["max_steps"]
    return (step + 1) / warmup if step < warmup else (total - step) / max(1, total - warmup)


def optimize(model, store, train, val, labels, cfg, optimizer, run, run_identity, first, best,
             early, device, checkpoint_callback=None):
    import torch

    training = cfg["training"]
    terminal = terminal_metrics(run, first, {"max_steps": training["max_steps"],
                                             "early_stopping_patience": training["early_stopping_patience"]},
                                early)
    if terminal is not None:
        return best, terminal
    by_source = defaultdict(list)
    for item in train:
        by_source[item["architecture_id"]].append(item)
    sources = sorted(by_source)
    stale, monitor_best = early.get("stale_evaluations", 0), early.get("monitor_best")
    history = list(early.get("validation_history", []))
    trim = int(cfg["objective"]["trimmed_fraction"] * training["batch_graphs_per_source"] * len(sources))
    metrics = {}
    model.train()
    for step in range(first, training["max_steps"]):
        factor = min(1.0, _lr_factor(step, cfg))
        for group in optimizer.param_groups:
            group["lr"] = training["learning_rate"] * factor
        chosen = [pool[random.randrange(len(pool))] for source in sources
                  for pool in [by_source[source]] for _ in range(training["batch_graphs_per_source"])]
        batch = _to(collate([build_graph(store, item["context"], item["target"]) for item in chosen]), device)
        optimizer.zero_grad(set_to_none=True)
        losses = head_losses(model(batch), batch)
        per_graph = sum(losses.values())
        if not torch.isfinite(per_graph).all():
            dump(run / "numerical-failure.json", {"step_attempt": step + 1, "completed_steps": step,
                                                  "phase": "forward", "optimizer_update_applied": False})
            raise ValueError("nonfinite P2.4 training loss; update not applied")
        kept = torch.sort(per_graph).values[:len(chosen) - trim]  # trimmed loss: drop the highest-loss graphs
        loss = kept.mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
        if not torch.isfinite(norm):
            dump(run / "numerical-failure.json", {"step_attempt": step + 1, "completed_steps": step,
                                                  "phase": "backward", "optimizer_update_applied": False})
            raise ValueError("nonfinite P2.4 gradients; update not applied")
        optimizer.step()
        metrics = {"step": step + 1, "train_trimmed_loss": float(loss.detach()), "gradient_norm": float(norm),
                   "learning_rate_factor": factor, "trimmed_graphs": trim, "optimizer_update_applied": True,
                   **{f"train_{name}": float(value.detach().mean()) for name, value in losses.items()}}
        should_stop = False
        checkpoint = None
        if (step + 1) % training["validation_steps"] == 0 or step + 1 == training["max_steps"]:
            validation = summarize(score_examples(model, store, val, device), labels)
            development = validation["source_development"]
            value = development["macro_f1"]
            checkpoint = run / "checkpoints" / f"step-{step + 1}"
            improved = best["f1"] is None or value > best["f1"]
            if improved:
                best = {"f1": value, "checkpoint": checkpoint.name, "source_development": development}
            if monitor_best is None or value > monitor_best + training["early_stopping_min_delta"]:
                monitor_best, stale = value, 0
            else:
                stale += 1
            history.append({"step": step + 1, "aggregate_score": validation["aggregate_score"],
                            "selection_value": value, "selection_improved": improved})
            should_stop = stale >= training["early_stopping_patience"]
            metrics.update(source_validation=validation, source_development_macro_f1=value,
                           early_stopping={"stale_evaluations": stale,
                                           "patience": training["early_stopping_patience"],
                                           "metric": "f1", "direction": "max", "triggered": should_stop})
            save_gtat_checkpoint(
                model, optimizer, torch.amp.GradScaler("cuda", enabled=False), checkpoint, step + 1,
                run_identity, best, {"stale_evaluations": stale, "monitor_best": monitor_best,
                                     "validation_history": history},
                {"kind": "LINEAR_WARMUP_THEN_LINEAR_DECAY", "last_completed_step": step + 1,
                 "learning_rate_factor": factor},
            )
        with (run / "logs/train.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if checkpoint is not None and checkpoint_callback:
            checkpoint_callback(run, checkpoint, best["checkpoint"])
        print(json.dumps({key: value for key, value in metrics.items() if key != "source_validation"}),
              flush=True)
        if should_stop:
            break
    return best, metrics


def run_training(repo_root, data_root, output_root, fold_id, run_id, resume_from=None,
                 expected_bundle_sha256=None, checkpoint_callback=None,
                 source_development_root=None, expected_source_development_sha256=None,
                 graph_root=None, expected_graph_sha256=None, device=None):
    import torch

    from .contracts import load_graph_nodes, load_graph_windows, validate_graph_bundle

    cfg = load_config(repo_root)
    if not expected_bundle_sha256 or not expected_graph_sha256:
        raise ValueError("trusted P2.4 sequence and graph bundle SHA-256 are required")
    validation = validate_bundle(data_root, fold_id, expected_bundle_sha256)
    graph = validate_graph_bundle(graph_root, fold_id, expected_graph_sha256, data_root)
    validate_run_id(run_id)
    if fold_id not in cfg["target_folds"] or run_id != development_run_id(fold_id, cfg["seed"]):
        raise ValueError("P2.4 run identity differs from frozen seed-42 matrix")
    if not source_development_root or not expected_source_development_sha256:
        raise ValueError("P2.4 requires the trusted source-development label sidecar")
    sidecar_hash = sha256_file(Path(source_development_root) / "manifest.json")
    if sidecar_hash != expected_source_development_sha256:
        raise ValueError("source-development manifest digest mismatch")
    training = cfg["training"]
    selected_val = select_windows(data_root, fold_id, "val", training["validation_windows_per_source"],
                                  cfg["seed"], replacement=False)
    labels = load_source_development(source_development_root, data_root, fold_id, selected_val)
    device = torch.device(device or "cuda")
    run = Path(output_root) / fold_id / cfg["expert_id"] / run_id
    resume_path = Path(resume_from).resolve() if resume_from else None
    resume_in_place = bool(resume_path and run.exists() and resume_path.is_relative_to(run.resolve()))
    if run.exists() and not resume_in_place:
        raise ValueError("P2.4 run directory exists without verified in-place resume")
    run.mkdir(parents=True, exist_ok=resume_in_place)
    (run / "logs").mkdir(exist_ok=resume_in_place)
    (run / "checkpoints").mkdir(exist_ok=resume_in_place)
    if resume_in_place:
        (run / "checksums.sha256").unlink(missing_ok=True)
    run_identity = {
        "implementation_sha256": implementation_digest(repo_root),
        "config_sha256": identity(cfg),
        "bundle_sha256": validation["bundle_sha256"],
        "graph_bundle_sha256": graph["bundle_sha256"],
        "fold_id": fold_id,
        "precision": "torch.float32",
        "adapter": "none",
        "source_development_manifest_sha256": sidecar_hash,
    }
    resume = verify_checkpoint(resume_from, run_identity) if resume_from else None
    dump(run / "config.yaml", cfg)
    export_environment(run)
    seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    store = NodeStore(load_graph_nodes(graph_root, fold_id), cfg)
    train = [item for item in examples(store, load_graph_windows(graph_root, fold_id, "train"))
             if item["status"] == "SCORED"]
    val_rows = {row["window_id"]: row for row in examples(store, load_graph_windows(graph_root, fold_id, "val"))}
    missing = [row["window_id"] for row in selected_val if row["window_id"] not in val_rows]
    if missing:
        raise ValueError("selected validation windows are missing from the graph view")
    val = [val_rows[row["window_id"]] for row in selected_val]
    dump(run / "selection.json", {"val": [{"architecture_id": item["architecture_id"],
                                           "window_id": item["window_id"]} for item in val]})
    model = build_model(cfg)
    if resume_from:
        load_weights(model, resume_from)
    model.to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    seed_all(cfg["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    first, best = 0, {"f1": None, "checkpoint": None}
    early = {"stale_evaluations": 0, "monitor_best": None, "validation_history": []}
    if resume:
        state = torch.load(Path(resume_from) / "state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        first, best, early = state["step"], state["best"], state["early_stopping"]
        if state.get("scheduler", {}).get("last_completed_step") != first:
            raise ValueError("P2.4 resume lacks matching scheduler state")
        previous_best = Path(resume_from).parent / best["checkpoint"]
        if not previous_best.exists():
            raise ValueError("P2.4 resume best checkpoint missing")
        if not previous_best.resolve().is_relative_to(run.resolve()):
            shutil.copytree(previous_best, run / "checkpoints/resume-best")
            best["checkpoint"] = "resume-best"
    try:
        best, metrics = optimize(model, store, train, val, labels, cfg, optimizer, run, run_identity,
                                 first, best, early, device, checkpoint_callback)
        selected_checkpoint = run / "checkpoints" / best["checkpoint"]
        load_weights(model, selected_checkpoint)
        timer = time.monotonic()
        rows = score_examples(model, store, val, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds_per_window = (time.monotonic() - timer) / len(val)
        final = summarize(rows, labels)
        checkpoint_hash = sha256_file(selected_checkpoint / WEIGHTS)
        sample = next(row for row in rows if row["status"] == "SCORED")
        item = next(entry for entry in val if entry["window_id"] == sample["window_id"])
        dump(run / "validation-scores.json", rows)
        dump(run / "sanity-evidence.json", {
            "schema_version": "SEQLOGAD-EVIDENCE-DRAFT-P2.4", "expert_id": cfg["expert_id"],
            "expert_role": cfg["role"], "fold_id": fold_id, "architecture_id": item["architecture_id"],
            "unit_id": item["window_id"],
            "record_ids": [f"{item['architecture_id']}:{item['target_rank']}"],
            "score_raw": sample["nll"], "score_direction": cfg["score"]["direction"],
            "score_calibrated": None, "calibration_status": "NOT_CALIBRATED",
            "uncertainty": None, "uncertainty_method": None,
            "coverage": sample["coverage"],
            "evidence": {name: sample[name] for name in COMPONENTS} | {"context_records": sample["context_records"]},
            "localisation": {"target_rank": item["target_rank"]},
            "status": "SCORED", "reason": "UNCERTAINTY_NOT_ESTIMATED_IN_PHASE2",
            "model_revision": cfg["id"], "checkpoint_sha256": checkpoint_hash,
            "input_manifest_sha256": graph["bundle_sha256"], "context_config_sha256": identity(cfg["graph_view"]),
        })
        metrics.update(
            source_development_macro_f1=best["f1"],
            source_development_best=best["source_development"],
            source_development_manifest_sha256=sidecar_hash,
            final_source_validation=final,
            runtime_seconds=time.monotonic() - start,
            best=best,
            peak_vram_allocated=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            peak_vram_reserved=torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            resources={"total_parameters": parameters, "trainable_parameters": parameters,
                       "parameter_criterion_met": parameters <= cfg["model"]["max_total_parameters"],
                       "validation_seconds_per_window": seconds_per_window},
            natural_anomaly_metrics={"status": "TARGET_NOT_EVALUATED_SOURCE_DEVELOPMENT_ONLY",
                                     "accuracy": None, "f1": None, "precision": None, "recall": None,
                                     "auroc": None, "average_precision": None},
        )
        dump(run / "metrics.json", metrics)
        dump(run / "manifest.json", {
            "run_id": run_id, "fold_id": fold_id, "expert_id": cfg["expert_id"], "identity": run_identity,
            "base_model_id": None, "pretrained_weights": False, "tokenizer_id": None,
            "adapter_name": None, "seed": cfg["seed"], "precision": "torch.float32", "model_config": cfg,
            "resume_parent": str(resume_from) if resume_from else None,
            "data_manifest_sha256": validation["bundle_sha256"],
            "graph_manifest_sha256": graph["bundle_sha256"],
            "git": git_state(repo_root), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "completion_status": "TRAINING_COMPLETED_REQUIRES_REVIEW", "scientific_results": "NOT_EVALUATED",
            "real_training_completed": True, "environment": "environment.json", "metrics": "metrics.json",
            "dependency_lock": "dependency-lock.txt", "checkpoint": str(selected_checkpoint.relative_to(run)),
            "best_checkpoint": best["checkpoint"], "checkpoint_sha256": checkpoint_hash,
        })
        seal(run)
        verify_seal(run, require_nested_ledgers=True)
        return {"run_directory": str(run), "metrics": metrics}
    except Exception as exc:
        dump(run / "failure.json", {"type": type(exc).__name__, "message": str(exc), "training_completed": False})
        raise
