"""P2.3 lightweight Sequence expert: a small causal Transformer trained from scratch.

It shares the P2.2 tokenizer, windows, prompt, objective, context views and
source-development checkpoint selection, so only model capacity and pretraining
differ. Importing never loads weights.
"""
from __future__ import annotations

import os
import random
import shutil
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

from seqlogad.common.checksum import sha256_file
from seqlogad.semantic.checkpoint import save_checkpoint, seal, verify_checkpoint, verify_seal
from seqlogad.semantic.contracts import dump
from seqlogad.semantic.prepare import git_state
from seqlogad.semantic.runtime import export_environment, seed_all

from .config import TARGETS, identity, load_config as load_reference_config
from .contracts import validate_bundle
from .encoding import encoded_windows
from .implementation import implementation_digest
from .model import load_tokenizer, precision, verify_access
from .objective import select_windows
from .optimization import optimize
from .source_development import load_source_development
from .source_validation import score_validation
from .train import preflight, validate_run_id

CONFIG = "configs/models/sequence-lightweight-v1.yaml"
# Same file name as a P2.2 adapter so checkpoint verification and Drive recovery stay shared;
# for P2.3 it holds every weight of the small model.
WEIGHTS = "adapter/adapter_model.safetensors"
LORA_KEYS = ("adapter_name", "training_mode", "rank", "alpha", "use_rslora", "dropout",
             "target_modules", "model_id")
ALLOWED_OVERRIDES = {"id", "version", "expert_id", "inherits", "implementation_contract",
                     "architecture", "learning_rate", "max_steps", "warmup_steps",
                     "resource_criterion", "capability_comparison"}


def development_run_id(fold_id, seed=42):
    target = fold_id.removeprefix("FOLD-TARGET-ARCH-")
    if target not in TARGETS or seed != 42:
        raise ValueError("P2.3 development run is locked to seed 42 and the HDFS/BGL target folds")
    return f"P2.3-DEVELOPMENT-{target}-S42"


def load_config(repo_root):
    root = Path(repo_root)
    light = yaml.safe_load((root / CONFIG).read_text())["sequence_lightweight"]
    if set(light) - ALLOWED_OVERRIDES:
        raise ValueError("P2.3 may override only model capacity and optimizer budget")
    if light["inherits"] != "configs/models/sequence-reference-v13r1.yaml":
        raise ValueError("P2.3 must inherit the V13R1 reference contract")
    if light["expert_id"] != "SEQUENCE_LIGHTWEIGHT" or light["architecture"]["pretrained_weights"]:
        raise ValueError("P2.3 expert identity or from-scratch policy changed")
    reference, base = load_reference_config(root, candidate="v13r1")
    cfg = {key: value for key, value in deepcopy(reference).items() if key not in LORA_KEYS}
    cfg.update(deepcopy(light), model_revision="NONE_TRAINED_FROM_SCRATCH")
    if cfg["warmup_steps"] >= cfg["max_steps"]:
        raise ValueError("invalid P2.3 optimizer budget")
    return cfg, base


def parameter_budget(cfg):
    criterion = cfg["resource_criterion"]
    return int(criterion["reference_total_parameters"] * criterion["max_total_parameter_fraction"])


def build_model(cfg):
    """Seeded random init; fp32 weights, BF16 autocast during training."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    arch = cfg["architecture"]
    torch.manual_seed(cfg["seed"])
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=arch["vocab_size"], hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"], num_hidden_layers=arch["num_hidden_layers"],
        num_attention_heads=arch["num_attention_heads"],
        num_key_value_heads=arch["num_key_value_heads"],
        max_position_embeddings=cfg["context_length"], rope_theta=arch["rope_theta"],
        tie_word_embeddings=arch["tie_word_embeddings"], bos_token_id=128000,
        eos_token_id=128001, pad_token_id=128001, use_cache=False,
        attn_implementation="sdpa",
    ))
    if sum(parameter.numel() for parameter in model.parameters()) > parameter_budget(cfg):
        raise ValueError("P2.3 model exceeds the preregistered parameter budget")
    return model


def assert_lightweight(model):
    if hasattr(model, "peft_config"):
        raise ValueError("P2.3 must not carry a PEFT adapter")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("P2.3 trains every parameter of the small model")


def _write_weights(model, directory):
    from safetensors.torch import save_model

    Path(directory).mkdir(parents=True)
    save_model(model, str(Path(directory) / "adapter_model.safetensors"))


def save_lightweight_checkpoint(*args, **kwargs):
    kwargs.update(adapter_name="none", write_weights=_write_weights)
    return save_checkpoint(*args, **kwargs)


def load_weights(model, checkpoint):
    from safetensors.torch import load_model

    load_model(model, str(Path(checkpoint) / WEIGHTS))


def run_training(repo_root, data_root, output_root, fold_id, run_id, resume_from=None,
                 expected_bundle_sha256=None, checkpoint_callback=None,
                 source_development_root=None, expected_source_development_sha256=None):
    import torch

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    cfg, base = load_config(repo_root)
    if not expected_bundle_sha256:
        raise ValueError("trusted P2.3 bundle SHA-256 is required")
    validation = validate_bundle(data_root, fold_id, expected_bundle_sha256)
    validate_run_id(run_id)
    if fold_id not in cfg["target_folds"] or run_id != development_run_id(fold_id, cfg["seed"]):
        raise ValueError("P2.3 run identity differs from frozen seed-42 matrix")
    if not source_development_root or not expected_source_development_sha256:
        raise ValueError("P2.3 requires the trusted source-development label sidecar")
    sidecar_hash = sha256_file(Path(source_development_root) / "manifest.json")
    if sidecar_hash != expected_source_development_sha256:
        raise ValueError("source-development manifest digest mismatch")
    selected = {
        "train": select_windows(data_root, fold_id, "train", cfg["train_windows_per_source"],
                                cfg["seed"], replacement=True),
        "val": select_windows(data_root, fold_id, "val", cfg["validation_windows_per_source"],
                              cfg["seed"], replacement=False),
    }
    development_labels = load_source_development(
        source_development_root, data_root, fold_id, selected["val"]
    )
    run = Path(output_root) / fold_id / cfg["expert_id"] / run_id
    resume_path = Path(resume_from).resolve() if resume_from else None
    resume_in_place = bool(resume_path and run.exists() and resume_path.is_relative_to(run.resolve()))
    if run.exists() and not resume_in_place:
        raise ValueError("P2.3 run directory exists without verified in-place resume")
    run.mkdir(parents=True, exist_ok=resume_in_place)
    (run / "logs").mkdir(exist_ok=resume_in_place)
    (run / "checkpoints").mkdir(exist_ok=resume_in_place)
    if resume_in_place:
        (run / "checksums.sha256").unlink(missing_ok=True)
    run_identity = {
        "implementation_sha256": implementation_digest(repo_root),
        "config_sha256": identity(cfg),
        "bundle_sha256": validation["bundle_sha256"],
        "fold_id": fold_id,
        "precision": str(precision(torch)),
        "adapter": "none",
        "source_development_manifest_sha256": sidecar_hash,
    }
    resume = verify_checkpoint(resume_from, run_identity) if resume_from else None
    dump(run / "config.yaml", cfg)
    export_environment(run)
    dump(run / "selection.json", {
        partition: [{"architecture_id": row["architecture_id"], "window_id": row["window_id"]}
                    for row in rows]
        for partition, rows in selected.items()
    })
    seed_all(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    verify_access(base)  # gated tokenizer only; base weights are never downloaded
    dtype = precision(torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = load_tokenizer(cfg)
    model = build_model(cfg)
    if resume_from:
        load_weights(model, resume_from)
    model.cuda()
    assert_lightweight(model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    train, rejected_train = encoded_windows(selected["train"], tokenizer, cfg, "train")
    val, rejected_val = encoded_windows(selected["val"], tokenizer, cfg, "validation")
    if not set(development_labels) <= {row["window_id"] for row in val}:
        raise ValueError("source development encoding coverage is incomplete")
    raw_train_windows = {
        (row["architecture_id"], row["window_id"]): row for row in selected["train"]
    }
    dump(run / "coverage.json", {
        "train_rejected": rejected_train,
        "validation_rejected": rejected_val,
        "train_mean_target_coverage": sum(row["coverage"] for row in train) / len(train),
        "validation_mean_target_coverage": sum(row["coverage"] for row in val) / len(val),
        "train_mean_retained_context_records": sum(row["retained_context_records"] for row in train) / len(train),
        "validation_mean_retained_context_records": sum(row["retained_context_records"] for row in val) / len(val),
    })
    dump(run / "preflight.json", preflight(model, tokenizer, train, dtype))
    seed_all(cfg["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(dtype == torch.float16), init_scale=cfg["fp16_initial_scale"],
        growth_interval=cfg["fp16_growth_interval"],
    )
    first, best = 0, {"f1": None, "checkpoint": None}
    early = {"stale_evaluations": 0, "monitor_best": None, "validation_history": []}
    if resume:
        state = torch.load(Path(resume_from) / "state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        first, best = state["step"], state["best"]
        early = state.get("early_stopping", early)
        if not {"stale_evaluations", "monitor_best", "validation_history"} <= set(early):
            raise ValueError("P2.3 resume lacks complete early-stopping state")
        if state.get("scheduler", {}).get("last_completed_step") != first:
            raise ValueError("P2.3 resume lacks matching scheduler state")
        if first > cfg["max_steps"]:
            raise ValueError("P2.3 checkpoint exceeds configured budget")
        previous_best = Path(resume_from).parent / best["checkpoint"]
        if not previous_best.exists():
            raise ValueError("P2.3 resume best checkpoint missing")
        if not previous_best.resolve().is_relative_to(run.resolve()):
            shutil.copytree(previous_best, run / "checkpoints/resume-best")
            best["checkpoint"] = "resume-best"
    try:
        best, metrics = optimize(
            model, tokenizer, dtype, train, val, cfg, optimizer, scaler, run,
            run_identity, first, best, early, checkpoint_callback, raw_train_windows,
            development_labels, check_model=assert_lightweight,
            save=save_lightweight_checkpoint,
        )
        selected_checkpoint = run / "checkpoints" / best["checkpoint"]
        load_weights(model, selected_checkpoint)
        timer = time.monotonic()
        score_validation(model, tokenizer, val, dtype)
        torch.cuda.synchronize()
        seconds_per_window = (time.monotonic() - timer) / len(val)
        from .scoring import score_next_log
        sample = selected["val"][0]
        checkpoint_hash = sha256_file(selected_checkpoint / WEIGHTS)
        metadata = {
            "fold_id": fold_id,
            "architecture_id": sample["architecture_id"],
            "unit_id": sample["window_id"],
            "record_ids": [f"{sample['architecture_id']}:{rank}" for rank in
                           list(sample["context_ranks"]) + [sample["target_rank"]]],
            "checkpoint_sha256": checkpoint_hash,
            "input_manifest_sha256": validation["bundle_sha256"],
            "context_config_sha256": identity(cfg),
        }
        dump(run / "sanity-evidence.json", score_next_log(
            model, tokenizer, cfg, sample["context_messages"], sample["target_message"], metadata
        ))
        from .diagnostics import run_probes
        probe_meta = {key: value for key, value in metadata.items()
                      if key in {"fold_id", "checkpoint_sha256", "input_manifest_sha256", "context_config_sha256"}}
        probe_meta.update(architecture_id="SYNTHETIC", unit_id="probe", record_ids=[])
        dump(run / "sequence-probes.json", run_probes(
            lambda context, target: score_next_log(model, tokenizer, cfg, context, target, probe_meta)
        ))
        torch.cuda.synchronize()
        metrics.update(
            source_development_macro_f1=best["f1"],
            source_development_best=best["source_development"],
            source_development_manifest_sha256=sidecar_hash,
            runtime_seconds=time.monotonic() - start,
            best=best,
            peak_vram_allocated=torch.cuda.max_memory_allocated(),
            peak_vram_reserved=torch.cuda.max_memory_reserved(),
            resources={
                "total_parameters": parameters,
                "trainable_parameters": parameters,
                "reference_total_parameters": cfg["resource_criterion"]["reference_total_parameters"],
                "total_parameter_fraction": parameters / cfg["resource_criterion"]["reference_total_parameters"],
                "parameter_criterion_met": parameters <= parameter_budget(cfg),
                "validation_seconds_per_window": seconds_per_window,
            },
            natural_anomaly_metrics={
                "status": "TARGET_NOT_EVALUATED_SOURCE_DEVELOPMENT_ONLY",
                "accuracy": None, "f1": None, "precision": None, "recall": None,
                "auroc": None, "average_precision": None,
            },
        )
        dump(run / "metrics.json", metrics)
        cfg["real_training_completed"] = True
        dump(run / "config.yaml", cfg)
        dump(run / "manifest.json", {
            "run_id": run_id,
            "fold_id": fold_id,
            "expert_id": cfg["expert_id"],
            "identity": run_identity,
            "base_model_id": None,
            "pretrained_weights": False,
            "tokenizer_id": cfg["tokenizer_id"],
            "tokenizer_revision": cfg["tokenizer_revision"],
            "adapter_name": None,
            "semantic_adapter_used": False,
            "seed": cfg["seed"],
            "precision": str(dtype),
            "model_config": cfg,
            "resume_parent": str(resume_from) if resume_from else None,
            "data_manifest_sha256": validation["bundle_sha256"],
            "data_manifest_path": "manifests/bundle.json",
            "git": git_state(repo_root),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "completion_status": "TRAINING_COMPLETED_REQUIRES_REVIEW",
            "scientific_results": "NOT_EVALUATED",
            "real_training_completed": True,
            "environment": "environment.json",
            "metrics": "metrics.json",
            "dependency_lock": "dependency-lock.txt",
            "checkpoint": str(selected_checkpoint.relative_to(run)),
            "best_checkpoint": best["checkpoint"],
            "checkpoint_sha256": checkpoint_hash,
        })
        seal(run)
        verify_seal(run, require_nested_ledgers=True)
        return {"run_directory": str(run), "metrics": metrics}
    except Exception as exc:
        dump(run / "failure.json", {
            "type": type(exc).__name__, "message": str(exc), "training_completed": False,
        })
        raise
