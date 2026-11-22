"""Train one frozen Phase-2 expert on a confirmation seed for both target folds (SEED-CONFIRM-001).

Same persistence, recovery and packaging as the seed-42 runner, which stays unchanged; every artifact is
named with the seed and written under the seed's own Drive folder. Usage from a notebook:
python -m seqlogad.seeds.runner --task-id P2.3 --seed 3407 --repo-root ... --data-root ... ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from seqlogad.common.checksum import sha256_file
from seqlogad.phase2_execution.archive import create_zip, seal_tree
from seqlogad.phase2_execution.artifacts import package_target
from seqlogad.phase2_execution.persistence import publish_atomic, restore_previous_if_broken, verify_persisted_zip
from seqlogad.phase2_execution.recovery import create_recovery, restore_recovery
from seqlogad.phase2_execution.runner import (LIGHTWEIGHT_BACKUP_SECONDS, _json_atomic, _validate_semantic_data,
                                              _validate_sequence_data, cleanup_gpu)
from seqlogad.phase2_execution.specs import FOLDS, TARGETS, load_development_spec

from . import adapters
from .protocol import CONFIRMATION_SEEDS, final_name, load_protocol, run_id

LABELED = {"P2.2", "P2.3", "P2.4"}  # checkpoint selection on source validation labels (P3.1 amendment, C2)


def _train(task_id, seed, target, spec, *, repo_root, data_root, output_root, bundle_sha256, resume_from,
           recovery_zip, labels, graph):
    last_backup = [time.monotonic()]

    def checkpoint_callback(run, checkpoint, best_checkpoint):
        if task_id == "P2.3":  # same 30-minute cap as seed 42: a P2.3 snapshot holds ~0.9 GB of weights
            if time.monotonic() - last_backup[0] < LIGHTWEIGHT_BACKUP_SECONDS:
                return
            last_backup[0] = time.monotonic()
        receipt = create_recovery(run, checkpoint, recovery_zip, task_id=task_id, target=target, seed=seed,
                                  best_checkpoint=best_checkpoint)
        print("PERSISTENT_BACKUP=PASS", flush=True)
        print(f"PATH={receipt['path']}", flush=True)
        print(f"SHA256={receipt['sha256']}", flush=True)

    fold, rid = FOLDS[target], run_id(task_id, target, seed, spec.get("sequence_candidate", "v13r1"))
    common = dict(resume_from=str(resume_from) if resume_from else None,
                  expected_bundle_sha256=bundle_sha256, checkpoint_callback=checkpoint_callback)
    llama = {"seed": seed, "training_mode": "QLORA_NF4_DOUBLE_QUANT"}
    if task_id == "P2.1":
        from seqlogad.semantic.train import run_training
        return run_training(repo_root, data_root, output_root, fold, rid, overrides=llama, run_mode="final", **common)
    with adapters.confirmation_seed(task_id, seed) as module:
        if task_id == "P2.2":
            return module.run_training(repo_root, data_root, output_root, fold, rid, overrides=llama,
                                       run_mode="development", candidate=spec["sequence_candidate"],
                                       **labels, **common)
        return module.run_training(repo_root, data_root, output_root, fold, rid, **labels, **graph, **common)


def _completed(archive, rid):
    with zipfile.ZipFile(archive) as handle:
        manifest = json.loads(handle.read(f"{rid}/manifest.json"))
    if manifest.get("run_id") != rid or manifest.get("real_training_completed") is not True:
        raise ValueError("persisted target ZIP is not this completed confirmation run")


def _package_expert(task_id, seed, archives, summary, destination):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / f"{task_id}_S{seed}"
        (root / "targets").mkdir(parents=True)
        for archive in archives.values():
            shutil.copy2(archive, root / "targets" / archive.name)
            shutil.copy2(f"{archive}.sha256", root / "targets" / f"{archive.name}.sha256")
        (root / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        seal_tree(root)
        return create_zip(root, destination, root.name)


def run_seed(*, task_id, seed, repo_root, data_root, output_root, drive_root, expected_bundle_sha256,
             source_development_root=None, expected_source_development_sha256=None,
             graph_root=None, expected_graph_sha256=None):
    repo_root, data_root, output_root, drive_root = map(Path, (repo_root, data_root, output_root, drive_root))
    load_protocol(repo_root, seed)
    amendment = load_development_spec(repo_root)
    spec = amendment["experts"][task_id]
    if spec["implementation_status"] != "READY":
        raise ValueError(f"{task_id} is not ready in the frozen amendment")
    if (task_id in LABELED) != bool(source_development_root and expected_source_development_sha256):
        raise ValueError("the source-development sidecar is required by P2.2-P2.4 and forbidden for P2.1")
    if (task_id == "P2.4") != bool(graph_root and expected_graph_sha256):
        raise ValueError("the graph view (root and trusted SHA-256) is required by P2.4 only")
    if task_id in LABELED and sha256_file(Path(source_development_root) / "manifest.json") \
            != expected_source_development_sha256:
        raise ValueError("source-development manifest SHA-256 mismatch")
    if sha256_file(data_root / "manifests/bundle.json") != expected_bundle_sha256:
        raise ValueError("data bundle SHA-256 mismatch")
    (_validate_semantic_data if task_id == "P2.1" else _validate_sequence_data)(data_root, expected_bundle_sha256)
    if spec["requires_hf_token"] and not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN must come from the runtime secret/environment")
    labels = ({"source_development_root": source_development_root,
               "expected_source_development_sha256": expected_source_development_sha256}
              if task_id in LABELED else {})
    graph = {"graph_root": graph_root, "expected_graph_sha256": expected_graph_sha256} if task_id == "P2.4" else {}

    export = output_root / "exports" / task_id
    export.mkdir(parents=True, exist_ok=True)
    results, archives = {}, {}
    for target in TARGETS:
        rid = run_id(task_id, target, seed, spec.get("sequence_candidate", "v13r1"))
        target_zip = drive_root / task_id / "targets" / f"{task_id}_{target}_S{seed}.zip"
        recovery_zip = drive_root / task_id / "recovery" / target / "LATEST_RECOVERY.zip"
        for path in (target_zip, recovery_zip):
            if restore_previous_if_broken(path):
                print(f"{target}=RESTORED_PREVIOUS {path.name}", flush=True)
        if target_zip.exists():
            verify_persisted_zip(target_zip)
            _completed(target_zip, rid)
            archives[target], results[target] = target_zip, {"status": "SKIPPED_COMPLETED", "run_id": rid}
            print(f"{target}=SKIP_COMPLETED_VERIFIED", flush=True)
            continue
        run_directory = output_root / FOLDS[target] / spec["expert_id"] / rid
        resume_from = None
        if recovery_zip.exists():
            if run_directory.exists():
                raise ValueError("ambiguous local run and persistent recovery both exist")
            resume_from, receipt = restore_recovery(recovery_zip, run_directory, task_id=task_id, target=target,
                                                    seed=seed, run_id=rid)
            print(f"{target}=RESUME_STEP_{receipt['latest_step']}", flush=True)
        elif run_directory.exists():
            raise ValueError("unsealed local run exists without trustworthy recovery")
        try:
            result = _train(task_id, seed, target, spec, repo_root=repo_root, data_root=data_root,
                            output_root=output_root, bundle_sha256=expected_bundle_sha256,
                            resume_from=resume_from, recovery_zip=recovery_zip, labels=labels, graph=graph)
            local = export / target_zip.name
            package_target(Path(result["run_directory"]), local)
            persisted = publish_atomic(local, target_zip)
            verify_persisted_zip(target_zip)
        except Exception:
            print("RUN_COMPLETED_BUT_BACKUP_FAILED_OR_TRAINING_FAILED", flush=True)
            raise
        archives[target] = target_zip
        results[target] = {"status": "RESUMED_AND_COMPLETED" if resume_from else "COMPLETED", "run_id": rid,
                           "target_zip_sha256": persisted["sha256"]}
        print(f"{target}=COMPLETED SHA256={persisted['sha256']}", flush=True)
        print(json.dumps(cleanup_gpu(), sort_keys=True), flush=True)

    summary = {
        "schema": "SEQLOGAD-ONECLICK-SUMMARY-1", "task_id": task_id, "expert_id": spec["expert_id"],
        "run_mode": "development", "seed": seed, "run_role": "CONFIRMATION_SEED", "primary_seed": 42,
        "protocol": "SEED-CONFIRM-001", "targets": results,
        "source_label_status": ("SOURCE_VALIDATION_MODEL_SELECTION_ONLY" if task_id in LABELED
                                else amendment["source_label_status"]),
        "multi_seed_stability": "NOT_CLAIMED", "scientific_result": "NOT_EVALUATED",
        "safe_to_stop_runtime": True, "safe_to_disconnect_colab": True,
    }
    name = final_name(spec["final_zip"], seed)
    _package_expert(task_id, seed, archives, summary, export / name)
    final_drive = drive_root / task_id / name
    final = publish_atomic(export / name, final_drive)
    verify_persisted_zip(final_drive)
    _json_atomic(drive_root / task_id / "SUMMARY.json", summary)
    print(f"FINAL_DRIVE_PATH={final_drive}", flush=True)
    print(f"FINAL_DRIVE_SHA256={final['sha256']}", flush=True)
    print("SAFE_TO_DISCONNECT_COLAB=YES", flush=True)
    return dict(summary, final_zip_sha256=final["sha256"], final_drive_path=str(final_drive))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True, choices=["P2.1", "P2.2", "P2.3", "P2.4"])
    parser.add_argument("--seed", required=True, type=int, choices=CONFIRMATION_SEEDS)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--drive-root", required=True, help="this seed's own folder, e.g. .../Seeds/S3407")
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--source-development-root")
    parser.add_argument("--expected-source-development-sha256")
    parser.add_argument("--graph-root")
    parser.add_argument("--expected-graph-sha256")
    args = parser.parse_args()
    result = run_seed(task_id=args.task_id, seed=args.seed, repo_root=args.repo_root, data_root=args.data_root,
                      output_root=args.output_root, drive_root=args.drive_root,
                      expected_bundle_sha256=args.expected_bundle_sha256,
                      source_development_root=args.source_development_root,
                      expected_source_development_sha256=args.expected_source_development_sha256,
                      graph_root=args.graph_root, expected_graph_sha256=args.expected_graph_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
