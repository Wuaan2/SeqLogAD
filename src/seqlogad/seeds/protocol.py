"""SEED-CONFIRM-001 identities: seed 42 stays primary; seeds 3407 and 8675309 only confirm stability."""
from pathlib import Path

import yaml

from seqlogad.phase2_execution.specs import deterministic_run_id, load_development_spec

PRIMARY_SEED = 42
CONFIRMATION_SEEDS = (3407, 8675309)
# Checkpoint selection uses the seed-42 selection windows for every training seed, so the P2.6
# held-out windows (validation minus those windows) stay unseen by every selection.
SELECTION_SEED = 42
PROTOCOL = "configs/protocols/seed-confirmation-v1.yaml"


def load_protocol(repo_root, seed):
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("not a preregistered confirmation seed; seed 42 runs through the frozen runner")
    protocol = yaml.safe_load((Path(repo_root) / PROTOCOL).read_text(encoding="utf-8"))["seed_confirmation"]
    if (protocol["status"] != "PREREGISTERED" or protocol["primary_seed"] != PRIMARY_SEED
            or protocol["confirmation_seeds"] != list(CONFIRMATION_SEEDS)
            or protocol["selection_seed"] != SELECTION_SEED):
        raise ValueError("SEED-CONFIRM-001 preregistration is not in force")
    load_development_spec(repo_root)  # also checks that the amendment defers exactly these seeds
    return protocol


def run_id(task_id, target, seed, sequence_candidate="v13r1"):
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("not a preregistered confirmation seed")
    candidate = sequence_candidate if task_id == "P2.2" else "v12"
    return deterministic_run_id(task_id, target, sequence_candidate=candidate).replace("-S42", f"-S{seed}")


def final_name(spec_name, seed):
    if "_S42_" not in spec_name or seed not in CONFIRMATION_SEEDS:
        raise ValueError("unexpected FINAL ZIP name or seed")
    return spec_name.replace("_S42_", f"_S{seed}_")
