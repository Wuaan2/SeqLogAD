"""Load and guard the preregistered GTAT-ST-001 config."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from seqlogad.phase2_execution.specs import TARGETS

CONFIG = "configs/models/gtat-v1.yaml"
PACKAGES = ("common", "protocol", "semantic", "sequence", "graph", "phase2_execution")


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def development_run_id(fold_id, seed=42):
    target = fold_id.removeprefix("FOLD-TARGET-ARCH-")
    if target not in TARGETS or seed != 42:
        raise ValueError("P2.4 development run is locked to seed 42 and the HDFS/BGL target folds")
    return f"P2.4-DEVELOPMENT-{target}-S42"


def load_config(repo_root):
    cfg = yaml.safe_load((Path(repo_root) / CONFIG).read_text())["gtat"]
    if cfg["expert_id"] != "GTAT_STRUCTURAL_TEMPORAL" or cfg["model_identity"]["pretrained_weights"]:
        raise ValueError("P2.4 expert identity or from-scratch policy changed")
    if cfg["seed"] != 42 or cfg["run_mode"] != "development":
        raise ValueError("P2.4 is locked to the seed-42 development run")
    if cfg["objective"]["target_node_visible_fields"]:
        raise ValueError("P2.4 target node must stay fully masked")
    if (cfg["source_development"]["training_labels_allowed"]
            or cfg["source_development"]["held_out_target_labels_allowed"]):
        raise ValueError("P2.4 label policy changed")
    if cfg["satisfies_sequence_role"] or cfg["clean_normal_claim"]:
        raise ValueError("P2.4 role or normality claim changed")
    if cfg["training"]["warmup_steps"] >= cfg["training"]["max_steps"]:
        raise ValueError("invalid P2.4 optimizer budget")
    return cfg


def implementation_digest(repo_root):
    """Bind a run to the packaged code, including the graph package."""
    base = Path(repo_root) / "src/seqlogad"
    files = sorted([base / "__init__.py"] + [path for package in PACKAGES
                                             for path in (base / package).glob("*.py")])
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(Path(repo_root)).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()
