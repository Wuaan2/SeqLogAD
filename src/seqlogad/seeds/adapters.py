"""Run the frozen seed-42 trainers on a confirmation seed without editing them (SEED-CONFIRM-001).

Each sequence/graph trainer locks seed 42 through three names it holds: `load_config`,
`development_run_id` and `select_windows`. Inside `confirmation_seed` those names are swapped, in the
trainer's own module only and restored afterwards, for versions that (1) set the training seed,
(2) name the run with that seed and (3) keep the seed-42 checkpoint-selection windows. The training loop,
objective, model and selection rule are the frozen code. P2.1 needs no adapter: its frozen final protocol
already lists seeds 3407 and 8675309.
"""
from __future__ import annotations

import importlib
from contextlib import contextmanager

from .protocol import CONFIRMATION_SEEDS, SELECTION_SEED

MODULES = {"P2.2": "seqlogad.sequence.train", "P2.3": "seqlogad.sequence.lightweight",
           "P2.4": "seqlogad.graph.train"}


def fixed_selection(select_windows):
    """Training windows follow the training seed; validation (selection) windows stay those of seed 42."""
    def select(data_root, fold_id, partition, per_source, seed, *, replacement):
        return select_windows(data_root, fold_id, partition, per_source,
                              SELECTION_SEED if partition == "val" else seed, replacement=replacement)
    return select


def seeded_run_id(development_run_id, seed):
    def run_id(fold_id, cfg_seed, *candidate):
        if cfg_seed != seed:
            raise ValueError("run seed differs from the confirmation seed")
        return development_run_id(fold_id, 42, *candidate).replace("-S42", f"-S{seed}")
    return run_id


def _reference_config(load_config, seed):
    def load(repo_root, overrides=None, **kwargs):
        cfg, base = load_config(repo_root, dict(overrides or {}, seed=42), **kwargs)
        cfg["seed"] = seed
        return cfg, base
    return load


def _lightweight_config(load_config, seed):
    def load(repo_root):
        cfg, base = load_config(repo_root)
        cfg["seed"] = seed
        return cfg, base
    return load


def _gtat_config(load_config, seed):
    def load(repo_root):
        cfg = load_config(repo_root)
        cfg["seed"] = seed
        return cfg
    return load


CONFIGS = {"P2.2": _reference_config, "P2.3": _lightweight_config, "P2.4": _gtat_config}


@contextmanager
def confirmation_seed(task_id, seed):
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("not a preregistered confirmation seed")
    module = importlib.import_module(MODULES[task_id])
    originals = {name: getattr(module, name) for name in ("load_config", "development_run_id", "select_windows")}
    swapped = {"load_config": CONFIGS[task_id](originals["load_config"], seed),
               "development_run_id": seeded_run_id(originals["development_run_id"], seed),
               "select_windows": fixed_selection(originals["select_windows"])}
    for name, value in swapped.items():
        setattr(module, name, value)
    try:
        yield module
    finally:
        for name, value in originals.items():
            setattr(module, name, value)
