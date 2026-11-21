"""P2.6 evaluation windows: source validation windows held out from checkpoint selection (DIV-P2-001 §2)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from seqlogad.sequence.objective import select_windows

CONFIG = "configs/protocols/p2.6-diversity-v1.yaml"
LABELED_UNITS = {"ARCH-HDFS": "sequence_id", "ARCH-BGL": "window_id"}  # HDFS block, BGL target line


def load_config(repo_root):
    cfg = yaml.safe_load((Path(repo_root) / CONFIG).read_text())["p2_6_diversity"]
    if cfg["seed"] != 42 or cfg["combinations"]["learned_weights"]:
        raise ValueError("P2.6 preregistration changed")
    return cfg


def evaluation_windows(data_root, fold_id, seed=42, selection_per_source=512):
    """Label-free: every val window of the fold minus the checkpoint-selection windows (and their HDFS blocks)."""
    windows = pl.read_parquet(Path(data_root) / "folds" / fold_id / "sequence/val/windows.parquet")
    selected = select_windows(data_root, fold_id, "val", selection_per_source, seed, replacement=False)
    chosen = {row["window_id"] for row in selected}
    blocks = {row["sequence_id"] for row in selected if row["architecture_id"] == "ARCH-HDFS"}
    held_out = windows.filter(
        ~pl.col("window_id").is_in(list(chosen))
        & ~((pl.col("architecture_id") == "ARCH-HDFS") & pl.col("sequence_id").is_in(list(blocks)))
    )
    if held_out.is_empty() or held_out["window_id"].n_unique() != held_out.height:
        raise ValueError("empty or duplicate P2.6 evaluation windows")
    unit = pl.when(pl.col("architecture_id") == "ARCH-HDFS").then(pl.col("sequence_id")).otherwise(pl.col("window_id"))
    return (held_out.with_columns(unit_id=unit,
                                  labeled=pl.col("architecture_id").is_in(list(LABELED_UNITS)),
                                  target_record_id=pl.concat_str([pl.col("architecture_id"), pl.lit(":"),
                                                                  pl.col("target_rank").cast(pl.Utf8)]))
            .sort("architecture_id", "target_rank"))
