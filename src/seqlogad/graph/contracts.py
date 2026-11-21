"""P2.4 graph-view contract (GTAT-ST-001 §3.1): schemas, fail-closed validator, loaders."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from seqlogad.common.checksum import sha256_file
from seqlogad.semantic.contracts import read, safe_path

VIEW_SCHEMA = "SEQLOGAD-GRAPH-VIEW-1"
BUNDLE_SCHEMA = "SEQLOGAD-GRAPH-BUNDLE-1"
BUNDLE_ID = "P2.4-GRAPH-VIEW-S42-001"
BUNDLE_MANIFEST = "manifests/graph-bundle.json"
LEDGER = "manifests/checksums.sha256"
PARTITIONS = {"train": "SOURCE_TRAIN", "val": "SOURCE_VALIDATION"}
STATUSES = ("SCORED", "ABSTAIN")
GRAPH_SCHEMA = {
    "window_id": pl.String,
    "architecture_id": pl.String,
    "partition": pl.String,
    "sequence_id": pl.String,
    "target_rank": pl.Int64,
    "context_ranks": pl.List(pl.Int64),
    "status": pl.String,
}
NODE_SCHEMA = {
    "architecture_id": pl.String,
    "partition": pl.String,
    "rank": pl.Int64,
    "effective_ts": pl.Int64,
    "event_text": pl.String,
    "component_text": pl.String,
    "severity": pl.Int8,
    "location": pl.String,
    "request_group": pl.String,
    "continuation_lines": pl.Int32,
}
NODE_KEY = ["architecture_id", "partition", "rank"]
WINDOW_KEY = ["window_id", "architecture_id", "partition", "sequence_id", "target_rank"]
# Only these architectures may carry a non-null value in the field.
FIELD_OWNERS = {"location": {"ARCH-BGL"}, "request_group": {"ARCH-OPENSTACK"},
                "continuation_lines": {"ARCH-HADOOP"}}
FORBIDDEN_MARKERS = ("label", "anomaly", "session", "trace", "node_id", "source_file", "logfile")
MAX_CONTEXT = 64


def fold_files(fold_id: str) -> list[str]:
    prefix = f"folds/{fold_id}/graph"
    return [f"{prefix}/train/graphs.parquet", f"{prefix}/val/graphs.parquet",
            f"{prefix}/metadata/nodes.parquet", f"{prefix}/metadata/view.json"]


def check_columns(frame: pl.DataFrame, schema: dict, what: str) -> None:
    """Exact column names, order and dtypes; nothing extra, nothing label-like."""
    if dict(frame.schema) != schema or list(frame.columns) != list(schema):
        raise ValueError(f"{what} column allowlist violation: {list(frame.columns)}")
    bad = [c for c in frame.columns if any(m in c.lower() for m in FORBIDDEN_MARKERS)]
    if bad:
        raise ValueError(f"{what} forbidden column: {bad}")


def check_graphs(graphs: pl.DataFrame, partition: str, sources: set[str], target: str) -> None:
    check_columns(graphs, GRAPH_SCHEMA, "graphs")
    if graphs.is_empty():
        raise ValueError("empty graph partition")
    archs = set(graphs["architecture_id"].unique())
    if target in archs:
        raise ValueError("target-architecture row in graph view")
    if not archs <= sources:
        raise ValueError("graph row from a non-source architecture")
    if set(graphs["partition"].unique()) != {partition}:
        raise ValueError("graph partition mismatch")
    if graphs.drop("context_ranks").null_count().row(0) != (0,) * (len(GRAPH_SCHEMA) - 1):
        raise ValueError("null graph field")
    if graphs["window_id"].n_unique() != graphs.height:
        raise ValueError("duplicate window_id")
    if not set(graphs["status"].unique()) <= set(STATUSES):
        raise ValueError("unknown graph status")
    stats = graphs.select(
        pl.col("context_ranks").list.len().alias("n"),
        (pl.col("context_ranks").list.max() < pl.col("target_rank")).fill_null(True).alias("causal"),
        pl.col("context_ranks").list.eval(pl.element().diff().drop_nulls() > 0).list.all().alias("asc"),
        pl.col("status"),
    )
    if (stats["n"] > MAX_CONTEXT).any() or not stats["causal"].all() or not stats["asc"].all():
        raise ValueError("context must be ascending, causal and at most 64 records")
    if not ((stats["n"] > 0) == (stats["status"] == "SCORED")).all():
        raise ValueError("status must be SCORED iff the context is non-empty")


def check_nodes(nodes: pl.DataFrame, sources: set[str], target: str) -> None:
    check_columns(nodes, NODE_SCHEMA, "nodes")
    archs = set(nodes["architecture_id"].unique())
    if target in archs:
        raise ValueError("target-architecture row in graph view")
    if not archs <= sources:
        raise ValueError("node row from a non-source architecture")
    if nodes.select(NODE_KEY + ["effective_ts", "event_text", "severity", "continuation_lines"]).null_count().sum_horizontal().item():
        raise ValueError("null required node field")
    if nodes.select(NODE_KEY).is_duplicated().any():
        raise ValueError("duplicate node key")
    if not nodes["severity"].is_in([-1, 0, 1, 2, 3]).all() or (nodes["continuation_lines"] < 0).any():
        raise ValueError("node value out of range")
    for column, owners in FIELD_OWNERS.items():
        foreign = nodes.filter(~pl.col("architecture_id").is_in(list(owners)))
        value = foreign[column]
        if (value.null_count() != foreign.height) if column != "continuation_lines" else (value != 0).any():
            raise ValueError(f"{column} set outside {sorted(owners)}")


def referenced_keys(graphs: pl.DataFrame) -> pl.DataFrame:
    context = graphs.select("architecture_id", "partition", pl.col("context_ranks").alias("rank")).explode("rank")
    target = graphs.select("architecture_id", "partition", pl.col("target_rank").alias("rank"))
    return pl.concat([context.drop_nulls("rank"), target]).unique()


def _same_rows(left: pl.DataFrame, right: pl.DataFrame, key: list[str]) -> bool:
    return (left.height == right.height and left.join(right, on=key, how="anti").is_empty()
            and right.join(left, on=key, how="anti").is_empty())


def _ledger_ok(root: Path, files: dict) -> bool:
    expected = dict(files, **{BUNDLE_MANIFEST: sha256_file(root / BUNDLE_MANIFEST)})
    lines = (root / LEDGER).read_text().splitlines()
    return lines == [f"{digest}  {name}" for name, digest in sorted(expected.items())]


def validate_graph_bundle(graph_root, fold_id, expected_bundle_sha256, sequence_root) -> dict:
    """Fail-closed check of one fold of an extracted graph bundle against its sequence bundle."""
    root, seq_root = Path(graph_root), Path(sequence_root)
    bundle_path = root / BUNDLE_MANIFEST
    digest = sha256_file(bundle_path)
    if expected_bundle_sha256 and digest != expected_bundle_sha256:
        raise ValueError("trusted graph bundle digest mismatch")
    bundle = read(bundle_path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA or bundle.get("bundle_id") != BUNDLE_ID:
        raise ValueError("graph bundle schema mismatch")
    if bundle.get("target_labels_read") is not False or bundle.get("label_columns") != []:
        raise ValueError("graph bundle label policy violation")
    folds = bundle.get("folds", [])
    if fold_id not in folds or len(folds) != len(set(folds)):
        raise ValueError("unknown/duplicate graph fold")
    shape = {name for fold in folds for name in fold_files(fold)}
    if set(bundle.get("files", {})) != shape:
        raise ValueError("graph bundle file allowlist violation")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != shape | {BUNDLE_MANIFEST, LEDGER}:
        raise ValueError("unexpected/missing graph bundle files")
    for name, file_digest in bundle["files"].items():
        if sha256_file(safe_path(root, name)) != file_digest:
            raise ValueError(f"graph bundle checksum mismatch: {name}")
    if not _ledger_ok(root, bundle["files"]):
        raise ValueError("graph bundle checksum ledger mismatch")

    lineage = bundle["lineage"]
    if sha256_file(seq_root / "manifests/bundle.json") != lineage["sequence_bundle"]["manifest_sha256"]:
        raise ValueError("graph bundle is not bound to this sequence bundle")
    seq_fold = read(safe_path(seq_root, f"folds/{fold_id}/manifests/fold.json"))
    target, sources = seq_fold["target_architecture"], set(seq_fold["source_architectures"])
    prefix = f"folds/{fold_id}/graph"
    view = read(root / prefix / "metadata/view.json")
    if (view.get("schema_version") != VIEW_SCHEMA or view.get("fold_id") != fold_id
            or view.get("target_architecture") != target or set(view.get("source_architectures", [])) != sources
            or target in sources):
        raise ValueError("graph view fold/source identity violation")
    if view.get("target_labels_read") is not False or view.get("label_columns") != []:
        raise ValueError("graph view label policy violation")
    if view.get("graph_view_sha256") != lineage["graph_view_sha256"]:
        raise ValueError("graph_view config binding mismatch")

    frames, counts = [], {}
    for directory, partition in PARTITIONS.items():
        graphs = pl.read_parquet(root / prefix / f"{directory}/graphs.parquet")
        check_graphs(graphs, partition, sources, target)
        windows = pl.read_parquet(safe_path(seq_root, f"folds/{fold_id}/sequence/{directory}/windows.parquet"))
        if not _same_rows(graphs.select(WINDOW_KEY), windows.select(WINDOW_KEY), WINDOW_KEY):
            raise ValueError("graph windows differ from the sequence windows of the fold")
        counts[partition] = {
            arch: {status: int(n) for status, n in
                   graphs.filter(pl.col("architecture_id") == arch).group_by("status").len().iter_rows()}
            for arch in sorted(sources)}
        frames.append(graphs)
    nodes = pl.read_parquet(root / prefix / "metadata/nodes.parquet")
    check_nodes(nodes, sources, target)
    if not _same_rows(nodes.select(NODE_KEY), referenced_keys(pl.concat(frames)), NODE_KEY):
        raise ValueError("nodes must be exactly the records used by the fold's graphs")
    node_counts = {f"{a}|{p}": int(n) for a, p, n in nodes.group_by("architecture_id", "partition").len().iter_rows()}
    if view.get("counts") != counts or view.get("node_counts") != node_counts:
        raise ValueError("graph view manifest count mismatch")
    return {"status": "PASS", "fold_id": fold_id, "bundle_sha256": digest, "counts": counts,
            "node_counts": node_counts, "view": view}


def load_graph_windows(graph_root, fold_id, partition) -> pl.DataFrame:
    if partition not in PARTITIONS:
        raise ValueError("partition must be 'train' or 'val'")
    frame = pl.read_parquet(safe_path(Path(graph_root), f"folds/{fold_id}/graph/{partition}/graphs.parquet"))
    check_columns(frame, GRAPH_SCHEMA, "graphs")
    return frame


def load_graph_nodes(graph_root, fold_id) -> pl.DataFrame:
    frame = pl.read_parquet(safe_path(Path(graph_root), f"folds/{fold_id}/graph/metadata/nodes.parquet"))
    check_columns(frame, NODE_SCHEMA, "nodes")
    return frame
