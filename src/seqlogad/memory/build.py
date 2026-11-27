"""ADAPT-MEM-001 (P3.2): target normal memory from the approved burn-in buffer only.

Reads the Phase-1 stream of the target and the raw lines of buffer records. Never opens a label file,
never reads a record outside TARGET_BURN_IN and never uses a model score. Each component of a fold's
memory is written once, made read-only and re-verified on every load; evaluation code only reads it.
Windows and graphs reuse the frozen P2.2/P2.4 builders unchanged, with the partition TARGET_BURN_IN.
"""
from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml

from seqlogad.common.checksum import sha256_file
from seqlogad.graph.prepare import _attach_fields, _scan, build_partition_graphs, declared_files, load_graph_view
from seqlogad.protocol.architectures import ADAPTERS, ARCHITECTURES
from seqlogad.protocol.buffer import build_buffer, write_buffer
from seqlogad.protocol.chronology import SEG_EARLY, load_stream_frame, load_stream_index
from seqlogad.protocol.folds import build_all_folds, write_fold
from seqlogad.protocol.normalizer import load_normalizer
from seqlogad.protocol.nul import canonicalize_nul_message
from seqlogad.semantic.contracts import membership
from seqlogad.semantic.prepare import hash64
from seqlogad.sequence.prepare import _bounded_windows, _sequence_id

CONTRACT = "configs/protocols/adapt-mem-v1.yaml"
NORMALIZER = "configs/parsing/normalizer-cs-v1.yaml"
MEMORY_ROOT = "data/processed/memory/ADAPT-MEM-001"
PARTITION = "TARGET_BURN_IN"


class AdaptMemoryError(RuntimeError):
    """Raised when memory would read outside the approved buffer, or a stored component changed."""


def load_contract(root):
    path = Path(root) / CONTRACT
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))["adapt_mem"]
    if spec["partition"] != PARTITION or spec["status"] != "PREREGISTERED":
        raise AdaptMemoryError("ADAPT-MEM-001 must be preregistered on TARGET_BURN_IN")
    return dict(spec, sha256=sha256_file(path))


def memory_key(target, membership_sha256):
    return f"ADAPT-MEM-001/{target}/{membership_sha256}"


def check_buffer(manifest, index, expected, normalizer_sha256):
    """The rebuilt buffer must be the one approved at G0: same records, ranks, content and raw bytes."""
    rule = manifest["rejection_rule"]
    found = {
        "acceptance": rule["acceptance"], "adaptation_use_permitted": rule["adaptation_use_permitted"],
        "records": manifest["records"], "membership_sha256": manifest["buffer_membership_sha256"],
        "distinct_normalized_messages": manifest["distinct_normalized_messages"],
        "first_timestamp_utc": manifest["first_timestamp_utc"], "last_timestamp_utc": manifest["last_timestamp_utc"],
        "raw_sha256": [f["sha256"] for f in index["source_files"]],
        "normalizer_rule_file_sha256": index["normalizer"]["rule_file_sha256"],
        "labels_read": manifest["target_labels_read"] or index["labels_read"],
    }
    wanted = {k: expected[k] for k in ("records", "membership_sha256", "distinct_normalized_messages",
                                       "first_timestamp_utc", "last_timestamp_utc", "raw_sha256")}
    wanted.update(acceptance="ACCEPTED", adaptation_use_permitted=True, labels_read=False,
                  normalizer_rule_file_sha256=normalizer_sha256)
    bad = {k: (found[k], v) for k, v in wanted.items() if found[k] != v}
    if bad:
        raise AdaptMemoryError(f"{manifest['buffer_id']} is not the approved buffer: {bad}")


def rebuild_buffers(root, contract):
    """Regenerate the P1.5 folds and P1.6 buffers from the frozen streams (label-free) and pin them."""
    for fold in build_all_folds(root, list(ARCHITECTURES)):
        write_fold(root, fold)
    buffers = {}
    for fold_id, expected in contract["buffers"].items():
        manifest = build_buffer(root, fold_id)
        index = load_stream_index(root, expected["target_architecture"])
        check_buffer(manifest, index, expected, contract["normalizer_rule_file_sha256"])
        write_buffer(root, manifest)
        buffers[fold_id] = manifest
    return buffers


def buffer_frame(root, target, membership_sha256):
    """The buffer records of the frozen stream; fails unless their ranks hash to the approved membership."""
    frame = load_stream_frame(root, target).filter(pl.col("segment") == SEG_EARLY).sort("rank")
    if membership(frame["rank"]) != membership_sha256:
        raise AdaptMemoryError(f"{target}: stream EARLY ranks differ from the approved buffer")
    return frame.select("rank", "effective_ts", "gid", "norm_h")


def transition_counts(buffer):
    """First-order norm_h counts over every buffer record; history resets at each group (GLOBAL if none)."""
    frame = buffer.sort("rank").with_columns(
        pl.when(pl.col("gid") >= 0).then(pl.col("gid")).otherwise(-1).alias("group"))
    frame = frame.with_columns(pl.col("norm_h").shift(1).over("group").alias("prev_h"))
    unigrams = frame.group_by("norm_h").agg(pl.len().alias("count"), pl.col("rank").min().alias("first_rank"))
    transitions = frame.drop_nulls("prev_h").group_by("prev_h", "norm_h").agg(pl.len().alias("count"))
    return (unigrams.sort("norm_h"),
            transitions.rename({"norm_h": "next_h"}).sort("prev_h", "next_h"))


def normalized_texts(root, target, positions, buffer):
    """One raw pass: NORM-CS-001 text of the wanted buffer ranks, each checked against the stream norm_h."""
    spec = ARCHITECTURES[target]
    adapter, normalizer = ADAPTERS[spec.adapter], load_normalizer(root)
    wanted = positions.join(buffer.select("rank", "norm_h"), on="rank", how="inner").sort("file_position")
    if wanted.height != positions.height:
        raise AdaptMemoryError(f"{target}: a requested rank is outside the buffer")
    targets = iter(wanted.select("file_position", "rank", "norm_h").iter_rows())
    current, position, out = next(targets, None), 0, []
    for path, rel in declared_files(target, Path(root) / spec.raw_root):
        with path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                if current and position == current[0]:
                    record = adapter(line, rel, number)
                    text = normalizer.normalize(canonicalize_nul_message(record.raw_message_bytes).canonical_message)
                    if hash64(text.encode()) != current[2]:
                        raise AdaptMemoryError(f"{target}: normalized text differs from Phase 1 at rank {current[1]}")
                    out.append((current[1], text))
                    current = next(targets, None)
                position += 1
    if current is not None:
        raise AdaptMemoryError(f"{target}: raw files end before every wanted record")
    return pl.DataFrame(out, schema={"rank": pl.Int64, "normalized_message": pl.String}, orient="row")


def _quantiles(series):
    series = series.drop_nulls().cast(pl.Float64)
    return {f"p{q}": round(float(series.quantile(q / 100, "nearest")), 3) for q in (5, 50, 95)} | {
        "min": round(float(series.min()), 3), "max": round(float(series.max()), 3)}


def graph_statistics(graphs, nodes, windows):
    attributes = pl.concat([
        nodes.group_by(pl.col(field).cast(pl.String).alias("value")).agg(pl.len().alias("count"))
        .with_columns(pl.lit(field).alias("field")) for field in ("component_text", "severity", "location", "request_group")
    ]).select("field", "value", "count").sort("field", "count", "value", descending=[False, True, False])
    summary = {
        "windows": graphs.height, "scored": int((graphs["status"] == "SCORED").sum()),
        "abstain": int((graphs["status"] == "ABSTAIN").sum()), "nodes": nodes.height,
        "context_records": _quantiles(graphs["context_ranks"].list.len()),
        "target_gap_seconds": _quantiles((windows["target_effective_ts"] - windows["context_last_effective_ts"]) / 1e6),
        "distinct": {f: int(nodes[f].drop_nulls().n_unique()) for f in ("component_text", "severity", "location")},
    }
    return attributes, summary


def build_statistics(root, fold_id, contract):
    """Transitions, the TARGET_BURN_IN windows and their GTAT graphs, all from the approved buffer."""
    expected = contract["buffers"][fold_id]
    target = expected["target_architecture"]
    spec = ARCHITECTURES[target]
    buffer = buffer_frame(root, target, expected["membership_sha256"])
    unigrams, transitions = transition_counts(buffer)

    windows_cfg = contract["components"]["windows"]
    windows, eligible = _bounded_windows(buffer, target, PARTITION, windows_cfg["limit"],
                                         windows_cfg["max_predecessor_records"])
    files = declared_files(target, Path(root) / spec.raw_root)
    scanned, digests = _scan(files, ADAPTERS[spec.adapter])
    if [d for d, _ in digests] != expected["raw_sha256"]:
        raise AdaptMemoryError(f"{target}: raw files differ from the approved buffer")
    members = scanned.join(buffer.select("rank"), on="rank", how="semi").sort("rank")
    if not members.select("rank", "effective_ts", "gid").equals(buffer.select("rank", "effective_ts", "gid")):
        raise AdaptMemoryError(f"{target}: raw chronology differs from the frozen stream")
    _, graph_view = load_graph_view(Path(root))
    graphs, keys = build_partition_graphs(target, PARTITION, members.drop("file_position"), windows, graph_view)

    required = set(windows["target_rank"].to_list()) | set(keys["rank"].to_list()) | set(unigrams["first_rank"].to_list())
    for values in windows["context_ranks"].to_list():
        required.update(values)
    positions = members.filter(pl.col("rank").is_in(list(required))).select("rank", "file_position")
    texts = normalized_texts(root, target, positions, buffer)

    unigrams = unigrams.join(texts.rename({"rank": "first_rank"}), on="first_rank", how="left")
    sequence_ranks = set(windows["target_rank"].to_list())
    for values in windows["context_ranks"].to_list():
        sequence_ranks.update(values)
    events = buffer.select("rank", "effective_ts", "gid").filter(pl.col("rank").is_in(list(sequence_ranks))).join(
        texts, on="rank", how="inner", validate="1:1").sort("rank")
    events = events.with_columns(
        pl.lit(target).alias("architecture_id"), pl.lit(PARTITION).alias("partition"),
        pl.concat_str([pl.lit(target), pl.lit(":"), pl.col("rank")]).alias("record_id"),
        pl.Series("sequence_id", [_sequence_id(target, PARTITION, f"G{g}" if g >= 0 else "GLOBAL")
                                  for g in events["gid"].to_list()]),
    ).select("architecture_id", "partition", "rank", "record_id", "effective_ts", "sequence_id", "normalized_message")
    if events.height != len(sequence_ranks):
        raise AdaptMemoryError(f"{target}: a window references a record with no text")

    nodes = keys.join(members.select("rank", "file_position", "effective_ts"), on="rank", how="left").join(
        texts, on="rank", how="left").with_columns(pl.lit(PARTITION).alias("partition"))
    nodes = _attach_fields(target, files, nodes, graph_view, Path(root) / NORMALIZER, Path(root))
    attributes, summary = graph_statistics(graphs, nodes, windows)
    tables = {"unigrams": unigrams, "transitions": transitions, "windows": windows, "events": events,
              "graphs": graphs, "nodes": nodes, "node_attributes": attributes}
    counts = {"buffer_records": buffer.height, "distinct_events": unigrams.height,
              "distinct_transitions": transitions.height, "transition_pairs": int(transitions["count"].sum()),
              "eligible_windows": eligible, "windows": windows.height, "events": events.height}
    return tables, dict(counts=counts, graph_statistics=summary)


def write_component(root, fold_id, component, tables, manifest):
    """Write one memory component once; every file is hashed into its manifest and made read-only."""
    out = Path(root) / MEMORY_ROOT / fold_id / component
    if out.exists():
        raise AdaptMemoryError(f"{out} exists; a memory component is written once")
    staging = out.with_name(component + ".building")
    staging.mkdir(parents=True)
    files = {}
    for name, frame in tables.items():
        frame.write_parquet(staging / f"{name}.parquet", compression="zstd")
        files[f"{name}.parquet"] = sha256_file(staging / f"{name}.parquet")
    manifest = dict(manifest, files=files)
    (staging / "component.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.rename(out)
    for path in out.iterdir():
        os.chmod(path, stat.S_IREAD)
    return out


def load_component(root, fold_id, component):
    """Read a component's manifest after re-hashing every file and the key it was built for."""
    out = Path(root) / MEMORY_ROOT / fold_id / component
    manifest = json.loads((out / "component.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha256_file(out / name) != digest:
            raise AdaptMemoryError(f"{out / name} changed after the memory was written")
    if manifest["memory_key"] != memory_key(manifest["target_architecture"], manifest["buffer_membership_sha256"]):
        raise AdaptMemoryError(f"{out}: memory key does not name its target and buffer")
    return manifest


def build_memory_statistics(root, repo_root, folds=None):
    """P3.2 CPU part: rebuild and pin the buffers, then write each fold's statistics component."""
    contract = load_contract(repo_root)
    buffers = rebuild_buffers(root, contract)
    report = {}
    for fold_id in folds or list(contract["buffers"]):
        buffer = buffers[fold_id]
        target = buffer["target_architecture"]
        tables, details = build_statistics(root, fold_id, contract)
        manifest = {
            "artifact_id": "ADAPT-MEM-001", "component": "statistics", "fold_id": fold_id,
            "target_architecture": target, "buffer_id": buffer["buffer_id"],
            "buffer_membership_sha256": buffer["buffer_membership_sha256"],
            "memory_key": memory_key(target, buffer["buffer_membership_sha256"]),
            "buffer_provenance": buffer["provenance"], "contract_sha256": contract["sha256"],
            "builder_sha256": sha256_file(Path(__file__)), "partition": PARTITION,
            "created_at": datetime.now(timezone.utc).isoformat(), "target_labels_read": False,
            "records_outside_buffer_read": False, **details,
        }
        write_component(root, fold_id, "statistics", tables, manifest)
        report[fold_id] = load_component(root, fold_id, "statistics")["counts"]
    return report
