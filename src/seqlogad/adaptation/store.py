"""Frozen-memory reader and write-once artifact store shared by CAL-CS-001 and DRIFT-CS-001."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import polars as pl
import yaml

from seqlogad.common.checksum import sha256_file
from seqlogad.memory.build import MEMORY_ROOT, load_component

ADAPT_ROOT = "data/processed/adaptation"
COMPONENTS = ("statistics", "prototypes", "buffer_scores")


class AdaptationError(RuntimeError):
    pass


def load_spec(repo_root, path, key):
    """A PREREGISTERED spec with its SHA-256, so every artifact binds the exact rules it was built under."""
    full = Path(repo_root) / path
    spec = yaml.safe_load(full.read_text(encoding="utf-8"))[key]
    if spec["status"] != "PREREGISTERED":
        raise AdaptationError(f"{path} is not in force")
    return dict(spec, sha256=sha256_file(full))


def load_frozen_memory(root, fold_id):
    """memory.json must be FROZEN and bind exactly the three component manifests on disk, each re-hashed."""
    folder = Path(root) / MEMORY_ROOT / fold_id
    path = folder / "memory.json"
    if not path.exists():
        raise AdaptationError(f"{fold_id}: no memory.json; the fold memory is not frozen yet (P3.2)")
    memory = json.loads(path.read_text(encoding="utf-8"))
    if memory["status"] != "FROZEN" or set(memory["components"]) != set(COMPONENTS):
        raise AdaptationError(f"{fold_id}: the fold memory is not frozen")
    manifests = {}
    for name, digest in memory["components"].items():
        manifests[name] = load_component(root, fold_id, name)
        if sha256_file(folder / name / "component.json") != digest or manifests[name]["memory_key"] != memory["memory_key"]:
            raise AdaptationError(f"{fold_id}: {name} is not the component memory.json froze")
    return dict(memory, sha256=sha256_file(path)), manifests


def memory_table(root, fold_id, component, name):
    return pl.read_parquet(Path(root) / MEMORY_ROOT / fold_id / component / f"{name}.parquet")


def write_artifact(root, artifact_id, fold_id, tables, manifest):
    """Write one fold of an artifact once; every file is hashed into artifact.json and made read-only."""
    out = Path(root) / ADAPT_ROOT / artifact_id / fold_id
    if out.exists():
        raise AdaptationError(f"{out} exists; {artifact_id} is written once")
    staging = out.with_name(fold_id + ".building")
    staging.mkdir(parents=True)
    files = {}
    for name, frame in tables.items():
        if isinstance(frame, bytes):  # a named binary file, e.g. model weights
            (staging / name).write_bytes(frame)
            files[name] = sha256_file(staging / name)
            continue
        frame.write_parquet(staging / f"{name}.parquet", compression="zstd")
        files[f"{name}.parquet"] = sha256_file(staging / f"{name}.parquet")
    (staging / "artifact.json").write_text(json.dumps(dict(manifest, files=files), indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    staging.rename(out)
    for path in out.iterdir():
        os.chmod(path, stat.S_IREAD)
    return out


def load_artifact(root, artifact_id, fold_id):
    """Read an artifact's manifest after re-hashing every file it lists."""
    out = Path(root) / ADAPT_ROOT / artifact_id / fold_id
    if not (out / "artifact.json").exists():
        raise AdaptationError(f"{fold_id}: {artifact_id} is not built")
    manifest = json.loads((out / "artifact.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha256_file(out / name) != digest:
            raise AdaptationError(f"{out / name} changed after {artifact_id} was written")
    return manifest
