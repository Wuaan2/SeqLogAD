"""Build the P2.4 graph view (GTAT-ST-001 §3, §3.1) from P2.1/P2.2 bundles and raw logs.

Label-free and fail-closed. Ranks are rebuilt exactly as ``protocol.chronology`` does
(declared file order, per-file timestamp carry, stable sort on effective_ts) and bound to
the semantic lineage raw-file digests. Every node's re-normalised raw message must equal
the bundled ``normalized_message`` byte for byte, or the build stops.

Hadoop: context continuation lines fold into their parent first, then the 64 most recent
parent records are kept; the target is kept as it is. ``continuation_lines`` (a trainer count
feature) is CONTINUATION_LINES_RULE below: one value per rank, causal for every window.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
import shutil
import tempfile
import time
import zipfile
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import polars as pl
import yaml

from seqlogad.common.checksum import sha256_file
from seqlogad.evaluation.test_seal import canonical_json
from seqlogad.protocol.architectures import ADAPTERS, ARCHITECTURES, TimestampStatus
from seqlogad.protocol.normalizer import load_normalizer
from seqlogad.protocol.nul import canonicalize_nul_message
from seqlogad.semantic.contracts import dump, membership, read
from seqlogad.semantic.contracts import validate_bundle as validate_semantic_bundle
from seqlogad.semantic.prepare import git_state
from seqlogad.sequence.contracts import validate_bundle as validate_sequence_bundle
from seqlogad.sequence.prepare import _sequence_id

from .contracts import (BUNDLE_ID, BUNDLE_MANIFEST, BUNDLE_SCHEMA, GRAPH_SCHEMA, LEDGER, NODE_SCHEMA,
                        PARTITIONS, VIEW_SCHEMA, WINDOW_KEY, check_graphs, check_nodes, fold_files,
                        referenced_keys, validate_graph_bundle)

CONFIG = "configs/models/gtat-v1.yaml"
P2_2_CONTEXT = 8
REQUEST_KEY = b"GTAT-ST-001"
NORMALIZER = "metadata/normalizer-cs-v1.yaml"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
ZIP_ROOT = "phase2_graph"  # extracts to <runtime>/phase2_graph/, the graph_root of the contracts
CONTINUATION_LINES_RULE = (
    "Hadoop parent records only: continuation lines from the parent up to the next parsed record, "
    "stopping before the earliest target of the fold inside that run; never counts a line at or after "
    "any target. 0 for targets that are continuation lines and for every other source.")

# ----------------------------------------------------------------- field rules (§3.1)

_HOST_PORT = re.compile(r"[\w.@-]+:\d+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_MIDPLANE = re.compile(r"R\d\d-M\d")


def component_text(raw: str | None) -> str | None:
    """Lowercase words split on . $ _ and camelCase; digits and host:port removed."""
    if raw is None:
        return None
    text = _CAMEL.sub(" ", _HOST_PORT.sub(" ", raw))
    words = re.findall(r"[a-z]+", re.sub(r"\d+", " ", text).lower())
    return " ".join(words) or None


def severity_code(level: str | None, severity_map: dict, unknown: int = -1) -> int:
    return int(severity_map.get(level, unknown)) if level is not None else unknown


def midplane(node_id: str | None) -> str | None:
    match = _MIDPLANE.match(node_id or "")
    return match.group(0) if match else None


def request_group(request_id: str | None) -> str | None:
    if not request_id:
        return None
    return hashlib.blake2b(request_id.encode(), digest_size=8, key=REQUEST_KEY).hexdigest()


# --------------------------------------------------------------- context rules (§3)

def window_context(p: int, ranks, ts, cont, *, global_group: bool, fold: bool,
                   max_context: int, gap_us: int) -> list[int]:
    """Ranks of the <=max_context most recent records before history position p.

    Walks back from the target; in GLOBAL groups it stops at the first gap > gap_us.
    With ``fold`` (Hadoop), continuation lines are not records of their own: they belong to
    the nearest preceding parsed record, and orphans with no parsed record before them drop.
    """
    out: list[int] = []
    i = p - 1
    while i >= 0 and len(out) < max_context:
        if global_group and ts[i + 1] - ts[i] > gap_us:
            break
        if not (fold and cont[i]):
            out.append(ranks[i])
        i -= 1
    out.reverse()
    return out


def continuation_counts(cont, target_positions: list[int], parents: set[int]) -> dict[int, int]:
    """Folded continuation lines per parent position, never counting a line at or after a target.

    A parent's run is the continuation lines up to the next parsed record; when a target of
    the fold falls inside that run, only the lines before the earliest such target count.
    """
    targets = sorted(target_positions)
    counts = {}
    n = len(cont)
    for i in parents:
        j = i + 1
        while j < n and cont[j]:
            j += 1
        k = bisect.bisect_right(targets, i)
        stop = targets[k] if k < len(targets) and targets[k] < j else j
        counts[i] = stop - i - 1
    return counts


def build_partition_graphs(architecture: str, partition: str, members: pl.DataFrame,
                           windows: pl.DataFrame, graph_view: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Graphs of one (architecture, partition) and the node keys they use.

    ``members``: every record of the partition, columns rank, effective_ts, gid, continuation.
    ``windows``: the P2.2 windows of the partition. Fails closed on any P2.2 disagreement.
    """
    fold = graph_view["hadoop_continuation"].startswith("FOLD") and architecture == "ARCH-HADOOP"
    max_context = int(graph_view["max_context_records"])
    gap_us = int(graph_view["global_group_gap_cut_seconds"]) * 1_000_000
    members = members.sort("rank")
    targets = windows.join(members.select("rank", "gid", "effective_ts"), left_on="target_rank",
                           right_on="rank", how="left")
    if targets["gid"].null_count():
        raise ValueError(f"{architecture}/{partition}: window target outside the partition")
    if (targets["effective_ts"] != targets["target_effective_ts"]).any():
        raise ValueError(f"{architecture}/{partition}: target effective_ts differs from P2.2")
    expected = [_sequence_id(architecture, partition, f"G{g}" if g >= 0 else "GLOBAL")
                for g in targets["gid"].to_list()]
    if expected != targets["sequence_id"].to_list():
        raise ValueError(f"{architecture}/{partition}: graph sequence_id differs from P2.2")
    by_gid: dict[int, list] = {}
    for row in targets.select("gid", "target_rank", "context_ranks", "window_id").iter_rows():
        by_gid.setdefault(row[0], []).append(row[1:])
    groups = members.filter(pl.col("gid").is_in(list(by_gid))).group_by("gid", maintain_order=True).agg(
        "rank", "effective_ts", "continuation")
    contexts, node_rows = {}, []
    for gid, ranks, ts, cont in groups.iter_rows():
        positions = []
        for target_rank, p22_context, window_id in by_gid[gid]:
            p = bisect.bisect_left(ranks, target_rank)
            if list(p22_context) != ranks[max(0, p - P2_2_CONTEXT):p]:
                raise ValueError(f"{architecture}/{partition}: P2.2 context is not the group-history tail")
            contexts[window_id] = window_context(p, ranks, ts, cont, global_group=gid < 0, fold=fold,
                                                 max_context=max_context, gap_us=gap_us)
            positions.append(p)
        used = {bisect.bisect_left(ranks, r) for _, _, w in by_gid[gid] for r in contexts[w]}
        folded = continuation_counts(cont, positions, used) if fold else {}
        node_rows.extend((ranks[p], folded.get(p, 0)) for p in used | set(positions))
    graphs = windows.select("window_id", "architecture_id", "partition", "sequence_id", "target_rank").with_columns(
        pl.Series("context_ranks", [contexts[w] for w in windows["window_id"]], dtype=pl.List(pl.Int64)))
    graphs = graphs.with_columns(
        pl.when(pl.col("context_ranks").list.len() > 0).then(pl.lit("SCORED")).otherwise(pl.lit("ABSTAIN"))
        .alias("status")).select(list(GRAPH_SCHEMA))
    nodes = pl.DataFrame(node_rows, schema={"rank": pl.Int64, "continuation_lines": pl.Int32}, orient="row")
    return graphs, nodes.unique("rank").sort("rank")


# ------------------------------------------------------------------ raw-log access

def declared_files(architecture: str, raw_root: Path) -> list[tuple[Path, str]]:
    """Raw files in the declared P1 order, with the project-relative name the adapters saw."""
    spec = ARCHITECTURES[architecture]
    files = []
    for pattern in spec.file_glob:
        if any(ch in pattern for ch in "*?["):
            found = sorted(raw_root.glob(pattern), key=lambda p: PurePosixPath(p.relative_to(raw_root).as_posix()).parts)
            files.extend(found)
        elif (raw_root / pattern).is_file():
            files.append(raw_root / pattern)
    if not files:
        raise ValueError(f"no raw files for {architecture} under {raw_root}")
    return [(path, f"{spec.raw_root}/{path.relative_to(raw_root).as_posix()}") for path in files]


def _scan(files, adapter):
    """Pass 1: per-file timestamp carry, continuation flag, session gid, raw digests."""
    ts, cont, gid = array("q"), array("b"), array("i")
    sessions: dict[str, int] = {}
    digests = []
    for path, rel in files:
        digest, size, last = hashlib.sha256(), 0, 0
        with path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                digest.update(line)
                size += len(line)
                record = adapter(line, rel, number)
                if record.timestamp_us is not None:
                    last = record.timestamp_us
                ts.append(last)
                cont.append(record.timestamp_status is TimestampStatus.CONTINUATION)
                if record.session_id is None:
                    gid.append(-1)
                else:
                    gid.append(sessions.setdefault(record.session_id, len(sessions)))
        digests.append((digest.hexdigest(), size))
    frame = pl.DataFrame({"effective_ts": pl.Series(ts, dtype=pl.Int64), "gid": pl.Series(gid, dtype=pl.Int32),
                          "continuation": pl.Series(cont, dtype=pl.Int8).cast(pl.Boolean)})
    frame = frame.with_row_index("file_position").sort("effective_ts", maintain_order=True)
    return frame.with_row_index("rank").with_columns(pl.col("rank").cast(pl.Int64)), digests


def build_source(architecture: str, raw_root: str, semantic_root: str, sequence_root: str,
                 fold_ids: list[str], graph_view: dict, cache: str) -> dict:
    """Graphs and nodes of one source architecture, shared by every fold that uses it."""
    started = time.time()
    sem_root, seq_root, cache_dir = Path(semantic_root), Path(sequence_root), Path(cache)
    spec = ARCHITECTURES[architecture]
    adapter = ADAPTERS[spec.adapter]
    folds = [f for f in fold_ids if architecture in read(sem_root / f"folds/{f}/manifests/fold.json")["source_architectures"]]
    home = sem_root / f"folds/{folds[0]}"
    files = declared_files(architecture, Path(raw_root))
    frame, digests = _scan(files, adapter)
    lineage = read(home / "manifests/lineage.json")["sources"][architecture]["source_files"]
    raw_files = [{"source_id": f"file-{i:04}", "sha256": d, "bytes": n} for i, (d, n) in enumerate(digests)]
    if [(f["sha256"], f["bytes"]) for f in lineage] != digests:
        raise ValueError(f"{architecture}: raw files differ from the semantic lineage (content or order)")

    graphs_out, nodes_out, report = [], [], {"architecture_id": architecture, "raw_files": raw_files, "partitions": {}}
    for directory, partition in PARTITIONS.items():
        records = pl.read_parquet(home / f"semantic/{directory}/records.parquet").filter(
            pl.col("architecture_id") == architecture).select("rank", "normalized_message")
        for fold_id in folds:
            member = [m for m in read(sem_root / f"folds/{fold_id}/manifests/fold.json")["partitions"][partition]["members"]
                      if m["architecture_id"] == architecture]
            if len(member) != 1 or member[0]["records"] != records.height or membership(records["rank"]) != member[0]["membership_sha256"]:
                raise ValueError(f"{architecture}/{partition}: semantic membership differs from fold {fold_id}")
        windows = None
        for fold_id in folds:
            w = pl.read_parquet(seq_root / f"folds/{fold_id}/sequence/{directory}/windows.parquet").filter(
                pl.col("architecture_id") == architecture).sort("target_rank")
            if windows is not None and not w.equals(windows):
                raise ValueError(f"{architecture}/{partition}: P2.2 windows differ between folds")
            windows = w
        members = frame.join(records.select("rank"), on="rank", how="semi")
        if members.height != records.height:
            raise ValueError(f"{architecture}/{partition}: semantic rank missing from the raw chronology")
        graphs, keys = build_partition_graphs(architecture, partition, members.drop("file_position"), windows, graph_view)
        nodes = keys.join(members.select("rank", "file_position", "effective_ts"), on="rank", how="left").join(
            records, on="rank", how="left").with_columns(pl.lit(partition).alias("partition"))
        graphs_out.append(graphs)
        nodes_out.append(nodes)
        report["partitions"][partition] = {
            "members": records.height, "windows": graphs.height,
            "scored": int((graphs["status"] == "SCORED").sum()), "abstain": int((graphs["status"] == "ABSTAIN").sum()),
            "nodes": nodes.height}
    nodes = _attach_fields(architecture, files, pl.concat(nodes_out), graph_view, home / NORMALIZER, home)
    pl.concat(graphs_out).write_parquet(cache_dir / f"{architecture}-graphs.parquet", compression="zstd")
    nodes.write_parquet(cache_dir / f"{architecture}-nodes.parquet", compression="zstd")
    report["seconds"] = round(time.time() - started)
    print(f"[graph] {architecture}: {json.dumps(report['partitions'])} in {report['seconds']}s", flush=True)
    return report


def _attach_fields(architecture, files, nodes, graph_view, normalizer_path, normalizer_root):
    """Pass 2: re-parse node lines in file order, check the normalized message, read fields."""
    normalizer = load_normalizer(normalizer_root, normalizer_path.relative_to(normalizer_root).as_posix())
    adapter = ADAPTERS[ARCHITECTURES[architecture].adapter]
    severity_map = graph_view["severity_map"]
    unknown = int(graph_view["severity_unknown"])
    nodes = nodes.sort("file_position")
    wanted = nodes["file_position"].to_list()
    expected = nodes["normalized_message"].to_list()
    comp, sev, loc, req = [], [], [], []
    j, position = 0, 0
    for path, rel in files:
        with path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                if j < len(wanted) and position == wanted[j]:
                    record = adapter(line, rel, number)
                    text = normalizer.normalize(canonicalize_nul_message(record.raw_message_bytes).canonical_message)
                    if text != expected[j]:
                        raise ValueError(f"{architecture}: normalized message mismatch at rank lineage position {position}")
                    comp.append(component_text(record.component))
                    sev.append(severity_code(record.level, severity_map, unknown))
                    loc.append(midplane(record.node_id) if architecture == "ARCH-BGL" else None)
                    req.append(request_group(record.trace_id) if architecture == "ARCH-OPENSTACK" else None)
                    j += 1
                position += 1
    if j != len(wanted):
        raise ValueError(f"{architecture}: node outside the raw files")
    return nodes.with_columns(
        pl.lit(architecture).alias("architecture_id"),
        pl.col("normalized_message").alias("event_text"),
        pl.Series("component_text", comp, dtype=pl.String),
        pl.Series("severity", sev, dtype=pl.Int8),
        pl.Series("location", loc, dtype=pl.String),
        pl.Series("request_group", req, dtype=pl.String),
    ).select(list(NODE_SCHEMA)).sort("partition", "rank")


# ---------------------------------------------------------------- bundle materialisation

def write_fold(staging: Path, fold_id: str, target: str, sources: list[str], graphs: pl.DataFrame,
               nodes: pl.DataFrame, view_extra: dict) -> dict:
    """Write one fold's parquet files and view.json after the contract checks."""
    base = staging / f"folds/{fold_id}/graph"
    counts = {}
    for directory, partition in PARTITIONS.items():
        part = graphs.filter(pl.col("partition") == partition).sort("architecture_id", "target_rank")
        check_graphs(part, partition, set(sources), target)
        (base / directory).mkdir(parents=True, exist_ok=True)
        part.write_parquet(base / f"{directory}/graphs.parquet", compression="zstd")
        counts[partition] = {
            arch: {status: int(n) for status, n in part.filter(pl.col("architecture_id") == arch).group_by("status").len().iter_rows()}
            for arch in sorted(sources)}
    nodes = nodes.sort("architecture_id", "partition", "rank")
    check_nodes(nodes, set(sources), target)
    keys = referenced_keys(graphs)
    if keys.height != nodes.height or not keys.join(nodes, on=["architecture_id", "partition", "rank"], how="anti").is_empty():
        raise ValueError(f"{fold_id}: nodes are not exactly the records used by the graphs")
    (base / "metadata").mkdir(parents=True, exist_ok=True)
    nodes.write_parquet(base / "metadata/nodes.parquet", compression="zstd")
    node_counts = {f"{a}|{p}": int(n) for a, p, n in nodes.group_by("architecture_id", "partition").len().iter_rows()}
    view = dict(view_extra, schema_version=VIEW_SCHEMA, view_id="P2.4-GRAPH-VIEW-001", fold_id=fold_id,
                target_architecture=target, source_architectures=sorted(sources), counts=counts,
                node_counts=node_counts, continuation_lines_rule=CONTINUATION_LINES_RULE,
                target_labels_read=False, label_columns=[])
    dump(base / "metadata/view.json", view)
    return {"counts": counts, "node_counts": node_counts}


def write_manifests(staging: Path, bundle_extra: dict, folds: list[str]) -> str:
    """(Re)write graph-bundle.json and the checksum ledger; return the manifest SHA-256."""
    files = {name: sha256_file(staging / name) for name in sorted(n for f in folds for n in fold_files(f))}
    bundle = dict(bundle_extra, schema_version=BUNDLE_SCHEMA, bundle_id=BUNDLE_ID, folds=folds, files=files,
                  target_labels_read=False, label_columns=[])
    dump(staging / BUNDLE_MANIFEST, bundle)
    ledger = dict(files, **{BUNDLE_MANIFEST: sha256_file(staging / BUNDLE_MANIFEST)})
    (staging / LEDGER).write_text("".join(f"{d}  {n}\n" for n, d in sorted(ledger.items())))
    return ledger[BUNDLE_MANIFEST]


def write_zip(staging: Path, output: Path) -> None:
    """Reproducible zip under ZIP_ROOT/: sorted entries, fixed timestamps and attributes."""
    partial = output.with_name(output.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(f"{ZIP_ROOT}/{name}", date_time=ZIP_TIME)
            info.compress_type, info.create_system, info.external_attr = zipfile.ZIP_DEFLATED, 3, 0o644 << 16
            archive.writestr(info, (staging / name).read_bytes(), compresslevel=6)
    partial.replace(output)


def _extract(archive: Path, destination: Path) -> Path:
    """Extract a zip with zip-slip protection; return the directory holding manifests/."""
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
                raise ValueError(f"unsafe zip entry: {name}")
        handle.extractall(destination)
    roots = [p.parents[1] for p in destination.rglob("manifests/bundle.json")]
    if len(roots) != 1:
        raise ValueError(f"{archive.name}: expected one bundle root")
    return roots[0]


def load_graph_view(project_root: Path) -> tuple[dict, dict]:
    gtat = yaml.safe_load((project_root / CONFIG).read_text(encoding="utf-8"))["gtat"]
    return gtat, gtat["graph_view"]


def build_graph_bundle(project_root, semantic_zip, sequence_zip, output_zip, raw_roots: dict,
                       work_dir=None, jobs: int = 1) -> dict:
    """Build, zip and re-validate the graph view for every target fold of gtat-v1."""
    root = Path(project_root).resolve()
    output = Path(output_zip).resolve()
    if output.exists():
        raise ValueError("output exists; do not overwrite a prepared graph bundle")
    gtat, graph_view = load_graph_view(root)
    fold_ids = list(gtat["target_folds"])
    work = Path(tempfile.mkdtemp(prefix="graph-build-", dir=work_dir))
    try:
        sem_root = _extract(Path(semantic_zip), work / "semantic")
        seq_root = _extract(Path(sequence_zip), work / "sequence")
        seq_manifest = sha256_file(seq_root / "manifests/bundle.json")
        if seq_manifest != gtat["unit"]["sequence_bundle_manifest_sha256"]:
            raise ValueError("sequence bundle is not the one frozen in gtat-v1.yaml")
        normalizer_digests = set()
        folds = {}
        for fold_id in fold_ids:
            validate_semantic_bundle(sem_root, fold_id)
            validate_sequence_bundle(seq_root, fold_id, seq_manifest)
            fold = read(sem_root / f"folds/{fold_id}/manifests/fold.json")
            if read(seq_root / f"folds/{fold_id}/manifests/fold.json") != fold:
                raise ValueError(f"{fold_id}: semantic and sequence fold manifests differ")
            folds[fold_id] = fold
            normalizer_digests |= {sha256_file(r / f"folds/{fold_id}/{NORMALIZER}") for r in (sem_root, seq_root)}
        if len(normalizer_digests) != 1:
            raise ValueError("bundled normalizer copies differ")
        sources = sorted({a for f in folds.values() for a in f["source_architectures"]})
        cache = work / "cache"
        cache.mkdir()
        roots = {a: str(Path(raw_roots[a]).resolve()) for a in sources}
        args = [(a, roots[a], str(sem_root), str(seq_root), fold_ids, graph_view, str(cache)) for a in sources]
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                reports = dict(zip(sources, pool.map(build_source, *zip(*args))))
        else:
            reports = {a: build_source(*arg) for a, arg in zip(sources, args)}

        staging = work / "bundle"
        view_extra = {"graph_view": graph_view, "graph_view_sha256": hashlib.sha256(canonical_json(graph_view).encode()).hexdigest(),
                      "sequence_bundle_manifest_sha256": seq_manifest}
        fold_reports, memberships = {}, {}
        for fold_id, fold in folds.items():
            fold_sources = sorted(fold["source_architectures"])
            graphs = pl.concat([pl.read_parquet(cache / f"{a}-graphs.parquet") for a in fold_sources])
            nodes = pl.concat([pl.read_parquet(cache / f"{a}-nodes.parquet") for a in fold_sources])
            for directory, partition in PARTITIONS.items():
                windows = pl.read_parquet(seq_root / f"folds/{fold_id}/sequence/{directory}/windows.parquet").select(WINDOW_KEY)
                mine = graphs.filter(pl.col("partition") == partition).select(WINDOW_KEY)
                if windows.height != mine.height or not windows.join(mine, on=WINDOW_KEY, how="anti").is_empty():
                    raise ValueError(f"{fold_id}/{partition}: graph windows differ from the fold's sequence windows")
            fold_reports[fold_id] = write_fold(staging, fold_id, fold["target_architecture"], fold_sources, graphs, nodes, view_extra)
            memberships[fold_id] = {p: {m["architecture_id"]: m["membership_sha256"] for m in fold["partitions"][p]["members"]}
                                    for p in PARTITIONS.values()}
        code = Path(__file__).parent
        bundle_extra = {
            "lineage": {
                "semantic_bundle": {"zip_sha256": sha256_file(semantic_zip), "manifest_sha256": sha256_file(sem_root / "manifests/bundle.json")},
                "sequence_bundle": {"zip_sha256": sha256_file(sequence_zip), "manifest_sha256": seq_manifest},
                "fold_membership_sha256": memberships,
                "raw_files": {a: reports[a]["raw_files"] for a in sources},
                "normalizer_sha256": normalizer_digests.pop(),
                "graph_view_sha256": view_extra["graph_view_sha256"],
                "builder_git": git_state(root),
                "builder_code_sha256": {p.name: sha256_file(p) for p in (code / "prepare.py", code / "contracts.py")},
                "config_sha256": sha256_file(root / CONFIG),
            },
            "builder_revision": BUNDLE_SCHEMA,
        }
        manifest_sha = write_manifests(staging, bundle_extra, fold_ids)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_zip(staging, output)
        check = work / "verify"
        with zipfile.ZipFile(output) as handle:
            handle.extractall(check)
        try:
            results = {f: validate_graph_bundle(check / ZIP_ROOT, f, manifest_sha, seq_root) for f in fold_ids}
        except Exception:
            output.unlink(missing_ok=True)
            raise
        return {"zip": str(output), "zip_sha256": sha256_file(output), "bundle_sha256": manifest_sha,
                "folds": {f: {"counts": r["counts"], "node_counts": r["node_counts"]} for f, r in results.items()},
                "sources": {a: {k: v for k, v in r.items() if k != "raw_files"} for a, r in reports.items()}}
    finally:
        shutil.rmtree(work, ignore_errors=True)
