"""P2.4 graph view: causal context rules, field rules and the fail-closed validator (synthetic)."""
import zipfile
from pathlib import Path

import polars as pl
import pytest

from seqlogad.common.checksum import sha256_file
from seqlogad.graph import prepare
from seqlogad.graph.contracts import (GRAPH_SCHEMA, NODE_SCHEMA, load_graph_nodes, load_graph_windows,
                                      validate_graph_bundle)
from seqlogad.semantic.contracts import dump
from seqlogad.sequence.prepare import _sequence_id

ROOT = Path(__file__).resolve().parents[1]
GRAPH_VIEW = prepare.load_graph_view(ROOT)[1]
SEC = 1_000_000
FOLD = "FOLD-TARGET-ARCH-BGL"


def _members(ranks, gids, seconds=None, cont=None):
    return pl.DataFrame(
        {"rank": ranks, "effective_ts": [s * SEC for s in (seconds or ranks)], "gid": gids,
         "continuation": cont or [False] * len(ranks)},
        schema={"rank": pl.Int64, "effective_ts": pl.Int64, "gid": pl.Int32, "continuation": pl.Boolean})


def _windows(arch, partition, members, targets):
    """P2.2-shaped windows: context = last 8 records of the uncut group history."""
    rows = []
    for target in targets:
        row = members.filter(pl.col("rank") == target).row(0, named=True)
        history = members.filter((pl.col("gid") == row["gid"]) & (pl.col("rank") < target)).sort("rank")["rank"]
        group = f"G{row['gid']}" if row["gid"] >= 0 else "GLOBAL"
        rows.append((f"win-{arch}-{partition}-{target}", arch, partition, _sequence_id(arch, partition, group),
                     target, history.to_list()[-8:], row["effective_ts"]))
    return pl.DataFrame(rows, orient="row", schema={
        "window_id": pl.String, "architecture_id": pl.String, "partition": pl.String, "sequence_id": pl.String,
        "target_rank": pl.Int64, "context_ranks": pl.List(pl.Int64), "target_effective_ts": pl.Int64})


def _graphs(members, targets, arch="ARCH-HDFS", partition="SOURCE_TRAIN"):
    graphs, nodes = prepare.build_partition_graphs(arch, partition, members, _windows(arch, partition, members, targets),
                                                   GRAPH_VIEW)
    assert list(graphs.columns) == list(GRAPH_SCHEMA)
    return {r["target_rank"]: (r["context_ranks"], r["status"]) for r in graphs.iter_rows(named=True)}, nodes


def test_context_is_causal_and_capped_at_64():
    graphs, nodes = _graphs(_members(list(range(100)), [0] * 100), [90, 5])
    assert graphs[90] == (list(range(26, 90)), "SCORED")
    assert graphs[5] == (list(range(5)), "SCORED")
    assert set(nodes["rank"]) == set(range(26, 91)) | set(range(6))


def test_context_resets_at_group_and_partition():
    members = _members(list(range(12)), [r % 2 for r in range(12)])
    graphs, _ = _graphs(members, [11])
    assert graphs[11][0] == [1, 3, 5, 7, 9]
    validation = _members([100, 101, 102], [0, 0, 0])  # train records are not part of this partition
    graphs, _ = _graphs(validation, [102], partition="SOURCE_VALIDATION")
    assert graphs[102][0] == [100, 101]
    bad = _windows("ARCH-HDFS", "SOURCE_TRAIN", members, [11])
    with pytest.raises(ValueError, match="outside the partition"):
        prepare.build_partition_graphs("ARCH-HDFS", "SOURCE_TRAIN", validation, bad, GRAPH_VIEW)


def test_global_gap_cut_and_abstain():
    seconds = [0, 1, 2, 3, 4, 3605, 3606, 3607, 7207]  # gaps > 1 h before rank 5; exactly 1 h before rank 8
    ranks = list(range(9))
    graphs, nodes = _graphs(_members(ranks, [-1] * 9, seconds), [5, 7, 8])
    assert graphs[5] == ([], "ABSTAIN")
    assert graphs[7] == ([5, 6], "SCORED")
    assert graphs[8] == ([5, 6, 7], "SCORED")  # a gap of exactly 3600 s is not longer than 1 h
    assert 5 in set(nodes["rank"])  # an abstained target is still a node
    graphs, _ = _graphs(_members(ranks, [0] * 9, seconds), [5])  # named groups are never cut
    assert graphs[5] == ([0, 1, 2, 3, 4], "SCORED")


def test_hadoop_continuations_fold_into_parent():
    # rank:   0 1 2 3 4 5 6 7 8 9    c = continuation line, P = parsed record
    cont = [True, True, False, True, True, False, True, True, False, True]
    graphs, nodes = _graphs(_members(list(range(10)), [0] * 10, [0, 0, 1, 1, 1, 2, 2, 2, 3, 3], cont), [7, 9],
                            arch="ARCH-HADOOP")
    assert graphs[7] == ([2, 5], "SCORED")  # orphans 0-1 dropped; target 7 kept although it is a continuation
    assert graphs[9] == ([2, 5, 8], "SCORED")
    folded = dict(nodes.select("rank", "continuation_lines").iter_rows())
    assert folded == {2: 2, 5: 1, 7: 0, 8: 0, 9: 0}  # never counts a line at or after a target
    # 64 is applied after folding: 70 parents with 3 continuation lines each
    flags = ([False] + [True] * 3) * 70
    ranks = list(range(len(flags)))
    graphs, _ = _graphs(_members(ranks, [0] * len(ranks), [r // 4 for r in ranks], flags), [len(flags) - 1],
                        arch="ARCH-HADOOP")
    context = graphs[len(flags) - 1][0]
    assert len(context) == 64 and all(not flags[r] for r in context)
    graphs, _ = _graphs(_members(list(range(10)), [0] * 10, None, cont), [9])  # other sources never fold
    assert graphs[9][0] == list(range(9))


def test_component_text_and_field_rules():
    text = prepare.component_text
    assert text("dfs.DataNode$PacketResponder") == "dfs data node packet responder"
    assert text("org.apache.hadoop.mapreduce.v2.app.MRAppMaster") == "org apache hadoop mapreduce v app mr app master"
    assert text("nova.osapi_compute.wsgi.server") == "nova osapi compute wsgi server"
    assert text("LeaseRenewer:msrabi@msra-sa-41:9000") == "lease renewer"
    assert text("SERV_NET") == "serv net" and text("KERNEL") == "kernel"
    assert text("12345") is None and text(None) is None
    smap = GRAPH_VIEW["severity_map"]
    assert [prepare.severity_code(v, smap) for v in ("INFO", "WARN", "WARNING", "SEVERE", "FAILURE", "Kill", None)] == \
        [0, 1, 1, 2, 3, -1, -1]
    assert prepare.midplane("R02-M1-N0-C:J12-U11") == "R02-M1" and prepare.midplane("R00-M0-N4") == "R00-M0"
    assert prepare.midplane("UNKNOWN_LOCATION") is None and prepare.midplane(None) is None
    group = prepare.request_group("38101a0b-2096-447d-96ea-a692162415ae")
    assert len(group) == 16 and group == prepare.request_group("38101a0b-2096-447d-96ea-a692162415ae")
    assert group != prepare.request_group("113d3a99-c3da-401f-bd62-cc2caa5b96d2") and prepare.request_group(None) is None


# ------------------------------------------------------------------ validator

def _bundle(tmp_path: Path):
    graphs = pl.DataFrame([
        ("w1", "ARCH-HDFS", "SOURCE_TRAIN", "s1", 5, [3, 4], "SCORED"),
        ("w2", "ARCH-OPENSTACK", "SOURCE_TRAIN", "s2", 9, [], "ABSTAIN"),
        ("w3", "ARCH-HDFS", "SOURCE_VALIDATION", "s3", 20, [19], "SCORED"),
    ], schema=GRAPH_SCHEMA, orient="row")
    nodes = pl.DataFrame([
        (a, p, r, r * SEC, f"event {r}", "dfs data node", 0, None, "ab" if a == "ARCH-OPENSTACK" else None, 0)
        for a, p, r in [("ARCH-HDFS", "SOURCE_TRAIN", 3), ("ARCH-HDFS", "SOURCE_TRAIN", 4), ("ARCH-HDFS", "SOURCE_TRAIN", 5),
                        ("ARCH-OPENSTACK", "SOURCE_TRAIN", 9), ("ARCH-HDFS", "SOURCE_VALIDATION", 19),
                        ("ARCH-HDFS", "SOURCE_VALIDATION", 20)]
    ], schema=NODE_SCHEMA, orient="row")
    sequence = tmp_path / "sequence"
    dump(sequence / "manifests/bundle.json", {"bundle_id": "synthetic"})
    dump(sequence / f"folds/{FOLD}/manifests/fold.json",
         {"fold_id": FOLD, "target_architecture": "ARCH-BGL", "source_architectures": ["ARCH-HDFS", "ARCH-OPENSTACK"]})
    for directory, partition in (("train", "SOURCE_TRAIN"), ("val", "SOURCE_VALIDATION")):
        path = sequence / f"folds/{FOLD}/sequence/{directory}/windows.parquet"
        path.parent.mkdir(parents=True)
        graphs.filter(pl.col("partition") == partition).drop("context_ranks", "status").write_parquet(path)
    graph = tmp_path / "graph"
    prepare.write_fold(graph, FOLD, "ARCH-BGL", ["ARCH-HDFS", "ARCH-OPENSTACK"], graphs, nodes,
                       {"graph_view_sha256": "gv"})
    extra = {"lineage": {"sequence_bundle": {"manifest_sha256": sha256_file(sequence / "manifests/bundle.json")},
                         "graph_view_sha256": "gv"}}
    digest = prepare.write_manifests(graph, extra, [FOLD])
    return graph, sequence, extra, graphs, nodes, digest


def test_validator_accepts_a_sealed_bundle_and_loaders_read_it(tmp_path):
    graph, sequence, _, graphs, nodes, digest = _bundle(tmp_path)
    report = validate_graph_bundle(graph, FOLD, digest, sequence)
    assert report["bundle_sha256"] == digest == sha256_file(graph / "manifests/graph-bundle.json")
    assert report["counts"]["SOURCE_TRAIN"] == {"ARCH-HDFS": {"SCORED": 1}, "ARCH-OPENSTACK": {"ABSTAIN": 1}}
    assert load_graph_windows(graph, FOLD, "train").height == 2 and load_graph_nodes(graph, FOLD).height == 6
    with pytest.raises(ValueError, match="digest"):
        validate_graph_bundle(graph, FOLD, "0" * 64, sequence)


def test_zip_is_reproducible_and_rooted_at_phase2_graph(tmp_path):
    graph, sequence, _, _, _, digest = _bundle(tmp_path)
    first, second = tmp_path / "a.zip", tmp_path / "b.zip"
    prepare.write_zip(graph, first)
    prepare.write_zip(graph, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        archive.extractall(tmp_path / "x")
    assert names == sorted(names) and all(n.startswith("phase2_graph/") for n in names)
    assert validate_graph_bundle(tmp_path / "x/phase2_graph", FOLD, digest, sequence)["status"] == "PASS"


@pytest.mark.parametrize("column", ["label", "extra"])
def test_validator_rejects_an_extra_or_label_column(tmp_path, column):
    graph, sequence, extra, _, _, _ = _bundle(tmp_path)
    path = graph / f"folds/{FOLD}/graph/train/graphs.parquet"
    pl.read_parquet(path).with_columns(pl.lit(0).alias(column)).write_parquet(path)
    digest = prepare.write_manifests(graph, extra, [FOLD])  # resealed, so only the content check can fail
    with pytest.raises(ValueError, match="column"):
        validate_graph_bundle(graph, FOLD, digest, sequence)


def test_validator_and_builder_reject_a_target_architecture_row(tmp_path):
    graph, sequence, extra, graphs, nodes, _ = _bundle(tmp_path)
    path = graph / f"folds/{FOLD}/graph/metadata/nodes.parquet"
    leaked = pl.concat([nodes, nodes.head(1).with_columns(pl.lit("ARCH-BGL").alias("architecture_id"))])
    leaked.write_parquet(path)
    digest = prepare.write_manifests(graph, extra, [FOLD])
    with pytest.raises(ValueError, match="target-architecture"):
        validate_graph_bundle(graph, FOLD, digest, sequence)
    bad = pl.concat([graphs, graphs.head(1).with_columns(pl.lit("ARCH-BGL").alias("architecture_id"),
                                                         pl.lit("w9").alias("window_id"))])
    with pytest.raises(ValueError, match="target-architecture"):
        prepare.write_fold(tmp_path / "other", FOLD, "ARCH-BGL", ["ARCH-HDFS", "ARCH-OPENSTACK"], bad, nodes, {})
