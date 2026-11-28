"""FUS-GATE-001 (P3.7, exploratory): synthetic copies follow the preregistered mutations; the gate reads no label."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from seqlogad.adaptation import gate, synthetic
from seqlogad.common.checksum import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _base():
    """Two sources, three windows each; window a2 has a single distinct context message."""
    rows = []
    for source, tag in (("ARCH-A", "a"), ("ARCH-B", "b")):
        for k in range(3):
            context = [f"{tag}x", f"{tag}y", f"{tag}z"] if k != 2 else [f"{tag}x", f"{tag}x"]
            rows.append({"window_id": f"{tag}{k}", "architecture_id": source, "target_rank": 10 * k + 5,
                         "context_ranks": [10 * k + i for i in range(len(context))], "context_messages": context,
                         "target_message": f"{tag}t{k}", "target_record_id": f"{source}:{10 * k + 5}"})
    return rows


def test_spec_matches_the_code():
    spec = gate.load_spec(ROOT)
    assert tuple(spec["synthetic"])[:3] == synthetic.MUTATIONS and spec["mode"] == "EXPLORATORY"
    assert set(synthetic.READS["P2.1"]) == {"CLEAN", "FOREIGN_TARGET"}  # SEMANTIC_LLAMA reads the target message only
    assert "FOREIGN_TOPOLOGY" not in synthetic.READS["P2.2"] and len(synthetic.READS["P2.4"]) == 4


def test_mutations_draw_donors_from_another_source_and_are_deterministic():
    base = _base()
    table = synthetic.mutations(base)
    assert table.equals(synthetic.mutations(base))
    source = {r["window_id"]: r["architecture_id"] for r in base}
    for row in table.filter(pl.col("donor_window_id").is_not_null()).iter_rows(named=True):
        assert source[row["donor_window_id"]] != row["architecture_id"]
    shuffled = table.filter(pl.col("mutation") == "SHUFFLED_CONTEXT")
    assert set(shuffled["base_window_id"]) == {"a0", "a1", "b0", "b1"}  # a2/b2 have one distinct context message
    rows = synthetic.message_rows(base, table)
    by = {r["synthetic_id"]: r for r in rows.iter_rows(named=True)}
    assert "FOREIGN_TOPOLOGY:a0" not in by and by["CLEAN:a0"]["context_messages"] == ["ax", "ay", "az"]
    assert by["FOREIGN_TARGET:a0"]["target_message"].startswith("bt")
    assert by["SHUFFLED_CONTEXT:a0"]["context_messages"] != ["ax", "ay", "az"]
    assert sorted(by["SHUFFLED_CONTEXT:a0"]["context_messages"]) == ["ax", "ay", "az"]


def test_graph_copies_change_only_what_the_mutation_says():
    nodes = pl.DataFrame({"architecture_id": ["ARCH-A"] * 3 + ["ARCH-B"] * 2, "partition": ["SOURCE_VALIDATION"] * 5,
                          "rank": [1, 2, 3, 1, 2], "effective_ts": [0] * 5,
                          "event_text": ["m1", "m2", "m3", "n1", "n2"], "component_text": ["c"] * 3 + ["k1", "k2"],
                          "severity": [0] * 5, "location": ["L"] * 3 + ["Q1", "Q2"], "request_group": [None] * 5,
                          "continuation_lines": [0] * 5}, schema_overrides={"request_group": pl.String})
    graphs = pl.DataFrame({"window_id": ["w", "v"], "architecture_id": ["ARCH-A", "ARCH-B"],
                           "partition": ["SOURCE_VALIDATION"] * 2, "sequence_id": ["s", "t"], "target_rank": [3, 2],
                           "context_ranks": [[2], [1]], "status": ["SCORED"] * 2})
    table = pl.DataFrame({"synthetic_id": ["FOREIGN_TARGET:w", "SHUFFLED_CONTEXT:w", "FOREIGN_TOPOLOGY:w"],
                          "mutation": ["FOREIGN_TARGET", "SHUFFLED_CONTEXT", "FOREIGN_TOPOLOGY"],
                          "base_window_id": ["w"] * 3, "architecture_id": ["ARCH-A"] * 3, "target_rank": [3] * 3,
                          "context_ranks": [[1, 2]] * 3, "donor_window_id": ["v", None, "v"],
                          "permutation": [None, [1, 0], None]})
    extra_nodes, extra_graphs = synthetic.graph_copies(nodes, graphs, table, {"SHUFFLED_CONTEXT:w": ["m2", "m1"]})
    part = {p: f.sort("rank") for (p,), f in extra_nodes.partition_by("partition", as_dict=True).items()}
    assert part["SYN:FOREIGN_TARGET:w"]["event_text"].to_list() == ["m2", "n2"]  # graph context is rank 2 only
    assert part["SYN:SHUFFLED_CONTEXT:w"]["event_text"].to_list() == ["m1", "m3"]  # rank 2 takes position 2's message
    topology = part["SYN:FOREIGN_TOPOLOGY:w"]
    assert topology["component_text"].to_list() == ["k1", "k2"] and topology["event_text"].to_list() == ["m2", "m3"]
    assert set(extra_graphs["window_id"]) == set(table["synthetic_id"])


def test_assemble_fills_missing_features_and_averages_the_sequence_family():
    experts = pl.DataFrame({"unit": ["u", "u", "u", "v"],
                            "expert_id": ["SEMANTIC_LLAMA", "SEQUENCE_LLAMA_REFERENCE", "SEQUENCE_LIGHTWEIGHT",
                                          "GTAT_STRUCTURAL_TEMPORAL"],
                            "scored": [True, True, True, False], "score": [1.0, 2.0, 4.0, None],
                            "native": [0.2, 0.6, None, None], "coverage": [1.0, 0.5, 0.9, 1.0]})
    windows = pl.DataFrame({"unit": ["u", "v"], "event_coverage": [1.0, None], "transition_coverage": [0.5, 1.0],
                            "topology_coverage": [1.0, 1.0], "context_length": [1.0, 0.5]})
    x, s, m, units = gate.assemble(experts, windows)
    assert units == ["u", "v"] and s[0, 0] == 1.0 and s[0, 1] == 3.0 and np.isnan(s[0, 2])
    assert x[0, :3].tolist() == [0.2, 0.6, 0.5] and x[0, 7] == 0.75
    assert x[1, 0] == gate.MISSING and x[1, 2] == 0.0 and x[1, 3] == gate.MISSING and not m[1].any()


def test_copy_scores_reuse_the_clean_score_and_check_receipts(tmp_path):
    view = tmp_path / "view" / synthetic.VIEW / "F"
    view.mkdir(parents=True)
    table = synthetic.mutations(_base())
    table.write_parquet(view / "mutations.parquet")
    folder = tmp_path / "scores" / "F"
    folder.mkdir(parents=True)
    for expert, task in gate.TASK_OF.items():
        rows = table.filter(pl.col("mutation").is_in(list(synthetic.READS[task])))
        lines = [json.dumps({"synthetic_id": r, "score_raw": float(i), "status": "VALID", "coverage": 1.0, "uncertainty": None})
                 for i, r in enumerate(rows["synthetic_id"])]
        (folder / f"{expert}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (folder / f"{expert}.receipt.json").write_text(json.dumps({
            "scores_sha256": sha256_file(folder / f"{expert}.jsonl"), "labels_read": False,
            "sanity_reproduction": {"pass": True}, "mutations_sha256": sha256_file(view / "mutations.parquet")}))
    scores, _ = gate.copy_scores(tmp_path / "scores", tmp_path / "view", "F")
    semantic = scores.filter(pl.col("expert_id") == "SEMANTIC_LLAMA")
    assert semantic.filter(pl.col("synthetic_id") == "SHUFFLED_CONTEXT:a0")["scored_as"][0] == "CLEAN:a0"
    (folder / "GTAT_STRUCTURAL_TEMPORAL.jsonl").write_text("{}\n")
    with pytest.raises(gate.AdaptationError, match="receipt"):
        gate.copy_scores(tmp_path / "scores", tmp_path / "view", "F")


def test_the_gate_learns_to_trust_the_family_that_separates_the_copies():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    rows, scores = [], []
    for episode in ("E1", "E2", "E3"):
        for k in range(60):
            for mutation in ("CLEAN", "SHUFFLED_CONTEXT"):
                rows.append({"unit": f"{mutation}:{episode}{k}", "episode": episode, "mutation": mutation,
                             "base": f"{episode}{k}"})
                shift = 2.0 if mutation != "CLEAN" else 0.0
                scores.append([rng.normal(0, 1), rng.normal(shift, 0.3), rng.normal(0, 1)])  # SEQUENCE separates
    windows = pl.DataFrame(rows)
    s = np.asarray(scores)
    x = np.tile([0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], (len(rows), 1))
    x[:, 0] = rng.uniform(size=len(rows))  # an uninformative feature
    m = np.ones_like(s, dtype=bool)
    cfg = dict(gate.load_spec(ROOT)["gate"]["training"], max_epochs=60)
    model, epochs, curve, report = gate.fit_gate(x, s, m, windows, cfg)
    assert 1 <= epochs <= 60 and len(curve) == 60
    shuffled = [r for r in report if r["mutation"] == "SHUFFLED_CONTEXT"]
    assert all(r["GATE"] > r["EQUAL_WEIGHT"] for r in shuffled)
    weights, _ = gate.fused(model, x, s, m)
    assert float(weights[:, 1].mean()) > 0.5  # the SEQUENCE family gets the largest weight
