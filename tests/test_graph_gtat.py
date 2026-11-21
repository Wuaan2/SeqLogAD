"""P2.4 GTAT-ST-001: preregistered config, graph construction rules, model budget and checkpoints."""
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from seqlogad.graph import config as gtat
from seqlogad.graph.features import EDGE, NodeStore, build_graph, examples
from seqlogad.phase2_execution.specs import TARGETS, deterministic_run_id

ROOT = Path(__file__).resolve().parents[1]
CFG = gtat.load_config(ROOT)


def nodes(events, seconds=None, components=None, locations=None, requests=None):
    n = len(events)
    return pl.DataFrame({
        "architecture_id": ["ARCH-X"] * n, "partition": ["SOURCE_TRAIN"] * n, "rank": list(range(n)),
        "effective_ts": [s * 1_000_000 for s in (seconds or list(range(n)))],
        "event_text": events, "component_text": components or [None] * n,
        "severity": [0] * n, "location": locations or [None] * n,
        "request_group": requests or [None] * n, "continuation_lines": [0] * n,
    }, schema_overrides={"severity": pl.Int8, "continuation_lines": pl.Int32,
                         "component_text": pl.Utf8, "location": pl.Utf8, "request_group": pl.Utf8})


def graph(events, **fields):
    store = NodeStore(nodes(events, **fields), CFG)
    return store, build_graph(store, np.arange(len(events) - 1), len(events) - 1)


def test_run_identity_and_config_guards():
    assert [deterministic_run_id("P2.4", t) for t in TARGETS] == [
        gtat.development_run_id(f"FOLD-TARGET-ARCH-{t}") for t in TARGETS] == [
        "P2.4-DEVELOPMENT-BGL-S42", "P2.4-DEVELOPMENT-HDFS-S42"]
    assert CFG["objective"]["target_node_visible_fields"] == []
    assert CFG["training"]["checkpoint_selection"] == "MAX_MACRO_SOURCE_DEVELOPMENT_F1"
    with pytest.raises(ValueError):
        gtat.development_run_id("FOLD-TARGET-ARCH-HADOOP")


def test_order_collision_abaca_acaba_gives_different_graphs_and_scores():
    torch = pytest.importorskip("torch")
    from seqlogad.graph.features import collate
    from seqlogad.graph.model import head_losses
    from seqlogad.graph.train import build_model

    _, abaca = graph(["a", "b", "a", "c", "a", "target"])
    _, acaba = graph(["a", "c", "a", "b", "a", "target"])
    assert abaca["event_tokens"] != acaba["event_tokens"]
    model = build_model(CFG).eval()
    with torch.no_grad():
        scores = [sum(head_losses(model(batch), batch).values()).item()
                  for batch in (collate([abaca]), collate([acaba]))]
    assert scores[0] != pytest.approx(scores[1])


def test_graph_rules_recurrence_masked_target_and_age():
    _, g = graph(["a", "b", "a", "a"], seconds=[0, 1, 5, 5000])
    assert g["edges"][EDGE["RECURRENCE"], 3, 1]  # second "a" points back to the first
    assert g["edges"][EDGE["NEXT"], 0, 3] and g["edges"][EDGE["CONTEXT"], 0, 1:4].all()
    assert g["event_tokens"][0] == [] and g["severity"][0] == 0  # target content masked
    assert g["label_pointer"] == 3 and g["label_delta"] == len(CFG["features"]["time_buckets"]) - 1
    assert g["time"][0, 1] == g["time"][0, 2]  # age is relative to the last context record


def test_ids_are_structure_not_features():
    _, first = graph(["a", "b", "c"], locations=["R00-M0", "R00-M0", "R01-M1"])
    _, second = graph(["a", "b", "c"], locations=["R17-M1", "R17-M1", "R02-M0"])
    for key in ("node_type", "offset", "count", "degree", "edges", "time"):
        assert np.array_equal(first[key], second[key]), key


def test_examples_reject_non_causal_context():
    store = NodeStore(nodes(["a", "b", "c"]), CFG)
    window = pl.DataFrame({"window_id": ["w"], "architecture_id": ["ARCH-X"], "partition": ["SOURCE_TRAIN"],
                           "sequence_id": ["s"], "target_rank": [1], "context_ranks": [[0, 2]],
                           "status": ["SCORED"]})
    with pytest.raises(ValueError, match="causal"):
        examples(store, window)


def test_parameter_budget_and_checkpoint_roundtrip(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from seqlogad.graph.train import build_model, load_weights, save_gtat_checkpoint
    from seqlogad.semantic.checkpoint import verify_checkpoint

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    model = build_model(CFG)
    assert sum(p.numel() for p in model.parameters()) <= CFG["model"]["max_total_parameters"]
    identity = {"adapter": "none"}
    checkpoint = save_gtat_checkpoint(model, torch.optim.AdamW(model.parameters()),
                                      torch.amp.GradScaler("cuda", enabled=False), tmp_path / "step-1", 1,
                                      identity, {"f1": None, "checkpoint": "step-1"})
    assert verify_checkpoint(checkpoint, identity)["adapter"] == "none"
    other = build_model(dict(CFG, seed=7))
    load_weights(other, checkpoint)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))


def test_budget_violation_fails_closed():
    pytest.importorskip("torch")
    from seqlogad.graph.train import build_model

    with pytest.raises(ValueError, match="parameter budget"):
        build_model(dict(CFG, model=dict(CFG["model"], hidden_size=1024, ffn_size=4096)))
