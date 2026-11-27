"""ADAPT-MEM-001 prototypes (P3.2): nested window subset, resumable pass, k-means and the fold-memory freeze."""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from seqlogad.memory import prototypes
from seqlogad.memory.build import AdaptMemoryError, memory_key, write_component
from seqlogad.sequence.prepare import _candidate_score

ROOT = Path(__file__).resolve().parents[1]
FOLD, ARCH = "FOLD-TARGET-ARCH-BGL", "ARCH-BGL"


def _manifest(membership="ab"):
    return {"artifact_id": "ADAPT-MEM-001", "fold_id": FOLD, "target_architecture": ARCH, "buffer_id": "BUFFER-ARCH-BGL",
            "buffer_membership_sha256": membership, "memory_key": memory_key(ARCH, membership),
            "partition": "TARGET_BURN_IN"}


def _statistics(root, n=20):
    windows = pl.DataFrame({
        "window_id": [f"w{r}" for r in range(1, n + 1)], "architecture_id": [ARCH] * n,
        "partition": ["TARGET_BURN_IN"] * n, "sequence_id": ["s"] * n, "target_rank": list(range(1, n + 1)),
        "context_ranks": [[r - 1] for r in range(1, n + 1)]})
    events = pl.DataFrame({"architecture_id": [ARCH] * (n + 1), "rank": list(range(n + 1)),
                           "normalized_message": [f"m{r}" for r in range(n + 1)]})
    write_component(root, FOLD, "statistics", {"windows": windows, "events": events}, _manifest())


def test_spec_is_preregistered():
    spec = prototypes.load_spec(ROOT)
    assert spec["windows"]["limit"] == 8192 and spec["prototypes"]["k"] == 64
    assert set(spec["experts"]) == {"P2.1", "P2.2", "P2.3", "P2.4"}


def test_buffer_rows_are_the_lowest_candidate_scores_in_rank_order(tmp_path):
    _statistics(tmp_path)
    rows, manifest = prototypes.buffer_rows(tmp_path, FOLD, 5)
    expected = sorted(range(1, 21), key=lambda r: _candidate_score(ARCH, "TARGET_BURN_IN", r))[:5]
    assert [r["target_rank"] for r in rows] == sorted(expected)
    assert rows[0]["target_message"] == f"m{rows[0]['target_rank']}"
    assert rows[0]["context_messages"] == [f"m{rows[0]['target_rank'] - 1}"]
    assert manifest["memory_key"] == memory_key(ARCH, "ab")


def test_chunk_loop_resumes_without_rescoring(tmp_path, monkeypatch):
    monkeypatch.setattr(prototypes, "CHUNK", 2)
    monkeypatch.setattr(prototypes, "convert", lambda native: native)
    rows = [{"target_record_id": f"{ARCH}:{i}"} for i in range(5)]
    calls = []

    def run(chunk, fail_at=None):
        calls.append(chunk[0]["target_record_id"])
        if fail_at and chunk[0]["target_record_id"] == fail_at:
            raise RuntimeError("disconnect")
        return [({"target_record_id": r["target_record_id"]}, None if r is rows[3] else np.ones(3)) for r in chunk]

    with pytest.raises(RuntimeError):
        prototypes.score_chunks(rows, lambda c: run(c, f"{ARCH}:2"), 3, tmp_path)
    calls.clear()
    envelopes, vectors = prototypes.score_chunks(rows, run, 3, tmp_path)
    assert calls == [f"{ARCH}:2", f"{ARCH}:4"]  # chunk 0 was already on disk
    assert [e["target_record_id"] for e in envelopes] == [r["target_record_id"] for r in rows]
    assert np.isnan(vectors[3]).all() and vectors.dtype == np.float16 and not np.isnan(vectors[0]).any()


def test_kmeans_separates_clusters_and_skips_missing_vectors():
    rng = np.random.default_rng(0)
    a = rng.normal([5, 0, 0], 0.1, (20, 3))
    b = rng.normal([0, 5, 0], 0.1, (20, 3))
    vectors = np.vstack([a, b, np.full((1, 3), np.nan)]).astype(np.float16)
    centroids, cluster, distance = prototypes.kmeans_prototypes(vectors, 2, 42, 4)
    assert centroids.shape == (2, 3) and cluster[-1] == -1 and np.isnan(distance[-1])
    assert len(set(cluster[:20])) == 1 and len(set(cluster[20:40])) == 1 and cluster[0] != cluster[20]
    assert float(np.nanmax(distance)) < 0.01


def test_freeze_binds_components_once(tmp_path):
    for component in ("statistics", "prototypes", "buffer_scores"):
        write_component(tmp_path, FOLD, component, {"t": pl.DataFrame({"a": [1]})}, _manifest())
    memory = prototypes.freeze_memory(tmp_path, FOLD)
    assert memory["status"] == "FROZEN" and set(memory["components"]) == {"statistics", "prototypes", "buffer_scores"}
    with pytest.raises(AdaptMemoryError):
        prototypes.freeze_memory(tmp_path, FOLD)


def test_llama_embedding_is_the_last_token_hidden_state_of_the_scorer_forward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32))
    model = peft.get_peft_model(base, peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"])).eval()
    inputs = {"r1": torch.tensor([[1, 5, 7, 9]]), "r2": torch.tensor([[3, 4]])}

    def one(row):  # stands in for score_next_log: one forward of a batch of one window
        with torch.no_grad():
            model(input_ids=inputs[row["id"]])
        return {"id": row["id"]}

    run, dim, hook = prototypes._with_embeddings("P2.2", None, one, model)
    pairs = run([{"id": "r1"}, {"id": "r2"}])
    hook.remove()
    assert dim == 16
    for row_id, (native, vector) in zip(("r1", "r2"), pairs):
        with torch.no_grad():
            expected = model(input_ids=inputs[row_id], output_hidden_states=True).hidden_states[-1][0, -1]
        assert native["id"] == row_id and np.allclose(vector, expected.numpy(), atol=1e-6)


def test_notebook_pins_the_shipped_code_zip_and_runs_the_buffer_pass():
    import json

    from seqlogad.common.checksum import sha256_file

    notebook = json.loads((ROOT / "colab/P3.2_Memory/P3.2_Prototypes.ipynb").read_text(encoding="utf-8"))
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    digest = sha256_file(ROOT / "colab/P3.2_Memory/uploads/seqlogad-code-memory-v1.zip")
    assert f'EXPECTED_CODE_ZIP_SHA256 = "{digest}"' in source and "EXPECTED_VIEW_ZIP_SHA256" in source
    assert '"seqlogad.memory.prototypes", "pass"' in source and 'PASS_TASKS = "P2.3,P2.2,P2.1"' in source
    assert "assert native_bf16" in source and notebook["metadata"]["colab"]["gpuType"] == "L4"
