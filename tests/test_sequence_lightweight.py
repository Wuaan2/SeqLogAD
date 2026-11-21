"""P2.3 lightweight Sequence expert: frozen parity with P2.2 and the preregistered budget."""
from pathlib import Path

import pytest

from seqlogad.phase2_execution.specs import TARGETS, deterministic_run_id, load_development_spec
from seqlogad.sequence import lightweight
from seqlogad.sequence.config import load_config as load_reference

ROOT = Path(__file__).resolve().parents[1]
LORA_ONLY = {"adapter_name", "training_mode", "rank", "alpha", "use_rslora", "dropout",
             "target_modules", "model_id", "model_revision"}


def test_p2_3_is_ready_and_shares_the_reference_contract():
    spec = load_development_spec(ROOT)["experts"]["P2.3"]
    assert spec["implementation_status"] == "READY" and spec["blockers"] == []
    cfg, _ = lightweight.load_config(ROOT)
    reference, _ = load_reference(ROOT, candidate="v13r1")
    changed = {key for key in set(cfg) | set(reference) if cfg.get(key) != reference.get(key)}
    assert changed - LORA_ONLY <= lightweight.ALLOWED_OVERRIDES
    assert cfg["target_folds"] == ["FOLD-TARGET-ARCH-BGL", "FOLD-TARGET-ARCH-HDFS"]
    assert cfg["checkpoint_selection"] == "MAX_MACRO_SOURCE_DEVELOPMENT_F1"
    assert [deterministic_run_id("P2.3", t) for t in TARGETS] == [
        lightweight.development_run_id(f"FOLD-TARGET-ARCH-{t}") for t in TARGETS] == [
        "P2.3-DEVELOPMENT-BGL-S42", "P2.3-DEVELOPMENT-HDFS-S42"]
    with pytest.raises(ValueError):
        lightweight.development_run_id("FOLD-TARGET-ARCH-HADOOP")


def test_model_fits_the_preregistered_parameter_budget_and_roundtrips(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    import torch
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    cfg, _ = lightweight.load_config(ROOT)
    model = lightweight.build_model(cfg)
    assert sum(p.numel() for p in model.parameters()) == 37_030_144 <= lightweight.parameter_budget(cfg)
    lightweight.assert_lightweight(model)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    identity = {"adapter": "none"}
    checkpoint = lightweight.save_lightweight_checkpoint(
        model, optimizer, scaler, tmp_path / "step-1", 1, identity, {"f1": None, "checkpoint": "step-1"},
        adapter_name="sequence",
    )
    from seqlogad.semantic.checkpoint import verify_checkpoint
    assert verify_checkpoint(checkpoint, identity)["adapter"] == "none"
    torch.manual_seed(0)
    other = lightweight.build_model(dict(cfg, seed=7))
    lightweight.load_weights(other, checkpoint)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))


def test_budget_violation_fails_closed():
    pytest.importorskip("transformers")
    cfg, _ = lightweight.load_config(ROOT)
    cfg["architecture"] = dict(cfg["architecture"], hidden_size=1024, intermediate_size=4096)
    with pytest.raises(ValueError, match="parameter budget"):
        lightweight.build_model(cfg)


def test_lightweight_drive_backup_is_throttled(monkeypatch):
    from seqlogad.phase2_execution import runner

    captured, calls = {}, []
    monkeypatch.setattr(lightweight, "run_training", lambda *a, **k: captured.update(k) or {})
    monkeypatch.setattr(runner, "create_recovery",
                        lambda *a, **k: calls.append(1) or {"path": "p", "size": 1, "sha256": "x"})
    runner._lightweight_train("repo", "data", "out", "BGL", "bundle", None, "dest",
                              source_development_root="labels",
                              expected_source_development_sha256="digest")
    callback = captured["checkpoint_callback"]
    callback("run", "checkpoint", "best")
    assert calls == []  # inside the 30-minute window
    monkeypatch.setattr(runner, "LIGHTWEIGHT_BACKUP_SECONDS", 0)
    callback("run", "checkpoint", "best")
    assert calls == [1]
