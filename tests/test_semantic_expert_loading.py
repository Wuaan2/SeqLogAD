"""SemanticExpert must load a finished run whose config.yaml was annotated after training."""
from pathlib import Path

import pytest

from seqlogad.semantic import expert as semantic_expert
from seqlogad.semantic.checkpoint import seal
from seqlogad.semantic.config import identity, load_config
from seqlogad.semantic.contracts import dump

ROOT = Path(__file__).resolve().parents[1]


def _finished_run(tmp_path, recorded_cfg, final_cfg):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/step-1"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter/adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "state.pt").write_bytes(b"state")
    run_identity = {"config_sha256": identity(recorded_cfg), "fold_id": "FOLD-TARGET-ARCH-BGL",
                    "bundle_sha256": "b" * 64, "precision": "torch.bfloat16"}
    dump(checkpoint / "manifest.json", {"identity": run_identity, "step": 1})
    seal(checkpoint)
    dump(run / "config.yaml", final_cfg)
    dump(run / "manifest.json", {"identity": run_identity, "best_checkpoint": "step-1",
                                 "fold_id": "FOLD-TARGET-ARCH-BGL", "precision": "torch.bfloat16",
                                 "data_manifest_sha256": "b" * 64})
    return run


@pytest.fixture
def stub_model(monkeypatch):
    model = type("Model", (), {"eval": lambda self: self})()
    monkeypatch.setattr(semantic_expert, "load_model", lambda cfg, base, path: (model, None, "torch.bfloat16"))


def test_finished_run_with_completion_flag_loads(tmp_path, stub_model):
    cfg, _ = load_config(ROOT)
    run = _finished_run(tmp_path, dict(cfg, real_training_completed=False), dict(cfg, real_training_completed=True))
    assert semantic_expert.SemanticExpert.from_checkpoint(ROOT, run).metadata["fold_id"] == "FOLD-TARGET-ARCH-BGL"


def test_any_other_config_change_is_rejected(tmp_path, stub_model):
    cfg, _ = load_config(ROOT)
    run = _finished_run(tmp_path, dict(cfg, real_training_completed=False), dict(cfg, seed=7))
    with pytest.raises(ValueError, match="differs from the trained run"):
        semantic_expert.SemanticExpert.from_checkpoint(ROOT, run)
