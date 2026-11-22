"""SEED-CONFIRM-001: confirmation seeds reuse the frozen seed-42 code and keep the seed-42 selection windows."""
import io
import zipfile
from pathlib import Path

import pytest

from seqlogad.common.checksum import sha256_file
from seqlogad.seeds import adapters
from seqlogad.seeds.protocol import final_name, load_protocol, run_id
from seqlogad.seeds.score import seeded_extract_run, seeded_scoring

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_names_runs_with_the_seed_and_rejects_the_primary_seed():
    assert load_protocol(ROOT, 3407)["primary_seed"] == 42
    for seed in (42, 1234):
        with pytest.raises(ValueError):
            load_protocol(ROOT, seed)
    assert run_id("P2.1", "BGL", 3407) == "P2.1-FINAL-BGL-S3407"
    assert run_id("P2.2", "HDFS", 8675309) == "P2.2-DEVELOPMENT-HDFS-S8675309-V13R1"
    assert run_id("P2.4", "BGL", 8675309) == "P2.4-DEVELOPMENT-BGL-S8675309"
    assert final_name("seqlogad_outputs_P2.3_S42_FINAL.zip", 3407) == "seqlogad_outputs_P2.3_S3407_FINAL.zip"


def test_selection_windows_stay_those_of_seed_42():
    calls = []
    select = adapters.fixed_selection(lambda *args, replacement: calls.append((args[2], args[4], replacement)))
    select("data", "FOLD-TARGET-ARCH-BGL", "train", 8, 3407, replacement=True)
    select("data", "FOLD-TARGET-ARCH-BGL", "val", 512, 3407, replacement=False)
    assert calls == [("train", 3407, True), ("val", 42, False)]


def test_adapters_change_only_the_training_seed_and_restore_the_frozen_names():
    from seqlogad.sequence import lightweight, train

    frozen = (lightweight.load_config, train.load_config, train.select_windows)
    reference, _ = train.load_config(ROOT, {"seed": 42}, run_mode="development", candidate="v13r1")
    light, _ = lightweight.load_config(ROOT)
    with adapters.confirmation_seed("P2.2", 3407) as module:
        cfg, _ = module.load_config(ROOT, {"seed": 3407}, run_mode="development", candidate="v13r1")
        assert module.development_run_id("FOLD-TARGET-ARCH-BGL", 3407, "v13r1") == "P2.2-DEVELOPMENT-BGL-S3407-V13R1"
        with pytest.raises(ValueError):
            module.development_run_id("FOLD-TARGET-ARCH-BGL", 42, "v13r1")
    assert {k for k in cfg.keys() | reference.keys() if cfg.get(k) != reference.get(k)} == {"seed"}
    with adapters.confirmation_seed("P2.3", 8675309) as module:
        seeded, _ = module.load_config(ROOT)
        assert module.development_run_id("FOLD-TARGET-ARCH-HDFS", 8675309) == "P2.3-DEVELOPMENT-HDFS-S8675309"
    assert {k for k in seeded.keys() | light.keys() if seeded.get(k) != light.get(k)} == {"seed"}
    assert (lightweight.load_config, train.load_config, train.select_windows) == frozen


def test_gtat_adapter_sets_the_seed():
    pytest.importorskip("torch")
    from seqlogad.graph import config, train

    with adapters.confirmation_seed("P2.4", 3407) as module:
        cfg = module.load_config(ROOT)
        assert module.development_run_id("FOLD-TARGET-ARCH-BGL", 3407) == "P2.4-DEVELOPMENT-BGL-S3407"
    assert cfg["seed"] == 3407 and train.load_config(ROOT)["seed"] == 42 and train.load_config is config.load_config


def test_seed_notebooks_pin_the_shipped_code_zip_and_their_own_seed():
    import json

    digest = sha256_file(ROOT / "colab/Seeds/uploads/seqlogad-code-seeds-v1.zip")
    for seed in (3407, 8675309):
        for task in ("P2.1", "P2.2", "P2.3", "P2.4", "P2.6"):
            notebook = json.loads((ROOT / f"colab/Seeds/S{seed}/{task}_S{seed}.ipynb").read_text(encoding="utf-8"))
            source = "".join("".join(cell["source"]) for cell in notebook["cells"])
            assert f'EXPECTED_CODE_ZIP_SHA256 = "{digest}"' in source
            assert f"SEED = {seed}" in source and f"Seeds/S{seed}" in source and "seqlogad.seeds." in source
            assert "phase2_execution.cli" not in source and "seqlogad_code_all_experts_s42" not in source


def _final_zip(path, task, target, seed):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as handle:
        handle.writestr(f"{task}-DEVELOPMENT-{target}-S{seed}/manifest.json", "{}")
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr(f"{task}_S{seed}/targets/{task}_{target}_S{seed}.zip", inner.getvalue())
    return path


def test_seed_scoring_reads_the_seed_run_and_restores_the_frozen_scorer(tmp_path):
    from seqlogad.diversity import score as frozen
    from seqlogad.graph import config as gconfig

    final = _final_zip(tmp_path / "final.zip", "P2.3", "BGL", 3407)
    run = seeded_extract_run(3407, {"P2.3": sha256_file(final)})(final, "P2.3", "BGL", tmp_path / "work")
    assert run.name == "P2.3-DEVELOPMENT-BGL-S3407"
    digests, extract, load = frozen.FINAL_SHA256, frozen.extract_run, gconfig.load_config
    with seeded_scoring(3407, {"P2.3": final}) as scorer:
        assert scorer.FINAL_SHA256 == {"P2.3": sha256_file(final)}
        assert gconfig.load_config(ROOT)["seed"] == 3407
    assert (frozen.FINAL_SHA256, frozen.extract_run, gconfig.load_config) == (digests, extract, load)
    assert gconfig.load_config(ROOT)["seed"] == 42
