"""P2.2 Sequence V13R1 is locked to the EXC-007 HDFS/BGL target folds."""
from pathlib import Path

import pytest

from seqlogad.phase2_execution.specs import TARGETS, deterministic_run_id, load_development_spec
from seqlogad.sequence.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_p2_2_is_ready_for_hdfs_and_bgl_only():
    spec = load_development_spec(ROOT)["experts"]["P2.2"]
    assert spec["implementation_status"] == "READY" and spec["sequence_candidate"] == "v13r1"
    cfg, _ = load_config(ROOT, candidate="v13r1")
    assert cfg["target_folds"] == ["FOLD-TARGET-ARCH-BGL", "FOLD-TARGET-ARCH-HDFS"]
    assert [deterministic_run_id("P2.2", t, sequence_candidate="v13r1") for t in TARGETS] == [
        "P2.2-DEVELOPMENT-BGL-S42-V13R1", "P2.2-DEVELOPMENT-HDFS-S42-V13R1"]
    with pytest.raises(ValueError):
        deterministic_run_id("P2.2", "HADOOP", sequence_candidate="v13r1")
