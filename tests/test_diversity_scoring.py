"""P2.6 scoring pass: resumable evidence files and verified FINAL ZIP inputs."""
import json

import pytest

pytest.importorskip("polars")

from seqlogad.diversity import score


def test_resume_drops_a_partial_trailing_line(tmp_path):
    path = tmp_path / "EXPERT.jsonl"
    path.write_text(json.dumps({"target_record_id": "ARCH-X:1"}) + "\n" + '{"target_record_id": "ARCH-X:2"')
    assert score._done(path) == {"ARCH-X:1"}
    assert path.read_text().count("\n") == 1


def test_unverified_final_zip_is_refused(tmp_path):
    fake = tmp_path / "final.zip"
    fake.write_bytes(b"not the frozen FINAL ZIP")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        score.extract_run(fake, "P2.2", "BGL", tmp_path / "work")
