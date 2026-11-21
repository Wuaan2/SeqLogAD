"""verify_persisted_zip re-reads a stale Drive view before failing."""
import pytest

from seqlogad.phase2_execution import persistence


def test_stale_read_recovers(monkeypatch, tmp_path):
    calls = []

    def flaky(path):
        calls.append(path)
        if len(calls) < 3:
            raise ValueError("persistent ZIP checksum sidecar mismatch")
        return {"sha256": "ok"}

    monkeypatch.setattr(persistence, "_verify_persisted_zip_once", flaky)
    monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
    assert persistence.verify_persisted_zip(tmp_path / "a.zip") == {"sha256": "ok"}
    assert len(calls) == 3


def test_real_mismatch_still_fails(monkeypatch, tmp_path):
    def broken(path):
        raise ValueError("persistent ZIP checksum sidecar mismatch")

    monkeypatch.setattr(persistence, "_verify_persisted_zip_once", broken)
    monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="sidecar mismatch"):
        persistence.verify_persisted_zip(tmp_path / "a.zip", attempts=4)


def _zip(tmp_path, name, text):
    from seqlogad.phase2_execution.archive import create_zip, seal_tree
    root = tmp_path / f"src_{name}"
    root.mkdir()
    (root / "payload.txt").write_text(text)
    seal_tree(root)
    return create_zip(root, tmp_path / f"{name}.zip", "run") and tmp_path / f"{name}.zip"


def test_publish_keeps_previous_and_restores_broken_destination(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
    first, second = _zip(tmp_path, "a", "one"), _zip(tmp_path, "b", "two")
    destination = tmp_path / "drive" / "LATEST_RECOVERY.zip"
    one = persistence.publish_atomic(first, destination)
    two = persistence.publish_atomic(second, destination)
    previous = tmp_path / "drive" / "LATEST_RECOVERY.zip.previous"
    assert persistence.verify_persisted_zip(previous)["sha256"] == one["sha256"]
    destination.write_bytes(b"not a zip")  # simulate a broken Drive copy
    assert persistence.restore_previous_if_broken(destination) is True
    assert persistence.verify_persisted_zip(destination, attempts=1)["sha256"] == one["sha256"]
    assert two["sha256"] != one["sha256"]
    assert not persistence.restore_previous_if_broken(destination)
