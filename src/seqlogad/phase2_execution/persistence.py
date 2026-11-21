"""Bounded atomic Google Drive persistence with end-to-end digest checks."""
from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path

from seqlogad.common.checksum import sha256_file

from .archive import verify_zip


def _write_sidecar(path, digest):
    Path(str(path) + ".sha256").write_text(f"{digest}  {Path(path).name}\n")


def _verify_persisted_zip_once(path):
    result = verify_zip(path)
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise ValueError("persistent ZIP checksum sidecar missing")
    digest, name = sidecar.read_text().strip().split("  ", 1)
    if name != path.name or digest != result["sha256"]:
        raise ValueError("persistent ZIP checksum sidecar mismatch")
    return result


def verify_persisted_zip(path, attempts=12, delay_seconds=5.0):
    # Colab's Google Drive FUSE mount can serve stale bytes for a short time,
    # so a mismatch is re-read before it is treated as real.
    path = Path(path)
    for attempt in range(attempts):
        try:
            return _verify_persisted_zip_once(path)
        except (ValueError, OSError, zipfile.BadZipFile):
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds)


def _copy_verified(source, target, digest):
    # Write in place; Drive FUSE can serve wrong bytes for minutes after os.replace.
    shutil.copyfile(source, target)
    _write_sidecar(target, digest)
    return verify_persisted_zip(target)


def _verified_or_none(path, attempts=12):
    try:
        return verify_persisted_zip(path, attempts=attempts)
    except (ValueError, OSError, zipfile.BadZipFile):
        return None


def publish_atomic(local_zip, destination):
    local_zip, destination = Path(local_zip), Path(destination)
    source = verify_zip(local_zip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = Path(str(destination) + ".previous")
    if destination.exists():
        current = _verified_or_none(destination, attempts=1)
        if current is not None:  # never overwrite a good fallback with a broken copy
            _copy_verified(destination, previous, current["sha256"])
    return _copy_verified(local_zip, destination, source["sha256"])


def restore_previous_if_broken(destination):
    """Replace a broken persisted ZIP by its verified .previous copy; return True if restored."""
    destination = Path(destination)
    previous = Path(str(destination) + ".previous")
    if not destination.exists() or _verified_or_none(destination) is not None:
        return False
    fallback = _verified_or_none(previous)
    if fallback is None:
        return False
    _copy_verified(previous, destination, fallback["sha256"])
    return True
