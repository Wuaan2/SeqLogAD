"""Verify completed Sequence runs before reusing a persisted target ZIP."""
import json
import zipfile

from seqlogad.sequence.config import identity, load_config
from seqlogad.sequence.implementation import implementation_digest


def _verify_manifest(archive, run_id, expected):
    with zipfile.ZipFile(archive) as handle:
        manifest = json.loads(handle.read(f"{run_id}/manifest.json"))
    if manifest.get("run_id") != run_id or manifest.get("real_training_completed") is not True:
        raise ValueError("completed Sequence run identity mismatch")
    if any(manifest.get("identity", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("completed Sequence config/data/code/labels identity mismatch")


def _expected(repo_root, cfg, fold_id, bundle_sha256, sidecar_sha256):
    return {"config_sha256": identity(cfg), "fold_id": fold_id,
            "bundle_sha256": bundle_sha256,
            "implementation_sha256": implementation_digest(repo_root),
            "source_development_manifest_sha256": sidecar_sha256}


def verify_completed_sequence(archive, repo_root, run_id, fold_id, candidate,
                              bundle_sha256, sidecar_sha256):
    cfg, _ = load_config(repo_root, candidate=candidate)
    _verify_manifest(archive, run_id, _expected(repo_root, cfg, fold_id, bundle_sha256, sidecar_sha256))


def verify_completed_gtat(archive, repo_root, run_id, fold_id, bundle_sha256, sidecar_sha256,
                          graph_sha256):
    from seqlogad.graph import config as gtat

    _verify_manifest(archive, run_id, {
        "config_sha256": gtat.identity(gtat.load_config(repo_root)), "fold_id": fold_id,
        "bundle_sha256": bundle_sha256, "graph_bundle_sha256": graph_sha256,
        "implementation_sha256": gtat.implementation_digest(repo_root),
        "source_development_manifest_sha256": sidecar_sha256,
    })


def verify_completed_lightweight(archive, repo_root, run_id, fold_id, bundle_sha256, sidecar_sha256):
    from seqlogad.sequence.lightweight import load_config as load_lightweight_config

    cfg, _ = load_lightweight_config(repo_root)
    _verify_manifest(archive, run_id, _expected(repo_root, cfg, fold_id, bundle_sha256, sidecar_sha256))
