"""ADAPT-MEM-001 (P3.2): memory reads only the approved buffer, keys on it and cannot change after writing."""
import os
import stat
from pathlib import Path

import polars as pl
import pytest

from seqlogad.memory.build import (AdaptMemoryError, check_buffer, load_component, load_contract, memory_key,
                                   transition_counts, write_component)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract(ROOT)


def test_contract_covers_the_two_target_folds_and_forbids_labels():
    assert set(CONTRACT["buffers"]) == {"FOLD-TARGET-ARCH-HDFS", "FOLD-TARGET-ARCH-BGL"}  # EXC-007
    assert CONTRACT["partition"] == "TARGET_BURN_IN" and CONTRACT["components"]["prototypes"]["status"] == "PENDING_GPU"
    assert any("label" in rule for rule in CONTRACT["forbidden"])
    assert memory_key("ARCH-BGL", "ab") == "ADAPT-MEM-001/ARCH-BGL/ab"


def test_p3_2_status_update_has_exact_provenance():
    import openpyxl
    import yaml

    from seqlogad.common.checksum import sha256_file

    update = CONTRACT["roadmap_update"]
    handoff = yaml.safe_load((ROOT / "configs/protocols/p3.1-g2-handoff-v1.yaml").read_text(encoding="utf-8"))
    after = yaml.safe_load((ROOT / "configs/protocols/cal-cs-v1.yaml").read_text(encoding="utf-8"))["cal_cs"]
    before_path, after_path = ROOT / update["previous_workbook"], ROOT / after["roadmap_update"]["previous_workbook"]
    assert sha256_file(before_path) == update["from_sha256"] == handoff["g2_handoff"]["roadmap_update"]["to_sha256"]
    assert sha256_file(after_path) == update["to_sha256"]
    books = [openpyxl.load_workbook(p) for p in (before_path, after_path)]
    before, after = [{f"{s.title}!{c.coordinate}": c.value for s in b for row in s for c in row if c.value is not None}
                     for b in books]
    changed = {k: {"before": before.get(k), "after": after.get(k)}
               for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changed == update["changed_cells"] and update["gate_signature_created"] is False
    for sheet in books[0]:
        assert str(sheet.data_validations) == str(books[1][sheet.title].data_validations)


def test_transitions_reset_at_each_group_and_keep_the_global_chronology():
    buffer = pl.DataFrame({"rank": [0, 1, 2, 3, 4, 5], "effective_ts": [0] * 6,
                           "gid": [7, -1, 7, -1, 8, 7], "norm_h": [1, 2, 3, 2, 1, 1]})
    unigrams, transitions = transition_counts(buffer)
    assert dict(unigrams.select("norm_h", "count").iter_rows()) == {1: 3, 2: 2, 3: 1}
    # group 7: 1->3->1; GLOBAL: 2->2; group 8 has one record and no transition
    assert sorted(transitions.iter_rows()) == [(1, 3, 1), (2, 2, 1), (3, 1, 1)]


def _approved(expected, **changes):
    manifest = {"buffer_id": "BUFFER-ARCH-BGL", "records": expected["records"],
                "buffer_membership_sha256": expected["membership_sha256"],
                "distinct_normalized_messages": expected["distinct_normalized_messages"],
                "first_timestamp_utc": expected["first_timestamp_utc"], "last_timestamp_utc": expected["last_timestamp_utc"],
                "target_labels_read": False,
                "rejection_rule": {"acceptance": "ACCEPTED", "adaptation_use_permitted": True}}
    index = {"source_files": [{"sha256": s} for s in expected["raw_sha256"]], "labels_read": False,
             "normalizer": {"rule_file_sha256": CONTRACT["normalizer_rule_file_sha256"]}}
    for key, value in changes.items():
        (manifest["rejection_rule"] if key == "acceptance" else manifest)[key] = value
    return manifest, index


def test_only_the_approved_buffer_passes():
    expected = CONTRACT["buffers"]["FOLD-TARGET-ARCH-BGL"]
    check_buffer(*_approved(expected), expected, CONTRACT["normalizer_rule_file_sha256"])
    for change in ({"buffer_membership_sha256": "0" * 64}, {"records": 1}, {"acceptance": "REJECTED"},
                   {"target_labels_read": True}):
        with pytest.raises(AdaptMemoryError):
            check_buffer(*_approved(expected, **change), expected, CONTRACT["normalizer_rule_file_sha256"])


def test_component_is_written_once_and_detects_tampering(tmp_path):
    manifest = {"target_architecture": "ARCH-BGL", "buffer_membership_sha256": "ab",
                "memory_key": memory_key("ARCH-BGL", "ab")}
    out = write_component(tmp_path, "FOLD-TARGET-ARCH-BGL", "statistics", {"unigrams": pl.DataFrame({"a": [1]})},
                          manifest)
    assert load_component(tmp_path, "FOLD-TARGET-ARCH-BGL", "statistics")["files"]
    assert not os.access(out / "unigrams.parquet", os.W_OK)
    with pytest.raises(AdaptMemoryError):
        write_component(tmp_path, "FOLD-TARGET-ARCH-BGL", "statistics", {}, manifest)
    os.chmod(out / "unigrams.parquet", stat.S_IWRITE | stat.S_IREAD)
    pl.DataFrame({"a": [2]}).write_parquet(out / "unigrams.parquet")
    with pytest.raises(AdaptMemoryError):
        load_component(tmp_path, "FOLD-TARGET-ARCH-BGL", "statistics")


def test_statistics_receipt_binds_the_current_contract_and_builder():
    import yaml

    from seqlogad.common.checksum import sha256_file

    receipt = yaml.safe_load((ROOT / "configs/protocols/adapt-mem-statistics-v1.yaml").read_text(encoding="utf-8"))
    receipt = receipt["adapt_mem_statistics"]
    assert set(receipt["folds"]) == set(CONTRACT["buffers"]) and receipt["fold_memory_frozen"] is False
    for fold_id, fold in receipt["folds"].items():
        expected = CONTRACT["buffers"][fold_id]
        assert fold["memory_key"] == memory_key(expected["target_architecture"], expected["membership_sha256"])
        assert fold["counts"]["buffer_records"] == expected["records"]
        assert fold["counts"]["distinct_events"] == expected["distinct_normalized_messages"]
        # editing either file after the build needs a new memory version
        assert fold["contract_sha256"] == CONTRACT["sha256"]
        assert fold["builder_sha256"] == sha256_file(ROOT / "src/seqlogad/memory/build.py")
