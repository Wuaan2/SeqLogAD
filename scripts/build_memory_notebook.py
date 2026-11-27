"""Generate colab/P3.2_Memory/P3.2_Prototypes.ipynb from the seed P2.6 scoring notebook (same setup cells).

Usage: python scripts/build_memory_notebook.py <code_zip_sha256> <view_zip_sha256>
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "colab/Seeds/S3407/P2.6_S3407.ipynb"
OUT = REPO / "colab/P3.2_Memory/P3.2_Prototypes.ipynb"
DRIVE = "/content/drive/MyDrive/SeqLogAD"

INTRO = """# P3.2 — Target normal memory: buffer pass (ADAPT-MEM-001)

Runs the frozen seed-42 experts **P2.1 Semantic**, **P2.2 Sequence reference** and **P2.3 Sequence lightweight**
(P2-FREEZE-001) on 8,192 burn-in windows of each target fold (HDFS, BGL). For every window it stores the expert's raw
score as an EVID-CS-001 envelope and its embedding (last-token hidden state). Nothing is trained, no label is read and
no score is calibrated; P3.3 does that only once authorized. P2.4 GTAT runs on a local CPU. Each expert must first
reproduce its stored sanity score. Spec: `configs/protocols/adapt-mem-prototypes-v1.yaml`.

## Setup

1. In Colab, choose **Runtime → Change runtime type** and select an **L4** (native BF16, at least 20 GiB), the GPU the
   experts were trained on.
2. Upload `seqlogad-code-memory-v1.zip` and `p3.2-buffer-view-v1.zip` to `MyDrive/SeqLogAD/P3.2_Memory/uploads/`.
3. The seed-42 FINAL ZIPs must be on Drive where the P2.6 pass read them:
   `P2.1_Semantic/P2.1/`, `P2.2_Sequence/P2.2/`, `P2.3_Lightweight/P2.3/` (under `MyDrive/SeqLogAD/`). The data ZIPs are
   read from the seed-42 upload folders (`P2.1_Semantic/uploads/`, `P2.2_Sequence/uploads/`, `P2.4_GTAT/uploads/`).
4. Keep the Hugging Face token in **Colab Secrets** as `HF_TOKEN` with notebook access. Never paste it into a cell.
5. Choose **Runtime → Run all**. Expect about 3.5 hours on an L4.

Progress is saved on Drive every 256 windows. If Colab disconnects, run the notebook again: finished chunks and
experts are skipped. Download `MyDrive/SeqLogAD/P3.2_Memory/P3.2_buffer_pass.zip` and its `.sha256` only after
`SAFE_TO_DISCONNECT_COLAB=YES`.
"""

CONFIG = '''# HUMAN CONFIG — fixed by ADAPT-MEM-001 (configs/protocols/adapt-mem-prototypes-v1.yaml).
from pathlib import Path
import os, uuid
from google.colab import drive

drive.mount("/content/drive")

TASK_ID = "P3.2"
NOTEBOOK_STATUS = "READY"
BLOCKERS = []
TARGETS = ["BGL", "HDFS"]  # EXC-007: Hadoop/OpenStack are source-only
DRIVE_ROOT = Path("{drive}/P3.2_Memory")
UPLOAD_ROOT = DRIVE_ROOT / "uploads"  # the P3.2 code ZIP and the burn-in view ZIP
# Data ZIPs come from the seed-42 upload folders; nothing is written there.
UPLOAD_ROOTS = [UPLOAD_ROOT] + [Path("{drive}") / folder / "uploads" for folder in ("P2.2_Sequence", "P2.1_Semantic", "P2.4_GTAT")]
SEQLOGAD = Path("{drive}")
FINAL_ZIPS = {{  # seed-42 deliverables of P2-FREEZE-001
    "P2.1": SEQLOGAD / "P2.1_Semantic/P2.1/seqlogad_outputs_P2.1_S42_FINAL.zip",
    "P2.2": SEQLOGAD / "P2.2_Sequence/P2.2/seqlogad_outputs_P2.2_S42_FINAL.zip",
    "P2.3": SEQLOGAD / "P2.3_Lightweight/P2.3/seqlogad_outputs_P2.3_S42_FINAL.zip",
}}
PASS_TASKS = "P2.3,P2.2,P2.1"  # P2.4 runs on a local CPU
PERSISTENT_ROOT = DRIVE_ROOT
# Keep Python, extracted source, tokenizer cache and temporary outputs on Colab's local disk.
STATE_ROOT = Path("/content/SeqLogAD")
HF_HOME = STATE_ROOT / "hf-cache"
VENV = STATE_ROOT / "venv-py312"
RUNTIME_ROOT = STATE_ROOT / "runtime" / "memory-{short}"
REPO_ROOT = str(RUNTIME_ROOT / "seqlogad_code_memory")
VIEW_ROOT = str(RUNTIME_ROOT)  # holds data/processed/memory/ADAPT-MEM-001/<fold>/statistics
DATA_ROOT = str(RUNTIME_ROOT / "phase2_sequence")
SEMANTIC_ROOT = str(RUNTIME_ROOT / "phase2")
GRAPH_ROOT = str(RUNTIME_ROOT / "phase2_graph")
PASS_ROOT = DRIVE_ROOT / "buffer_pass"  # chunks and receipts persist on Drive, so a rerun resumes
WORK_ROOT = STATE_ROOT / "p3.2-work"
SESSION_ROOT = Path("/content/seqlogad_sessions") / uuid.uuid4().hex
EXPECTED_CODE_ZIP_SHA256 = "{code}"
EXPECTED_VIEW_ZIP_SHA256 = "{view}"
EXPECTED_DATA_ZIP_SHA256 = "4ed76b92e4b001138b28b4b3b4d3017d6ff7f30dd31e8cca0649bfdf34ede781"
EXPECTED_SEMANTIC_ZIP_SHA256 = "ebcfdfd6eb7b1607eb6d57e60fae88456d03bcb8ae2c6688ea398042c856224b"
EXPECTED_BUNDLE_SHA256 = "1268b669a8b1b474658cfa7daa6556c987b0f31f07e0101ed30af250a8d7f129"
EXPECTED_GRAPH_ZIP_SHA256 = "595b5dc7ec1b96abc80b4722b5f39dcf1c5653ec25aefebcce7ee74dbf36f9ac"
EXPECTED_GRAPH_MANIFEST_SHA256 = "8918172b4376fd822ec812e9b46b2725f83b517a43c94f3efe16f55623c9d059"
MIN_VRAM_GIB = 20

assert TARGETS == ["BGL", "HDFS"]
assert all(path.is_file() for path in FINAL_ZIPS.values()), "A seed-42 FINAL ZIP is missing on Drive"
assert NOTEBOOK_STATUS == "READY", "Notebook is fail-closed: " + ", ".join(BLOCKERS)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
for directory in (STATE_ROOT, PERSISTENT_ROOT, HF_HOME, SESSION_ROOT, PASS_ROOT, WORK_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
print({{"session_root": str(SESSION_ROOT), "persistent_root": str(PERSISTENT_ROOT),
       "upload_folders": [str(root) for root in UPLOAD_ROOTS]}})
'''

RUN = '''# Scores and embeds the burn-in windows; one progress line per chunk of 256 windows.
args = [
    PYTHON, "-m", "seqlogad.memory.prototypes", "pass",
    "--repo-root", REPO_ROOT,
    "--view-root", VIEW_ROOT,
    "--data-root", DATA_ROOT,
    "--semantic-root", SEMANTIC_ROOT,
    "--graph-root", GRAPH_ROOT,
    "--output-root", str(PASS_ROOT),
    "--work-root", str(WORK_ROOT),
    "--tasks", PASS_TASKS,
]
for task, path in FINAL_ZIPS.items():
    args += ["--final", f"{task}={path}"]
run_live(args, check=True)
'''

FINAL = '''# Every expert and fold must have a complete pass and a passing sanity check; package them for download.
experts = {"P2.1": "SEMANTIC_LLAMA", "P2.2": "SEQUENCE_LLAMA_REFERENCE", "P2.3": "SEQUENCE_LIGHTWEIGHT"}
package = DRIVE_ROOT / "P3.2_buffer_pass.zip"
with zipfile.ZipFile(SESSION_ROOT / package.name, "w", zipfile.ZIP_DEFLATED) as handle:
    for target in TARGETS:
        fold = f"FOLD-TARGET-ARCH-{target}"
        for task, expert in experts.items():
            receipt = json.loads((PASS_ROOT / fold / f"{expert}.receipt.json").read_text())
            assert receipt["sanity_reproduction"]["pass"] and not receipt["labels_read"]
            for suffix, key in ((".jsonl", "evidence_sha256"), (".embeddings.npy", "embeddings_sha256")):
                path = PASS_ROOT / fold / f"{expert}{suffix}"
                assert receipt[key] == digest(path), f"{task} {target} {path.name} changed"
                handle.write(path, f"buffer_pass/{fold}/{path.name}")
            handle.write(PASS_ROOT / fold / f"{expert}.receipt.json", f"buffer_pass/{fold}/{expert}.receipt.json")
            print(task, target, receipt["windows"], receipt["status_counts"], receipt["embedding_dim"])
shutil.copyfile(SESSION_ROOT / package.name, package)
package_hash = digest(package)
Path(str(package) + ".sha256").write_text(f"{package_hash}  {package.name}\\n")
shutil.rmtree(SESSION_ROOT)
print("====================================")
print("SEQLOGAD COLAB EXECUTION SUMMARY - P3.2 BUFFER PASS")
print(f"BUFFER_PASS_ZIP={package}")
print(f"BUFFER_PASS_ZIP_SHA256={package_hash}")
print("PERSISTENT_STORAGE_VERIFIED=YES")
print("SAFE_TO_DISCONNECT_COLAB=YES")
print("====================================")
'''


def put(text, anchor, addition):
    assert text.count(anchor) == 1, anchor
    return text.replace(anchor, anchor + addition)


def main(code_sha, view_sha):
    assert re.fullmatch(r"[0-9a-f]{64}", code_sha) and re.fullmatch(r"[0-9a-f]{64}", view_sha)
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    text = ["".join(cell["source"]) for cell in nb["cells"]]
    setup = text[4].replace("seqlogad_code_seed_confirmation", "seqlogad_code_memory")
    setup = put(setup, "graph_zip = locate(EXPECTED_GRAPH_ZIP_SHA256)\n", "view_zip = locate(EXPECTED_VIEW_ZIP_SHA256)\n")
    setup = put(setup, '    "graph_sha256": EXPECTED_GRAPH_ZIP_SHA256,\n', '    "view_sha256": EXPECTED_VIEW_ZIP_SHA256,\n')
    setup = put(setup, "    safe_extract(graph_zip, partial)\n", "    safe_extract(view_zip, partial)\n")
    setup = put(setup, '    assert (partial / "phase2_graph/manifests/graph-bundle.json").is_file()\n',
                '    assert (partial / "data/processed/memory/ADAPT-MEM-001").is_dir()\n')
    setup = setup.replace("P2.6 scoring of P2.1/P2.2 requires", "The P3.2 pass of P2.1/P2.2 requires")
    assert "seqlogad_code_memory" in setup and "P3.2 pass" in setup
    text[0] = INTRO
    text[1] = CONFIG.format(drive=DRIVE, short=code_sha[:12], code=code_sha, view=view_sha)
    text[4] = setup
    text[7] = "## Buffer pass (resumable; progress persists on Drive)"
    run_index = next(i for i, t in enumerate(text) if '"-m", "seqlogad.seeds.score"' in t)
    text[run_index] = RUN
    text[-1 if text[-1].strip() else -2] = FINAL
    for cell, value in zip(nb["cells"], text):
        cell["source"] = value.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    joined = "".join(text)
    assert "SEED" not in joined.replace("SEED-CONFIRM", "") and "seeds.score" not in joined, "seed leftovers"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(OUT.relative_to(REPO))


if __name__ == "__main__":
    main(*sys.argv[1:3])
