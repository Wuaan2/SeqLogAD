"""Generate colab/P3.7_Gate/P3.7_Synthetic.ipynb from the P3.2 buffer-pass notebook (same setup cells).

Usage: python scripts/build_gate_notebook.py <code_zip_sha256> <view_zip_sha256>
"""
import json
import re
import sys
from pathlib import Path

from build_memory_notebook import CONFIG as MEMORY_CONFIG

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "colab/P3.2_Memory/P3.2_Prototypes.ipynb"
OUT = REPO / "colab/P3.7_Gate/P3.7_Synthetic.ipynb"
DRIVE = "/content/drive/MyDrive/SeqLogAD"

INTRO = """# P3.7 — Exploratory gate: synthetic scoring pass (FUS-GATE-001)

Runs the frozen seed-42 experts **P2.1 Semantic**, **P2.2 Sequence reference** and **P2.3 Sequence lightweight**
(P2-FREEZE-001) on the preregistered synthetic copies of the checkpoint-selection windows of each target fold
(CLEAN, FOREIGN_TARGET, SHUFFLED_CONTEXT). An expert scores only the copies whose input differs from CLEAN. Nothing is
trained and no label is read. P2.4 GTAT and the gate run on a local CPU. Each expert must first reproduce its stored
sanity score. The gate is EXPLORATORY: G2 stays PARTIAL. Spec: `configs/protocols/fus-gate-v1.yaml`.

## Setup

1. In Colab, choose **Runtime → Change runtime type** and select an **L4** (native BF16, at least 20 GiB).
2. Upload `seqlogad-code-gate-v1.zip` and `p3.7-synthetic-view-v1.zip` to `MyDrive/SeqLogAD/P3.7_Gate/uploads/`.
3. The seed-42 FINAL ZIPs and the data ZIPs must be where the P3.2 pass read them (`P2.1_Semantic/`, `P2.2_Sequence/`,
   `P2.3_Lightweight/`, `P2.4_GTAT/` under `MyDrive/SeqLogAD/`).
4. Keep the Hugging Face token in **Colab Secrets** as `HF_TOKEN` with notebook access. Never paste it into a cell.
5. Choose **Runtime → Run all**. Expect about 2 hours on an L4.

Progress is saved on Drive every 256 copies. If Colab disconnects, run the notebook again: finished copies and experts
are skipped. Download `MyDrive/SeqLogAD/P3.7_Gate/P3.7_synthetic_scores.zip` and its `.sha256` only after
`SAFE_TO_DISCONNECT_COLAB=YES`.
"""

RUN = '''# Scores the synthetic copies; one progress line per chunk of 256 copies.
args = [
    PYTHON, "-m", "seqlogad.adaptation.synthetic", "score",
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
package = DRIVE_ROOT / "P3.7_synthetic_scores.zip"
with zipfile.ZipFile(SESSION_ROOT / package.name, "w", zipfile.ZIP_DEFLATED) as handle:
    for target in TARGETS:
        fold = f"FOLD-TARGET-ARCH-{target}"
        for task, expert in experts.items():
            receipt = json.loads((PASS_ROOT / fold / f"{expert}.receipt.json").read_text())
            assert receipt["sanity_reproduction"]["pass"] and not receipt["labels_read"]
            path = PASS_ROOT / fold / f"{expert}.jsonl"
            assert receipt["scores_sha256"] == digest(path), f"{task} {target} {path.name} changed"
            handle.write(path, f"synthetic_scores/{fold}/{path.name}")
            handle.write(PASS_ROOT / fold / f"{expert}.receipt.json", f"synthetic_scores/{fold}/{expert}.receipt.json")
            print(task, target, receipt["copies"], receipt["status_counts"])
shutil.copyfile(SESSION_ROOT / package.name, package)
package_hash = digest(package)
Path(str(package) + ".sha256").write_text(f"{package_hash}  {package.name}\\n")
shutil.rmtree(SESSION_ROOT)
print("====================================")
print("SEQLOGAD COLAB EXECUTION SUMMARY - P3.7 SYNTHETIC SCORING")
print(f"SYNTHETIC_SCORES_ZIP={package}")
print(f"SYNTHETIC_SCORES_ZIP_SHA256={package_hash}")
print("PERSISTENT_STORAGE_VERIFIED=YES")
print("SAFE_TO_DISCONNECT_COLAB=YES")
print("====================================")
'''


def swap(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main(code_sha, view_sha):
    assert re.fullmatch(r"[0-9a-f]{64}", code_sha) and re.fullmatch(r"[0-9a-f]{64}", view_sha)
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    text = ["".join(cell["source"]) for cell in nb["cells"]]
    config = MEMORY_CONFIG.format(drive=DRIVE, short=code_sha[:12], code=code_sha, view=view_sha)
    for old, new in (("# HUMAN CONFIG — fixed by ADAPT-MEM-001 (configs/protocols/adapt-mem-prototypes-v1.yaml).",
                      "# HUMAN CONFIG — fixed by FUS-GATE-001 (configs/protocols/fus-gate-v1.yaml); EXPLORATORY."),
                     ('TASK_ID = "P3.2"', 'TASK_ID = "P3.7"'),
                     ('DRIVE_ROOT = Path("/content/drive/MyDrive/SeqLogAD/P3.2_Memory")',
                      'DRIVE_ROOT = Path("/content/drive/MyDrive/SeqLogAD/P3.7_Gate")'),
                     ("# the P3.2 code ZIP and the burn-in view ZIP", "# the P3.7 code ZIP and the synthetic view ZIP"),
                     ('PASS_TASKS = "P2.3,P2.2,P2.1"  # P2.4 runs on a local CPU',
                      'PASS_TASKS = "P2.3,P2.2,P2.1"  # P2.4 and the gate run on a local CPU'),
                     (f'"runtime" / "memory-{code_sha[:12]}"', f'"runtime" / "gate-{code_sha[:12]}"'),
                     ('RUNTIME_ROOT / "seqlogad_code_memory"', 'RUNTIME_ROOT / "seqlogad_code_gate"'),
                     ("VIEW_ROOT = str(RUNTIME_ROOT)  # holds data/processed/memory/ADAPT-MEM-001/<fold>/statistics",
                      "VIEW_ROOT = str(RUNTIME_ROOT)  # holds synthetic/FUS-GATE-001/<fold>/{mutations,rows}.parquet"),
                     ('PASS_ROOT = DRIVE_ROOT / "buffer_pass"', 'PASS_ROOT = DRIVE_ROOT / "synthetic_scores"'),
                     ('WORK_ROOT = STATE_ROOT / "p3.2-work"', 'WORK_ROOT = STATE_ROOT / "p3.7-work"')):
        config = swap(config, old, new)
    setup = text[4].replace("seqlogad_code_memory", "seqlogad_code_gate")
    setup = swap(setup, '    assert (partial / "data/processed/memory/ADAPT-MEM-001").is_dir()\n',
                 '    assert (partial / "synthetic/FUS-GATE-001").is_dir()\n')
    setup = setup.replace("The P3.2 pass of P2.1/P2.2 requires", "The P3.7 pass of P2.1/P2.2 requires")
    text[0], text[1], text[4] = INTRO, config, setup
    text[7] = "## Synthetic scoring pass (resumable; progress persists on Drive)"
    text[8] = RUN
    text[10] = FINAL
    for cell, value in zip(nb["cells"], text):
        cell["source"] = value.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    joined = "".join(text)
    assert "memory.prototypes" not in joined and "buffer_pass" not in joined and "ADAPT-MEM-001/" not in joined
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(OUT.relative_to(REPO))


if __name__ == "__main__":
    main(*sys.argv[1:3])
