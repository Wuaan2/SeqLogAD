"""Write the SEED-CONFIRM-001 Colab notebooks (one set per confirmation seed) from the seed-42 notebooks.

The seed-42 notebooks are read, never modified. Each transformation asserts that it applied, so a change in
a template fails loudly instead of producing a half-converted notebook.
Usage: python scripts/build_seed_notebooks.py <seed code ZIP sha256>
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS = (3407, 8675309)
CODE_ZIP = "seqlogad-code-seeds-v1.zip"
CODE_ROOT = "seqlogad_code_seed_confirmation"
DRIVE = "/content/drive/MyDrive/SeqLogAD"
TEMPLATES = {
    "P2.1": ("colab/P2.1_Semantic/P2.1_Semantic.ipynb", "Semantic expert", ("P2.1_Semantic",)),
    "P2.2": ("colab/P2.2_Sequence/P2.2_Sequence.ipynb", "Sequence reference expert (V13R1)", ("P2.2_Sequence",)),
    "P2.3": ("colab/P2.3_Lightweight/P2.3_Lightweight.ipynb", "Lightweight Sequence expert", ("P2.2_Sequence",)),
    "P2.4": ("colab/P2.4_GTAT/P2.4_GTAT.ipynb", "GTAT structural/temporal expert", ("P2.4_GTAT", "P2.2_Sequence")),
    "P2.6": ("colab/P2.6_Diversity/P2.6_Scoring.ipynb", "P2.6 scoring pass",
             ("P2.2_Sequence", "P2.1_Semantic", "P2.4_GTAT")),
}
EXPERTS = {"P2.1": "P2.1_Semantic", "P2.2": "P2.2_Sequence", "P2.3": "P2.3_Lightweight", "P2.4": "P2.4_GTAT"}
GRAPH_ZIP = "595b5dc7ec1b96abc80b4722b5f39dcf1c5653ec25aefebcce7ee74dbf36f9ac"
GRAPH_MANIFEST = "8918172b4376fd822ec812e9b46b2725f83b517a43c94f3efe16f55623c9d059"
MULTI_ROOT_LOCATE = '''def locate(expected):
    # First match wins, so a ZIP copied into two upload folders is not ambiguous.
    for root in UPLOAD_ROOTS:
        for path in (sorted(root.glob("*.zip")) if root.is_dir() else []):
            if digest(path) == expected:
                return path
    raise AssertionError(f"Put the ZIP with SHA-256 {expected} in one of {[str(r) for r in UPLOAD_ROOTS]}")
'''


def sub(text, pattern, replacement, flags=re.M):
    result, count = re.subn(pattern, replacement, text, flags=flags)
    assert count >= 1, pattern
    return result


def put(text, anchor, addition):
    assert text.count(anchor) == 1, anchor
    return text.replace(anchor, anchor + addition)


def intro(task, seed, name, uploads, original):
    gpu = next(line for line in original.splitlines() if line.startswith("1. "))
    token = ("3. Keep the Hugging Face token in **Colab Secrets** as `HF_TOKEN` with notebook access. "
             "Never paste the token into a cell.\n" if task != "P2.4" else "3. No Hugging Face token is needed.\n")
    where = f"Seeds/S{seed}/P2.6/" if task == "P2.6" else f"Seeds/S{seed}/{task}/"
    what = ("Scores the held-out DIV-P2-001 windows with the four seed-"
            f"{seed} experts (same windows, envelopes and sanity-reproduction guard as seed 42). "
            f"Run it after the four training notebooks of seed {seed} have finished.\n"
            if task == "P2.6" else
            f"Retrains the frozen {task} expert with training seed **{seed}** for both target folds (BGL, HDFS). "
            "Only the training seed changes: data, config, step budget and checkpoint-selection rule are those of "
            "seed 42, and for P2.2–P2.4 the selection windows are also those of seed 42.\n")
    folders = ", ".join(f"`MyDrive/SeqLogAD/{folder}/uploads/`" for folder in uploads)
    return (f"# {task} — {name} — confirmation seed {seed}\n\n{what}\n"
            "**Seed 42 stays the primary seed.** This notebook never writes to the seed-42 folders and changes "
            "nothing in P2-FREEZE-001 or Phase 3. Protocol: `docs/protocol/SEED-CONFIRM-001-seed-confirmation.md` "
            "(also inside the code ZIP).\n\n## Setup\n\n"
            f"{gpu}\n"
            f"2. Upload `{CODE_ZIP}` once to `MyDrive/SeqLogAD/Seeds/uploads/` (both seeds share it). "
            f"The data ZIPs are read from the seed-42 upload folders ({folders}); no second upload is needed.\n"
            f"{token}4. Choose **Runtime → Run all**.\n\n"
            f"Outputs go to `MyDrive/SeqLogAD/{where}`. If Colab disconnects, run the notebook again: finished "
            "parts are skipped and an unfinished fold resumes from its last recovery checkpoint. Disconnect only "
            "after `SAFE_TO_DISCONNECT_COLAB=YES`.\n")


def f_string_s42(text):
    def fix(match):
        return 'f"' + match.group(2).replace("S42", "S{SEED}") + '"'
    return re.sub(r'(f?)"([^"\n]*S42[^"\n]*)"', fix, text)


def convert(task, seed, code_sha):
    path, name, uploads = TEMPLATES[task]
    nb = json.loads((REPO / path).read_text(encoding="utf-8"))
    cells = nb["cells"]
    text = ["".join(cell["source"]) for cell in cells]
    original_intro = text[0]
    text[0] = intro(task, seed, name, uploads, original_intro)

    config = text[1]
    config = sub(config, r"^SEED = 42$", f"SEED = {seed}  # SEED-CONFIRM-001 confirmation seed; seed 42 stays primary")
    drive_root = f"{DRIVE}/Seeds/S{seed}" + ("/P2.6" if task == "P2.6" else "")
    config = sub(config, r"^DRIVE_ROOT = .*$", f'DRIVE_ROOT = Path("{drive_root}")  # this seed\'s own folder')
    config = re.sub(r"^# The .*ZIPs.*\n", "", config, flags=re.M)
    config = re.sub(r"^UPLOAD_ROOTS = \[.*?\]\n", "", config, flags=re.M | re.S)
    roots = ", ".join(f'"{folder}"' for folder in uploads)
    config = sub(config, r'^UPLOAD_ROOT = DRIVE_ROOT / "uploads"$',
                 f'UPLOAD_ROOT = Path("{DRIVE}/Seeds/uploads")  # the seed code ZIP, shared by both seeds\n'
                 f'# Data ZIPs come from the seed-42 upload folders; nothing is written there.\n'
                 f'UPLOAD_ROOTS = [UPLOAD_ROOT] + [Path("{DRIVE}") / folder / "uploads" for folder in ({roots},)]')
    config = sub(config, r"^RUNTIME_ROOT = .*$", f'RUNTIME_ROOT = STATE_ROOT / "runtime" / "seeds-{code_sha[:12]}-{task}"')
    config = sub(config, r'^EXPECTED_CODE_ZIP_SHA256 = ".*"$', f'EXPECTED_CODE_ZIP_SHA256 = "{code_sha}"')
    config = sub(config, r"SEED == 42", "SEED in (3407, 8675309)")
    if task == "P2.6":
        finals = "".join(f'    "{t}": SEQLOGAD / "Seeds/S{seed}/{t}/seqlogad_outputs_{t}_S{seed}_FINAL.zip",\n'
                         for t in EXPERTS)
        config = sub(config, r"^FINAL_ZIPS = \{\n.*?^\}$", "FINAL_ZIPS = {\n" + finals + "}", flags=re.M | re.S)
        config = sub(config, r"^SCORING_TASKS = .*$", 'SCORING_TASKS = "P2.4,P2.3,P2.2,P2.1"')
        # Extracted runs are cached per seed, so scoring both seeds in one runtime cannot mix them.
        config = sub(config, r'^WORK_ROOT = STATE_ROOT / "p2\.6-work"$', f'WORK_ROOT = STATE_ROOT / "p2.6-work-S{seed}"')
        config = put(config, 'SEMANTIC_ROOT = str(RUNTIME_ROOT / "phase2")\n',
                     'GRAPH_ROOT = str(RUNTIME_ROOT / "phase2_graph")\n')
        config = put(config, "EXPECTED_BUNDLE_SHA256 = \"1268b669a8b1b474658cfa7daa6556c987b0f31f07e0101ed30af250a8d7f129\"\n",
                     f'EXPECTED_GRAPH_ZIP_SHA256 = "{GRAPH_ZIP}"\nEXPECTED_GRAPH_MANIFEST_SHA256 = "{GRAPH_MANIFEST}"\n')
    text[1] = config.replace("seqlogad_code_all_experts_s42", CODE_ROOT)

    setup = text[4].replace("seqlogad_code_all_experts_s42", CODE_ROOT)
    assert CODE_ROOT in setup
    if "UPLOAD_ROOTS" not in setup:
        setup = sub(setup, r"^def locate\(expected\):\n.*?return matches\[0\]\n", MULTI_ROOT_LOCATE, flags=re.M | re.S)
    if task == "P2.6":
        setup = put(setup, "semantic_zip = locate(EXPECTED_SEMANTIC_ZIP_SHA256)\n",
                    "graph_zip = locate(EXPECTED_GRAPH_ZIP_SHA256)\n")
        setup = put(setup, '    "semantic_sha256": EXPECTED_SEMANTIC_ZIP_SHA256,\n',
                    '    "graph_sha256": EXPECTED_GRAPH_ZIP_SHA256,\n')
        setup = put(setup, "    safe_extract(semantic_zip, partial)\n", "    safe_extract(graph_zip, partial)\n")
        setup = put(setup, '    assert (partial / "phase2/manifests/bundle.json").is_file()\n',
                    '    assert (partial / "phase2_graph/manifests/graph-bundle.json").is_file()\n')
        setup = put(setup, 'assert digest(Path(DATA_ROOT) / "manifests/bundle.json") == EXPECTED_BUNDLE_SHA256\n',
                    'assert digest(Path(GRAPH_ROOT) / "manifests/graph-bundle.json") == EXPECTED_GRAPH_MANIFEST_SHA256\n')
    text[4] = setup

    run_index = next(i for i, t in enumerate(text) if '"-m", "seqlogad.' in t and ("phase2_execution.cli" in t
                                                                              or "diversity.score" in t))
    run = text[run_index]
    if task == "P2.6":
        run = sub(run, r'"seqlogad\.diversity\.score",', '"seqlogad.seeds.score",\n    "--seed", str(SEED),')
        run = put(run, '    "--semantic-root", SEMANTIC_ROOT,\n', '    "--graph-root", GRAPH_ROOT,\n')
    else:
        run = sub(run, r'"seqlogad\.phase2_execution\.cli",', '"seqlogad.seeds.runner",')
        run = sub(run, r'^    "--task-id", TASK_ID,\n', '    "--task-id", TASK_ID,\n    "--seed", str(SEED),\n')
        run = run.replace('"--persistent-root"', '"--drive-root"')
        run = re.sub(r'^    "--sequence-candidate", SEQUENCE_CANDIDATE,\n', "", run, flags=re.M)
    text[run_index] = run

    final_index = len(text) - 1 if text[-1].strip() else len(text) - 2
    final = f_string_s42(text[final_index])
    if task == "P2.6":
        final = sub(final, r'"P2\.3": "SEQUENCE_LIGHTWEIGHT"\}',
                    '"P2.3": "SEQUENCE_LIGHTWEIGHT",\n           "P2.4": "GTAT_STRUCTURAL_TEMPORAL"}')
    assert "S42" not in final and "{SEED}" in final, task
    text[final_index] = final

    for cell, value in zip(cells, text):
        cell["source"] = value.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    assert not any("S42" in t and "seed-42" not in t.lower() for t in text[1:]), task
    out = REPO / "colab" / "Seeds" / f"S{seed}" / f"{task}_S{seed}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return out


def main(code_sha):
    assert re.fullmatch(r"[0-9a-f]{64}", code_sha)
    for seed in SEEDS:
        for task in TEMPLATES:
            print(convert(task, seed, code_sha).relative_to(REPO))


if __name__ == "__main__":
    main(sys.argv[1])
