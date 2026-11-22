# Confirmation seeds (SEED-CONFIRM-001)

Notebooks that retrain the four Phase-2 experts with training seeds **3407** and **8675309**, then score them
for P2.6. **Seed 42 stays the primary seed**: these runs only report how stable the Phase-2 findings are. They
never write to the seed-42 Drive folders and change nothing in P2-FREEZE-001 or Phase 3.
Protocol: [`docs/protocol/SEED-CONFIRM-001-seed-confirmation.md`](../../docs/protocol/SEED-CONFIRM-001-seed-confirmation.md).

| Folder | Notebooks |
|---|---|
| `S3407/` | `P2.1_S3407`, `P2.2_S3407`, `P2.3_S3407`, `P2.4_S3407`, `P2.6_S3407` |
| `S8675309/` | `P2.1_S8675309`, `P2.2_S8675309`, `P2.3_S8675309`, `P2.4_S8675309`, `P2.6_S8675309` |

P2.5 has no notebook: the evidence contract is unchanged and the P2.6 notebook writes the envelopes.

## Uploads

Upload `uploads/seqlogad-code-seeds-v1.zip` once to `MyDrive/SeqLogAD/Seeds/uploads/`. Both seeds share it. The
data ZIPs are read from the seed-42 upload folders that are already on Drive:

| Data | Read from |
|---|---|
| Semantic data `phase2-final.zip` | `MyDrive/SeqLogAD/P2.1_Semantic/uploads/` |
| Sequence data `phase2-sequence-s42.zip` and labels `phase2-source-development-v13r1.zip` | `MyDrive/SeqLogAD/P2.2_Sequence/uploads/` |
| Graph view `phase2-graph-s42.zip` | `MyDrive/SeqLogAD/P2.4_GTAT/uploads/` |

## Order and GPU

For each seed, run the four training notebooks in any order, then the P2.6 notebook of that seed. The two seeds
are independent and can run in parallel.

Before starting two notebooks of the same seed at the same time, create `MyDrive/SeqLogAD/Seeds/S<seed>/` in
the Drive web UI. Two Colab runtimes that create the same missing folder within seconds of each other each get
their own copy (Drive allows duplicate names), which splits the seed's outputs. If that happened, wait until the
runs in those folders print `SAFE_TO_DISCONNECT_COLAB=YES`, move the `P2.x` subfolder into one copy and delete
the empty copy.

| Notebook | GPU | Hugging Face token |
|---|---|---|
| P2.1, P2.2 | Native BF16, at least 20 GiB (L4, A100, H100, G4) | Yes |
| P2.3 | Native BF16 (L4, A100, H100, G4) | Yes (tokenizer only) |
| P2.4 | At least 12 GiB, FP32 (T4 works) | No |
| P2.6 | Native BF16, at least 20 GiB | Yes |

Keep the token in Colab Secrets as `HF_TOKEN`; never paste it into a cell. If Colab disconnects, run the notebook
again: finished folds are skipped and an unfinished fold resumes from its last recovery checkpoint.

## Outputs to download

Mirror `MyDrive/SeqLogAD/Seeds/` into the local `Seeds/` folder next to the `P2.x_*` folders:

| Per seed | File |
|---|---|
| P2.1–P2.4 | `Seeds/S<seed>/P2.x/seqlogad_outputs_P2.x_S<seed>_FINAL.zip` and its `.sha256` |
| P2.6 | `Seeds/S<seed>/P2.6/P2.6_evidence_S<seed>.zip` and its `.sha256` |

The `recovery/` and `targets/` folders do not need to be downloaded; the FINAL ZIP already contains the targets.

## How the code stays separate

The notebooks call the new package `seqlogad.seeds` (`runner`, `score`). It reuses the seed-42 modules
unchanged: it trains through the frozen trainers with only the training seed swapped, keeps the seed-42
checkpoint-selection windows, and names every run and file with the seed. The notebooks are generated from the
seed-42 notebooks by `scripts/build_seed_notebooks.py`.
