# P2.6_Diversity — expert diversity evaluation DIV-P2-001, seed 42

P2.6 asks whether each expert adds anomaly evidence beyond the best single expert, on **source** data only. The design and every decision rule were frozen before scoring: `docs/protocol/P2.6-expert-diversity-preregistration.md` and `configs/protocols/p2.6-diversity-v1.yaml`. The label scope comes from the C2 amendment of 2026-11-22: HDFS (block) and BGL (log line) source-validation labels, on windows **not** used for checkpoint selection. Target labels are never read.

The work has three steps:

1. **Scoring (label-free).** Each expert's selected checkpoint scores the evaluation windows and writes one EVID-CS-001 envelope per line. P2.4 was scored locally on CPU. P2.1, P2.2 and P2.3 need a GPU, so they run in `P2.6_Scoring.ipynb` on Colab.
2. **Comparators (label-free, CPU).** Four cheap baselines: rarity, count vector, isolation forest and n-gram Markov.
3. **Analysis (CPU, uses labels).** AP, AUROC, FAR, double-fault, equal-weight combinations, bootstrap intervals and the recommended G2 verdict. The researcher signs G2.

| Fold | Evaluation windows | Labeled source |
|---|---|---|
| FOLD-TARGET-ARCH-BGL | 4,598 (HDFS 1,526 windows in 1,509 blocks; Hadoop 1,536; OpenStack 1,536) | HDFS blocks |
| FOLD-TARGET-ARCH-HDFS | 4,608 (BGL 1,536; Hadoop 1,536; OpenStack 1,536) | BGL lines |

Before scoring, every expert must reproduce the score in its own `sanity-evidence.json`. Later commits refactored shared helpers, so this behavioural check replaces the training-time code digest. For P2.4 the difference was 5.8e-7 (BGL fold) and 1.6e-9 (HDFS fold).

## Files

| File | Role | SHA-256 |
|---|---|---|
| `P2.6_Scoring.ipynb` | Colab scoring notebook for P2.1–P2.3 | — |
| `uploads/seqlogad-code-s42-p2.6-scoring-v2.zip` | Code (commit `f286358` of the earlier repository; v2 fixes the P2.1 checkpoint loader) | `9a429cc53096ab6c11446c7c3743b71c6b2eb5c62a89e82d81801748c98ab172` |

## Running the scoring notebook on Colab

1. Upload `uploads/seqlogad-code-s42-p2.6-scoring-v2.zip` to `MyDrive/SeqLogAD/P2.6_Diversity/uploads/`. The notebook reads everything else from where it already is on Drive:
   - the sequence data ZIP from `P2.2_Sequence/uploads/`;
   - the semantic data ZIP from `P2.1_Semantic/uploads/`;
   - the three FINAL ZIPs from `P2.1_Semantic/P2.1/`, `P2.2_Sequence/P2.2/` and `P2.3_Lightweight/P2.3/`.
2. Keep `HF_TOKEN` in Colab Secrets. The Llama-3.1 base model and tokenizer are needed.
3. Pick a GPU with native BF16 and at least 20 GiB (L4, A100, H100 or G4).
4. **Run all.** Order: P2.3, then P2.2, then P2.1, each for the BGL fold then the HDFS fold. Progress is saved to Drive every 64 windows. After a disconnect, run all again; finished parts are skipped.
5. Wait for `SAFE_TO_DISCONNECT_COLAB=YES`. Download `MyDrive/SeqLogAD/P2.6_Diversity/P2.6_evidence_S42.zip` and its `.sha256` into this folder.

About 9,200 windows per expert. Expected time on an L4: roughly 20–30 minutes for P2.3 and about 1 hour each for P2.2 and P2.1. This has not been measured yet.
