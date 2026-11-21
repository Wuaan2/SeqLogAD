# P2.2_Sequence — Sequence LoRA V13R1 (Llama-3.1-8B), seed 42

Protocol EXC-007 (2026-11-20): only HDFS and BGL are target folds. Hadoop and OpenStack are source-only. Each fold has three sources.

| Fold | Target | Sources | Labeled source for checkpoint selection |
|---|---|---|---|
| FOLD-TARGET-ARCH-BGL | BGL | HDFS, Hadoop, OpenStack | HDFS (block level) |
| FOLD-TARGET-ARCH-HDFS | HDFS | BGL, Hadoop, OpenStack | BGL (log line) |

Training is label-free next-log NLL on source windows. Following the P3.1 amendment (2026-09-27), checkpoints are selected and early-stopped by macro F1 on labeled **source validation** windows; target labels are never read. Reports must name this as "label-free source training with labeled source development".

## Files

| File | Role | SHA-256 |
|---|---|---|
| `P2.2_Sequence.ipynb` | Colab notebook | — |
| `uploads/seqlogad-code-s42-p2.2-hdfs-bgl-v2.zip` | Code (built from commit `d713713` of the earlier repository) | `feef719adcde698b507a9b0ffe46e97ca178371cc8c3d215dfd16f26b758edba` |
| `uploads/phase2-sequence-s42.zip` | Sequence-view data bundle (unchanged from V13R1; `bundle.json` `1268b669…`) | `4ed76b92e4b001138b28b4b3b4d3017d6ff7f30dd31e8cca0649bfdf34ede781` |
| `uploads/phase2-source-development-v13r1.zip` | Source-validation label sidecar (unchanged; `manifest.json` `885b5382…`) | `c765eb88a4feebc8ceb157110e5a6bd28dec942a7fb9343d7f6b969b3e79b295` |

The data bundle and label sidecar still contain the old Hadoop fold; the code only reads the BGL and HDFS folds.

The git history of the earlier repository is no longer available, so verify every ZIP by its SHA-256; the notebook refuses any other file.

## Running on Colab

1. Upload the three files in `uploads/` to `MyDrive/SeqLogAD/P2.2_Sequence/uploads/`.
2. Add `HF_TOKEN` to Colab Secrets. Never paste the token into a cell.
3. Pick a GPU with native BF16 and at least 20 GiB VRAM (L4, A100, H100 or G4; not T4). The 22-10 run peaked at about 12 GB VRAM and took about 1 h per fold on a G4.
4. Open `P2.2_Sequence.ipynb` and choose **Run all**. BGL trains first, then HDFS.
   If Colab disconnects, run the notebook again to resume from the last checkpoint on Drive.
5. Wait for `SAFE_TO_DISCONNECT_COLAB=YES`.

Do not run P2.1 and P2.2 in the same Colab runtime at the same time.

## Result (seed 42, NVIDIA L4)

| Fold | Best / last step | Source-development macro F1 | AUROC | AP | Runtime | Peak VRAM allocated |
|---|---|---:|---:|---:|---:|---:|
| BGL (HDFS blocks) | 500 / 800 | 0.149 | 0.759 | 0.060 | 15,580 s | 10.2 GB |
| HDFS (BGL lines) | 100 / 400 | 0.208 | 0.816 | 0.099 | 3,979 s (resumed; last session only) | 10.2 GB |

Both folds stopped early (patience 3), far below the 1,536-step budget.

## Sequence output

`MyDrive/SeqLogAD/P2.2_Sequence/P2.2/seqlogad_outputs_P2.2_S42_FINAL.zip` (with a `.sha256` file), SHA-256 `6c009f2cd91c37899e340d877e4902197f9f5f96b4e164dbf86cb749583265c6`:

```
P2.2_S42/SUMMARY.json
P2.2_S42/checksums.sha256
P2.2_S42/targets/P2.2_BGL_S42.zip   (+ .sha256)  LoRA-Sequence adapter run for the BGL fold, selected checkpoint
P2.2_S42/targets/P2.2_HDFS_S42.zip  (+ .sha256)  LoRA-Sequence adapter run for the HDFS fold
```

Each target ZIP holds one run directory `P2.2-DEVELOPMENT-<TARGET>-S42-V13R1/` with `metrics.json`, `logs/train.jsonl`, `manifest.json`, `config.yaml`, the selected checkpoint and diagnostics. The last notebook cell checks that both targets are present.

## Changes from the V13R1 RunPod/Colab kit (22-10)

- Target folds: BGL and HDFS only (the old kit also trained Hadoop).
- Projmem/Aletheia evidence collection (`collect_v13r1.py`, `collection_archive_v13r1.py`, Projmem wheel) is removed; it served a separate data-collection job and does not affect training.
- The A40-only GPU guard is relaxed to native BF16 with at least 20 GiB VRAM.
- Drive persistence never renames files: ZIPs are written in place, re-verified, and a verified `.previous` copy is restored on resume if the latest one is broken (fix from the P2.1 run).
- Training configuration, data, labels, selection rule and seed are unchanged apart from the target list.
