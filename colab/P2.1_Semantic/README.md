# P2.1_Semantic — Semantic LoRA (Llama-3.1-8B), seed 42

Protocol EXC-007 (2026-11-20): only HDFS and BGL are target folds. Hadoop and OpenStack are source-only. Each fold has three sources.

| Fold | Target | Sources |
|---|---|---|
| FOLD-TARGET-ARCH-BGL | BGL | HDFS, Hadoop, OpenStack |
| FOLD-TARGET-ARCH-HDFS | HDFS | BGL, Hadoop, OpenStack |

The semantic expert reads one log line at a time. Llama-3.1-8B with a LoRA-Semantic adapter (QLoRA NF4) learns to reconstruct a corrupted normalized message; the anomaly score is the mean reconstruction token NLL. Training and checkpoint selection are label-free.

## Files

| File | Role | SHA-256 |
|---|---|---|
| `P2.1_Semantic.ipynb` | Colab notebook | — |
| `uploads/seqlogad-code-s42-hdfs-bgl-v3.zip` | Code (built from commit `d713713` of the earlier repository) | `feef719adcde698b507a9b0ffe46e97ca178371cc8c3d215dfd16f26b758edba` |
| `uploads/phase2-final.zip` | Semantic-view data bundle from Phase 1 | `ebcfdfd6eb7b1607eb6d57e60fae88456d03bcb8ae2c6688ea398042c856224b` |

The git history of the earlier repository is no longer available, so verify every ZIP by its SHA-256; the notebook refuses any other file.

## Running on Colab

1. Upload the two files in `uploads/` to `MyDrive/SeqLogAD/P2.1_Semantic/uploads/`.
2. Add `HF_TOKEN` to Colab Secrets. Never paste the token into a cell.
3. Open `P2.1_Semantic.ipynb`, pick a GPU and choose **Run all**. BGL trains first, then HDFS. If Colab disconnects, run the notebook again to resume from the checkpoint on Drive.
4. Wait for `SAFE_TO_DISCONNECT_COLAB=YES`.

## Result (seed 42, NVIDIA L4)

| Fold | Last step | Source-validation reconstruction NLL | Runtime | Peak VRAM allocated |
|---|---:|---:|---:|---:|
| BGL | 1,100 | 0.0290 | 5,410 s | 9.5 GB |
| HDFS | 1,200 | 0.3324 | 3,917 s (resumed; last session only) | 9.9 GB |

## Output

`MyDrive/SeqLogAD/P2.1_Semantic/P2.1/seqlogad_outputs_P2.1_S42_FINAL.zip` (with a `.sha256` file), SHA-256 `8308a1b28875c1012b8a1942ca90898a4ca39e6c68f52e46bdd3e9601c7e603f`:

```
P2.1_S42/SUMMARY.json
P2.1_S42/checksums.sha256
P2.1_S42/targets/P2.1_BGL_S42.zip   (+ .sha256)  LoRA-Semantic run of the BGL fold, with the selected checkpoint
P2.1_S42/targets/P2.1_HDFS_S42.zip  (+ .sha256)  LoRA-Semantic run of the HDFS fold
```

The `recovery/` and `targets/` folders on Drive are only needed while training; the FINAL ZIP already contains the targets.
