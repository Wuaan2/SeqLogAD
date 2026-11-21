# P2.3_Lightweight — Lightweight Sequence expert LIGHT-SEQ-001, seed 42

Protocol EXC-007 (2026-11-20): only HDFS and BGL are target folds. Hadoop and OpenStack are source-only. Each fold has three sources.

P2.3 asks whether a much cheaper sequence model keeps the next-log anomaly signal of P2.2 (Llama-3.1-8B + LoRA-Sequence V13R1). The model is a causal Transformer with Llama decoder blocks: 4 layers, hidden 256, 4 heads, FFN 1024, tied embeddings, randomly initialised. It has 37,030,144 parameters, 0.46% of the 8B reference.

Everything else is inherited unchanged from P2.2 V13R1: the Llama-3.1 tokenizer, the prompt, up to 8 predecessor records in 512 tokens, the training windows, the multiscale context views, the `CAUSAL_NEXT_LOG_TARGET_TOKEN_NLL` objective and checkpoint selection by macro F1 on labeled source validation windows (P3.1 amendment). Only the optimizer budget differs: lr 5e-4, warmup 200, up to 3072 steps. The frozen design and resource criterion are in `docs/protocol/P2.3-lightweight-sequence-expert.md`.

| Fold | Target | Sources | Labeled source for checkpoint selection |
|---|---|---|---|
| FOLD-TARGET-ARCH-BGL | BGL | HDFS, Hadoop, OpenStack | HDFS (block level) |
| FOLD-TARGET-ARCH-HDFS | HDFS | BGL, Hadoop, OpenStack | BGL (log line) |

## Files

| File | Role | SHA-256 |
|---|---|---|
| `P2.3_Lightweight.ipynb` | Colab notebook | — |
| `seqlogad-code-s42-p2.3-hdfs-bgl-v1.zip` | Code, built from `../Ma_nguon/SeqLogAD` commit `9ad5285` | `4aa643397864922455295e701dd484129ff08083e7731689aaf193024e4eb515` |
| `../P2.2_Sequence/phase2-sequence-s42.zip` | Sequence data bundle, shared with P2.2 | `4ed76b92e4b001138b28b4b3b4d3017d6ff7f30dd31e8cca0649bfdf34ede781` |
| `../P2.2_Sequence/phase2-source-development-v13r1.zip` | Source-validation label sidecar, shared with P2.2 | `c765eb88a4feebc8ceb157110e5a6bd28dec942a7fb9343d7f6b969b3e79b295` |

Rebuild the code ZIP (reproducible, same hash):

```powershell
python ..\Ma_nguon\SeqLogAD\scripts\package_colab_s42.py out.zip
```

## Running on Colab

1. Upload `seqlogad-code-s42-p2.3-hdfs-bgl-v1.zip` to `MyDrive/SeqLogAD/P2.3_Lightweight/uploads/`. The notebook reads the two data ZIPs from `MyDrive/SeqLogAD/P2.2_Sequence/uploads/`, which already holds them.
2. Keep `HF_TOKEN` in Colab Secrets. It is needed only for the gated Llama-3.1 tokenizer; the 8B weights are not downloaded.
3. Pick a GPU with native BF16 (L4, A100, H100 or G4; not T4).
4. Open `P2.3_Lightweight.ipynb` and choose **Run all**. BGL trains first, then HDFS. If Colab disconnects, run the notebook again to resume from the last checkpoint on Drive.
   A P2.3 recovery backup holds the full weights and optimizer state (about 0.9 GB), so it is copied to Drive at most every 30 minutes instead of every 100 steps. A disconnect loses at most about 30 minutes of training. A finished fold is always kept.
5. Wait for `SAFE_TO_DISCONNECT_COLAB=YES`.

Run time on a GPU has not been measured yet.

## Output

`MyDrive/SeqLogAD/P2.3_Lightweight/P2.3/seqlogad_outputs_P2.3_S42_FINAL.zip` (with a `.sha256` file):

```
P2.3_S42/SUMMARY.json
P2.3_S42/checksums.sha256
P2.3_S42/targets/P2.3_BGL_S42.zip   (+ .sha256)
P2.3_S42/targets/P2.3_HDFS_S42.zip  (+ .sha256)
```

Each target ZIP holds one run directory `P2.3-DEVELOPMENT-<TARGET>-S42/`. Its `metrics.json` has `source_development_macro_f1` (compare with P2.2) and a `resources` block: total parameters, parameter fraction of the reference, `parameter_criterion_met`, and validation seconds per window. Runtime and peak VRAM are top-level fields, as in P2.2. The selected checkpoint stores the whole small model at `checkpoints/<step>/adapter/adapter_model.safetensors`; the file name is kept from P2.2 so checkpoint verification and Drive recovery are shared.
