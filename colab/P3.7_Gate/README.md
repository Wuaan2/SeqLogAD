# P3.7 — Exploratory gate: synthetic scoring pass (FUS-GATE-001)

`P3.7_Synthetic.ipynb` runs the frozen seed-42 experts P2.1, P2.2 and P2.3 on the preregistered synthetic copies of
the checkpoint-selection windows of each target fold. P2.4 GTAT and the gate itself run on a local CPU. Nothing is
trained on the GPU and no label is read. The gate is EXPLORATORY: G2 stays PARTIAL. Spec:
[`configs/protocols/fus-gate-v1.yaml`](../../configs/protocols/fus-gate-v1.yaml); design:
[`docs/protocol/P3.7-exploratory-gate.md`](../../docs/protocol/P3.7-exploratory-gate.md).

## Uploads to `MyDrive/SeqLogAD/P3.7_Gate/uploads/`

| File | Where it comes from |
|---|---|
| `seqlogad-code-gate-v1.zip` | `scripts/package_colab_gate.py` |
| `p3.7-synthetic-view-v1.zip` | the synthetic view of both folds, built locally (not in git: generated data) |

The seed-42 FINAL ZIPs and the data ZIPs must be where the P3.2 pass read them.

## Run

On an L4 GPU (native BF16, at least 20 GiB), with `HF_TOKEN` in Colab Secrets, choose **Run all**. It takes about 2
hours. Progress is saved on Drive every 256 copies, so rerun the notebook after a disconnect.

When the log shows `SAFE_TO_DISCONNECT_COLAB=YES`, download `P3.7_synthetic_scores.zip` and its `.sha256`.
