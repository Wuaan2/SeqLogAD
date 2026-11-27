# P3.2 — Target normal memory: buffer pass (ADAPT-MEM-001)

`P3.2_Prototypes.ipynb` runs the frozen seed-42 experts P2.1, P2.2 and P2.3 on 8,192 burn-in windows per target fold
and stores, per window, the raw score (EVID-CS-001) and the embedding. P2.4 GTAT runs on a local CPU with the same code
(`python -m seqlogad.memory.prototypes pass --tasks P2.4 --device cpu …`). Nothing is trained, no label is read and no
score is calibrated. Spec: [`configs/protocols/adapt-mem-prototypes-v1.yaml`](../../configs/protocols/adapt-mem-prototypes-v1.yaml);
design: [`docs/protocol/P3.2-target-memory.md`](../../docs/protocol/P3.2-target-memory.md).

## Uploads to `MyDrive/SeqLogAD/P3.2_Memory/uploads/`

| File | Where it comes from |
|---|---|
| `seqlogad-code-memory-v1.zip` | `uploads/` in this folder (`scripts/package_colab_memory.py`) |
| `p3.2-buffer-view-v1.zip` | the statistics component of both folds, built locally (not in git: generated data) |

Also needed on Drive, as for the P2.6 pass: the seed-42 FINAL ZIPs in `P2.1_Semantic/P2.1/`, `P2.2_Sequence/P2.2/`,
`P2.3_Lightweight/P2.3/`, and the data ZIPs in `P2.1_Semantic/uploads/`, `P2.2_Sequence/uploads/`, `P2.4_GTAT/uploads/`.

## Run

L4 GPU (native BF16, at least 20 GiB), `HF_TOKEN` in Colab Secrets, **Run all**; about 3.5 hours. Progress is saved
on Drive every 256 windows; rerun the notebook after a disconnect. Download `P3.2_buffer_pass.zip` and its `.sha256`
after `SAFE_TO_DISCONNECT_COLAB=YES`.

## After the pass (local)

Unpack the ZIP next to the local GTAT outputs so that `buffer_pass/<fold>/` holds all four experts, then:

```
python -m seqlogad.memory.prototypes build --root <folder holding data/processed/memory> --repo-root . --pass-root <buffer_pass>
```

This fits the k-means prototypes, writes the `prototypes` and `buffer_scores` components and freezes each fold's
memory with `memory.json`.
