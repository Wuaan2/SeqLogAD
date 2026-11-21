# P2.4_GTAT — GTAT structural/temporal expert GTAT-ST-001, seed 42

Protocol EXC-007 (2026-11-20): only HDFS and BGL are target folds. Hadoop and OpenStack are source-only. Each fold has three sources.

P2.4 scores the same windows as P2.2 (one target record each), but from a causal graph instead of text. The graph has up to 64 previous records of the same group (HDFS block, Hadoop application, OpenStack instance, BGL global stream), plus component, BGL location and OpenStack request nodes. Edges carry quantised time gaps. The target is masked, and the model predicts three things: its arrival-time bucket, which earlier record it repeats (or "new"), and its event class. The score is the sum of the three negative log-likelihoods; each part is kept as evidence for fusion.

The model is a two-stream graph topology attention network (core idea from GTAT, Shen et al., Scientific Reports 2025). It has 3 layers, hidden size 128 and 4,047,335 parameters, is randomly initialised and trains in FP32. Checkpoints are selected by macro F1 on labeled source validation windows (P3.1 amendment), exactly as in P2.2 and P2.3. Target labels are never read. The frozen design is in `docs/protocol/P2.4-gtat-structural-temporal-expert.md` and `configs/models/gtat-v1.yaml`.

| Fold | Target | Sources | Labeled source for checkpoint selection |
|---|---|---|---|
| FOLD-TARGET-ARCH-BGL | BGL | HDFS, Hadoop, OpenStack | HDFS (block level) |
| FOLD-TARGET-ARCH-HDFS | HDFS | BGL, Hadoop, OpenStack | BGL (log line) |

## Files

| File | Role | SHA-256 |
|---|---|---|
| `P2.4_GTAT.ipynb` | Colab notebook | — |
| `uploads/seqlogad-code-s42-p2.4-hdfs-bgl-v1.zip` | Code, built from `../Ma_nguon/SeqLogAD` commit `997a7fc` | `cb2e0741dc6d1b701b113f7e1beb698ecf8dd02e25ccec1d603973c920210758` |
| `uploads/phase2-graph-s42.zip` | Graph view (22.8 MB), built locally from the P2.1/P2.2 bundles and raw logs | `595b5dc7ec1b96abc80b4722b5f39dcf1c5653ec25aefebcce7ee74dbf36f9ac` |
| ↳ `phase2_graph/manifests/graph-bundle.json` | Graph view manifest | `8918172b4376fd822ec812e9b46b2725f83b517a43c94f3efe16f55623c9d059` |
| `../P2.2_Sequence/uploads/phase2-sequence-s42.zip` | Sequence data bundle, shared with P2.2 | `4ed76b92e4b001138b28b4b3b4d3017d6ff7f30dd31e8cca0649bfdf34ede781` |
| `../P2.2_Sequence/uploads/phase2-source-development-v13r1.zip` | Source-validation label sidecar, shared with P2.2 | `c765eb88a4feebc8ceb157110e5a6bd28dec942a7fb9343d7f6b969b3e79b295` |

The graph view manifest binds the semantic and sequence bundles, the fold memberships, every raw file, the normalizer and the builder code. The recorded `prepare.py` and `contracts.py` hashes equal the files in commit `997a7fc`. Every node's message was checked byte for byte against the bundled normalized message.

Rebuild the code ZIP (reproducible, same hash):

```powershell
python ..\Ma_nguon\SeqLogAD\scripts\package_colab_s42.py out.zip
```

Rebuild the graph view (about 30 minutes on a local CPU; the example in `scripts/build_graph_view.py` shows the arguments).

## Running on Colab

1. Upload both files in `uploads/` to `MyDrive/SeqLogAD/P2.4_GTAT/uploads/`. The notebook reads the sequence data ZIP and the label sidecar from `MyDrive/SeqLogAD/P2.2_Sequence/uploads/`, which already holds them.
2. No Hugging Face token is needed.
3. Pick any GPU with at least 12 GiB (T4, L4, A100, H100 or G4). BF16 is not required.
4. Open `P2.4_GTAT.ipynb` and choose **Run all**. BGL trains first, then HDFS. Recovery checkpoints are small (about 50 MB) and are copied to Drive every 250 steps. If Colab disconnects, run the notebook again to resume from the last checkpoint on Drive.
5. Wait for `SAFE_TO_DISCONNECT_COLAB=YES`.

Each fold has at most 6,000 steps of 96 graphs, with early stopping after 6 validations without improvement. Measured on an L4 (2026-11-22): the BGL fold stopped at step 3,500 (best checkpoint step 2,000) in 836 s, and the HDFS fold stopped at step 2,000 (best step 500) in 549 s. Peak VRAM was 0.8–1.1 GB.

Result: `seqlogad_outputs_P2.4_S42_FINAL.zip`, SHA-256 `ff82c65343229183fd6caa7c0518961e58301ee5c7f71eb2d5fdce5eabbcfde3`.

## Output

`MyDrive/SeqLogAD/P2.4_GTAT/P2.4/seqlogad_outputs_P2.4_S42_FINAL.zip` (with a `.sha256` file):

```
P2.4_S42/SUMMARY.json
P2.4_S42/checksums.sha256
P2.4_S42/targets/P2.4_BGL_S42.zip   (+ .sha256)
P2.4_S42/targets/P2.4_HDFS_S42.zip  (+ .sha256)
```

Each target ZIP holds one run directory `P2.4-DEVELOPMENT-<TARGET>-S42/`. It contains:

- `metrics.json`:
  - `source_development_macro_f1` (compare with P2.2 and P2.3);
  - `final_source_validation`, with per-source mean scores, the three score components and abstain counts;
  - `resources`.
- `validation-scores.json`, with per-window scores and components.
- `sanity-evidence.json`, one example in the common evidence format.

The selected checkpoint stores the whole model at `checkpoints/<step>/adapter/adapter_model.safetensors`. The file name is kept from P2.2 so that checkpoint verification and Drive recovery are shared.
