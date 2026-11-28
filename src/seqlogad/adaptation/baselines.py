"""FUS-BASE-001 (P3.6): fusion baselines with no learned target weight.

Spec: configs/protocols/fus-base-v1.yaml. The singles and equal-weight fusion use the CAL-CS-001 calibrated scores
and the UNC-CS-001 states. The Transformer encoder and the Isolation Forest are fitted on the unlabeled target
burn-in buffer only (the frozen ADAPT-MEM-001 statistics windows). Each baseline has one preregistered
configuration, so none is tuned. No label is read here; the oracle strongest single is chosen in the final
evaluation, not here.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from seqlogad.adaptation import calibration
from seqlogad.adaptation.store import (ADAPT_ROOT, AdaptationError, load_artifact, load_frozen_memory, memory_table,
                                       write_artifact)
from seqlogad.adaptation.store import load_spec as _load_spec
from seqlogad.adaptation.uncertainty import REVIEW
from seqlogad.common.checksum import sha256_file
from seqlogad.diversity.comparators import _chains, _features

SPEC = "configs/protocols/fus-base-v1.yaml"
ARTIFACT = "FUS-BASE-001"
PAD, MASK, UNK = 0, 1, 2
LENGTH = 9  # 8 context records and the target


def load_spec(repo_root):
    spec = _load_spec(repo_root, SPEC, "fus_base")
    if spec["learned_target_weights"] is not False or spec["tuning"] != "none" \
            or spec["baselines"]["STRONGEST_SINGLE"]["rule"] != "ORACLE_UPPER_BOUND":
        raise AdaptationError("FUS-BASE-001 rule changed")
    return spec


# ---------------------------------------------------------------- singles and equal weight

def fusion_predictions(states, quorum):
    """SINGLE_<expert> and EQUAL_WEIGHT rows from per-expert rows (expert_id, target_rank, expert_state, reason,
    score_calibrated); an UNKNOWN_REVIEW expert or window gets no score."""
    scored = pl.col("expert_state") == "SCORED"
    singles = states.select(
        ("SINGLE_" + pl.col("expert_id")).alias("baseline_id"), "target_rank",
        pl.when(scored).then(pl.col("score_calibrated")).otherwise(None).alias("score_raw"),
        pl.when(scored).then(pl.lit("SCORED")).otherwise(pl.lit(REVIEW)).alias("status"), "reason")
    fused = (states.group_by("target_rank")
             .agg(scored.sum().alias("k"), pl.col("score_calibrated").filter(scored).mean().alias("mean"))
             .select(pl.lit("EQUAL_WEIGHT").alias("baseline_id"), "target_rank",
                     pl.when(pl.col("k") >= quorum).then(pl.col("mean")).otherwise(None).alias("score_raw"),
                     pl.when(pl.col("k") >= quorum).then(pl.lit("SCORED")).otherwise(pl.lit(REVIEW)).alias("status"),
                     pl.when(pl.col("k") >= quorum).then(pl.lit(None, pl.String)).otherwise(pl.lit("QUORUM")).alias("reason")))
    return pl.concat([singles, fused]).sort("baseline_id", "target_rank")


# ---------------------------------------------------------------- Transformer encoder

def vocabulary(messages):
    return {m: i + 3 for i, m in enumerate(sorted(set(messages)))}


def encode(chains, vocab):
    """Message ids of each window, left-padded to LENGTH; a message outside the vocabulary is UNK."""
    x = np.full((len(chains), LENGTH), PAD, dtype=np.int64)
    for i, chain in enumerate(chains):
        ids = [vocab.get(m, UNK) for m in chain][-LENGTH:]
        x[i, LENGTH - len(ids):] = ids
    return x


def _model(vocab_size, cfg):
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg["d_model"]
            self.token = torch.nn.Embedding(vocab_size, d, padding_idx=PAD)
            self.position = torch.nn.Embedding(LENGTH, d)
            layer = torch.nn.TransformerEncoderLayer(d, cfg["heads"], cfg["feedforward"], cfg["dropout"], batch_first=True)
            self.encoder = torch.nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
            self.out = torch.nn.Linear(d, vocab_size)

        def forward(self, x, position):
            h = self.token(x) + self.position(torch.arange(LENGTH, device=x.device))
            h = self.encoder(h, src_key_padding_mask=x == PAD)
            return self.out(h[torch.arange(x.shape[0]), position])

    return Encoder()


def masked_nll(model, x, position):
    """NLL of the record at `position` of each window, with that position masked."""
    import torch

    rows = torch.arange(x.shape[0])
    masked = x.clone()
    masked[rows, position] = MASK
    return torch.nn.functional.cross_entropy(model(masked, position), x[rows, position], reduction="none")


def fit_encoder(x, vocab_size, spec_encoder):
    """Train on the first 90% of the windows, stop early on the target NLL of the last 10%; returns the best state."""
    import torch

    cfg, train_cfg = spec_encoder["model"], spec_encoder["training"]
    torch.manual_seed(train_cfg["seed"])
    rng = np.random.default_rng(train_cfg["seed"])
    model = _model(vocab_size, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    split = int(len(x) * 0.9)
    train, valid = torch.as_tensor(x[:split]), torch.as_tensor(x[split:])
    filled = (train != PAD).sum(dim=1).numpy()
    best, wait, state, history = float("inf"), 0, None, []
    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        for batch in np.array_split(rng.permutation(split), max(1, split // train_cfg["batch"])):
            position = torch.as_tensor(LENGTH - 1 - rng.integers(0, filled[batch]))  # a non-pad position
            loss = masked_nll(model, train[batch], position).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = float(score_encoder(model, valid.numpy()).mean())
        history.append({"epoch": epoch, "validation_target_nll": loss})
        if loss < best:
            best, wait, state = loss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= train_cfg["patience"]:
                break
    model.load_state_dict(state)
    return model, history


def score_encoder(model, x, batch=4096):
    import torch

    model.eval()
    with torch.no_grad():
        return np.concatenate([masked_nll(model, torch.as_tensor(x[i:i + batch]),
                                          torch.full((len(x[i:i + batch]),), LENGTH - 1)).numpy()
                               for i in range(0, len(x), batch)])


# ---------------------------------------------------------------- Isolation Forest

def isolation_scores(train_chains, eval_chains, cfg):
    from sklearn.ensemble import IsolationForest

    forest = IsolationForest(n_estimators=cfg["n_estimators"], random_state=cfg["random_state"])
    forest.fit(_features(train_chains, cfg["buckets"]))
    return -forest.score_samples(_features(eval_chains, cfg["buckets"]))


# ---------------------------------------------------------------- build

def _buffer_states(root, fold_id, unc, cal):
    """UNC-CS-001 buffer states joined to the raw buffer scores, with the CAL-CS-001 calibrated score."""
    states = pl.read_parquet(Path(root) / ADAPT_ROOT / "UNC-CS-001" / fold_id / "buffer_states.parquet")
    scores = memory_table(root, fold_id, "buffer_scores", "scores").select("expert_id", "unit_id", "score_raw")
    states = states.join(scores, on=["expert_id", "unit_id"], how="left", validate="1:1")
    return pl.concat([
        part.with_columns(pl.Series("score_calibrated", calibration.calibrate(cal["experts"][e], part["score_raw"].to_list()), dtype=pl.Float64))
        for (e,), part in states.group_by("expert_id", maintain_order=True)])


def build_baselines(root, repo_root, folds=None):
    """Fit the encoder and the forest on the buffer and freeze them with the example buffer predictions, per fold."""
    import torch

    spec = load_spec(repo_root)
    base = spec["baselines"]
    report = {}
    for fold_id in folds or spec["folds"]:
        memory, _ = load_frozen_memory(root, fold_id)
        cal, unc = load_artifact(root, "CAL-CS-001", fold_id), load_artifact(root, "UNC-CS-001", fold_id)
        if cal["memory_sha256"] != memory["sha256"] or unc["memory_sha256"] != memory["sha256"]:
            raise AdaptationError(f"{fold_id}: CAL-CS-001 or UNC-CS-001 was built on another memory")
        architecture = memory["memory_key"].split("/")[1]
        windows = memory_table(root, fold_id, "statistics", "windows").sort("target_rank")
        events = memory_table(root, fold_id, "statistics", "events")
        messages = dict(events.select("rank", "normalized_message").iter_rows())
        chains = _chains(windows, messages)
        vocab = vocabulary(messages.values())
        x = encode(chains, vocab)

        start = time.perf_counter()
        model, history = fit_encoder(x, len(vocab) + 3, base["TRANSFORMER_ENCODER"])
        encoder_fit = time.perf_counter() - start
        start = time.perf_counter()
        forest_train = isolation_scores(chains, chains, base["ISOLATION_FOREST"]["config"])
        forest_fit = time.perf_counter() - start

        fused = fusion_predictions(_buffer_states(root, fold_id, unc, cal), unc["quorum"])
        fused = fused.filter(pl.col("target_rank").is_in(windows["target_rank"].implode()))  # one window set for all
        example = set(fused["target_rank"].to_list())
        keep = windows["target_rank"].is_in(list(example)).to_numpy()
        start = time.perf_counter()
        encoder_scores = score_encoder(model, x[keep])
        encoder_score = (time.perf_counter() - start) / max(1, int(keep.sum()))
        ranks = windows["target_rank"].filter(pl.Series(keep)).to_list()
        fitted = pl.concat([
            pl.DataFrame({"baseline_id": name, "target_rank": ranks, "score_raw": values.tolist(), "status": "SCORED",
                          "reason": None}, schema_overrides={"reason": pl.String, "score_raw": pl.Float64})
            for name, values in (("TRANSFORMER_ENCODER", encoder_scores), ("ISOLATION_FOREST", forest_train[keep]))])
        predictions = (pl.concat([fused, fitted])
                       .with_columns(pl.lit(fold_id).alias("fold_id"),
                                     (pl.lit(architecture + ":") + pl.col("target_rank").cast(pl.String)).alias("target_record_id"),
                                     pl.lit("HIGHER_IS_MORE_ANOMALOUS").alias("score_direction"),
                                     pl.col("baseline_id").is_in(["TRANSFORMER_ENCODER", "ISOLATION_FOREST"]).alias("in_sample"))
                       .select(*spec["predictions"]["schema"], "in_sample").sort("baseline_id", "target_rank"))
        if not predictions.filter(pl.col("status") == "SCORED")["score_raw"].is_finite().all():
            raise AdaptationError(f"{fold_id}: a non-finite baseline score")

        weights = io.BytesIO()
        torch.save(model.state_dict(), weights)
        budget = {
            "TRANSFORMER_ENCODER": {"trainable_parameters": sum(p.numel() for p in model.parameters()),
                                    "fit_seconds": encoder_fit, "score_seconds_per_window": encoder_score,
                                    "device": "cpu", "fitted_on": f"{len(x)} buffer statistics windows",
                                    "epochs": len(history), "vocabulary": len(vocab)},
            "ISOLATION_FOREST": {"trainable_parameters": None, "n_estimators": base["ISOLATION_FOREST"]["config"]["n_estimators"],
                                 "fit_seconds": forest_fit, "device": "cpu",
                                 "fitted_on": f"{len(chains)} buffer statistics windows"},
            "SINGLE_AND_EQUAL_WEIGHT": {"trainable_parameters": 0, "fitted_on": "nothing; frozen experts and CAL-CS-001"},
        }
        summary = (predictions.group_by("baseline_id", "status").len().sort("baseline_id", "status")
                   .to_dicts())
        manifest = {"artifact_id": ARTIFACT, "fold_id": fold_id, "memory_key": memory["memory_key"],
                    "memory_sha256": memory["sha256"],
                    "calibration_sha256": sha256_file(Path(root) / ADAPT_ROOT / "CAL-CS-001" / fold_id / "artifact.json"),
                    "uncertainty_sha256": sha256_file(Path(root) / ADAPT_ROOT / "UNC-CS-001" / fold_id / "artifact.json"),
                    "spec_sha256": spec["sha256"], "builder_sha256": sha256_file(Path(__file__)),
                    "torch_version": torch.__version__, "budget": budget, "predictions": summary,
                    "learned_target_weights": False, "target_labels_read": False}
        write_artifact(root, ARTIFACT, fold_id, {"predictions": predictions, "encoder_history": pl.DataFrame(history),
                                                 "encoder.pt": weights.getvalue()}, manifest)
        report[fold_id] = {"budget": budget, "predictions": summary}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="fit and freeze the FUS-BASE-001 baselines on the burn-in buffer")
    build.add_argument("--root", required=True, help="folder holding data/processed/memory")
    build.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    print(json.dumps(build_baselines(args.root, args.repo_root), indent=2, default=str))


if __name__ == "__main__":
    main()
