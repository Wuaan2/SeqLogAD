"""P2.6 label-free comparators (DIV-P2-001 §5), fitted on the same source's SOURCE_TRAIN material.

Train material = the source's P2.2 train windows and their events (bundle ``events.parquet``).
Messages are the bundled ``normalized_message`` strings. Add-one smoothing keeps one unseen slot:
p(m) = (c(m) + 1) / (N + V + 1), p(b | a) = (c(a, b) + 1) / (c(a, ·) + V + 1).
No label is read anywhere in this module.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.ensemble import IsolationForest

from .units import evaluation_windows, load_config

COMPARATORS = ("unseen_rarity", "count_vector", "isolation_forest", "ngram_markov")
COLUMNS = ["window_id", "architecture_id", "comparator", "score"]
HASH_KEY = b"DIV-P2-001"
REPO = Path(__file__).resolve().parents[3]


def _bucket(message: str, buckets: int) -> int:
    return int.from_bytes(hashlib.blake2b(message.encode(), digest_size=8, key=HASH_KEY).digest(), "big") % buckets


def _chains(windows: pl.DataFrame, messages: dict[int, str]) -> list[list[str]]:
    return [[messages[r] for r in context] + [messages[target]]
            for context, target in windows.select("context_ranks", "target_rank").iter_rows()]


def _features(chains: list[list[str]], buckets: int) -> csr_matrix:
    rows, cols = [], []
    for i, chain in enumerate(chains):
        for message in chain:
            rows.append(i)
            cols.append(_bucket(message, buckets))
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(chains), buckets))


def comparator_scores(train_windows: pl.DataFrame, train_events: pl.DataFrame, eval_windows: pl.DataFrame,
                      eval_events: pl.DataFrame, isolation: dict) -> pl.DataFrame:
    """One source: fit on its train material, score its evaluation windows (higher = more anomalous)."""
    train_msg = dict(train_events.select("rank", "normalized_message").iter_rows())
    eval_msg = dict(eval_events.select("rank", "normalized_message").iter_rows())
    unigram = Counter(train_msg.values())
    total, vocab = sum(unigram.values()), len(unigram)

    def nll(message: str) -> float:
        return -math.log((unigram.get(message, 0) + 1) / (total + vocab + 1))

    pairs = {(a, b) for context, target in train_windows.select("context_ranks", "target_rank").iter_rows()
             for a, b in zip(list(context), list(context)[1:] + [target])}  # each record pair once
    bigram = Counter((train_msg[a], train_msg[b]) for a, b in pairs)
    outgoing = Counter()
    for (a, _), count in bigram.items():
        outgoing[a] += count

    forest = IsolationForest(n_estimators=int(isolation["n_estimators"]), random_state=int(isolation["random_state"]))
    forest.fit(_features(_chains(train_windows, train_msg), int(isolation["buckets"])))
    chains = _chains(eval_windows, eval_msg)
    forest_score = -forest.score_samples(_features(chains, int(isolation["buckets"])))

    scores = {
        "unseen_rarity": [nll(chain[-1]) for chain in chains],
        "count_vector": [sum(map(nll, chain)) / len(chain) for chain in chains],  # mean: windows hold 2-9 records
        "isolation_forest": forest_score.tolist(),
        "ngram_markov": [-math.log((bigram.get((chain[-2], chain[-1]), 0) + 1) / (outgoing.get(chain[-2], 0) + vocab + 1))
                         for chain in chains],
    }
    frame = pl.concat([
        eval_windows.select("window_id", "architecture_id").with_columns(
            pl.lit(name).alias("comparator"), pl.Series("score", scores[name], dtype=pl.Float64))
        for name in COMPARATORS])
    if not frame["score"].is_finite().all():
        raise ValueError("non-finite comparator score")
    return frame.select(COLUMNS)


def score_comparators(data_root, fold_id) -> pl.DataFrame:
    """Every comparator on every P2.6 evaluation window of the fold, each source fitted on its own train."""
    cfg = load_config(REPO)["comparators"]
    if cfg["labels_used_for_fitting"] or cfg["fit_partition"] != "SOURCE_TRAIN_OF_THE_SAME_SOURCE":
        raise ValueError("comparator preregistration changed")
    root = Path(data_root) / "folds" / fold_id / "sequence"
    evaluation = evaluation_windows(data_root, fold_id)
    train = pl.read_parquet(root / "train/windows.parquet")
    events = pl.read_parquet(root / "metadata/events.parquet")
    frames = []
    for arch in sorted(evaluation["architecture_id"].unique()):
        source = events.filter(pl.col("architecture_id") == arch)
        frames.append(comparator_scores(
            train.filter(pl.col("architecture_id") == arch), source.filter(pl.col("partition") == "SOURCE_TRAIN"),
            evaluation.filter(pl.col("architecture_id") == arch), source.filter(pl.col("partition") == "SOURCE_VALIDATION"),
            cfg["isolation_forest"]))
    result = pl.concat(frames)
    if result.height != evaluation.height * len(COMPARATORS):
        raise ValueError("comparator coverage incomplete")
    return result
