"""GTAT-ST-001: two-stream graph topology attention with typed, time-aware edge biases (spec §5)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .features import EDGE_TYPES, NODE_TYPES, NO_OFFSET


class GTATLayer(nn.Module):
    """Node stream H and topology stream T; each attends over graph edges with keys from the other stream."""

    def __init__(self, hidden, heads, ffn, dropout, time_values):
        super().__init__()
        self.heads = heads
        self.h_norm, self.t_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.h_q, self.t_k, self.h_v, self.h_o = (nn.Linear(hidden, hidden) for _ in range(4))
        self.t_q, self.h_k, self.t_v, self.t_o = (nn.Linear(hidden, hidden) for _ in range(4))
        self.edge_bias = nn.Parameter(torch.zeros(2, len(EDGE_TYPES), heads))
        self.time_bias = nn.Embedding(time_values, 2 * heads)
        nn.init.zeros_(self.time_bias.weight)
        self.h_ffn = self._ffn(hidden, ffn, dropout)
        self.t_ffn = self._ffn(hidden, ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _ffn(hidden, ffn, dropout):
        return nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ffn), nn.GELU(),
                             nn.Dropout(dropout), nn.Linear(ffn, hidden))

    def _attend(self, query, key, value, bias, mask):
        batch, size, hidden = query.shape
        split = lambda x: x.view(batch, size, self.heads, hidden // self.heads).transpose(1, 2)
        query, key, value = split(query), split(key), split(value)
        scores = query @ key.transpose(-1, -2) / math.sqrt(hidden // self.heads) + bias
        scores = scores.masked_fill(~mask[:, None], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return (weights @ value).transpose(1, 2).reshape(batch, size, hidden)

    def forward(self, h, t, edges, time, mask):
        type_bias = torch.einsum("benm,seh->bshnm", edges, self.edge_bias)
        time_bias = self.time_bias(time).permute(0, 3, 1, 2)
        bias_h = type_bias[:, 0] + time_bias[:, :self.heads]
        bias_t = type_bias[:, 1] + time_bias[:, self.heads:]
        hn, tn = self.h_norm(h), self.t_norm(t)
        h = h + self.h_o(self._attend(self.h_q(hn), self.t_k(tn), self.h_v(hn), bias_h, mask))
        t = t + self.t_o(self._attend(self.t_q(tn), self.h_k(hn), self.t_v(tn), bias_t, mask))
        return h + self.h_ffn(h), t + self.t_ffn(t)


class GTATModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        features, model = cfg["features"], cfg["model"]
        hidden = model["hidden_size"]
        time_values = len(features["time_buckets"]) + 1  # + "no time"
        counts = len(features["degree_buckets"]) + 1
        self.event_bag = nn.EmbeddingBag(features["event_hash_buckets"], hidden, mode="mean")
        self.component_bag = nn.EmbeddingBag(features["component_hash_buckets"], hidden, mode="mean")
        self.severity = nn.Embedding(5, hidden)  # unknown, 0..3
        self.mask = nn.Parameter(torch.zeros(hidden))
        self.node_type = nn.Embedding(len(NODE_TYPES), hidden)
        self.offset = nn.Embedding(NO_OFFSET + 1, hidden)
        self.delta = nn.Embedding(time_values, hidden)
        self.age = nn.Embedding(time_values, hidden)
        self.recurrence = nn.Embedding(len(features["recurrence_buckets"]) + 1, hidden)
        self.first = nn.Embedding(3, hidden)
        self.count = nn.Embedding(counts, hidden)
        self.degree = nn.Embedding(counts, hidden)
        self.layers = nn.ModuleList(
            GTATLayer(hidden, model["num_heads"], model["ffn_size"], model["dropout"], time_values)
            for _ in range(model["num_layers"])
        )
        self.readout = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden), nn.GELU())
        self.delta_head = nn.Linear(hidden, len(features["time_buckets"]))
        self.event_head = nn.Linear(hidden, features["event_class_buckets"])
        self.pointer_query = nn.Linear(hidden, hidden)
        self.pointer_key = nn.Linear(2 * hidden, hidden)
        self.pointer_new = nn.Linear(hidden, 1)

    def forward(self, batch):
        size = batch["node_type"].shape[1]
        events = self.event_bag(batch["event_tokens"], batch["event_tokens_offsets"])
        components = self.component_bag(batch["component_tokens"], batch["component_tokens_offsets"])
        h = (events + components).view(-1, size, events.shape[-1]) + self.severity(batch["severity"])
        is_target = (batch["node_type"] == 0)[..., None]
        h = torch.where(is_target, self.mask.expand_as(h), h)
        t = (self.node_type(batch["node_type"]) + self.offset(batch["offset"]) + self.delta(batch["delta"])
             + self.age(batch["age"]) + self.recurrence(batch["recurrence"]) + self.first(batch["first"])
             + self.count(batch["count"]) + self.degree(batch["degree"]))
        edges = batch["edges"].float()
        eye = torch.eye(size, dtype=torch.bool, device=edges.device)
        mask = batch["edges"].any(1) | eye  # padded nodes attend to themselves only
        for layer in self.layers:
            h, t = layer(h, t, edges, batch["time"], mask)
        state = torch.cat([h, t], dim=-1)
        readout = self.readout(state[:, 0])
        keys = self.pointer_key(state)
        pointer = (keys @ self.pointer_query(readout)[..., None]).squeeze(-1) / math.sqrt(keys.shape[-1])
        pointer = pointer.masked_fill(~batch["context"], float("-inf"))
        pointer = torch.cat([pointer, self.pointer_new(readout)], dim=-1)  # last = NEW event
        return {"delta": self.delta_head(readout), "pointer": pointer, "event": self.event_head(readout)}


def head_losses(outputs, batch):
    """Per-graph NLL of each head (nats); their sum is the anomaly score."""
    return {
        "temporal_nll": F.cross_entropy(outputs["delta"].float(), batch["label_delta"], reduction="none"),
        "structural_nll": F.cross_entropy(outputs["pointer"].float(), batch["label_pointer"], reduction="none"),
        "content_nll": F.cross_entropy(outputs["event"].float(), batch["label_event"], reduction="none"),
    }
