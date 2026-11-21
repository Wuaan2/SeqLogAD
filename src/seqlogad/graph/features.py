"""Deterministic GTAT-ST-001 graph construction (spec §4). Edges are derived here, never stored."""
from __future__ import annotations

import bisect
import hashlib
import re
from collections import Counter

import numpy as np

KEY = b"GTAT-ST-001"
TOKEN = re.compile(r"[a-z]+")
NODE_TYPES = ("TARGET", "OCCURRENCE", "COMPONENT", "LOCATION", "REQUEST")
EDGE_TYPES = ("SELF", "NEXT", "RECURRENCE", "HAS_COMPONENT", "COMPONENT_OF",
              "AT_LOCATION", "LOCATION_OF", "IN_REQUEST", "REQUEST_OF", "CONTEXT")
NODE = {name: index for index, name in enumerate(NODE_TYPES)}
EDGE = {name: index for index, name in enumerate(EDGE_TYPES)}
MAX_OFFSET = 64
NO_OFFSET = MAX_OFFSET + 1  # attribute nodes have no position


def keyed_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8, key=KEY).digest(), "little")


def text_buckets(text, buckets, max_tokens):
    if not text:
        return []
    tokens = list(dict.fromkeys(TOKEN.findall(text.lower())))[:max_tokens]
    return [keyed_hash(token) % buckets for token in tokens]


def bucket(value, bounds):
    return max(0, bisect.bisect_right(bounds, value) - 1)


def _codes(values):
    """Dense ids in first-seen order; None -> -1."""
    index, codes = {}, np.empty(len(values), dtype=np.int64)
    for position, value in enumerate(values):
        codes[position] = -1 if value is None else index.setdefault(value, len(index))
    return codes, list(index)


class NodeStore:
    """Per-fold node table with hashed text features computed once per distinct text."""

    def __init__(self, nodes, cfg):
        features = cfg["features"]
        nodes = nodes.sort(["architecture_id", "partition", "rank"])
        self.cfg = features
        self.seconds = nodes["effective_ts"].to_numpy() // 1_000_000
        self.severity = nodes["severity"].to_numpy().astype(np.int64) + 1  # 0 = unknown
        self.continuation = nodes["continuation_lines"].to_numpy().astype(np.int64)
        self.event, events = _codes(nodes["event_text"].to_list())
        self.component, components = _codes(nodes["component_text"].to_list())
        self.location, _ = _codes(nodes["location"].to_list())
        self.request, _ = _codes(nodes["request_group"].to_list())
        limit = features["max_tokens_per_text"]
        self.event_tokens = [text_buckets(text, features["event_hash_buckets"], limit) for text in events]
        self.event_class = np.array([keyed_hash(text) % features["event_class_buckets"] for text in events],
                                    dtype=np.int64)
        self.component_tokens = [text_buckets(text, features["component_hash_buckets"], limit)
                                 for text in components]
        self.blocks = {}
        ranks = nodes["rank"].to_numpy()
        keys = list(zip(nodes["architecture_id"].to_list(), nodes["partition"].to_list()))
        start = 0
        for end in range(1, len(keys) + 1):
            if end == len(keys) or keys[end] != keys[start]:
                block = ranks[start:end]
                if len(block) > 1 and not (np.diff(block) > 0).all():
                    raise ValueError("duplicate node rank in nodes.parquet")
                self.blocks[keys[start]] = (start, block)
                start = end

    def locate(self, architecture, partition, ranks):
        start, block = self.blocks[(architecture, partition)]
        ranks = np.asarray(ranks, dtype=np.int64)
        position = np.searchsorted(block, ranks)
        if (position >= len(block)).any() or (block[np.minimum(position, len(block) - 1)] != ranks).any():
            raise ValueError("graph references a node missing from nodes.parquet")
        return start + position


def examples(store, windows):
    """Graph rows -> compact examples (node indices only)."""
    result = []
    for row in windows.iter_rows(named=True):
        context = list(row["context_ranks"] or [])
        if any(rank >= row["target_rank"] for rank in context) or context != sorted(set(context)):
            raise ValueError("graph context is not strictly causal and ordered")
        if len(context) > MAX_OFFSET:
            raise ValueError("graph context exceeds 64 records")
        scored = row["status"] == "SCORED"
        if scored != bool(context):
            raise ValueError("graph status disagrees with its context")
        result.append({
            "window_id": row["window_id"], "architecture_id": row["architecture_id"],
            "target_rank": row["target_rank"], "status": row["status"],
            "context": store.locate(row["architecture_id"], row["partition"], context) if context else None,
            "target": int(store.locate(row["architecture_id"], row["partition"], [row["target_rank"]])[0]),
        })
    return result


def build_graph(store, context, target):
    """One masked-target graph: node features, typed edges [type, dst, src], labels, coverage."""
    features = store.cfg
    time_bounds, clip = features["time_buckets"], features["time_clip_seconds"]
    none_time = len(time_bounds)
    n = len(context)
    seconds = store.seconds[context]
    events = store.event[context]
    components = store.component[context]
    locations = store.location[context]
    requests = store.request[context]
    request_counts = Counter(int(value) for value in requests if value >= 0)
    attributes = (
        [("COMPONENT", int(value)) for value in dict.fromkeys(components.tolist()) if value >= 0]
        + [("LOCATION", int(value)) for value in dict.fromkeys(locations.tolist()) if value >= 0]
        + [("REQUEST", int(value)) for value in dict.fromkeys(requests.tolist())
           if value >= 0 and request_counts[value] >= features["request_node_min_members"]]
    )
    size = 1 + n + len(attributes)
    edges = np.zeros((len(EDGE_TYPES), size, size), dtype=bool)
    time = np.full((size, size), none_time, dtype=np.int64)

    def gap(later, earlier):
        return bucket(min(int(later) - int(earlier), clip), time_bounds)

    edges[EDGE["SELF"], np.arange(size), np.arange(size)] = True
    last_seen, seen = {}, Counter()
    recurrence = np.zeros(n, dtype=np.int64)
    for i in range(n):
        node = 1 + i
        if i:
            edges[EDGE["NEXT"], node, node - 1] = True
            time[node, node - 1] = gap(seconds[i], seconds[i - 1])
        event = int(events[i])
        if event in last_seen:
            j = last_seen[event]
            edges[EDGE["RECURRENCE"], node, 1 + j] = True
            time[node, 1 + j] = gap(seconds[i], seconds[j])
        recurrence[i] = seen[event]
        seen[event] += 1
        last_seen[event] = i
        edges[EDGE["CONTEXT"], 0, node] = True
        time[0, node] = gap(seconds[-1], seconds[i])  # age; the target's own time stays masked
    edges[EDGE["NEXT"], 0, n] = True
    links = {"COMPONENT": (components, "HAS_COMPONENT", "COMPONENT_OF"),
             "LOCATION": (locations, "AT_LOCATION", "LOCATION_OF"),
             "REQUEST": (requests, "IN_REQUEST", "REQUEST_OF")}
    members = []
    for position, (kind, value) in enumerate(attributes):
        node = 1 + n + position
        values, into, out = links[kind]
        owners = np.flatnonzero(values == value) + 1
        edges[EDGE[into], node, owners] = True
        edges[EDGE[out], owners, node] = True
        members.append(len(owners))

    degree_bounds = features["degree_buckets"]
    in_degree = (edges[1:].any(0)).sum(1)
    node_type = np.array([NODE["TARGET"]] + [NODE["OCCURRENCE"]] * n
                         + [NODE[kind] for kind, _ in attributes], dtype=np.int64)
    offset = np.array([0] + [n - i for i in range(n)] + [NO_OFFSET] * len(attributes), dtype=np.int64)
    delta = np.full(size, none_time, dtype=np.int64)
    age = np.full(size, none_time, dtype=np.int64)
    for i in range(n):
        if i:
            delta[1 + i] = gap(seconds[i], seconds[i - 1])
        age[1 + i] = gap(seconds[-1], seconds[i])
    recurrence_bucket = np.zeros(size, dtype=np.int64)
    first = np.zeros(size, dtype=np.int64)
    count = np.zeros(size, dtype=np.int64)
    severity = np.zeros(size, dtype=np.int64)
    event_tokens = [[] for _ in range(size)]
    component_tokens = [[] for _ in range(size)]
    for i in range(n):
        node = 1 + i
        recurrence_bucket[node] = 1 + bucket(int(recurrence[i]), features["recurrence_buckets"])
        first[node] = 1 if recurrence[i] == 0 else 2
        count[node] = 1 + bucket(int(store.continuation[context[i]]), degree_bounds)
        severity[node] = store.severity[context[i]]
        event_tokens[node] = store.event_tokens[events[i]]
        if components[i] >= 0:
            component_tokens[node] = store.component_tokens[components[i]]
    for position, ((kind, value), total) in enumerate(zip(attributes, members)):
        node = 1 + n + position
        count[node] = 1 + bucket(total, degree_bounds)
        if kind == "COMPONENT":
            component_tokens[node] = store.component_tokens[value]
    target_event = int(store.event[target])
    pointer = last_seen.get(target_event)
    covered = int(((components >= 0) & (store.severity[context] > 0)).sum())
    return {
        "node_type": node_type, "offset": offset, "delta": delta, "age": age,
        "recurrence": recurrence_bucket, "first": first, "count": count,
        "degree": np.array([bucket(int(value), degree_bounds) for value in in_degree], dtype=np.int64),
        "severity": severity, "event_tokens": event_tokens, "component_tokens": component_tokens,
        "edges": edges, "time": time,
        "label_delta": gap(store.seconds[target], seconds[-1]),
        "label_pointer": -1 if pointer is None else 1 + pointer,
        "label_event": int(store.event_class[target_event]),
        "coverage": covered / n, "context_records": n,
    }


def collate(graphs):
    """Pad graphs to one batch of torch tensors."""
    import torch

    batch, size = len(graphs), max(len(graph["node_type"]) for graph in graphs)
    none_time = int(graphs[0]["delta"][0])  # the target node always carries the "no time" bucket
    padding = {"offset": NO_OFFSET, "delta": none_time, "age": none_time}
    dense = {name: torch.full((batch, size), padding.get(name, 0), dtype=torch.long)
             for name in ("node_type", "offset", "delta", "age", "recurrence", "first",
                          "count", "degree", "severity")}
    valid = torch.zeros(batch, size, dtype=torch.bool)
    context = torch.zeros(batch, size, dtype=torch.bool)
    edges = torch.zeros(batch, len(EDGE_TYPES), size, size, dtype=torch.bool)
    time = torch.full((batch, size, size), none_time, dtype=torch.long)
    bags = {"event_tokens": ([], []), "component_tokens": ([], [])}
    for b, graph in enumerate(graphs):
        count = len(graph["node_type"])
        for name, tensor in dense.items():
            tensor[b, :count] = torch.from_numpy(graph[name])
        valid[b, :count] = True
        context[b, 1:1 + graph["context_records"]] = True
        edges[b, :, :count, :count] = torch.from_numpy(graph["edges"])
        time[b, :count, :count] = torch.from_numpy(graph["time"])
        for name, (values, offsets) in bags.items():
            for node in range(size):
                offsets.append(len(values))
                if node < count:
                    values.extend(graph[name][node])
    result = dict(dense, valid=valid, context=context, edges=edges, time=time)
    for name, (values, offsets) in bags.items():
        result[name] = torch.tensor(values, dtype=torch.long)
        result[name + "_offsets"] = torch.tensor(offsets, dtype=torch.long)
    result["label_delta"] = torch.tensor([graph["label_delta"] for graph in graphs])
    result["label_pointer"] = torch.tensor([size if graph["label_pointer"] < 0 else graph["label_pointer"]
                                            for graph in graphs])
    result["label_event"] = torch.tensor([graph["label_event"] for graph in graphs])
    return result
