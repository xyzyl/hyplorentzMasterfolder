"""Knot diagrams -> graph tensors for the Lorentz GNN (HGNN-L spec section 3).

Consumes the dart-based ``Diagram`` objects produced by the KnotCLR
augmentation engine (knotclr.diagram / knotclr.views). Encoding:

- nodes = crossings; features: sign (+-1) in channel 0, 8-d sinusoidal
  position index (channels 1..8, writhe-normalized order along the strand
  walk), zero-padded to ``in_features`` (default 16)
- edges = arcs, emitted in BOTH directions; the 8-d edge attribute is the
  one-hot rotational slot (0-3) of the source dart followed by the one-hot
  slot of the target dart -- this is what preserves over/under routing
- crossingless diagrams (the unknot as 0 crossings) become a single virtual
  node flagged in ``virtual_mask``; the GNN swaps in its learned feature

The Diagram type is duck-typed (needs .n, .theta, .sign(c)) so hyplorentz
does not import knotclr.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import torch

IN_FEATURES = 16
POS_DIM = 8
EDGE_DIM = 8


class GraphBatch(NamedTuple):
    x: torch.Tensor            # (N, in_features) float32
    edge_index: torch.Tensor   # (2, E) int64, directed
    edge_attr: torch.Tensor    # (E, EDGE_DIM) float32
    batch: torch.Tensor        # (N,) int64 node -> graph
    virtual_mask: torch.Tensor # (N,) bool
    num_graphs: int


def _sinusoidal(pos: float, dim: int) -> list[float]:
    out = []
    for i in range(dim // 2):
        freq = math.pi * (2.0 ** i)
        out.append(math.sin(freq * pos))
        out.append(math.cos(freq * pos))
    return out


def diagram_to_graph(
    diag, in_features: int = IN_FEATURES, position_index: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """One diagram -> (x, edge_index, edge_attr, is_virtual).

    ``diag`` may be None or have n == 0 (crossingless unknot): the graph is
    then a single virtual node with no edges.
    """
    if diag is None or diag.n == 0:
        x = torch.zeros(1, in_features)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, EDGE_DIM)
        return x, edge_index, edge_attr, True

    n = diag.n
    x = torch.zeros(n, in_features)
    for c in range(n):
        x[c, 0] = float(diag.sign(c))
    if position_index and in_features >= 1 + POS_DIM:
        # Crossing order along the strand walk from dart 2, normalized to
        # [0, 1) -- rotation-equivariant up to the walk's start choice, which
        # sinusoidal encoding tolerates.
        order = []
        d = 2
        while True:
            c = d >> 2
            if c not in order:
                order.append(c)
            d = diag.theta[d] ^ 2
            if d == 2:
                break
        for rank, c in enumerate(order):
            x[c, 1 : 1 + POS_DIM] = torch.tensor(_sinusoidal(rank / n, POS_DIM))

    srcs, dsts, attrs = [], [], []
    for d in range(4 * n):
        e = diag.theta[d]
        if d > e:
            continue
        u, su = d >> 2, d & 3
        v, sv = e >> 2, e & 3
        a_uv = torch.zeros(EDGE_DIM)
        a_uv[su] = 1.0
        a_uv[4 + sv] = 1.0
        a_vu = torch.zeros(EDGE_DIM)
        a_vu[sv] = 1.0
        a_vu[4 + su] = 1.0
        srcs += [u, v]
        dsts += [v, u]
        attrs += [a_uv, a_vu]

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    edge_attr = torch.stack(attrs)
    return x, edge_index, edge_attr, False


def collate_graphs(diagrams: Sequence, in_features: int = IN_FEATURES) -> GraphBatch:
    """Block-diagonal batching of a list of diagrams."""
    xs, eis, eas, batch_ids, virt = [], [], [], [], []
    offset = 0
    for g, diag in enumerate(diagrams):
        x, ei, ea, is_virtual = diagram_to_graph(diag, in_features=in_features)
        xs.append(x)
        eis.append(ei + offset)
        eas.append(ea)
        batch_ids.append(torch.full((x.shape[0],), g, dtype=torch.long))
        virt.append(torch.full((x.shape[0],), is_virtual, dtype=torch.bool))
        offset += x.shape[0]
    return GraphBatch(
        x=torch.cat(xs),
        edge_index=torch.cat(eis, dim=1),
        edge_attr=torch.cat(eas),
        batch=torch.cat(batch_ids),
        virtual_mask=torch.cat(virt),
        num_graphs=len(diagrams),
    )
