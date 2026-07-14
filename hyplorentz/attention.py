"""Distance-based attention for the Lorentz GNN (HGNN-L spec section 4.2).

Attention logits are negative geodesic distances between LorentzLinear
query/key projections, scaled by 1/sqrt(d), plus a Euclidean edge-attribute
bias. Everything here produces *weights*; points never leave the manifold
(the edge MLP is Euclidean by design and only touches logits).

No torch_geometric: scatter-softmax is ~20 lines of index_add and keeps the
constraint logic auditable (spec section 9).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .lorentz import LorentzLinear, lorentz_distance

# Set True in tests / debugging to assert attention logits are finite before
# softmax (numerical requirement N5). Off by default: the .all() forces a
# device sync per layer per step.
DEBUG_ASSERTS = False

EPS = 1e-12


def scatter_softmax(logits: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Softmax of ``logits`` grouped by ``index`` (shape (E,) each).

    Numerically stabilized with a per-segment max subtraction. Segments with
    no entries never appear in the output (nothing indexes them).
    """
    seg_max = logits.new_full((num_segments,), float("-inf"))
    seg_max.scatter_reduce_(0, index, logits, reduce="amax", include_self=True)
    ex = (logits - seg_max[index]).exp()
    denom = logits.new_zeros(num_segments).index_add_(0, index, ex)
    return ex / denom[index].clamp(min=EPS)


class LorentzDistanceAttention(nn.Module):
    """Per-edge attention weights from geodesic query/key distances.

    logit_{uv} = -d_L(q_v, k_u) / sqrt(d)  +  MLP_edge(e_{uv})
    alpha_{uv} = softmax over incoming edges of v
    """

    def __init__(
        self,
        ambient_dim: int,
        k: float = 1.0,
        edge_dim: int = 8,
        edge_hidden: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.sqrt_d = math.sqrt(ambient_dim - 1)
        self.q_lin = LorentzLinear(ambient_dim, ambient_dim, k=k, dropout=dropout)
        self.k_lin = LorentzLinear(ambient_dim, ambient_dim, k=k, dropout=dropout)
        self.edge_mlp = (
            nn.Sequential(
                nn.Linear(edge_dim, edge_hidden),
                nn.ReLU(),
                nn.Linear(edge_hidden, 1),
            )
            if edge_dim > 0
            else None
        )

    def forward(
        self,
        x: torch.Tensor,            # (N, d+1) on-manifold node states
        edge_index: torch.Tensor,   # (2, E) directed u -> v
        edge_attr: torch.Tensor | None = None,  # (E, edge_dim) Euclidean
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        q = self.q_lin(x)
        kk = self.k_lin(x)
        logits = -lorentz_distance(q[dst], kk[src], k=self.k) / self.sqrt_d
        if self.edge_mlp is not None and edge_attr is not None:
            logits = logits + self.edge_mlp(edge_attr).squeeze(-1)
        if DEBUG_ASSERTS:
            assert torch.isfinite(logits).all(), "non-finite attention logits (N5)"
        return scatter_softmax(logits, dst, x.shape[0])
