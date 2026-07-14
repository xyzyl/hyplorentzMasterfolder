"""Fully hyperbolic graph convolution on the Lorentz manifold (HGNN-L spec).

Every node state, every aggregation result, and the graph readout live on
L^d_k. There are no tangent-space round trips anywhere:

- messages:     LorentzLinear on the ambient concat of the two endpoint states
                (LorentzLinear accepts arbitrary ambient input and lands its
                output exactly on the manifold -- this is load-bearing)
- aggregation:  attention-weighted Lorentzian centroid (closed form, exact)
- update:       LorentzLinear on ambient concat of old state and aggregate
- residual:     two-point weighted centroid with a learnable gate beta,
                initialized near 0.8 (mostly-identity at init)
- readout:      uniform Lorentzian centroid over each graph's nodes

Invariant I1: |<z, z>_L + k| < 1e-4 in float32 after every stage.

Graph batching is block-diagonal: node tensors are concatenated, edge_index
is offset, and ``batch`` maps node -> graph. Scatter ops are hand-rolled
index_add (no torch_geometric; spec section 9).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from geoopt.manifolds import Lorentz

from .attention import LorentzDistanceAttention
from .lorentz import EPS, LorentzCentroid, LorentzLinear, lift_to_hyperboloid, lorentz_inner


def lorentz_centroid_scatter(
    x: torch.Tensor,        # (E, d+1) points on the manifold
    weights: torch.Tensor,  # (E,) nonnegative
    index: torch.Tensor,    # (E,) segment id per point
    num_segments: int,
    k: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-segment weighted Lorentzian centroid.

    Returns (centroids (S, d+1), valid (S,) bool). Segments with no points
    (or all-zero weight) are marked invalid and returned as the manifold
    origin; callers must mask them out. The denominator is clamped at 1e-6
    (numerical requirement N3: degenerate when the weighted sum approaches
    the light cone).
    """
    s = x.new_zeros(num_segments, x.shape[-1])
    s.index_add_(0, index, x * weights.unsqueeze(-1))
    sq = -lorentz_inner(s, s, keepdim=True)
    valid = sq.squeeze(-1) > EPS
    denom = torch.sqrt(sq.clamp(min=EPS))
    mu = math.sqrt(k) * s / denom
    # Invalid rows get the origin so downstream ops stay on-manifold even if
    # a caller forgets the mask; the mask is still the contract.
    origin = x.new_zeros(x.shape[-1])
    origin[0] = math.sqrt(k)
    mu = torch.where(valid.unsqueeze(-1), mu, origin)
    return mu, valid


class LorentzGraphConv(nn.Module):
    """One HGNN-L layer (spec section 4): message, distance attention,
    centroid aggregation, update, gated centroid residual."""

    def __init__(
        self,
        ambient_dim: int,       # d + 1
        k: float = 1.0,
        edge_dim: int = 8,
        dropout: float = 0.0,
        residual_init: float = 0.8,
    ) -> None:
        super().__init__()
        self.k = k
        self.msg_lin = LorentzLinear(
            2 * ambient_dim, ambient_dim, k=k, dropout=dropout, nonlin=nn.ReLU()
        )
        self.attention = LorentzDistanceAttention(
            ambient_dim, k=k, edge_dim=edge_dim, dropout=dropout
        )
        self.upd_lin = LorentzLinear(
            2 * ambient_dim, ambient_dim, k=k, dropout=dropout, nonlin=nn.ReLU()
        )
        self.centroid = LorentzCentroid(k=k)
        # Residual gate beta = sigmoid(b); beta ~ residual_init at init.
        self.gate = nn.Parameter(torch.tensor(math.log(residual_init / (1 - residual_init))))

    def forward(
        self,
        x: torch.Tensor,                       # (N, d+1) on-manifold
        edge_index: torch.Tensor,              # (2, E) directed u -> v
        edge_attr: torch.Tensor | None = None, # (E, edge_dim)
    ) -> torch.Tensor:
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        m = self.msg_lin(torch.cat([x[src], x[dst]], dim=-1))         # (E, d+1)
        alpha = self.attention(x, edge_index, edge_attr)              # (E,)
        mu, valid = lorentz_centroid_scatter(m, alpha, dst, n, k=self.k)
        # Nodes with no incoming edges (possible after DropEdge, or a lone
        # virtual node): aggregate falls back to the node's own state.
        mu = torch.where(valid.unsqueeze(-1), mu, x)

        x_new = self.upd_lin(torch.cat([x, mu], dim=-1))              # (N, d+1)

        beta = torch.sigmoid(self.gate)
        pair = torch.stack([x, x_new], dim=-2)                        # (N, 2, d+1)
        w = torch.stack([beta, 1.0 - beta]).to(x.dtype)               # (2,)
        return self.centroid(pair, w.expand(n, 2))


class LorentzGNN(nn.Module):
    """HGNN-L trunk: lift + input embed + LorentzGraphConv stack + centroid
    readout. Output: one point on L^d_k per graph, ready for
    HyperbolicProjectionHead(already_lifted=True)."""

    def __init__(
        self,
        in_features: int = 16,
        dim: int = 64,          # manifold dimension d (ambient d+1)
        num_layers: int = 4,
        k: float = 1.0,
        edge_dim: int = 8,
        dropout: float = 0.0,
        drop_edge: float = 0.1,
    ) -> None:
        super().__init__()
        self.k = k
        self.drop_edge = drop_edge
        self.manifold = Lorentz(k=k)
        self.manifold.k.requires_grad_(False)
        self.pre_norm = nn.BatchNorm1d(in_features)
        # Learned feature for virtual nodes (crossingless diagrams; they are
        # the identity of the augmentation group and must not be dropped).
        self.virtual_feat = nn.Parameter(torch.zeros(in_features))
        self.embed = LorentzLinear(in_features + 1, dim + 1, k=k)
        self.layers = nn.ModuleList(
            LorentzGraphConv(dim + 1, k=k, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_layers)
        )
        self.readout = LorentzCentroid(k=k)

    def forward(
        self,
        x: torch.Tensor,                        # (N, in_features) Euclidean
        edge_index: torch.Tensor,               # (2, E) directed
        edge_attr: torch.Tensor | None,         # (E, edge_dim)
        batch: torch.Tensor,                    # (N,) node -> graph id
        num_graphs: int | None = None,
        virtual_mask: torch.Tensor | None = None,  # (N,) bool
        return_nodes: bool = False,             # node states instead of readout
    ) -> torch.Tensor:
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        if virtual_mask is not None and virtual_mask.any():
            x = torch.where(virtual_mask.unsqueeze(-1), self.virtual_feat, x)
        # BatchNorm needs > 1 row to compute batch stats in training mode.
        if x.shape[0] > 1 or not self.training:
            x = self.pre_norm(x)

        z = lift_to_hyperboloid(x, self.manifold)
        z = self.embed(z)

        if self.training and self.drop_edge > 0 and edge_index.numel():
            keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= self.drop_edge
            edge_index = edge_index[:, keep]
            edge_attr = edge_attr[keep] if edge_attr is not None else None

        for layer in self.layers:
            z = layer(z, edge_index, edge_attr)

        if return_nodes:
            return z

        ones = z.new_ones(z.shape[0])
        out, valid = lorentz_centroid_scatter(z, ones, batch, num_graphs, k=self.k)
        if not bool(valid.all()):
            raise ValueError("readout hit an empty or light-cone-degenerate graph segment")
        return out
