"""Hyperbolic projection head: Euclidean encoder features -> hyperboloid.

Designed to bolt onto any Euclidean encoder (transformer, GNN, CNN) in a
SimCLR-style setup:

    h = encoder(x)                      # Euclidean, e.g. 256-d
    z = HyperbolicProjectionHead(...)(h)  # point on L^d_k, ambient d+1

Pipeline:
    1. lift: expmap at the hyperboloid origin (exact, cheap)
    2. stack of LorentzLinear layers (fully hyperbolic; nonlinearity folded in)

All outputs satisfy <z, z>_L = -k by construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from geoopt.manifolds import Lorentz

from .lorentz import LorentzLinear, lift_to_hyperboloid


class HyperbolicProjectionHead(nn.Module):
    """SimCLR-style projection head living on the Lorentz manifold.

    Parameters
    ----------
    in_features : dimension of the (Euclidean) encoder output.
    hidden_features : manifold dimension of the hidden layer.
    out_features : manifold dimension of the output embedding. The output
        tensor has ``out_features + 1`` ambient coordinates.
    k : curvature magnitude (curvature is -1/k). Learnable if
        ``learnable_k=False`` is left as is; geoopt treats k as a buffer —
        we keep it fixed here, which is the common, stable choice.
    num_layers : number of LorentzLinear layers (>= 1).
    dropout : dropout inside each LorentzLinear.
    already_lifted : set True when the encoder output is already a point on
        L^{in_features}_k (ambient in_features + 1 coordinates), e.g. from
        LorentzGNN. Skips the pre-norm and the origin lift; the input feeds
        the first LorentzLinear directly.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 128,
        out_features: int = 64,
        k: float = 1.0,
        num_layers: int = 2,
        dropout: float = 0.0,
        already_lifted: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.manifold = Lorentz(k=k)
        # geoopt registers k as an nn.Parameter; keep curvature fixed.
        self.manifold.k.requires_grad_(False)
        self.k = k
        self.already_lifted = already_lifted

        # Pre-lift BatchNorm keeps encoder features in a sane range so the
        # exponential map does not shoot points toward the boundary.
        self.pre_norm = None if already_lifted else nn.BatchNorm1d(in_features)

        dims_ambient = (
            [in_features + 1]
            + [hidden_features + 1] * (num_layers - 1)
            + [out_features + 1]
        )
        layers: list[nn.Module] = []
        for i in range(num_layers):
            is_last = i == num_layers - 1
            layers.append(
                LorentzLinear(
                    dims_ambient[i],
                    dims_ambient[i + 1],
                    k=k,
                    dropout=dropout,
                    # ReLU on hidden layers only, as in the Euclidean SimCLR head.
                    nonlin=None if is_last else nn.ReLU(),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.already_lifted:
            z = h
        else:
            z = lift_to_hyperboloid(self.pre_norm(h), self.manifold)
        for layer in self.layers:
            z = layer(z)
        return z
