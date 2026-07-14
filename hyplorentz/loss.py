"""Contrastive losses on the Lorentz manifold.

HyperbolicNTXent is a drop-in replacement for the SimCLR NT-Xent loss where
cosine similarity is replaced by negative hyperbolic distance:

    sim(z_i, z_j) = -d(z_i, z_j) / tau

Two distance flavors:
  * "geodesic"  — sqrt(k) * arccosh(-<x,y>_L / k). The true metric.
  * "sq_lorentz" — squared Lorentzian distance -2k - 2<x,y>_L. No acosh,
    smoother gradients near zero distance, usually trains more stably.
    (Law et al. 2019, "Lorentzian Distance Learning".)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import lorentz_distance, lorentz_sqdist


def pairwise_lorentz_dist(
    z: torch.Tensor, k: float = 1.0, kind: str = "geodesic"
) -> torch.Tensor:
    """All-pairs distance matrix for a batch of hyperboloid points.

    z : (N, d+1) points on L^d_k  ->  (N, N) distances.
    """
    x = z.unsqueeze(1)  # (N, 1, d+1)
    y = z.unsqueeze(0)  # (1, N, d+1)
    if kind == "geodesic":
        return lorentz_distance(x, y, k=k)
    if kind == "sq_lorentz":
        return lorentz_sqdist(x, y, k=k)
    raise ValueError(f"unknown distance kind: {kind!r}")


class HyperbolicNTXent(nn.Module):
    """NT-Xent (SimCLR) loss with hyperbolic similarity.

    Expects two batches of embeddings z1, z2 of shape (N, d+1) where row i of
    z1 and row i of z2 are two views of the same sample.
    """

    def __init__(self, temperature: float = 0.3, k: float = 1.0, kind: str = "geodesic") -> None:
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.kind = kind

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        n = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2N, d+1)

        logits = -pairwise_lorentz_dist(z, k=self.k, kind=self.kind) / self.temperature

        # Mask self-similarity.
        logits.fill_diagonal_(float("-inf"))

        # Positive for row i is i+N (mod 2N).
        targets = torch.arange(2 * n, device=z.device)
        targets = (targets + n) % (2 * n)

        return F.cross_entropy(logits, targets)
