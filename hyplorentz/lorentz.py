"""Lorentz-model (hyperboloid) primitives and fully hyperbolic layers.

Conventions
-----------
We work on the hyperboloid  L^n_k = { x in R^{n+1} : <x, x>_L = -k,  x_0 > 0 }
with the Lorentzian inner product

    <x, y>_L = -x_0 y_0 + sum_i x_i y_i,

curvature -1/k (k > 0). This matches ``geoopt.manifolds.Lorentz(k=k)``.

The "fully hyperbolic" linear layer follows Chen et al. 2021,
*Fully Hyperbolic Neural Networks* (ACL 2022): instead of the tangent-space
round trip (logmap -> Euclidean linear -> expmap), the layer applies a linear
map in ambient space and then analytically re-solves the time coordinate so
the output lands exactly on the hyperboloid. Activation, dropout, and
normalization are folded into the layer, operating on the ambient vector
before the constraint is re-imposed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from geoopt.manifolds import Lorentz

# Numerical guards (float32-friendly; important on consumer GPUs).
EPS = 1e-6
MAX_NORM = 1e6


def lorentz_inner(x: torch.Tensor, y: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """Lorentzian inner product <x, y>_L = -x0*y0 + <x_s, y_s>."""
    res = (x * y).sum(dim=-1, keepdim=keepdim)
    res = res - 2 * (x[..., :1] * y[..., :1] if keepdim else (x[..., 0] * y[..., 0]))
    return res


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Geodesic distance d(x, y) = sqrt(k) * arccosh(-<x, y>_L / k).

    The arccosh argument is clamped to >= 1 + EPS: for points on the manifold
    it is mathematically >= 1, but float32 roundoff can push it slightly
    below, which makes arccosh return NaN and (worse) its gradient blow up.
    """
    prod = -lorentz_inner(x, y, keepdim=False) / k
    prod = prod.clamp(min=1.0 + EPS)
    return math.sqrt(k) * torch.acosh(prod)


def lorentz_sqdist(x: torch.Tensor, y: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Squared *Lorentzian* distance  ||x - y||_L^2 = -2k - 2<x, y>_L.

    Cheaper and smoother than the geodesic distance (no acosh), often a
    better similarity for contrastive losses (cf. Law et al. 2019).
    """
    return (-2.0 * k - 2.0 * lorentz_inner(x, y, keepdim=False)).clamp(min=0.0)


def project_to_hyperboloid(x_space: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Given spatial coords, solve the time coord so the point is on L^n_k."""
    time = torch.sqrt(k + x_space.pow(2).sum(dim=-1, keepdim=True))
    return torch.cat([time, x_space], dim=-1)


MAX_TANGENT_NORM = 10.0  # cosh(10) ~ 1.1e4: safely inside float32 range


def lift_to_hyperboloid(u_euclidean: torch.Tensor, manifold: Lorentz) -> torch.Tensor:
    """Map a Euclidean feature vector onto the hyperboloid via expmap at the
    origin. A tangent vector at the origin o = (sqrt(k), 0, ..., 0) has time
    component 0, so we prepend a zero and exponentiate.

    The tangent norm is capped: expmap0 involves cosh(||u||), which overflows
    float32 for ||u|| > ~44 and loses all precision well before that. Capping
    at 10 keeps points in a numerically trustworthy region of the manifold.
    """
    norm = u_euclidean.norm(dim=-1, keepdim=True).clamp(min=EPS)
    factor = (MAX_TANGENT_NORM / norm).clamp(max=1.0)
    u_euclidean = u_euclidean * factor
    zeros = torch.zeros_like(u_euclidean[..., :1])
    u_tangent = torch.cat([zeros, u_euclidean], dim=-1)
    return manifold.expmap0(u_tangent)


class LorentzLinear(nn.Module):
    """Fully hyperbolic linear layer (Chen et al. 2021).

    y_space' = phi(W x + b)              (ambient linear map + nonlinearity)
    y_time   = sigmoid(raw_time) * scale + 1/sqrt(k) + eps   (learned, > sqrt(k))
    y_space  = y_space' * sqrt((y_time^2 - k) / ||y_space'||^2)

    The rescaling of the spatial part makes <y, y>_L = -k hold *exactly*
    (up to float precision), so the output never drifts off the manifold —
    no projection step, no tangent-space leakage.

    Parameters
    ----------
    in_features / out_features : ambient dimensions (manifold dim + 1).
    k : curvature magnitude of the target hyperboloid.
    nonlin : optional nonlinearity applied to the ambient pre-activation.
    dropout : dropout on the input, as in the reference implementation.
    scale : initial value of the learnable time-scale (sets how far from the
        origin the layer can place points; 10 is the paper's default).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: float = 1.0,
        bias: bool = True,
        dropout: float = 0.0,
        nonlin: nn.Module | None = None,
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.weight = nn.Linear(in_features, out_features, bias=bias)
        # Learnable global scale for the time coordinate, stored in log space
        # so it stays positive.
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Small init keeps early outputs near the hyperboloid origin, which
        # keeps acosh arguments in their well-conditioned range early in
        # training.
        nn.init.xavier_uniform_(self.weight.weight, gain=0.5)
        if self.weight.bias is not None:
            nn.init.zeros_(self.weight.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weight(self.dropout(x))
        if self.nonlin is not None:
            x = self.nonlin(x)

        raw_time = x.narrow(-1, 0, 1)
        space = x.narrow(-1, 1, x.shape[-1] - 1)

        # Learned time coordinate, guaranteed > sqrt(k).
        sqrt_k = math.sqrt(self.k)
        time = raw_time.sigmoid() * self.scale.exp() + sqrt_k + EPS

        # Rescale spatial part so the Lorentz constraint holds exactly.
        space_sq = space.pow(2).sum(dim=-1, keepdim=True).clamp(min=EPS)
        space = space * torch.sqrt((time.pow(2) - self.k) / space_sq)

        return torch.cat([time, space], dim=-1)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, k={self.k}"


class LorentzCentroid(nn.Module):
    """Lorentzian centroid (weighted midpoint) — useful for pooling.

    mu = sqrt(k) * sum_i w_i x_i / | ||sum_i w_i x_i||_L |
    """

    def __init__(self, k: float = 1.0) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, w: torch.Tensor | None = None, dim: int = -2) -> torch.Tensor:
        if w is not None:
            x = x * w.unsqueeze(-1)
        s = x.sum(dim=dim)
        norm = torch.sqrt((-lorentz_inner(s, s, keepdim=True)).clamp(min=EPS))
        return math.sqrt(self.k) * s / norm
