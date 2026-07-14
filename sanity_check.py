"""Sanity checks for the hyperbolic projection head.

Run:  python sanity_check.py

Checks
------
1. Manifold constraint: every layer output satisfies <z, z>_L = -k.
2. Gradient health: no NaN/inf grads through head + loss (float32).
3. Toy SimCLR: loss decreases and positives end up closer than negatives.
4. Hierarchy test: contrastively embedding a synthetic tree in 2-D hyperbolic
   space preserves tree distances better than a 2-D Euclidean head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from geoopt.optim import RiemannianAdam

from hyplorentz import (
    HyperbolicNTXent,
    HyperbolicProjectionHead,
    lorentz_inner,
    pairwise_lorentz_dist,
)

torch.manual_seed(0)
K = 1.0


def check_constraint() -> None:
    head = HyperbolicProjectionHead(in_features=32, hidden_features=16, out_features=8, k=K)
    head.eval()
    h = torch.randn(256, 32) * 5  # deliberately large inputs
    z = head(h)
    inner = lorentz_inner(z, z, keepdim=False)
    err = (inner + K).abs().max().item()
    assert err < 1e-4, f"constraint violated: max |<z,z>_L + k| = {err:.2e}"
    assert (z[..., 0] > 0).all(), "time coordinate must be positive"
    print(f"[1] manifold constraint OK  (max err {err:.2e})")


def check_gradients() -> None:
    head = HyperbolicProjectionHead(in_features=32, hidden_features=16, out_features=8, k=K)
    crit = HyperbolicNTXent(temperature=0.3, k=K)
    h = torch.randn(64, 32)
    loss = crit(head(h + 0.1 * torch.randn_like(h)), head(h + 0.1 * torch.randn_like(h)))
    loss.backward()
    for name, p in head.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad in {name}"
    print(f"[2] gradients finite OK  (initial loss {loss.item():.3f})")


def toy_simclr() -> None:
    """Cluster structure: 8 classes in 32-d, augmentations = Gaussian noise."""
    n_class, per, dim = 8, 16, 32
    centers = torch.randn(n_class, dim) * 3
    base = centers.repeat_interleave(per, dim=0)  # (128, 32)

    head = HyperbolicProjectionHead(in_features=dim, hidden_features=32, out_features=16, k=K)
    crit = HyperbolicNTXent(temperature=0.3, k=K)
    opt = RiemannianAdam(head.parameters(), lr=1e-3)

    first = last = None
    for step in range(400):
        v1 = base + 0.3 * torch.randn_like(base)
        v2 = base + 0.3 * torch.randn_like(base)
        loss = crit(head(v1), head(v2))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    head.eval()
    with torch.no_grad():
        z1, z2 = head(base), head(base + 0.3 * torch.randn_like(base))
        d = pairwise_lorentz_dist(torch.cat([z1, z2]), k=K)
        n = base.shape[0]
        pos = d[torch.arange(n), torch.arange(n) + n].mean().item()
        neg = d[:n, n:].sum().item() / (n * n - n)
        pos = pos  # mean positive-pair distance
    assert last < first, "loss did not decrease"
    assert pos < neg, f"positives ({pos:.2f}) not closer than negatives ({neg:.2f})"
    print(f"[3] toy SimCLR OK  (loss {first:.2f} -> {last:.2f}; pos {pos:.2f} < neg {neg:.2f})")


def _tree_features_and_distances(depth: int = 5):
    """Complete binary tree. Feature = noisy path encoding; target = tree metric."""
    nodes = list(range(2 ** (depth + 1) - 1))
    n = len(nodes)

    def ancestors(i):
        out = [i]
        while i > 0:
            i = (i - 1) // 2
            out.append(i)
        return out

    anc = [set(ancestors(i)) for i in nodes]
    dep = [len(a) - 1 for a in anc]
    dist = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            common = max(dep[a] for a in (anc[i] & anc[j]))
            dist[i, j] = dep[i] + dep[j] - 2 * common

    feat = torch.zeros(n, n)
    for i in range(n):
        for a in anc[i]:
            feat[i, a] = 1.0
    return feat, dist


def _distortion(emb_dist: torch.Tensor, tree_dist: torch.Tensor) -> float:
    """Mean relative distortion after optimal global scaling."""
    mask = tree_dist > 0
    scale = (emb_dist[mask] * tree_dist[mask]).sum() / (emb_dist[mask] ** 2).sum()
    return ((scale * emb_dist[mask] - tree_dist[mask]).abs() / tree_dist[mask]).mean().item()


def tree_test() -> None:
    feat, tree_dist = _tree_features_and_distances(depth=5)
    n, dim = feat.shape

    # --- hyperbolic head, 2-D manifold ---
    hyp = HyperbolicProjectionHead(in_features=dim, hidden_features=16, out_features=2, k=K)
    opt = RiemannianAdam(hyp.parameters(), lr=5e-3)
    for _ in range(600):
        z = hyp(feat)
        d = pairwise_lorentz_dist(z, k=K)
        loss = _stress(d, tree_dist)
        opt.zero_grad(); loss.backward(); opt.step()
    hyp.eval()
    with torch.no_grad():
        hyp_dist = pairwise_lorentz_dist(hyp(feat), k=K)

    # --- Euclidean baseline head, 2-D output ---
    euc = nn.Sequential(nn.BatchNorm1d(dim), nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.Adam(euc.parameters(), lr=5e-3)

    def _pdist(e: torch.Tensor) -> torch.Tensor:
        # sqrt of squared distances + eps: avoids cdist's NaN gradient at d=0.
        sq = (e.unsqueeze(1) - e.unsqueeze(0)).pow(2).sum(-1)
        return torch.sqrt(sq + 1e-8)

    for _ in range(600):
        d = _pdist(euc(feat))
        loss = _stress(d, tree_dist)
        opt.zero_grad(); loss.backward(); opt.step()
    euc.eval()
    with torch.no_grad():
        euc_dist = _pdist(euc(feat))

    dh, de = _distortion(hyp_dist, tree_dist), _distortion(euc_dist, tree_dist)
    print(f"[4] tree (63 nodes) in 2-D: distortion hyperbolic {dh:.3f} vs euclidean {de:.3f}"
          + ("  hyperbolic wins" if dh < de else "  FAIL"))
    assert dh < de, "hyperbolic should beat Euclidean on tree metric in 2-D"


def _stress(d: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target > 0
    scale = ((d[mask] * target[mask]).sum() / (d[mask] ** 2).sum()).detach().clamp(min=1e-6)
    return (((scale * d[mask] - target[mask]) / target[mask]) ** 2).mean()


if __name__ == "__main__":
    check_constraint()
    check_gradients()
    toy_simclr()
    tree_test()
    print("\nall checks passed")
