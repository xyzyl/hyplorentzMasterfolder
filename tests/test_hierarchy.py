"""Phase 2 harness (HGNN-L spec section 7): synthetic hierarchy recovery.

(a) Tree distortion: embed random trees with the *GNN* (node features are
    local structure only), train with a triplet ranking loss on the tree
    metric (see triplet_loss for why not stress regression), and compare
    mean relative distortion vs a width-matched Euclidean GIN at output
    dim d in {2, 8}.
    Gate: hyperbolic wins at d=2 by >= 25% relative.

(b) Link prediction on tree-like synthetic graphs (random trees + a few
    shortcut edges), scored by embedding distance.
    Gate: hyperbolic matches or beats GIN AUC at d=8 (tolerance 0.01).

These are training runs (~minutes on CPU), so the module is marked
``pytest.mark.slow``. Run explicitly:

    python -m pytest tests/test_hierarchy.py -q -m slow --no-header -s
or  python tests/test_hierarchy.py          (prints the full table)
"""

from __future__ import annotations

import math
import random

import pytest
import torch
import torch.nn as nn
from geoopt.optim import RiemannianAdam

from hyplorentz import LorentzGNN, LorentzLinear, pairwise_lorentz_dist

pytestmark = pytest.mark.slow

K = 1.0
EDGE_DIM = 8  # unused features; the GNN accepts edge_dim=0 to disable the MLP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 1000  # 300 was not enough: both models occasionally collapse to
              # near-constant pairwise distances (identical distortions)


# ------------------------------------------------------------- synthetic data

def random_tree(n: int, rng: random.Random) -> list[tuple[int, int]]:
    """Uniform random recursive tree (attach each new node to a random one)."""
    return [(rng.randint(0, v - 1), v) for v in range(1, n)]


def tree_distances(n: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    D = torch.full((n, n), -1.0)
    for s in range(n):
        D[s, s] = 0.0
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if D[s, v] < 0:
                    D[s, v] = D[s, u] + 1
                    stack.append(v)
    return D


def structural_features(n: int, edges: list[tuple[int, int]], dim: int = 16) -> torch.Tensor:
    """Local structure only: degree one-hot (capped) + strong random features.
    No positions, no identifiers -- hierarchy must come through message
    passing. The unit-scale noise channels are the standard symmetry-breaking
    device (random-feature GNNs, Sato et al. 2021): with degree-only inputs
    almost all tree nodes are locally identical and BOTH models collapse to
    the constant-distance saddle (observed: identical hyp/euc distortions)."""
    deg = torch.zeros(n, dtype=torch.long)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    x = torch.zeros(n, dim)
    x[torch.arange(n), deg.clamp(max=dim // 2 - 1)] = 1.0
    x[:, dim // 2 :] = torch.randn(n, dim - dim // 2)
    return x


def to_edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    return torch.tensor([src, dst], dtype=torch.long)


# ------------------------------------------------------------------ models
#
# Both models share a width-HIDDEN trunk and project to the target manifold /
# output dim d only at the last layer. The comparison isolates the GEOMETRY
# of the embedding space; running the whole trunk at width 2 would just
# strangle computation identically for both.

HIDDEN = 16


class HypNodeEmbed(nn.Module):
    """LorentzGNN trunk (width HIDDEN) + LorentzLinear projection to L^d_k."""

    def __init__(self, in_features: int, dim: int, num_layers: int = 4):
        super().__init__()
        self.gnn = LorentzGNN(in_features, HIDDEN, num_layers=num_layers, k=K,
                              edge_dim=0, drop_edge=0.0)
        self.proj = LorentzLinear(HIDDEN + 1, dim + 1, k=K)

    def forward(self, x, ei, batch):
        z = self.gnn(x, ei, None, batch, num_graphs=1, return_nodes=True)
        return self.proj(z)


class GIN(nn.Module):
    """Euclidean GIN, trunk width HIDDEN, linear projection to R^d."""

    def __init__(self, in_features: int, dim: int, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(in_features, HIDDEN)
        self.eps = nn.Parameter(torch.zeros(num_layers))
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(HIDDEN, 2 * HIDDEN), nn.ReLU(),
                          nn.Linear(2 * HIDDEN, HIDDEN))
            for _ in range(num_layers)
        )
        self.proj = nn.Linear(HIDDEN, dim)

    def forward(self, x, edge_index):
        h = self.embed(x)
        src, dst = edge_index
        for i, mlp in enumerate(self.mlps):
            agg = torch.zeros_like(h).index_add_(0, dst, h[src])
            h = mlp((1 + self.eps[i]) * h + agg)
        return self.proj(h)


def euclidean_pdist(e: torch.Tensor) -> torch.Tensor:
    sq = (e.unsqueeze(1) - e.unsqueeze(0)).pow(2).sum(-1)
    return torch.sqrt(sq + 1e-8)


# ----------------------------------------------------------------- objectives

def stress(d: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target > 0
    scale = ((d[mask] * target[mask]).sum() / (d[mask] ** 2).sum()).detach().clamp(min=1e-6)
    return (((scale * d[mask] - target[mask]) / target[mask]) ** 2).mean()


def triplet_loss(D: torch.Tensor, target: torch.Tensor, n_samples: int = 4096) -> torch.Tensor:
    """Rank-based training signal: for random triplets (i, j, l) with
    target[i,j] < target[i,l], demand D[i,j] + margin < D[i,l].

    Used for TRAINING both models in the distortion task; distortion is
    still the evaluation metric. The scale-invariant stress regression is a
    trap here: the constant-distance collapse is a stable degenerate optimum
    that both the Lorentz GNN and the GIN fall into within ~250 steps
    (observed: identical hyp/euc distortions equal to the constant-embedding
    distortion of the target tree). Ranking margins are zero at collapse, so
    the triplet objective actively repels it.
    """
    n = D.shape[0]
    dev = D.device
    i = torch.randint(0, n, (n_samples,), device=dev)
    j = torch.randint(0, n, (n_samples,), device=dev)
    l = torch.randint(0, n, (n_samples,), device=dev)
    keep = target[i, j] < target[i, l]
    return torch.relu(1.0 + D[i, j] - D[i, l])[keep].mean()


def distortion(d: torch.Tensor, target: torch.Tensor) -> float:
    mask = target > 0
    scale = (d[mask] * target[mask]).sum() / (d[mask] ** 2).sum()
    return ((scale * d[mask] - target[mask]).abs() / target[mask]).mean().item()


# ------------------------------------------------------------------- training

def fit_tree_embedding(dim: int, n: int, seed: int, steps: int = STEPS) -> tuple[float, float]:
    """Returns (hyperbolic distortion, euclidean distortion) on one tree."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    edges = random_tree(n, rng)
    target = tree_distances(n, edges).to(DEVICE)
    x = structural_features(n, edges).to(DEVICE)
    ei = to_edge_index(edges).to(DEVICE)
    batch = torch.zeros(n, dtype=torch.long, device=DEVICE)

    gnn = HypNodeEmbed(16, dim, num_layers=4).to(DEVICE)
    opt = RiemannianAdam(gnn.parameters(), lr=5e-3)
    for _ in range(steps):
        z = gnn(x, ei, batch)
        loss = triplet_loss(pairwise_lorentz_dist(z, k=K), target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
        opt.step()
    gnn.eval()
    with torch.no_grad():
        z = gnn(x, ei, batch)
        d_hyp = distortion(pairwise_lorentz_dist(z, k=K), target)

    torch.manual_seed(seed)
    gin = GIN(16, dim, num_layers=4).to(DEVICE)
    opt = torch.optim.Adam(gin.parameters(), lr=5e-3)
    for _ in range(steps):
        loss = triplet_loss(euclidean_pdist(gin(x, ei)), target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gin.parameters(), 1.0)
        opt.step()
    gin.eval()
    with torch.no_grad():
        d_euc = distortion(euclidean_pdist(gin(x, ei)), target)
    return d_hyp, d_euc


def link_prediction_auc(dim: int, n: int, seed: int, steps: int = STEPS) -> tuple[float, float]:
    """Tree + 25% shortcut edges; hold out HALF THE SHORTCUTS as test
    positives + equal non-edges. Score = -distance.

    Two design constraints make this task measure anything at all:
    - The spanning tree always stays in the training graph: holding out a
      tree edge disconnects it, the severed endpoints land in unrelated
      embedding regions, and every model scores below chance (observed
      AUC ~0.3 for both geometries under a naive uniform edge holdout).
    - Shortcuts are DISTANCE-BIASED (tree distance 2..4, triadic-closure
      style). Uniformly random shortcuts are statistically identical to the
      random negatives, so no model could beat AUC 0.5 even in principle.

    Returns (AUC hyperbolic, AUC euclidean)."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tree = random_tree(n, rng)
    tdist = tree_distances(n, tree)
    have = set(map(tuple, map(sorted, tree)))
    shortcuts: list[tuple[int, int]] = []
    while len(shortcuts) < max(4, n // 4):
        u = rng.randrange(n)
        near = [v for v in range(n) if 2 <= tdist[u, v] <= 4]
        if not near:
            continue
        v = rng.choice(near)
        if tuple(sorted((u, v))) not in have:
            shortcuts.append((u, v))
            have.add(tuple(sorted((u, v))))
    rng.shuffle(shortcuts)
    n_test = len(shortcuts) // 2
    test_pos = shortcuts[:n_test]
    train_edges = tree + shortcuts[n_test:]
    test_neg = []
    while len(test_neg) < n_test:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and tuple(sorted((u, v))) not in have:
            test_neg.append((u, v))

    x = structural_features(n, train_edges).to(DEVICE)
    ei = to_edge_index(train_edges).to(DEVICE)
    batch = torch.zeros(n, dtype=torch.long, device=DEVICE)

    def train_and_auc(model, is_hyp: bool) -> float:
        model = model.to(DEVICE)
        opt = (RiemannianAdam if is_hyp else torch.optim.Adam)(model.parameters(), lr=5e-3)
        pos = torch.tensor(train_edges, dtype=torch.long, device=DEVICE)
        for _ in range(steps):
            if is_hyp:
                z = model(x, ei, batch)
                D = pairwise_lorentz_dist(z, k=K)
            else:
                D = euclidean_pdist(model(x, ei))
            neg = torch.randint(0, n, pos.shape, device=DEVICE)
            d_pos = D[pos[:, 0], pos[:, 1]]
            d_neg = D[neg[:, 0], neg[:, 1]]
            # margin ranking: positives closer than sampled negatives
            loss = torch.relu(1.0 + d_pos - d_neg).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            if is_hyp:
                z = model(x, ei, batch)
                D = pairwise_lorentz_dist(z, k=K)
            else:
                D = euclidean_pdist(model(x, ei))
            Dc = D.cpu()
            sp = torch.tensor([-Dc[u, v] for u, v in test_pos])
            sn = torch.tensor([-Dc[u, v] for u, v in test_neg])
        # AUC = P(score_pos > score_neg), ties count half
        gt = (sp.unsqueeze(1) > sn.unsqueeze(0)).float().mean()
        eq = (sp.unsqueeze(1) == sn.unsqueeze(0)).float().mean()
        return float(gt + 0.5 * eq)

    torch.manual_seed(seed)
    gnn = HypNodeEmbed(16, dim, num_layers=4)
    auc_h = train_and_auc(gnn, True)
    torch.manual_seed(seed)
    auc_e = train_and_auc(GIN(16, dim, num_layers=4), False)
    return auc_h, auc_e


# ----------------------------------------------------------------- the gates

TREE_SIZES = (63, 255, 1023)
SEEDS = (0, 1, 2)


def run_distortion(dim: int) -> tuple[float, float]:
    hs, es = [], []
    for n in TREE_SIZES:
        for s in SEEDS:
            h, e = fit_tree_embedding(dim, n, seed=s)
            hs.append(h)
            es.append(e)
            print(f"  d={dim} n={n} seed={s}: hyp {h:.3f} euc {e:.3f}")
    return sum(hs) / len(hs), sum(es) / len(es)


def run_linkpred(dim: int) -> tuple[float, float]:
    hs, es = [], []
    for n in TREE_SIZES:
        for s in SEEDS:
            h, e = link_prediction_auc(dim, n, seed=s)
            hs.append(h)
            es.append(e)
            print(f"  d={dim} n={n} seed={s}: AUC hyp {h:.3f} euc {e:.3f}")
    return sum(hs) / len(hs), sum(es) / len(es)


def test_tree_distortion_d2_gate():
    h, e = run_distortion(dim=2)
    rel_win = (e - h) / e
    print(f"[gate a] d=2 distortion: hyp {h:.3f} vs euc {e:.3f} (rel win {rel_win:.1%})")
    assert rel_win >= 0.25, f"hyperbolic win {rel_win:.1%} < 25%"


def test_tree_distortion_d8():
    h, e = run_distortion(dim=8)
    print(f"[info] d=8 distortion: hyp {h:.3f} vs euc {e:.3f}")


def test_link_prediction_d8_gate():
    h, e = run_linkpred(dim=8)
    print(f"[gate b] d=8 link AUC: hyp {h:.3f} vs euc {e:.3f}")
    assert h >= e - 0.01, f"hyperbolic AUC {h:.3f} below GIN {e:.3f}"


if __name__ == "__main__":
    test_tree_distortion_d2_gate()
    test_tree_distortion_d8()
    test_link_prediction_d8_gate()
    print("phase 2 gates passed")
