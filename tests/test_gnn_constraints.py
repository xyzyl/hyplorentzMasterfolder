"""Phase 1 gate (HGNN-L spec section 7): unit correctness of the Lorentz GNN.

Covers invariant I1 (on-manifold after every stage), numerical requirements
N1-N6, gradient finiteness, permutation invariance of the readout, batch
sizes {1, 2, 256}, single-node graphs, and self-loops. All float32.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

import hyplorentz.attention as attention_mod
from hyplorentz import (
    HyperbolicNTXent,
    HyperbolicProjectionHead,
    LorentzGNN,
    LorentzGraphConv,
    lift_to_hyperboloid,
    lorentz_centroid_scatter,
    lorentz_distance,
    lorentz_inner,
    scatter_softmax,
)
from hyplorentz.data import collate_graphs, diagram_to_graph

KNOTS_ROOT = Path("D:/Iskander/knots")
if str(KNOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(KNOTS_ROOT))

K = 1.0
I1_TOL = 1e-4
DIM = 64          # manifold dim
AMB = DIM + 1
EDGE_DIM = 8


@pytest.fixture(autouse=True)
def _debug_asserts():
    """N5: attention logits are asserted finite in every test."""
    attention_mod.DEBUG_ASSERTS = True
    yield
    attention_mod.DEBUG_ASSERTS = False


def constraint_error(z: torch.Tensor, k: float = K) -> float:
    return float((lorentz_inner(z, z, keepdim=False) + k).abs().max().detach())


def assert_on_manifold(z: torch.Tensor, abs_tol: float | None = None):
    """I1 check. A point stored in float32 at time coordinate T carries
    representation-level constraint error ~ T^2 * eps (cancellation of T^2
    against the spatial norm), so the universally enforceable form of I1 is
    RELATIVE: |<z,z>_L + k| / max(1, z0^2) < 1e-5. The spec's absolute 1e-4
    is additionally asserted (abs_tol) on trunk/readout outputs, where the
    embed layer bounds T <= scale + sqrt(k) ~ 11 and the absolute form has
    ~10x headroom over the float32 eps floor.
    """
    assert torch.isfinite(z).all()
    assert (z[..., 0] > 0).all()
    err = (lorentz_inner(z, z, keepdim=False) + K).abs()
    rel = float((err / z[..., 0].pow(2).clamp(min=1.0)).max().detach())
    assert rel < 1e-5, f"I1 (relative) violated: {rel}"
    if abs_tol is not None:
        e = float(err.max().detach())
        assert e < abs_tol, f"I1 (absolute) violated: {e}"


def random_points(n: int, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    from geoopt.manifolds import Lorentz

    u = torch.randn(n, DIM, generator=g) * scale
    return lift_to_hyperboloid(u, Lorentz(k=K))


def random_graph(n: int, extra_self_loops: int = 0, seed: int = 0):
    """Connected-ish directed multigraph with both edge directions."""
    g = torch.Generator().manual_seed(seed)
    srcs, dsts = [], []
    for v in range(1, n):
        u = int(torch.randint(0, v, (1,), generator=g))
        srcs += [u, v]
        dsts += [v, u]
    for _ in range(n):  # extra random edges
        u = int(torch.randint(0, n, (1,), generator=g))
        v = int(torch.randint(0, n, (1,), generator=g))
        srcs += [u, v]
        dsts += [v, u]
    for _ in range(extra_self_loops):
        u = int(torch.randint(0, n, (1,), generator=g))
        srcs.append(u)
        dsts.append(u)
    ei = torch.tensor([srcs, dsts], dtype=torch.long)
    ea = torch.rand(ei.shape[1], EDGE_DIM, generator=g)
    return ei, ea


# ------------------------------------------------------------------ I1 stages

def test_i1_attention_weights_normalized():
    x = random_points(10)
    ei, ea = random_graph(10)
    conv = LorentzGraphConv(AMB, k=K, edge_dim=EDGE_DIM)
    alpha = conv.attention(x, ei, ea)
    sums = torch.zeros(10).index_add_(0, ei[1], alpha)
    assert torch.allclose(sums, torch.ones(10), atol=1e-5)


def test_i1_centroid_scatter():
    x = random_points(30, scale=3.0)
    idx = torch.arange(30) % 5  # every segment non-empty
    w = torch.rand(30, generator=torch.Generator().manual_seed(0)) + 0.01
    mu, valid = lorentz_centroid_scatter(x, w, idx, 5, k=K)
    assert_on_manifold(mu)
    assert valid.all()


def test_i1_layer_and_extreme_inputs():
    for scale in (1e-3, 1.0, 1e3):
        x = random_points(20, scale=scale)
        assert_on_manifold(x)
        ei, ea = random_graph(20, seed=1)
        conv = LorentzGraphConv(AMB, k=K, edge_dim=EDGE_DIM)
        out = conv(x, ei, ea)
        assert_on_manifold(out)


def test_i1_deep_stack():
    x = random_points(16, scale=10.0)
    ei, ea = random_graph(16, seed=2)
    convs = [LorentzGraphConv(AMB, k=K, edge_dim=EDGE_DIM) for _ in range(8)]
    z = x
    for c in convs:
        z = c(z, ei, ea)
        assert_on_manifold(z)


# ------------------------------------------------------------------- N1 - N6

def test_n1_zero_distance_gradient_finite():
    x = random_points(4, seed=3).requires_grad_(True)
    d = lorentz_distance(x, x.detach(), k=K)  # identical points: acosh at clamp
    d.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_n2_lift_tangent_cap():
    from geoopt.manifolds import Lorentz

    u = torch.randn(8, DIM) * 1e6
    z = lift_to_hyperboloid(u, Lorentz(k=K))
    assert_on_manifold(z)  # relative check; absolute is impossible at the cap in fp32
    assert (z[:, 0] <= math.cosh(10.0) * math.sqrt(K) * 1.01).all()


def test_n3_degenerate_centroid_masked():
    x = random_points(6)
    w = torch.zeros(6)  # all-zero weights: weighted sum is the zero vector
    mu, valid = lorentz_centroid_scatter(x, w, torch.zeros(6, dtype=torch.long), 2, k=K)
    assert not valid[0]
    assert not valid[1]  # empty segment
    assert torch.isfinite(mu).all()
    assert_on_manifold(mu)  # invalid rows are the origin, still on-manifold


def test_n4_constraint_precision_at_scale():
    # No clamp may degrade the exact rescale: even for far-out points the
    # constraint must hold to float32 precision relative to time^2.
    x = random_points(50, scale=10.0, seed=4)
    conv = LorentzGraphConv(AMB, k=K, edge_dim=EDGE_DIM)
    ei, ea = random_graph(50, seed=4)
    out = conv(x, ei, ea)
    assert_on_manifold(out)


def test_n5_logits_finite_assertion_runs():
    # DEBUG_ASSERTS is on via fixture; a normal forward must not trip it.
    x = random_points(12)
    ei, ea = random_graph(12, seed=5)
    conv = LorentzGraphConv(AMB, k=K, edge_dim=EDGE_DIM)
    conv(x, ei, ea)


def test_n6_sq_lorentz_loss_drop_in():
    gnn = LorentzGNN(in_features=16, dim=DIM, num_layers=2, k=K, drop_edge=0.0)
    head = HyperbolicProjectionHead(DIM, 128, 64, k=K, already_lifted=True)
    x = torch.randn(12, 16)
    ei, ea = random_graph(12, seed=6)
    batch = torch.tensor([0] * 6 + [1] * 6)
    z = head(gnn(x, ei, ea, batch, num_graphs=2))
    assert_on_manifold(z)
    for kind in ("geodesic", "sq_lorentz"):
        loss = HyperbolicNTXent(temperature=0.3, k=K, kind=kind)(z, z.roll(0, 0))
        assert torch.isfinite(loss)


# ------------------------------------------------------- invariance and grads

def test_permutation_invariance_of_readout():
    torch.manual_seed(7)
    gnn = LorentzGNN(in_features=16, dim=DIM, num_layers=3, k=K, drop_edge=0.0).eval()
    n = 14
    x = torch.randn(n, 16)
    ei, ea = random_graph(n, seed=7)
    batch = torch.zeros(n, dtype=torch.long)
    out = gnn(x, ei, ea, batch, num_graphs=1)

    perm = torch.randperm(n)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    out_p = gnn(x[perm], inv[ei], ea, batch, num_graphs=1)
    assert torch.allclose(out, out_p, atol=1e-4), (out - out_p).abs().max()


def test_gradient_finiteness():
    torch.manual_seed(8)
    gnn = LorentzGNN(in_features=16, dim=DIM, num_layers=4, k=K, drop_edge=0.0)
    head = HyperbolicProjectionHead(DIM, 128, 64, k=K, already_lifted=True)
    x = torch.randn(20, 16)
    x[3] = x[4]  # coincident features: near-zero distances downstream
    ei, ea = random_graph(20, extra_self_loops=3, seed=8)
    batch = torch.tensor([0] * 10 + [1] * 10)
    z = head(gnn(x, ei, ea, batch, num_graphs=2))
    loss = HyperbolicNTXent(temperature=0.3, k=K)(z[:1], z[1:])
    loss.backward()
    for name, p in list(gnn.named_parameters()) + list(head.named_parameters()):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


# ------------------------------------------------------------- shapes / edges

@pytest.mark.parametrize("num_graphs", [1, 2, 256])
def test_batch_sizes(num_graphs):
    torch.manual_seed(9)
    gnn = LorentzGNN(in_features=16, dim=32, num_layers=2, k=K)
    xs, eis, eas, bs = [], [], [], []
    off = 0
    for g in range(num_graphs):
        n = 3 + g % 5
        ei, ea = random_graph(n, extra_self_loops=1, seed=g)
        xs.append(torch.randn(n, 16))
        eis.append(ei + off)
        eas.append(ea)
        bs.append(torch.full((n,), g, dtype=torch.long))
        off += n
    out = gnn(torch.cat(xs), torch.cat(eis, 1), torch.cat(eas), torch.cat(bs),
              num_graphs=num_graphs)
    assert out.shape == (num_graphs, 33)
    assert_on_manifold(out, abs_tol=I1_TOL)


def test_single_node_no_edges():
    gnn = LorentzGNN(in_features=16, dim=32, num_layers=2, k=K).eval()
    x = torch.randn(1, 16)
    ei = torch.zeros(2, 0, dtype=torch.long)
    ea = torch.zeros(0, EDGE_DIM)
    out = gnn(x, ei, ea, torch.zeros(1, dtype=torch.long), num_graphs=1)
    assert_on_manifold(out)


def test_self_loops_only():
    gnn = LorentzGNN(in_features=16, dim=32, num_layers=2, k=K).eval()
    x = torch.randn(2, 16)
    ei = torch.tensor([[0, 1], [0, 1]])  # only self-loops
    ea = torch.rand(2, EDGE_DIM)
    out = gnn(x, ei, ea, torch.zeros(2, dtype=torch.long), num_graphs=1)
    assert_on_manifold(out)


def test_scatter_softmax_isolated_segments():
    logits = torch.tensor([0.0, 1.0, 2.0])
    index = torch.tensor([0, 0, 2])  # segment 1 empty
    alpha = scatter_softmax(logits, index, 3)
    assert torch.isfinite(alpha).all()
    assert abs(float(alpha[:2].sum()) - 1.0) < 1e-6
    assert abs(float(alpha[2]) - 1.0) < 1e-6


# --------------------------------------------------------------- knot data

def test_knot_diagram_pipeline():
    knotclr = pytest.importorskip("knotclr.census")
    census = knotclr.load_census(max_crossings=8)
    diagrams = [k.diagram() for k in census[:16]]
    batch = collate_graphs(diagrams)
    assert batch.x.shape[1] == 16
    assert batch.edge_attr.shape[1] == EDGE_DIM
    # 4-regular: every crossing has 4 incident darts -> in-degree 4
    deg = torch.zeros(batch.x.shape[0]).index_add_(
        0, batch.edge_index[1], torch.ones(batch.edge_index.shape[1]))
    assert (deg == 4).all()

    gnn = LorentzGNN(in_features=16, dim=DIM, num_layers=4, k=K).eval()
    out = gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
              num_graphs=batch.num_graphs, virtual_mask=batch.virtual_mask)
    assert out.shape == (16, AMB)
    assert_on_manifold(out, abs_tol=I1_TOL)


def test_unknot_virtual_node():
    batch = collate_graphs([None])
    assert batch.virtual_mask.all()
    gnn = LorentzGNN(in_features=16, dim=32, num_layers=2, k=K).eval()
    out = gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
              num_graphs=1, virtual_mask=batch.virtual_mask)
    assert_on_manifold(out)


def test_r1_move_creates_self_loop_and_runs():
    knotclr_d = pytest.importorskip("knotclr.diagram")
    R = pytest.importorskip("knotclr.reidemeister")
    trefoil = [(1, 4, 2, 5), (3, 6, 4, 1), (5, 2, 6, 3)]
    diag = knotclr_d.Diagram.from_pd(trefoil)
    moves = [m for m in R.enumerate_moves(diag, ("r1+",))]
    kinked = R.apply_move(diag, moves[0])
    x, ei, ea, virt = diagram_to_graph(kinked)
    assert not virt
    # the R1 kink is a self-loop arc in the crossing graph
    assert (ei[0] == ei[1]).any()
    gnn = LorentzGNN(in_features=16, dim=32, num_layers=2, k=K).eval()
    out = gnn(x, ei, ea, torch.zeros(x.shape[0], dtype=torch.long), num_graphs=1)
    assert_on_manifold(out)
