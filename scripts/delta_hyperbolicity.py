"""Phase 0 gate (HGNN-L spec section 7): Gromov delta-hyperbolicity of the knot data.

Two graph families are measured:

(a) Knot-diagram crossing graphs -- nodes = crossings, edges = arcs -- drawn
    from the actual KnotCLR training view distribution (knotclr.views.random_view,
    n in [14, 24]) plus the raw census diagrams. Graphs are small (<= 24 nodes),
    so delta and diameter are computed EXACTLY over all quadruples (stronger
    than the spec's 10^4-sample requirement).

(b) Reidemeister-orbit neighborhood graphs -- nodes = diagrams up to oriented
    isomorphism (canonical_code), edges = single R-moves. The orbit is infinite,
    so we take a BFS ball per base knot (crossing cap N_CAP, node cap NODE_CAP),
    BFS from <= N_LANDMARKS landmarks, and sample N_QUAD quadruples among
    landmarks. Distances within the sampled ball upper-bound true orbit
    distances (shortcuts may leave the ball), so the reported delta is an
    estimate; this is the standard sampled-delta protocol.

Gate: delta/diam <= 0.25 on at least one family. Run with the knots venv:

    D:/Iskander/knots/.venv/Scripts/python.exe scripts/delta_hyperbolicity.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

KNOTS_ROOT = Path("D:/Iskander/knots")
if str(KNOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(KNOTS_ROOT))

from knotclr.census import load_census  # noqa: E402
from knotclr.diagram import Diagram  # noqa: E402
from knotclr import reidemeister as R  # noqa: E402
from knotclr.views import random_view  # noqa: E402

SEED = 0
N_VIEWS = 400
# (a) scaling sweep: does delta grow with diameter (grid-like, bad) or
# saturate (hyperbolic, good)? Sizes beyond the training range on purpose.
VIEW_SCALES = (24, 48, 64, 96)
N_VIEWS_PER_SCALE = 60
# (b) crossing-capped orbit components are FINITE (finitely many diagrams of
# a knot type with <= n_cap crossings); enumerate them exactly.
ORBIT_N_CAPS = (7, 8, 9)
NODE_CAP = 200_000  # safety valve; 'truncated': True means it was hit
N_LANDMARKS = 512
N_QUAD = 10_000
ORBIT_BASES = ("3_1", "4_1")
GATE = 0.25

OUT_PATH = Path(__file__).resolve().parent.parent / "phase0_delta_report.json"


# --------------------------------------------------------------- graph utils

def crossing_graph_edges(diag: Diagram) -> list[tuple[int, int]]:
    """Edges of the crossing graph: one per arc (theta orbit), self-loops kept."""
    return [
        (d >> 2, diag.theta[d] >> 2)
        for d in range(4 * diag.n)
        if d < diag.theta[d]
    ]


def bfs_all_pairs(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    D = np.full((n, n), -1, dtype=np.int32)
    for s in range(n):
        dist = D[s]
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u] + 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = du
                    q.append(v)
    return D


def exact_delta(D: np.ndarray) -> float:
    """Four-point delta over ALL quadruples, chunked over the first point
    so memory stays O(n^3). Exact for any n where O(n^4) time is acceptable
    (fine up to n ~ 100)."""
    Df = D.astype(np.float64)
    best = 0.0
    for a in range(Df.shape[0]):
        s1 = Df[a, :, None, None] + Df[None, :, :]     # d(a,b) + d(c,d)
        s2 = Df[a, None, :, None] + Df[:, None, :]     # d(a,c) + d(b,d)
        s3 = Df[a, None, None, :] + Df[:, :, None]     # d(a,d) + d(b,c)
        sums = np.stack([s1, s2, s3])
        sums.sort(axis=0)
        best = max(best, float(((sums[2] - sums[1]) / 2.0).max()))
    return best


def sampled_delta(Dll: np.ndarray, n_quad: int, rng: np.random.Generator) -> float:
    """Four-point delta over sampled quadruples of a landmark-landmark matrix."""
    m = Dll.shape[0]
    idx = rng.integers(0, m, size=(4, n_quad))
    a, b, c, d = idx
    s1 = Dll[a, b] + Dll[c, d]
    s2 = Dll[a, c] + Dll[b, d]
    s3 = Dll[a, d] + Dll[b, c]
    sums = np.sort(np.stack([s1, s2, s3]).astype(np.float64), axis=0)
    return float(((sums[2] - sums[1]) / 2.0).max())


# ------------------------------------------------- (a) diagram crossing graphs

def measure_diagram_graphs() -> dict:
    rng = random.Random(SEED)
    census = load_census()

    def stats(diagrams: list[Diagram]) -> dict:
        ratios, deltas, diams = [], [], []
        for diag in diagrams:
            if diag.n < 4:
                continue  # diam can be 1; ratio degenerate
            D = bfs_all_pairs(diag.n, crossing_graph_edges(diag))
            diam = int(D.max())
            delta = exact_delta(D)
            deltas.append(delta)
            diams.append(diam)
            ratios.append(delta / diam if diam > 0 else 0.0)
        r = np.array(ratios)
        return {
            "count": len(ratios),
            "delta_mean": float(np.mean(deltas)),
            "diam_mean": float(np.mean(diams)),
            "ratio_mean": float(r.mean()),
            "ratio_median": float(np.median(r)),
            "ratio_p90": float(np.percentile(r, 90)),
            "ratio_max": float(r.max()),
            "frac_leq_gate": float((r <= GATE).mean()),
        }

    print(f"[a] census diagrams: {len(census)} knots")
    census_stats = stats([k.diagram() for k in census])

    print(f"[a] generating {N_VIEWS} training-distribution views ...")
    bases = [rng.choice(census).diagram() for _ in range(N_VIEWS)]
    views = [random_view(b, rng) for b in bases]
    view_stats = stats(views)

    # Scaling sweep: if delta grows proportionally with diam, the graphs are
    # grid-like at every scale and the small-size ratio is not an artifact.
    scaling = {}
    for max_n in VIEW_SCALES:
        t0 = time.time()
        vs = [
            random_view(rng.choice(census).diagram(), rng,
                        min_n=max_n - 4, max_n=max_n)
            for _ in range(N_VIEWS_PER_SCALE)
        ]
        scaling[max_n] = stats(vs)
        scaling[max_n]["seconds"] = round(time.time() - t0, 1)
        print(f"[a] scale n~{max_n}: delta_mean={scaling[max_n]['delta_mean']:.2f} "
              f"diam_mean={scaling[max_n]['diam_mean']:.2f} "
              f"ratio_median={scaling[max_n]['ratio_median']:.3f}")

    return {"census": census_stats, "views": view_stats, "scaling": scaling}


# --------------------------------------------- (b) Reidemeister orbit balls

def orbit_ball(base: Diagram, n_cap: int, node_cap: int) -> tuple[list, dict]:
    """BFS ball in move space. Returns (adjacency list, info)."""
    key0 = base.canonical_code()
    index = {key0: 0}
    diagrams = [base]
    adj: list[set[int]] = [set()]
    q = deque([0])
    while q and len(diagrams) < node_cap:
        i = q.popleft()
        diag = diagrams[i]
        for move in R.enumerate_moves(diag):
            try:
                new = R.apply_move(diag, move)
            except Exception:
                continue
            if new.n > n_cap:
                continue
            key = new.canonical_code()
            j = index.get(key)
            if j is None:
                if len(diagrams) >= node_cap:
                    continue
                j = len(diagrams)
                index[key] = j
                diagrams.append(new)
                adj.append(set())
                q.append(j)
            if j != i:
                adj[i].add(j)
                adj[j].add(i)
    info = {"nodes": len(diagrams), "truncated": bool(q)}
    return [sorted(s) for s in adj], info


def measure_orbit_graphs() -> dict:
    census = {k.name: k for k in load_census()}
    nrng = np.random.default_rng(SEED)
    out = {}
    for name, n_cap in [(b, c) for b in ORBIT_BASES for c in ORBIT_N_CAPS]:
        t0 = time.time()
        base = census[name].diagram()
        adj, info = orbit_ball(base, n_cap, NODE_CAP)
        n = len(adj)
        lm = nrng.choice(n, size=min(N_LANDMARKS, n), replace=False)
        Dl = np.full((len(lm), n), -1, dtype=np.int32)
        for row, s in enumerate(lm):
            dist = Dl[row]
            dist[s] = 0
            q = deque([int(s)])
            while q:
                u = q.popleft()
                du = dist[u] + 1
                for v in adj[u]:
                    if dist[v] < 0:
                        dist[v] = du
                        q.append(v)
        assert (Dl >= 0).all(), "orbit ball not connected?"
        Dll = Dl[:, lm]
        delta = sampled_delta(Dll, N_QUAD, nrng)
        if len(lm) <= 150:
            delta = max(delta, exact_delta(Dll))
        diam = int(Dl.max())
        key = f"{name}@n<={n_cap}"
        out[key] = {
            **info,
            "landmarks": int(len(lm)),
            "delta": delta,
            "diam": diam,
            "ratio": delta / diam,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[b] {key}: {out[key]}")
    return out


def main() -> None:
    t0 = time.time()
    report = {
        "config": {
            "seed": SEED, "n_views": N_VIEWS, "orbit_n_caps": ORBIT_N_CAPS,
            "orbit_node_cap": NODE_CAP, "landmarks": N_LANDMARKS,
            "quadruples": N_QUAD, "gate": GATE,
        },
        "diagram_graphs": measure_diagram_graphs(),
        "orbit_graphs": measure_orbit_graphs(),
    }

    a_ratio = report["diagram_graphs"]["views"]["ratio_median"]
    b_ratios = {k: v["ratio"] for k, v in report["orbit_graphs"].items()}
    a_pass = a_ratio <= GATE
    # Verdict from the best-resolved (largest n_cap) component per base.
    top_cap = max(ORBIT_N_CAPS)
    b_top = {k: r for k, r in b_ratios.items() if k.endswith(f"n<={top_cap}")}
    b_pass = len(b_top) > 0 and all(r <= GATE for r in b_top.values())
    report["gate"] = {
        "diagram_graphs_median_ratio": a_ratio,
        "diagram_graphs_pass": a_pass,
        "orbit_graph_ratios": b_ratios,
        "orbit_graphs_pass": b_pass,
        "PASS": a_pass or b_pass,
    }

    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gate"], indent=2))
    print(f"total {time.time() - t0:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
