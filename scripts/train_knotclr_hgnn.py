"""Phase 3 entrypoint (HGNN-L spec section 7): KnotCLR contrastive training
with the fully hyperbolic Lorentz GNN as the encoder.

Reuses the knots project's view store and batching (stored token views ARE
PD codes, decoded here straight into graph batches; gauge randomization is a
free graph-isomorphism augmentation for a GNN). Two geometries:

  --geometry lorentz    LorentzGNN -> HyperbolicProjectionHead(already_lifted)
                        -> HyperbolicNTXent (fully hyperbolic; fp32 only)
  --geometry euclidean  width/depth-matched Euclidean GIN -> MLP head
                        -> cosine NT-Xent (comparison (i) of the spec's
                        three-way; comparison (ii) is runs/contrastive/hyp50k)

Run with the knots venv from the knots repo root:

    D:/Iskander/knots/.venv/Scripts/python.exe \
        D:/Iskander/hyplorentzMasterfolder/scripts/train_knotclr_hgnn.py \
        --geometry lorentz --steps 50000 --name hgnn50k
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

KNOTS_ROOT = Path("D:/Iskander/knots")
sys.path.insert(0, str(KNOTS_ROOT))

from knotclr.runlog import RunLog  # noqa: E402
from knotclr.tokenize import ARC_BASE, SEP, XPOS  # noqa: E402
from train.dataset import ViewStore, contrastive_batches  # noqa: E402

from hyplorentz import (  # noqa: E402
    HyperbolicNTXent,
    HyperbolicProjectionHead,
    LorentzGNN,
    lorentz_distance,
)

EDGE_DIM = 8
IN_FEATURES = 16
POS_DIM = 8


# ---------------------------------------------------- tokens -> graph batch

def tokens_to_graphs(tokens: np.ndarray):
    """Vectorized batch decode: token rows [B, SEQ_LEN] -> graph tensors.

    Layout per row: CLS (SIGN a b c d) x n SEP PAD... Arc labels 1..2n, each
    appearing exactly twice across the row; matching the two occurrences
    yields the arcs (edges). Slot = column 0..3 inside the crossing tuple.
    Node features: sign channel + sinusoidal strand position (under-in label
    a normalized by 2n; the PD gauge shift makes this a free augmentation).
    """
    B, _ = tokens.shape
    seps = np.argmax(tokens == SEP, axis=1)
    ns = (seps - 1) // 5
    total = int(ns.sum())
    offsets = np.concatenate([[0], np.cumsum(ns)[:-1]])

    x = np.zeros((total, IN_FEATURES), np.float32)
    batch = np.repeat(np.arange(B), ns)

    srcs, dsts, attrs = [], [], []
    freqs = math.pi * (2.0 ** np.arange(POS_DIM // 2))
    for b in range(B):
        n = int(ns[b])
        if n == 0:
            continue
        body = tokens[b, 1 : 1 + 5 * n].reshape(n, 5).astype(np.int64)
        sign = np.where(body[:, 0] == XPOS, 1.0, -1.0)
        labels = body[:, 1:5] - ARC_BASE  # 0..2n-1
        o = int(offsets[b])
        x[o : o + n, 0] = sign
        pos = (labels[:, 0] / (2 * n))[:, None]  # under-in label = strand rank
        x[o : o + n, 1 : 1 + POS_DIM : 2] = np.sin(freqs * pos)
        x[o : o + n, 2 : 2 + POS_DIM : 2] = np.cos(freqs * pos)

        flat = labels.ravel()
        order = np.argsort(flat, kind="stable")  # the two darts of each arc are adjacent
        d1, d2 = order[0::2], order[1::2]       # 2n arcs
        c1, s1 = d1 // 4 + o, d1 % 4
        c2, s2 = d2 // 4 + o, d2 % 4
        m = 2 * n
        e = np.zeros((2 * m, EDGE_DIM), np.float32)
        e[np.arange(m), s1] = 1.0
        e[np.arange(m), 4 + s2] = 1.0
        e[m + np.arange(m), s2] = 1.0
        e[m + np.arange(m), 4 + s1] = 1.0
        srcs.append(np.concatenate([c1, c2]))
        dsts.append(np.concatenate([c2, c1]))
        attrs.append(e)

    edge_index = torch.from_numpy(
        np.stack([np.concatenate(srcs), np.concatenate(dsts)]).astype(np.int64)
    )
    return (
        torch.from_numpy(x),
        edge_index,
        torch.from_numpy(np.concatenate(attrs)),
        torch.from_numpy(batch.astype(np.int64)),
        B,
    )


# ------------------------------------------------------- euclidean baseline

class EuclideanGIN(nn.Module):
    """Width/depth-matched Euclidean encoder for comparison (i)."""

    def __init__(self, in_features=IN_FEATURES, dim=64, num_layers=4, edge_dim=EDGE_DIM):
        super().__init__()
        self.embed = nn.Linear(in_features, dim)
        self.eps = nn.Parameter(torch.zeros(num_layers))
        # message = [h_src, edge_attr] aggregated by sum; update mixes it
        # with the (1+eps)-scaled self state, GIN-style.
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * dim + edge_dim, 2 * dim), nn.ReLU(),
                          nn.Linear(2 * dim, dim))
            for _ in range(num_layers)
        )
        self.dim = dim

    def forward(self, x, edge_index, edge_attr, batch, num_graphs):
        h = self.embed(x)
        src, dst = edge_index
        for i, mlp in enumerate(self.mlps):
            msg = torch.cat([h[src], edge_attr], dim=-1)
            agg = h.new_zeros(h.shape[0], msg.shape[1]).index_add_(0, dst, msg)
            h = mlp(torch.cat([(1 + self.eps[i]) * h, agg], dim=-1))
        out = h.new_zeros(num_graphs, h.shape[1]).index_add_(0, batch, h)
        cnt = h.new_zeros(num_graphs).index_add_(0, batch, h.new_ones(h.shape[0]))
        return out / cnt.clamp(min=1).unsqueeze(-1)


def nt_xent(za, zb, temp):
    za = nn.functional.normalize(za, dim=1)
    zb = nn.functional.normalize(zb, dim=1)
    logits = za @ zb.t() / temp
    target = torch.arange(len(za), device=za.device)
    return 0.5 * (nn.functional.cross_entropy(logits, target)
                  + nn.functional.cross_entropy(logits.t(), target))


# ----------------------------------------------------------------- retrieval

@torch.no_grad()
def retrieval_eval(encode_fn, store, device, rng, geometry, k, max_knots=2000):
    classes = np.arange(store.n_knots)
    if len(classes) > max_knots:
        classes = rng.choice(classes, size=max_knots, replace=False)
    rows_q, rows_g = [], []
    for cls in classes:
        s, e = store.class_starts[cls], store.class_ends[cls]
        if e - s >= 2:
            a, b = rng.choice(e - s, size=2, replace=False)
        else:
            a = b = 0
        rows_q.append(s + a)
        rows_g.append(s + b)

    def embed(rows):
        out = []
        for i in range(0, len(rows), 512):
            z = encode_fn(store.tokens[np.array(rows[i : i + 512])])
            out.append(z.cpu())
        return torch.cat(out)

    q, g = embed(rows_q), embed(rows_g)
    if geometry == "lorentz":
        sim = -lorentz_distance(q.unsqueeze(1), g.unsqueeze(0), k=k)
    else:
        sim = nn.functional.normalize(q, dim=1) @ nn.functional.normalize(g, dim=1).t()
    rank_of_true = (sim >= sim.diag()[:, None]).sum(1)
    return {"r@1": float((rank_of_true <= 1).float().mean()),
            "r@5": float((rank_of_true <= 5).float().mean()),
            "n": len(classes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["lorentz", "euclidean"], default="lorentz")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch-knots", type=int, default=256)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--kind", choices=["geodesic", "sq_lorentz"], default="geodesic")
    ap.add_argument("--drop-edge", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--views", default=str(KNOTS_ROOT / "data/views"))
    ap.add_argument("--db", default=str(KNOTS_ROOT / "data/knots.sqlite"))
    ap.add_argument("--out", default=str(KNOTS_ROOT / "runs/contrastive"))
    ap.add_argument("--name", default="hgnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    if args.temp is None:
        args.temp = 0.3 if args.geometry == "lorentz" else 0.1

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_store = ViewStore(args.views, args.db, "train")
    val_store = ViewStore(args.views, args.db, "val")
    cn_map = train_store.labels("crossing_number")
    uniq = sorted(train_store.knot_index, key=train_store.knot_index.get)
    strat = np.array([cn_map[kid][0] for kid in uniq], np.int64)

    if args.geometry == "lorentz":
        encoder = LorentzGNN(IN_FEATURES, args.dim, num_layers=args.layers,
                             k=args.k, edge_dim=EDGE_DIM,
                             drop_edge=args.drop_edge).to(device)
        head = HyperbolicProjectionHead(args.dim, 128, args.dim, k=args.k,
                                        already_lifted=True).to(device)
        crit = HyperbolicNTXent(temperature=args.temp, k=args.k, kind=args.kind)
    else:
        encoder = EuclideanGIN(IN_FEATURES, args.dim, args.layers).to(device)
        head = nn.Sequential(nn.Linear(args.dim, 128), nn.ReLU(),
                             nn.Linear(128, args.dim)).to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    from geoopt.optim import RiemannianAdam

    # LorentzLinear log time-scales must not be weight-decayed (see the
    # knots train/contrastive.py hyperbolic path for the rationale).
    scale_params = [p for n_, p in
                    list(encoder.named_parameters()) + list(head.named_parameters())
                    if n_.endswith(".scale")]
    scale_ids = {id(p) for p in scale_params}
    opt = RiemannianAdam(
        [{"params": [p for p in params if id(p) not in scale_ids]},
         {"params": scale_params, "weight_decay": 0.0}],
        lr=args.lr, weight_decay=0.05,
    )

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"])
        head.load_state_dict(ck["proj"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    def encode_tokens(tok_rows, train=False):
        x, ei, ea, batch, ng = tokens_to_graphs(tok_rows)
        x, ei, ea, batch = x.to(device), ei.to(device), ea.to(device), batch.to(device)
        return encoder(x, ei, ea, batch, num_graphs=ng)

    run = RunLog("contrastive", args.name, vars(args))
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in params)
    print(f"geometry {args.geometry}, params {n_params}, device {device}", flush=True)

    batches = contrastive_batches(train_store, args.batch_knots, rng, strat)
    history = {}
    t0 = time.time()
    loss_acc, loss_n = 0.0, 0
    for step in range(start_step + 1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ta, tb, _ = next(batches)
        # fp32 end to end for lorentz (spec: no half precision on manifold)
        za = head(encode_tokens(ta, train=True))
        zb = head(encode_tokens(tb, train=True))
        loss = crit(za, zb) if args.geometry == "lorentz" else nt_xent(za, zb, args.temp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        loss_acc += loss.item()
        loss_n += 1

        if step % 500 == 0:
            rate = (step - start_step) / (time.time() - t0)
            print(f"step {step} loss {loss_acc/loss_n:.4f} ({rate:.1f} it/s)", flush=True)
            loss_acc, loss_n = 0.0, 0
        if step % args.eval_every == 0 or step == args.steps:
            encoder.eval()
            with torch.no_grad():
                ev_tr = retrieval_eval(lambda t: encode_tokens(t), train_store,
                                       device, rng, args.geometry, args.k)
                ev_va = retrieval_eval(lambda t: encode_tokens(t), val_store,
                                       device, rng, args.geometry, args.k)
            encoder.train()
            print(f"step {step} retrieval train {ev_tr} val {ev_va}", flush=True)
            history[step] = {"train": ev_tr, "val": ev_va}
            run.log_metrics({"history": history})
        if step % args.ckpt_every == 0 or step == args.steps:
            state = {"encoder": encoder.state_dict(), "proj": head.state_dict(),
                     "opt": opt.state_dict(), "step": step, "args": vars(args)}
            torch.save(state, out_dir / f"step{step:07d}.pt")
            torch.save(state, out_dir / "last.pt")

    print("done", flush=True)


if __name__ == "__main__":
    main()
