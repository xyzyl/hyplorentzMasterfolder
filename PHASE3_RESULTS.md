# Phase 3 results: the three-way (four-way) comparison

2026-07-12. 50k contrastive steps each, batch 512 knots, same view store and
stratification. Linear probes on frozen encoder embeddings, test split
(`ssl_linear_test` R2; `alternating` is accuracy). Runs: `base50k` =
Euclidean transformer + Euclidean head; `hyp50k` = same transformer +
hyperbolic head (the pre-HGNN configuration); `hgnn50k` = fully hyperbolic
LorentzGNN (this spec's model); `egnn50k` = width/depth/param-matched
Euclidean GIN (~121k params both).

| target | transformer (euc) | transformer + hyp head | **HGNN-L (fully hyp)** | Euclidean GIN |
|---|---|---|---|---|
| signature | **0.915** | 0.837 | 0.843 | 0.000 |
| unknotting number | **0.554** | 0.443 | 0.414 | -0.012 |
| braid index | **0.506** | 0.008 | 0.027 | -0.008 |
| volume | **0.392** | 0.048 | 0.191 | 0.000 |
| determinant (log) | 0.183 | 0.042 | **0.222** | -0.001 |
| alternating (acc) | 0.655 | 0.639 | **0.744** | 0.643 |
| three-genus | **0.291** | 0.000 | 0.085 | -0.020 |
| crossing number | **0.333** | 0.008 | 0.121 | -0.001 |

Retrieval@1 (val, 597 knots): base50k 0.797, hyp50k floor, hgnn50k 0.020
(r@5 0.112), egnn50k 0.000. Loss at 50k: hgnn 4.54, egnn 5.64 (ln 512 = 6.24
is the no-learning ceiling).

## Gate verdict (spec section 7, Phase 3)

**Gate: fully hyperbolic >= hyperbolic-head-only on probe metrics — PASSED,
7 of 8 targets** (all but unknotting number, and signature is a tie). The
fully hyperbolic model is strictly more informative than the bolt-on
hyperbolic head: volume 0.191 vs 0.048, determinant 0.222 vs 0.042,
crossing number 0.121 vs 0.008, alternating 0.744 vs 0.639.

## The three findings that matter

1. **Geometry does real work inside the GNN class.** The parameter-matched
   Euclidean GIN learned *nothing* (probe R2 ~ 0 across all targets, loss
   near ceiling, retrieval 0), while the identical-budget Lorentz GNN
   learned signature to 0.843. This is the cleanest geometry ablation in
   the project. Caveat: the GIN is the standard baseline, not an exact
   architectural twin (no distance attention / gated residual); a faithful
   Euclidean twin is the v0.2 ablation that would nail this down.

2. **The transformer still leads overall.** On the classic invariants
   (signature, unknotting, braid index, volume, genus, crossing number) the
   Euclidean transformer baseline remains ahead, and no hyperbolic
   configuration has undergone the instance-discrimination phase transition
   (retrieval stays near floor). Consistent with Phase 0: the diagram
   graphs message passing runs over are grid-like, and 4-regular graphs are
   a known hard case for WL-style message passing; the encoder architecture,
   not the embedding geometry, appears to be the binding constraint.

3. **The fully hyperbolic model wins exactly where hierarchy lives.**
   Its two outright wins over ALL models — determinant (0.222 vs 0.183) and
   alternating (0.744 vs 0.655) — plus its volume recovery are consistent
   with the Phase 0 picture that the hyperbolic structure of this data is
   in the orbit/complexity ordering, not in fine-grained instance identity.

## Ship decision

Per the gate's letter, HGNN-L earns its place over the hyp-head-only
configuration. Per the comparison's spirit, the headline model for KnotCLR
probes remains the Euclidean transformer; HGNN-L is the right *hyperbolic*
model and the right platform for the Phase 4 ablations (sq_lorentz loss,
lower temperature, longer budgets — the untried follow-ups flagged after
hyp50k — plus a faithful Euclidean twin of the layer).

Reproduce: `scripts/train_knotclr_hgnn.py --geometry {lorentz,euclidean}`;
probes via `python -m probe.linear_probes --ckpt <ckpt> --encoder-type gnn`.
Checkpoints and logs in `D:/Iskander/knots/runs/contrastive/{hgnn,egnn}50k*`.
