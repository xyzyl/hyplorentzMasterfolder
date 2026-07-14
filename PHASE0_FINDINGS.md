# Phase 0 findings: δ-hyperbolicity of the knot data

**Verdict: conditional PASS** (2026-07-12, `scripts/delta_hyperbolicity.py`,
full numbers in `phase0_delta_report.json`). The spec's gate — δ/diam ≤ 0.25
on at least one of the two graph families — is met by the Reidemeister-orbit
family, at the boundary. The diagram-graph family fails decisively.

## (a) Knot-diagram crossing graphs: NOT hyperbolic

Exact four-point δ and exact diameter, per graph, on views drawn from the
actual KnotCLR training distribution plus a size scaling sweep:

| view size | δ mean | diam mean | δ/diam median |
|-----------|--------|-----------|----------------|
| ~24 (training range) | 1.98 | 5.78 | 0.333 |
| ~48 | 2.71 | 9.35 | 0.300 |
| ~64 | 3.02 | 10.90 | 0.273 |
| ~96 | 3.52 | 13.42 | 0.269 |

δ grows with diameter across a 4× size sweep — grid-like scaling, consistent
with theory (4-regular planar maps are quadrangulation-like). The graphs the
GNN message-passes over are not tree-like, and no encoder geometry changes
that.

## (b) Reidemeister-orbit neighborhood graphs: borderline hyperbolic

Nodes = diagrams up to oriented isomorphism, edges = single R-moves. Key
device: with a crossing cap the component of a knot type is **finite** and
can be enumerated completely (no ball-truncation bias):

| component | nodes | complete? | δ | diam | δ/diam |
|-----------|-------|-----------|---|------|--------|
| 3_1, n≤7 | 53,863 | yes | 2.5 | 10 | **0.25** |
| 4_1, n≤7 | 17,700 | yes | 3.0 | 10 | 0.30 |
| 3_1, n≤8/9 | 200k cap | no | 2.0 | 8 | 0.25 |
| 4_1, n≤8/9 | 200k cap | no | 2.0 | 8–10 | 0.20–0.25 |

(512 landmarks, 10⁴ sampled quadruples; truncated rows underestimate diam.)

Unlike (a), δ stays flat at 2–3 as the explored region grows — the
qualitative signature of hyperbolicity — but the ratio sits at the gate line
rather than below it. Read: the orbit structure is *weakly* tree-like, not
emphatically so.

## Implications for HGNN-L

1. The hierarchy, if the model can exploit it, lives at the **orbit/embedding
   level** (what the contrastive loss sees), not inside individual diagram
   graphs (what message passing sees). This matches the spec §1 motivation
   and is consistent with the 50k hyp-head run's outcome (coarse cluster
   structure emerged; instance discrimination did not).
2. Expectations should be calibrated: this is not a "trees embed with low
   distortion" slam dunk. The Phase 3 three-way comparison (fully hyperbolic
   vs Euclidean-encoder+hyperbolic-head vs fully Euclidean) is the empirical
   arbiter, and the Phase 3 gate ("ship the simpler model if not better")
   carries real weight.
3. The decisive (a)-family negative result is worth reporting in the KnotCLR
   appendix regardless of Phase 3's outcome.
