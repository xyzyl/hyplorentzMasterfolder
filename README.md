# hyplorentz — fully hyperbolic projection head (Lorentz model)

A minimal, tested implementation of a **fully hyperbolic** SimCLR-style
projection head on the **hyperboloid (Lorentz) model**, built on `geoopt`.

- **Lorentz model, curvature -1/k**: numerically far more stable than the
  Poincaré ball in float32 — matters on an 8GB RTX 4060.
- **Fully hyperbolic linear layers** (Chen et al. 2021, *Fully Hyperbolic
  Neural Networks*): no tangent-space round trips. The layer maps in ambient
  space, learns the time coordinate, and analytically rescales the spatial
  part so `<z, z>_L = -k` holds *exactly* at every layer.
- **Hyperbolic NT-Xent** loss: SimCLR's InfoNCE with negative geodesic (or
  squared Lorentzian) distance as similarity.

## Install

```bash
pip install torch geoopt
```

## Usage

```python
import torch
from geoopt.optim import RiemannianAdam
from hyplorentz import HyperbolicProjectionHead, HyperbolicNTXent

encoder = MyKnotTransformer()          # any Euclidean encoder, output (B, 256)
head = HyperbolicProjectionHead(
    in_features=256, hidden_features=128, out_features=64, k=1.0, num_layers=2,
)
criterion = HyperbolicNTXent(temperature=0.3, k=1.0, kind="geodesic")

# RiemannianAdam == Adam for Euclidean params, Riemannian updates for any
# manifold params. Safe to use for the whole model.
opt = RiemannianAdam(list(encoder.parameters()) + list(head.parameters()), lr=3e-4)

for x1, x2 in loader:                  # two augmented views (e.g. Reidemeister moves)
    z1, z2 = head(encoder(x1)), head(encoder(x2))   # points on the hyperboloid
    loss = criterion(z1, z2)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
```

Embeddings have `out_features + 1` ambient coordinates. For downstream use
(kNN, probing), compare with `pairwise_lorentz_dist`, or map to the Poincaré
ball for visualization: `x_poincare = z_space / (z_time + sqrt(k))`.

## Verify

```bash
python sanity_check.py
```

Checks: (1) exact manifold constraint through the stack, (2) finite float32
gradients, (3) a toy SimCLR run where positives collapse together, and
(4) a 63-node binary tree embedded in **2-D**, where the hyperbolic head
reaches ~0.33 mean distortion vs ~0.57 for a matched Euclidean head — the
canonical "hyperbolic is doing its job" test.

## Numerical-stability notes (the stuff that actually bites)

- `arccosh` inputs are clamped to `>= 1 + 1e-6`; unclamped, float32 roundoff
  produces NaNs *and* infinite gradients at zero distance.
- The origin lift caps tangent norms at 10: `expmap0` involves `cosh(‖u‖)`,
  which loses all float32 precision long before it overflows at ‖u‖ ≈ 44.
- A BatchNorm before the lift keeps encoder features from shoving points
  toward the boundary early in training.
- Temperature ~0.3 (higher than the Euclidean-SimCLR 0.07 default) works
  better because hyperbolic distances are unbounded, unlike cosine similarity.
- `kind="sq_lorentz"` in the loss avoids `acosh` entirely (Law et al. 2019)
  — try it if you see instability at larger depths/curvatures.

## Files

- `hyplorentz/lorentz.py` — inner product, distances, origin lift, `LorentzLinear`, `LorentzCentroid`
- `hyplorentz/head.py` — `HyperbolicProjectionHead`
- `hyplorentz/loss.py` — `HyperbolicNTXent`, `pairwise_lorentz_dist`
- `sanity_check.py` — the four checks above
