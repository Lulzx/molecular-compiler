"""Track I encoder: decode the durable state u from measured responses (M13-R3 ii).

Two ridge decoders into the Stage I0 latent space:

- `decode_with_basis`: a response vector over (responder, stimulated) pairs is
  modelled as P + V u (the I0 `factor-r` model), so u is a ridge solution on
  the observed pairs. No training animals are needed beyond V and P.
- `fit_readout`: a generic linear readout from activity-window features to u,
  fitted on windows with known u (for example emulations of training animals).

`nearest_individual` is the identity check of M13-R3 (ii): the decoded u must
be closer to the animal's own u than to the u of every other animal.
"""

from dataclasses import dataclass

import numpy as np


def decode_with_basis(V, P, y, pairs=None, lam=1.0):
    """Ridge u minimizing |y - P - V u|^2 + lam |u|^2 over the observed `pairs`.

    V [pairs, r] and P [pairs] cover all pairs; y holds the measurements at
    `pairs` (default: all, in order).
    """
    V, P, y = np.asarray(V, float), np.asarray(P, float), np.asarray(y, float)
    if pairs is not None:
        V, P = V[pairs], P[pairs]
    if len(y) != len(P):
        raise ValueError("y must have one value per observed pair")
    ok = np.isfinite(y)
    A, r = V[ok], y[ok] - P[ok]
    return np.linalg.solve(A.T @ A + lam * np.eye(V.shape[1]), A.T @ r)


@dataclass
class Readout:
    weights: np.ndarray  # [features, r]
    bias: np.ndarray  # [r]
    mean: np.ndarray  # feature means used to center inputs

    def decode(self, features):
        return (np.asarray(features, float) - self.mean) @ self.weights + self.bias


def fit_readout(features, latents, lam=1.0):
    """Ridge readout [n windows, d] -> [n windows, r]; the bias is unpenalized."""
    X, U = np.asarray(features, float), np.asarray(latents, float)
    if len(X) != len(U):
        raise ValueError("one latent per window")
    mean, bias = X.mean(axis=0), U.mean(axis=0)
    Xc = X - mean
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (U - bias))
    return Readout(W, bias, mean)


def nearest_individual(decoded, animal, latents):
    """Is `decoded` closer to latents[animal] than to every other animal's?

    Returns (passes, margin) with margin = distance to the nearest other
    animal minus distance to the animal's own u (positive passes).
    """
    d = np.linalg.norm(np.asarray(latents, float) - np.asarray(decoded, float), axis=1)
    margin = float(np.min(np.delete(d, animal)) - d[animal])
    return margin > 0, margin


def sustains_identity(decoded_checkpoints, animal, latents):
    """M13-R3 (ii): `nearest_individual` at every checkpoint in [0, H]."""
    margins = [nearest_individual(u, animal, latents)[1] for u in decoded_checkpoints]
    return all(m > 0 for m in margins), margins
