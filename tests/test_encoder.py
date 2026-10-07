import numpy as np
import pytest

from molecular_compiler import encoder as E


def _planted(animals=12, pairs=80, rank=3, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0, 1, pairs),
        rng.normal(0, 1, (pairs, rank)),
        rng.normal(0, 1, (animals, rank)),
    )


def test_M13_R3_ii_basis_decoder_recovers_planted_latent():
    P, V, U = _planted()
    rng = np.random.default_rng(1)
    y = P + V @ U[4] + rng.normal(0, 0.05, len(P))
    u = E.decode_with_basis(V, P, y, lam=0.1)
    assert np.linalg.norm(u - U[4]) < 0.15
    ok, margin = E.nearest_individual(u, 4, U)
    assert ok and margin > 0


def test_M13_R3_ii_basis_decoder_uses_observed_pairs_and_skips_nan():
    P, V, U = _planted()
    pairs = np.arange(0, 80, 2)
    y = (P + V @ U[2])[pairs]
    y[3] = np.nan
    u = E.decode_with_basis(V, P, y, pairs=pairs, lam=1e-6)
    np.testing.assert_allclose(u, U[2], atol=1e-4)
    with pytest.raises(ValueError):
        E.decode_with_basis(V, P, y[:-1], pairs=pairs)


def test_M13_R3_ii_readout_maps_activity_features_to_latent():
    rng = np.random.default_rng(2)
    U = rng.normal(0, 1, (200, 3))
    M = rng.normal(0, 1, (3, 20))
    X = U @ M + rng.normal(0, 0.1, (200, 20)) + 5.0
    readout = E.fit_readout(X[:150], U[:150], lam=1.0)
    err = np.linalg.norm(readout.decode(X[150:]) - U[150:], axis=1).mean()
    assert err < 0.3 < np.linalg.norm(U[150:], axis=1).mean()
    with pytest.raises(ValueError):
        E.fit_readout(X, U[:-1])


def test_M13_R3_ii_nearest_individual_fails_for_drift_to_another_animal():
    _, _, U = _planted()
    ok, _ = E.nearest_individual(U[0], 0, U)
    assert ok
    ok, margin = E.nearest_individual(0.2 * U[0] + 0.8 * U[1], 0, U)
    assert not ok and margin < 0
    passes, margins = E.sustains_identity([U[0], U[0] + 0.01, U[1]], 0, U)
    assert not passes and margins[0] > 0 > margins[2]
