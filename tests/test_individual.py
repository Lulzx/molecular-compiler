import numpy as np
import pytest

from molecular_compiler import individual as I


def _synthetic(
    animals=30,
    events=10,
    pairs_per_event=12,
    n_pairs=60,
    rank=2,
    drift=False,
    scale=1.0,
    seed=0,
):
    """Trials y = P_p + V_p . u_a + noise; each event samples its own pairs."""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, n_pairs)
    V = rng.normal(0, 1, (n_pairs, rank))
    rows = {"animal": [], "pair": [], "y": [], "position": []}
    for a in range(animals):
        u = scale * rng.normal(0, 1, rank)
        for e in range(events):
            current = -u if drift and e >= events // 2 else u
            pairs = rng.choice(n_pairs, pairs_per_event, replace=False)
            rows["animal"] += [a] * len(pairs)
            rows["pair"] += pairs.tolist()
            rows["y"] += (
                P[pairs] + V[pairs] @ current + rng.normal(0, 0.3, len(pairs))
            ).tolist()
            rows["position"] += [e] * len(pairs)
    table = {k: np.asarray(v) for k, v in rows.items()}
    table["y"] = table["y"].astype(float)
    return table, {a: events for a in range(animals)}


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(I, "GRID", (1.0,))
    monkeypatch.setattr(I, "ALS_ITERATIONS", 15)


def _run(table, events):
    return I.run(
        table, events, folds=3, ranks=(1, 2, 4), samples=200, log=lambda *_: None
    )


def test_I0_lookup_returns_zero_for_pairs_unseen_in_training():
    table = (np.array([2, 5, 9]), np.array([[1.0], [2.0], [3.0]]))
    np.testing.assert_array_equal(
        I.lookup(table, np.array([5, 4, 9, 10]))[:, 0], [2, 0, 3, 0]
    )
    assert I.lookup((np.array([]), np.zeros((0,))), np.array([1])).tolist() == [0.0]


def test_I0_splits_are_chronological_halves_and_alternate_events():
    table = {"animal": np.zeros(7, int), "position": np.arange(7)}
    fit, score = I.split_halves(table, np.arange(7), {0: 7}, "chronological")
    assert fit.tolist() == [0, 1, 2] and score.tolist() == [3, 4, 5, 6]
    fit, score = I.split_halves(table, np.arange(7), {0: 7}, "interleaved")
    assert fit.tolist() == [0, 2, 4, 6] and score.tolist() == [1, 3, 5]


def test_I0_folds_match_held_out_animal_analysis():
    animals = np.arange(20)
    rng = np.random.default_rng(I.SEED)
    expected = dict(zip(rng.permutation(animals), np.arange(20) % I.FOLDS))
    assert I.assign_folds(animals) == expected


def test_I0_planted_individual_latent_passes_I0_1_and_I0_2(small):
    table, events = _synthetic()
    result = _run(table, events)
    gates = result["gates"]
    assert gates["I0-1"]["pass"] and gates["I0-2"]["pass"]
    # Ridge shrinkage can leave the next rank within one SE of the best.
    assert result["r_star"] in (2, 4)
    chron = result["gain_over_pop"]["chronological"]
    assert chron["factor-2"]["mean"] > 3 * chron["factor-1"]["mean"]
    assert gates["I0-3"]["verdict"] == "durable_over_session"


def test_I0_no_individual_state_fails_I0_1(small):
    table, events = _synthetic(scale=0.0)
    assert not _run(table, events)["gates"]["I0-1"]["pass"]


def test_I0_drifting_latent_is_flagged_by_I0_3(small):
    table, events = _synthetic(drift=True)
    assert _run(table, events)["gates"]["I0-3"]["verdict"] == "drifts_within_session"
