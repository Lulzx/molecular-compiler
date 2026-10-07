"""Track I, Stage I0: per-animal durable latent on recorded responses (spec 10.4).

Each off-diagonal trial of pair p = (responder, stimulated) in animal a is
predicted as P_p plus a per-animal deviation:

- `pop`: no deviation;
- `gain`: (u_a - 1) P_p with scalar u_a, ridge toward 1;
- `factor-r`: V_p . u_a, V fit by ridge ALS on training animals, u_a by ridge
  on the held-out animal's fit-half trials with V fixed;
- `swap-r`: factor-r scored with another held-out animal's u (derangement).

Outer folds are the `leave_animal_out` folds of `held_out_animals`.
Hyperparameters (kappa, lambda_V, lambda_u) are chosen per outer fold and
model by the chronological protocol run inside the training animals. Results
are restricted (`data_tier`) and stay local.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .phase0_worm import FOLDS, SEED, _dump, cluster_bootstrap

RANKS = (1, 2, 4, 8, 16)
GRID = (0.1, 1.0, 10.0)
INNER_FOLDS = 4
MIN_EVENTS = 6
ALS_ITERATIONS = 25
SAMPLES = 2000
COVARIATE_R2 = 0.5
DRIFT_MARGIN = 0.75  # Revision 11: durable needs G_chron / G_interleaved > 0.75


# ---------------------------------------------------------------- data


def trial_table(responses, n):
    """Off-diagonal trials with pair ids and within-animal event positions."""
    off = responses["stimulated"] != responses["responder"]
    animal = responses["animal"][off]
    event = responses["event"][off]
    pair = responses["responder"][off] * n + responses["stimulated"][off]
    y = responses["mean_dff"][off].astype(float)
    position = np.zeros(len(event), dtype=int)
    events = {}
    for a in np.unique(animal):
        rows = np.flatnonzero(animal == a)
        # Event ids are assigned in stimulation order, which is chronological.
        order = np.unique(event[rows])
        position[rows] = np.searchsorted(order, event[rows])
        events[int(a)] = len(order)
    return {"animal": animal, "pair": pair, "y": y, "position": position}, events


def split_halves(table, rows, events, split):
    """(fit, score) row indices for a set of rows of scored animals."""
    position = table["position"][rows]
    count = np.array([events[int(a)] for a in table["animal"][rows]])
    if split == "chronological":
        fit = position < count // 2
    elif split == "interleaved":
        fit = position % 2 == 0
    else:
        raise ValueError(split)
    return rows[fit], rows[~fit]


def assign_folds(animals, folds=FOLDS, seed=SEED):
    """Same animal folds as `phase1_worm.held_out_animals`."""
    rng = np.random.default_rng(seed)
    return dict(zip(rng.permutation(animals), np.arange(len(animals)) % folds))


# ---------------------------------------------------------------- models


def population(table, rows, kappa):
    pairs, inverse = np.unique(table["pair"][rows], return_inverse=True)
    total = np.bincount(inverse, weights=table["y"][rows])
    count = np.bincount(inverse)
    return pairs, total / (count + kappa)


def lookup(table, pairs):
    """Values of a (sorted keys, values) table at `pairs`; zero when absent."""
    keys, values = table
    index = np.clip(np.searchsorted(keys, pairs), 0, max(len(keys) - 1, 0))
    found = (keys[index] == pairs) if len(keys) else np.zeros(len(pairs), bool)
    out = np.zeros((len(pairs),) + values.shape[1:])
    out[found] = values[index[found]]
    return out


def fit_basis(table, rows, P, rank, lam_v, lam_u, seed=SEED, iterations=ALS_ITERATIONS):
    """Ridge ALS on (animal, pair) cell sums of deviations from P."""
    dev = table["y"][rows] - lookup(P, table["pair"][rows])
    animals, a_index = np.unique(table["animal"][rows], return_inverse=True)
    pairs, p_index = np.unique(table["pair"][rows], return_inverse=True)
    cell = a_index * len(pairs) + p_index
    cells, c_index = np.unique(cell, return_inverse=True)
    s = np.bincount(c_index, weights=dev)
    w = np.bincount(c_index).astype(float)
    ca, cp = cells // len(pairs), cells % len(pairs)
    rng = np.random.default_rng(seed)
    V = 0.1 * rng.standard_normal((len(pairs), rank))
    U = np.zeros((len(animals), rank))
    eye = np.eye(rank)

    def group(index, groups):
        return sparse.csr_matrix(
            (np.ones(len(index)), (index, np.arange(len(index)))),
            shape=(groups, len(index)),
        )

    by_animal, by_pair = group(ca, len(animals)), group(cp, len(pairs))

    def solve(G, other, lam):
        outer = (w[:, None] * other)[:, :, None] * other[:, None, :]
        A = (G @ outer.reshape(len(w), -1)).reshape(-1, rank, rank)
        b = G @ (s[:, None] * other)
        return np.linalg.solve(A + lam * eye, b[..., None])[..., 0]

    for _ in range(iterations):
        U = solve(by_animal, V[cp], lam_u)
        V = solve(by_pair, U[ca], lam_v)
    return pairs, V


def fit_latent(table, rows, P, V, rank, lam_u):
    if V is None:  # gain model: y = u P, ridge toward u = 1
        p = lookup(P, table["pair"][rows])
        return (p @ table["y"][rows] + lam_u) / (p @ p + lam_u)
    X = lookup(V, table["pair"][rows])
    dev = table["y"][rows] - lookup(P, table["pair"][rows])
    return np.linalg.solve(X.T @ X + lam_u * np.eye(rank), X.T @ dev)


def predict(table, rows, P, V, rank, u):
    p = lookup(P, table["pair"][rows])
    if V is None:
        return p if u is None else u * p
    return p + lookup(V, table["pair"][rows]) @ u


def score_animals(table, train, scored, events, model, params, split, swap=None):
    """Squared errors on score-half rows of each scored animal, plus latents.

    With `swap`, also returns errors when each animal is scored with the latent
    of `swap[a]` (the basis is fit once and shared).
    """
    kappa, lam_v, lam_u, rank = params
    P = population(table, train, kappa)
    V = None
    if model.startswith("factor"):
        V = fit_basis(table, train, P, rank, lam_v, lam_u)
    latents, rows_out = {}, {}
    for a in scored:
        rows = np.flatnonzero(table["animal"] == a)
        fit, score = split_halves(table, rows, events, split)
        latents[a] = (
            None if model == "pop" else fit_latent(table, fit, P, V, rank, lam_u)
        )
        rows_out[a] = score

    def errors(source):
        return {
            a: (
                table["y"][rows_out[a]]
                - predict(table, rows_out[a], P, V, rank, latents[source[a]])
            )
            ** 2
            for a in scored
        }

    own = errors({a: a for a in scored})
    return own, None if swap is None else errors(swap), latents


def eligible(animals, events):
    return [a for a in animals if events[int(a)] >= MIN_EVENTS]


def select(table, train_animals, events, model, rank, seed=SEED):
    """Inner leave-animal-out chronological selection on training animals."""
    animals = np.asarray(sorted(train_animals))
    inner = assign_folds(animals, INNER_FOLDS, seed + 1)
    if model == "pop":
        grid = [(k, None, None) for k in GRID]
    elif model == "gain":
        grid = [(k, None, lu) for k in GRID for lu in GRID]
    else:
        grid = list(itertools.product(GRID, GRID, GRID))
    best, best_loss = None, np.inf
    for kappa, lam_v, lam_u in grid:
        loss = 0.0
        for fold in range(INNER_FOLDS):
            held = [a for a in animals if inner[a] == fold]
            fit_rows = np.flatnonzero(
                np.isin(table["animal"], [a for a in animals if inner[a] != fold])
            )
            errors, _, _ = score_animals(
                table,
                fit_rows,
                eligible(held, events),
                events,
                model,
                (kappa, lam_v, lam_u, rank),
                "chronological",
            )
            loss += sum(e.sum() for e in errors.values())
        if loss < best_loss:
            best, best_loss = (kappa, lam_v, lam_u), loss
    return best


def derangement(items, rng):
    items = list(items)
    if len(items) < 2:
        return None
    while True:
        perm = rng.permutation(len(items))
        if np.all(perm != np.arange(len(items))):
            return {a: items[p] for a, p in zip(items, perm)}


# ---------------------------------------------------------------- analysis


def gain_statistic(errors, model, reference="pop"):
    def stat(index):
        return 1 - errors[model][index].sum() / errors[reference][index].sum()

    return stat


def difference_statistic(errors, first, second, reference="pop"):
    def stat(index):
        base = errors[reference][index].sum()
        return (errors[second][index].sum() - errors[first][index].sum()) / base

    return stat


def trial_noise(table, rows):
    """Pooled within-animal, within-pair trial variance."""
    key = table["animal"][rows].astype(np.int64) * 10**6 + table["pair"][rows]
    _, inverse, count = np.unique(key, return_inverse=True, return_counts=True)
    mean = np.bincount(inverse, weights=table["y"][rows]) / count
    resid = table["y"][rows] - mean[inverse]
    dof = (count - 1).sum()
    return float((resid**2).sum() / dof) if dof > 0 else None


def covariate_audit(latents, metadata, seed=SEED, permutations=1000):
    """OLS R^2 of each latent from recording covariates, with permutation p."""
    animals = sorted(latents)
    U = np.array([np.atleast_1d(latents[a]) for a in animals], dtype=float)
    meta = [metadata[int(a)] for a in animals]
    batches = sorted({m["batch"] for m in meta})
    X = np.column_stack(
        [np.ones(len(animals))]
        + [[m[k] for m in meta] for k in ("confident", "duration_s", "date")]
        + [[m["batch"] == b for m in meta] for b in batches[1:]]
    ).astype(float)
    X[:, 1:4] = (X[:, 1:4] - X[:, 1:4].mean(0)) / X[:, 1:4].std(0)

    def r2(Y):
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ coef
        total = ((Y - Y.mean(0)) ** 2).sum()
        return 1 - (resid**2).sum() / total if total > 0 else 0.0

    observed = r2(U)
    rng = np.random.default_rng(seed)
    null = [r2(U[rng.permutation(len(U))]) for _ in range(permutations)]
    return {
        "animals": len(animals),
        "covariates": ["confident_traces", "duration_s", "date"]
        + [f"batch={b}" for b in batches[1:]],
        "r2": float(observed),
        "permutation_p": float(
            (1 + np.sum(np.array(null) >= observed)) / (1 + permutations)
        ),
        "measurement_state": bool(observed > COVARIATE_R2),
    }


def recording_metadata(project):
    report = json.loads(
        (Path(project) / "responses" / "ingestion-report.json").read_text()
    )
    out = {}
    for row in report["wt"]["animals"]:
        parts = [p for p in row["dataset"].split("/") if p]
        stamp = parts[-1].split("_")[1]
        date = np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}")
        out[row["animal"]] = {
            "confident": row["confident"],
            "duration_s": row["duration_s"],
            "date": float((date - np.datetime64("2020-01-01")).astype(int)),
            "batch": parts[-3],
        }
    return out


def run(
    table,
    events,
    folds=FOLDS,
    seed=SEED,
    ranks=RANKS,
    samples=SAMPLES,
    metadata=None,
    errors_path=None,
    log=print,
):
    animals = np.unique(table["animal"])
    assignment = assign_folds(animals, folds, seed)
    models = ["pop", "gain"] + [f"factor-{r}" for r in ranks]
    splits = ("chronological", "interleaved")
    errors = {s: {m: [] for m in models + [f"swap-{r}" for r in ranks]} for s in splits}
    clusters = {s: [] for s in splits}
    selected = {m: [] for m in models}
    latents = {m: {} for m in models}
    rng = np.random.default_rng(seed + 2)
    for fold in range(folds):
        train = [a for a in animals if assignment[a] != fold]
        held = eligible([a for a in animals if assignment[a] == fold], events)
        train_rows = np.flatnonzero(np.isin(table["animal"], train))
        swap = derangement(held, rng)
        for model in models:
            rank = int(model.split("-")[1]) if model.startswith("factor") else 0
            kappa, lam_v, lam_u = select(table, train, events, model, rank, seed)
            selected[model].append(
                {"fold": fold, "kappa": kappa, "lambda_V": lam_v, "lambda_u": lam_u}
            )
            params = (kappa, lam_v, lam_u, rank)
            for split in splits:
                use_swap = swap if model.startswith("factor") else None
                err, swapped, lat = score_animals(
                    table, train_rows, held, events, model, params, split, use_swap
                )
                errors[split][model].append(np.concatenate([err[a] for a in held]))
                if split == "chronological":
                    latents[model].update(lat)
                if model == "pop":
                    clusters[split].append(
                        np.concatenate([np.full(len(err[a]), a) for a in held])
                    )
                if swapped is not None:
                    errors[split][f"swap-{rank}"].append(
                        np.concatenate([swapped[a] for a in held])
                    )
            log(
                f"fold {fold} {model} selected kappa={kappa} lambda_V={lam_v} lambda_u={lam_u}"
            )
    errors = {
        s: {m: np.concatenate(v) for m, v in e.items()} for s, e in errors.items()
    }
    clusters = {s: np.concatenate(v) for s, v in clusters.items()}
    if errors_path is not None:
        np.savez_compressed(
            errors_path,
            **{
                f"{s}/{m}": e for s, models in errors.items() for m, e in models.items()
            },
            **{f"{s}/animal": c for s, c in clusters.items()},
        )

    def boot(split, statistic):
        return cluster_bootstrap(None, clusters[split], statistic, samples, seed)

    gains = {
        s: {m: boot(s, gain_statistic(errors[s], m)) for m in errors[s] if m != "pop"}
        for s in splits
    }
    chron = gains["chronological"]
    best = max(ranks, key=lambda r: chron[f"factor-{r}"]["mean"])
    floor = chron[f"factor-{best}"]["mean"] - chron[f"factor-{best}"]["se"]
    r_star = min(r for r in ranks if chron[f"factor-{r}"]["mean"] >= floor)
    star = f"factor-{r_star}"
    i01_swap = boot(
        "chronological",
        difference_statistic(errors["chronological"], star, f"swap-{r_star}"),
    )
    i02 = boot(
        "chronological", difference_statistic(errors["chronological"], star, "gain")
    )

    i03 = durability(errors, clusters, star, samples, seed)
    noise_rows = np.flatnonzero(
        np.isin(table["animal"], np.unique(clusters["chronological"]))
    )
    sigma2 = trial_noise(table, noise_rows)
    mse_pop = float(errors["chronological"]["pop"].mean())

    def passes(b):
        return b["interval"] is not None and b["interval"][0] > 0

    result = {
        "pre_registered": True,
        "spec": "Revision 10, Section 10.4, Stage I0",
        "split": "leave_animal_out with within_animal_chronological / within_animal_interleaved",
        "folds": folds,
        "animals": len(animals),
        "scored_animals": len(np.unique(clusters["chronological"])),
        "score_trials": {s: len(clusters[s]) for s in splits},
        "selected": selected,
        "gain_over_pop": gains,
        "r_star": r_star,
        "r_best": best,
        "ceiling": {
            "trial_variance": sigma2,
            "mse_pop_chronological": mse_pop,
            "gain_ceiling": None if sigma2 is None else 1 - sigma2 / mse_pop,
        },
        "gates": {
            "I0-1": {
                "gain": chron[star],
                "minus_swap": i01_swap,
                "pass": passes(chron[star]) and passes(i01_swap),
            },
            "I0-2": {"minus_gain": i02, "pass": passes(i02)},
            "I0-3": i03,
        },
        "time_span": "one recording session (median about 32 min)",
        "data_tier": "restricted",
    }
    if metadata is not None:
        result["covariates"] = {
            "gain": covariate_audit(latents["gain"], metadata, seed),
            star: covariate_audit(latents[star], metadata, seed),
        }
    return result


def durability(errors, clusters, model, samples=SAMPLES, seed=SEED):
    """Persistence ratio G_chron / G_interleaved (spec Revision 11).

    Animals are bootstrapped jointly across both splits. `durable` needs the
    95% ratio interval above DRIFT_MARGIN; draws with G_interleaved <= 0 have
    no state to persist, and more than 2.5% of them rule `durable` out.
    `drifts` needs the ratio interval below 1 or the interval of
    G_chron - G_interleaved below 0 (strong drift can erase the interleaved
    gain itself). Anything else is `inconclusive`.
    """
    members = {
        s: {a: np.flatnonzero(clusters[s] == a) for a in np.unique(clusters[s])}
        for s in clusters
    }
    animals = sorted(set(members["chronological"]) & set(members["interleaved"]))

    def gain(split, drawn):
        rows = np.concatenate([members[split][a] for a in drawn])
        return 1 - errors[split][model][rows].sum() / errors[split]["pop"][rows].sum()

    def pair(drawn):
        return gain("chronological", drawn), gain("interleaved", drawn)

    chron, inter = pair(animals)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            pair(rng.choice(animals, size=len(animals), replace=True))
            for _ in range(samples)
        ]
    )
    positive = draws[:, 1] > 0
    ratios = draws[positive, 0] / draws[positive, 1]
    excluded = 1 - positive.mean()
    interval = np.quantile(ratios, [0.025, 0.975]).tolist() if len(ratios) else None
    difference = draws[:, 0] - draws[:, 1]
    difference_interval = np.quantile(difference, [0.025, 0.975]).tolist()
    ratio_valid = interval is not None and excluded <= 0.025
    if ratio_valid and interval[0] > DRIFT_MARGIN:
        verdict = "durable"
    elif difference_interval[1] < 0 or (ratio_valid and interval[1] < 1):
        verdict = "drifts"
    else:
        verdict = "inconclusive"
    return {
        "model": model,
        "margin": DRIFT_MARGIN,
        "ratio": float(chron / inter) if inter > 0 else None,
        "ratio_interval": interval,
        "draws_without_state": float(excluded),
        "difference": float(chron - inter),
        "difference_interval": difference_interval,
        "verdict": verdict,
    }


def individual_state(project, output, log=print):
    from .randi_traces import load_responses

    manifest = json.loads((Path(project) / "manifest.json").read_text())
    n = len(manifest["neuron_names"])
    table, events = trial_table(load_responses(project, "wt"), n)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    result = run(
        table,
        events,
        metadata=recording_metadata(project),
        log=log,
        errors_path=out / "individual-errors.npz",
    )
    return _dump(out / "individual-state.json", result)
