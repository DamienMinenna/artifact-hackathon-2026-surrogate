"""Score a PALLAS hackathon submission against an answer key.

Run: SUBMISSION=sub.csv KEY=key.csv /path/to/python pallas_score.py
Scoring rules in full: data/README.md; scales and baselines: DATA_CARD.md in the data pack.
A position-dependent task is scored against the PER-Z MEAN, not the global mean.

The key decides the scoring. The scorer reads the key's columns and picks the scoring
function from them: `config` alone gives one R2 per target against the global mean;
`config` plus `z_mm` or `plane_mm` gives the per-position R2 in the in-plasma zone, and then
`L_inj` must be in the key as well (it locates the plasma end and is not scored). An extra
or missing id column therefore calls the wrong scoring function. Every other column of the
key is a target and IS scored, so a key holds the id columns and the targets of the
objective and nothing else. The submission carries the same id columns and one column per
target, in physical units. Key columns per objective (targets: objectives.md):
    inverse A, direct B, inverse B        config
    trajectory (in-plasma, drift, OOD)    config, z_mm, L_inj
    moments at the planes                 config, plane_mm, L_inj
    particle clouds, field maps           folders: PARTICLES_SUBMISSION / PARTICLES_KEY,
                                          FIELD_SUBMISSION / FIELD_KEY
To score yourself, hold out whole configurations and write their true values in this shape
as your key; the organisers run this file with the hidden keys. "mean" is the plain mean
over the targets; a negative R2 means worse than predicting the mean.
"""

import os
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd

# Submit PHYSICAL values. The scale each target is scored on is applied here, so
# nobody has to remember which quantities are logged -- and so every submission is
# scored the way the reference numbers were.
# A target is multiplied by `multiply_by`, raised to `floor`, then logged if `log10`.
Scale = namedtuple("Scale", ["log10", "floor", "multiply_by"])
LINEAR = Scale(log10=False, floor=None, multiply_by=1.0)
SCALES = {
    "E_med_MeV": LINEAR,
    "E_mean_MeV": LINEAR,
    "dE_mad": Scale(log10=True, floor=1e-3, multiply_by=100.0),   # fraction -> percent
    "q_end": Scale(log10=True, floor=0.1, multiply_by=1e12),      # coulomb -> pC
    "p_1": LINEAR,
    "a_0": LINEAR,
    "c_N2": LINEAR,
    "E_MeV": LINEAR,
    "q_pC": Scale(log10=True, floor=0.1, multiply_by=1.0),
    "dE_pct": Scale(log10=True, floor=1e-3, multiply_by=1.0),
    "emit_um": Scale(log10=True, floor=1e-4, multiply_by=1.0),
    "div_mrad": Scale(log10=True, floor=1e-3, multiply_by=1.0),
    "sigz_um": Scale(log10=True, floor=1e-3, multiply_by=1.0),
    "q_tot_pC": Scale(log10=True, floor=0.1, multiply_by=1.0),
}
# The moment targets of test_moments_planes: six bunch widths on log, six
# centroids on their own linear scale. sigma_* are RMS widths, mean_* centroids;
# zeta/y/z are lengths in um, u_* are normalised momenta (dimensionless).
for _coord, _unit in (("zeta", "um"), ("y", "um"), ("z", "um"),
                      ("ux", ""), ("uy", ""), ("uz", "")):
    SCALES[f"sigma_{_coord}"] = Scale(log10=True, floor=1e-6, multiply_by=1.0)
    SCALES[f"mean_{_coord}"] = LINEAR

KEY_COLUMNS = ("row_id", "config", "z_mm", "plane_mm", "L_inj")   # anything not a target

PLASMA_END_OFFSET_MM = float(os.environ.get("PLASMA_END_OFFSET_MM", 3.2))
# The plasma ends at L_inj + 3.2 mm, PER CONFIGURATION: that
# expression is the shipped density profile's own support, measured, and it runs from
# 3.601 to 3.999 mm across the corpus. A key or submission carrying L_inj is split on
# its own end; the flat fallback below is only for a table that carries no L_inj, and
# it mislabels a median 2 and up to 5 positions per configuration.
PLASMA_END_MM = float(os.environ.get("PLASMA_END_MM", 3.8))

ZONES = ("pooled", "in_plasma", "drift")

# Scored INSIDE THE PLASMA ONLY. Past the plasma end these two
# columns measure a definition rather than the beam: div_mrad is
# sqrt(emit_g/beta_x + emit_g/beta_y), a waist-proxy that falls as beta grows through
# the drift, and emit_um is n_emit_x, which falls as the halo leaves the diagnostic.
# Rewarding a model for reproducing either in the drift rewards the wrong thing.
IN_PLASMA_ONLY = ("emit_um", "div_mrad")

# A position must carry at least this many configurations of the zone to be scored;
# below it the per-z mean is a mean over a handful of rows and the position is dropped
# from both sums.
MIN_ZONE_MEMBERS = int(os.environ.get("MIN_ZONE_MEMBERS", 30))

POSITION_COLUMNS = ("z_mm", "plane_mm")


def physical_to_normalised(name, values):
    """Physical values on the scale R2 is computed on: log10 for the quantities that span
    decades, unchanged otherwise. Not mean/std scaling."""
    scale = SCALES.get(name, LINEAR)
    values = np.asarray(values, float) * scale.multiply_by
    return np.log10(np.maximum(values, scale.floor)) if scale.log10 else values


def normalised_to_physical(name, values):
    """The inverse of physical_to_normalised: normalised values back to physical units."""
    scale = SCALES.get(name, LINEAR)
    values = np.asarray(values, float)
    return (10 ** values if scale.log10 else values) / scale.multiply_by


to_scored_scale = physical_to_normalised   # the old name, kept for scripts written against it


def align(submission, key):
    """Join a submission to its key and return (merged, join columns, targets)."""
    sub = pd.read_csv(submission) if isinstance(submission, str) else submission.copy()
    ans = pd.read_csv(key) if isinstance(key, str) else key.copy()

    # L_inj rides along in the position-dependent keys to locate the plasma end; in
    # test_inverse_B it is one of the four targets.
    positional = any(c in ans.columns for c in POSITION_COLUMNS)
    key_columns = KEY_COLUMNS if positional else tuple(c for c in KEY_COLUMNS if c != "L_inj")
    # L_inj only locates the plasma end: it is read from the key, never joined on (a float32
    # column does not survive a CSV round trip)
    join = [c for c in key_columns if c != "L_inj" and c in ans.columns and c in sub.columns]
    if not join:
        raise ValueError(f"submission has no join column; expected one of {key_columns}")
    targets = [c for c in ans.columns if c not in key_columns]
    extra = [c for c in sub.columns if c not in ans.columns and c not in key_columns]
    missing = [c for c in targets if c not in sub.columns]
    if missing:
        raise ValueError(f"submission is missing target column(s) {missing}. Every key column "
                         f"other than {key_columns} counts as a target: build a key from the "
                         f"id columns and the targets only")

    exact = {}                           # the key's own positions, for the zone test
    for c in join:                       # a CSV round trip must not break the join
        if c in POSITION_COLUMNS:
            exact[c] = ans[c].astype(float).to_numpy()
            ans[c], sub[c] = ans[c].astype(float).round(3), sub[c].astype(float).round(3)
    repeated = int(sub.duplicated(join).sum())
    if repeated:
        raise ValueError(f"submission repeats {repeated} key row(s) on {join}")
    merged = ans.merge(sub[join + targets + extra], on=join, suffixes=("", "_pred"))
    if len(merged) != len(ans):
        raise ValueError(f"submission covers {len(merged)} of {len(ans)} rows, joined on {join}"
                         + (" (positions are matched at 1 um: write them with 6 decimals)"
                            if any(c in POSITION_COLUMNS for c in join) else ""))
    for c in targets:                    # a NaN prediction would turn the score into NaN
        holes = int((merged[f"{c}_pred"].isna() & merged[c].notna()).sum())
        if holes:
            raise ValueError(f"submission has {holes} NaN prediction(s) in {c}")
    for c, values in exact.items():      # an inner merge keeps the key's row order
        merged[c] = values
    return merged, join, targets


def plasma_end(frame):
    """Where the plasma ends for each row, in mm.

    From the row's own L_inj when the table carries it, which is the ruled definition;
    the flat constant only when it does not.
    """
    if "L_inj" in frame.columns:
        return frame["L_inj"].to_numpy() + PLASMA_END_OFFSET_MM
    warnings.warn(f"the key has no L_inj column: every plasma end is taken as {PLASMA_END_MM} mm; "
                  "add L_inj to the key for each configuration's own end")
    return np.full(len(frame), PLASMA_END_MM)


def resolve_zone(frame, zone):
    """The default zone, None: inside the plasma for a trajectory (z_mm), every row otherwise.

    The moments planes (plane_mm) keep every plane: 5.0 mm lies past every plasma end and is
    read on its own, not dropped.
    """
    if zone is not None:
        return zone
    return "in_plasma" if "z_mm" in frame.columns else "pooled"


def zone_mask(frame, zone):
    """Rows of `frame` inside the plasma, in the drift after it, or all of them."""
    if zone == "pooled":
        return np.ones(len(frame), bool)
    position = "z_mm" if "z_mm" in frame.columns else "plane_mm"
    if position not in frame.columns:
        raise ValueError(f"zone {zone!r} needs a z_mm or plane_mm column")
    inside = frame[position].to_numpy() <= plasma_end(frame)
    if zone == "in_plasma":
        return inside
    if zone == "drift":
        return ~inside
    raise ValueError(f"unknown zone {zone!r}; use one of {ZONES}")


def scored_targets(targets, zone):
    """The targets that count in this zone. Two are in-plasma only; see IN_PLASMA_ONLY."""
    if zone == "in_plasma":
        return list(targets)
    return [t for t in targets if t not in IN_PLASMA_ONLY]


def r2(true, pred):
    """R2 on values already carried to the normalised scale, against the global mean.

    Used for the tasks with no position axis. For the position-dependent tasks see
    r2_per_z, which is what the reference numbers use.
    """
    resid = np.sum((true - pred) ** 2)
    spread = np.sum((true - true.mean()) ** 2)
    return float(1.0 - resid / spread)


def r2_per_z(true, pred, position):
    """R2 for a position-dependent task, against the PER-Z MEAN.

    The denominator is the spread across CONFIGURATIONS at each position, summed over
    positions -- not the spread over every cell at once. On a trajectory most of the
    total variance is the z-trend itself, and a global-mean denominator credits a model
    for reproducing the average curve shape, which is trivial. This baseline asks the
    question a surrogate is for: at this position, can you tell the configurations
    apart? It is also what every reference number in the pack uses.

    In zone mode the mean is taken over the configurations that are IN THIS ZONE at
    this z, and with a per-config plasma end that membership changes across
    3.6-4.0 mm. A position carrying fewer than MIN_ZONE_MEMBERS configurations is
    DROPPED from both sums rather than contributing a mean over a handful of rows.
    Returns (r2, n_positions_scored, n_positions_dropped).
    """
    position = np.asarray(position)
    resid, spread, scored, dropped = 0.0, 0.0, 0, 0
    for here in np.unique(position):
        at = position == here
        if int(at.sum()) < MIN_ZONE_MEMBERS:
            dropped += 1
            continue
        here_true, here_pred = true[at], pred[at]
        resid += float(np.sum((here_true - here_pred) ** 2))
        spread += float(np.sum((here_true - here_true.mean()) ** 2))
        scored += 1
    if spread <= 0.0:
        return float("nan"), scored, dropped
    return float(1.0 - resid / spread), scored, dropped


def score_submission(submission, key, zone=None):
    """R2 per target and the mean, on the agreed scale.

    `submission` and `key` are paths or DataFrames. They are joined on whichever of
    row_id / (config, z_mm) / (config, plane_mm) the key carries, so a submission in
    a different row order still scores correctly. `zone` restricts the rows to one
    side of the plasma end for the position-dependent tasks.
    """
    merged, _, targets = align(submission, key)
    zone = resolve_zone(merged, zone)
    keep = zone_mask(merged, zone)
    if not keep.any():
        raise ValueError(f"zone {zone!r} selects no rows")
    merged = merged[keep]
    targets = scored_targets(targets, zone)
    at = next((c for c in POSITION_COLUMNS if c in merged.columns), None)

    out, scored, dropped = {}, None, None
    for name in targets:
        true = physical_to_normalised(name, merged[name])
        pred = physical_to_normalised(name, merged[f"{name}_pred"])
        if at is None:
            out[name] = r2(true, pred)
        else:
            out[name], scored, dropped = r2_per_z(true, pred, merged[at].to_numpy())
    out["mean"] = float(np.mean([out[t] for t in targets]))
    out["n_rows"] = int(len(merged))
    if at is not None:
        out["n_z_scored"], out["n_z_dropped"] = scored, dropped
    return out


def score_zones(submission, key):
    """The same submission scored pooled, inside the plasma, and in the drift.

    The three columns are the point of the trajectory task: a model that looks
    strong pooled and weak in-plasma has learned the coasting, not the acceleration.
    Only in_plasma is ranked; the other two are for reading.
    """
    scored = {z: score_submission(submission, key, zone=z) for z in ZONES}
    _, _, targets = align(submission, key)
    extra = [c for c in ("n_rows", "n_z_scored", "n_z_dropped")
             if any(c in v for v in scored.values())]
    table = pd.DataFrame(scored).T.reindex(columns=list(targets) + ["mean"] + extra)
    table.index.name = "zone"
    # A blank cell is "not scored in this zone", not a failure: see IN_PLASMA_ONLY.
    return table


def score_budget(submissions, key, zone=None, target=0.90):
    """A data-budget curve: one submission per training-set size.

    `submissions` maps the number of training configs to a submission (path or
    frame). Returns the curve and the budget at which the mean score first reaches
    `target`, interpolated linearly between the two rungs that bracket it -- the
    number for "how much data does this approach need".
    """
    budgets = sorted(submissions)
    rows = [{"n_train_configs": b,
             **score_submission(submissions[b], key, zone=zone)} for b in budgets]
    table = pd.DataFrame(rows).set_index("n_train_configs")

    means = table["mean"].to_numpy()
    reached = np.where(means >= target)[0]
    if reached.size == 0:
        budget_at = float("nan")
    elif reached[0] == 0:
        budget_at = float(budgets[0])
    else:
        i = reached[0]
        lo, hi = means[i - 1], means[i]
        frac = (target - lo) / (hi - lo)
        budget_at = float(budgets[i - 1] + frac * (budgets[i] - budgets[i - 1]))
    return table, budget_at


def score_ood(in_dist, in_dist_key, ood, ood_key, zone=None):
    """One model on held-out configs inside its training range and beyond it.

    Both scores come from the SAME trained model; the drop is the answer to the
    organisers' third question. Nothing in the training set appears in either.
    """
    a = score_submission(in_dist, in_dist_key, zone=zone)
    b = score_submission(ood, ood_key, zone=zone)
    names = [k for k in a if k not in ("n_rows", "n_z_scored", "n_z_dropped")]
    table = pd.DataFrame({"in_distribution": [a[k] for k in names],
                          "out_of_distribution": [b[k] for k in names]}, index=names)
    table["drop"] = table["in_distribution"] - table["out_of_distribution"]
    table.index.name = "target"
    return table, {"n_rows_in_distribution": a["n_rows"], "n_rows_ood": b["n_rows"]}


def score_calibration(submission, key, level=0.9, zone=None):
    """Coverage and sharpness of predictive intervals, for the inverse task.

    The submission carries `<target>_lo` and `<target>_hi` beside each point
    prediction, an interval meant to hold the truth `level` of the time. Coverage
    alone is gameable -- an infinitely wide interval covers everything -- so the
    mean width on the normalised scale travels with it, and the two are read together.
    """
    merged, _, targets = align(submission, key)
    zone = resolve_zone(merged, zone)
    keep = zone_mask(merged, zone)
    merged = merged[keep]
    targets = scored_targets(targets, zone)

    rows = []
    for name in targets:
        bounds = [f"{name}_lo", f"{name}_hi"]
        if any(b not in merged.columns for b in bounds):
            raise ValueError(f"submission is missing interval column(s) for {name}")
        true = physical_to_normalised(name, merged[name])
        lo = physical_to_normalised(name, merged[bounds[0]])
        hi = physical_to_normalised(name, merged[bounds[1]])
        covered = (true >= np.minimum(lo, hi)) & (true <= np.maximum(lo, hi))
        rows.append({"target": name,
                     "coverage": float(covered.mean()),
                     "nominal": float(level),
                     "coverage_error": float(covered.mean() - level),
                     "mean_width_scored": float(np.mean(np.abs(hi - lo)))})
    table = pd.DataFrame(rows).set_index("target")
    return table, {"n_rows": int(len(merged))}


# Particle clouds are scored by a sliced Wasserstein-1 distance, the one our own
# particle models are judged by: both clouds are whitened on the TRUE
# cloud (the six coordinates differ in scale by ~100x, so an unwhitened distance
# measures uz and nothing else), the truth is charge-weighted, and the distance is
# divided by the truth's own resolution floor -- two random halves of the true cloud,
# rescaled by sqrt(2) to full size -- so a perfect prediction scores about 1.
# The submission is first redrawn (by its weights) to the true cloud's count, so
# submitting more particles does not lower the score. A plane with fewer than
# MIN_TRUE_PARTICLES true electrons is not scored: its floor is noise.
SW_PROJECTIONS = int(os.environ.get("SW_PROJECTIONS", 256))
SW_FLOOR_REPEATS = int(os.environ.get("SW_FLOOR_REPEATS", 4))
MIN_TRUE_PARTICLES = int(os.environ.get("MIN_TRUE_PARTICLES", 50))


def _probabilities(w, n):
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=np.float64)
    return w / w.sum() if np.isfinite(w.sum()) and w.sum() > 0 else np.full(n, 1.0 / n)


def _w1_1d(x1, p1, x2, p2):
    o1, o2 = np.argsort(x1), np.argsort(x2)
    x1, p1, x2, p2 = x1[o1], p1[o1], x2[o2], p2[o2]
    grid = np.sort(np.concatenate([x1, x2]))
    F1 = np.concatenate([[0.0], np.cumsum(p1)])[np.searchsorted(x1, grid, side="right")]
    F2 = np.concatenate([[0.0], np.cumsum(p2)])[np.searchsorted(x2, grid, side="right")]
    return float(np.sum(np.abs(F1[:-1] - F2[:-1]) * np.diff(grid)))


def sliced_wasserstein(X1, w1, X2, w2, directions):
    """Mean 1D Wasserstein-1 over the given unit directions, weights normalised."""
    p1, p2 = _probabilities(w1, len(X1)), _probabilities(w2, len(X2))
    P1, P2 = X1 @ directions, X2 @ directions
    return float(np.mean([_w1_1d(P1[:, j], p1, P2[:, j], p2) for j in range(P1.shape[1])]))


def cloud_ratio(X_true, w_true, X_pred, w_pred=None, seed=0):
    """One (configuration, plane): the distance to the truth over the truth's floor."""
    X_true = np.asarray(X_true, dtype=np.float64)
    X_pred = np.asarray(X_pred, dtype=np.float64)
    p = _probabilities(w_true, len(X_true))
    mean = p @ X_true
    cov = ((X_true - mean) * p[:, None]).T @ (X_true - mean)
    cov += np.eye(6) * 1e-12 * max(np.trace(cov) / 6, 1e-300)
    W = np.linalg.inv(np.linalg.cholesky(cov))   # whitening matrix of the truth
    Zt, Zp = (X_true - mean) @ W.T, (X_pred - mean) @ W.T
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((6, SW_PROJECTIONS))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    draw = np.random.default_rng([seed, 1])
    n = len(Zt)
    if w_pred is None and len(Zp) >= n:
        Zp = Zp[draw.choice(len(Zp), n, replace=False)]
    else:
        Zp = Zp[draw.choice(len(Zp), n, replace=True, p=_probabilities(w_pred, len(Zp)))]
    sw = sliced_wasserstein(Zt, w_true, Zp, None, directions)
    halves = []
    for _ in range(SW_FLOOR_REPEATS):
        m = rng.random(len(Zt)) < 0.5
        if m.sum() >= 2 and (~m).sum() >= 2:
            halves.append(sliced_wasserstein(Zt[m], p[m], Zt[~m], p[~m], directions))
    floor = float(np.mean(halves)) / np.sqrt(2.0) if halves else np.nan
    return {"sw": sw, "floor": floor, "ratio": sw / floor if floor > 0 else np.nan}


def load_cloud_file(path):
    """plane_mm, and one (N, 6) cloud with its weights per plane, lost particles dropped."""
    with np.load(path) as z:
        planes = np.asarray(z["plane_mm"], dtype=np.float64)
        state = np.asarray(z["state"], dtype=np.float64)
        present = z["present"] if "present" in z.files else np.isfinite(state).all(axis=2)
        weight = z["weight"] if "weight" in z.files else None
    clouds = {}
    for i, plane in enumerate(planes):
        keep = np.asarray(present[i], dtype=bool) & np.isfinite(state[i]).all(axis=1)
        w = None if weight is None else (weight[i] if np.ndim(weight) == 2 else weight)[keep]
        clouds[round(float(plane), 4)] = (state[i][keep], w)
    return clouds


def score_particles(submission_dir, key_dir, seed=0):
    """Median truth-referenced distance over every (configuration, plane) of the key.

    Both folders hold one `Config_<id>.npz` per configuration, with `plane_mm` and
    `state` (planes, particles, 6) in the pack's coordinate order; a submission may add
    `weight`, otherwise its particles count equally. A missing configuration or plane
    is an error, not a skip. A key plane with no particle present (the bunch is injected
    later) is not scored and is counted in `n_rows_no_bunch`; one with fewer than
    MIN_TRUE_PARTICLES electrons is counted in `n_rows_sparse`.
    """
    rows, skipped, sparse = [], 0, 0
    for name in sorted(os.listdir(key_dir)):
        if not name.endswith(".npz"):
            continue
        sub_path = os.path.join(submission_dir, name)
        if not os.path.exists(sub_path):
            raise ValueError(f"submission is missing {name}")
        truth, pred = load_cloud_file(os.path.join(key_dir, name)), load_cloud_file(sub_path)
        for plane, (X_true, w_true) in truth.items():
            if len(X_true) == 0:         # the bunch is not injected yet at this plane
                skipped += 1
                continue
            if len(X_true) < MIN_TRUE_PARTICLES:
                sparse += 1
                continue
            if plane not in pred:
                raise ValueError(f"{name} is missing plane {plane} mm")
            X_pred, w_pred = pred[plane]
            row = cloud_ratio(X_true, w_true, X_pred, w_pred, seed=seed)
            rows.append({"config": int(name[len("Config_"):-len(".npz")]),
                         "plane_mm": plane, "n_true": len(X_true), **row})
    table = pd.DataFrame(rows)
    return {"median_ratio": float(table.ratio.median()),
            "mean_ratio": float(table.ratio.mean()),
            "n_rows": int(len(table)), "n_config": int(table.config.nunique()),
            "n_rows_no_bunch": skipped, "n_rows_sparse": sparse}, table


FIELD_NAMES = ("ex", "rho_e", "laser")
# Truth and prediction are both averaged over this length along the box before the error
# is taken, so a sharp feature (the density spike at the bubble's back wall) placed a
# cell or two off counts as nearly right instead of being punished twice. 0 = raw maps.
FIELD_SMOOTH_UM = float(os.environ.get("FIELD_SMOOTH_UM", 1.0))
FIELD_DX_UM = 0.2003                      # spacing of the shipped field grid, um


def smooth_along_box(maps, smooth_um=None):
    """Running mean over `smooth_um` along the last axis, edges padded with the end value."""
    smooth_um = FIELD_SMOOTH_UM if smooth_um is None else smooth_um
    width = int(round(smooth_um / FIELD_DX_UM))
    if width < 2:
        return maps
    pad = [(0, 0)] * (maps.ndim - 1) + [(width // 2, width - 1 - width // 2)]
    padded = np.pad(maps, pad, mode="edge")
    csum = np.cumsum(padded, axis=-1)
    csum = np.concatenate([np.zeros_like(csum[..., :1]), csum], axis=-1)
    return (csum[..., width:] - csum[..., :-width]) / width


def load_field_dir(path):
    """config ids and {field: (N, dumps, points) float32} from a folder laid out like the download."""
    with np.load(os.path.join(path, "field_axes.npz")) as z:
        configs = np.asarray(z["config"], dtype=np.int64)
    fields = {name: np.load(os.path.join(path, f"field_{name}.npy"), mmap_mode="r")
              for name in FIELD_NAMES}
    return configs, fields


def score_field(submission_dir, key_dir, smooth_um=None):
    """Median over configurations of the per-map relative L2 error, one number per field.

    Both folders hold `field_axes.npz` (with `config`) and `field_<name>.npy` of shape
    (configs, dumps, points), the download's layout. Rows are matched by config; a
    missing configuration or a shape mismatch is an error, not a skip. Both maps are
    first smoothed over FIELD_SMOOTH_UM along the box (smooth_um=0 scores raw maps).
    """
    key_cfg, key = load_field_dir(key_dir)
    sub_cfg, sub = load_field_dir(submission_dir)
    where = {int(c): i for i, c in enumerate(sub_cfg)}
    missing = [int(c) for c in key_cfg if int(c) not in where]
    if missing:
        raise ValueError(f"submission is missing {len(missing)} configurations, e.g. {missing[:5]}")
    rows = []
    for k, c in enumerate(key_cfg):
        row = {"config": int(c)}
        for name in FIELD_NAMES:
            truth = np.asarray(key[name][k], dtype=np.float64)
            pred = np.asarray(sub[name][where[int(c)]], dtype=np.float64)
            if pred.shape != truth.shape:
                raise ValueError(f"Config_{c} {name}: shape {pred.shape}, key {truth.shape}")
            truth, pred = smooth_along_box(truth, smooth_um), smooth_along_box(pred, smooth_um)
            row[name] = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
        rows.append(row)
    table = pd.DataFrame(rows)
    summary = {f"median_rel_l2_{name}": float(table[name].median()) for name in FIELD_NAMES}
    summary["smooth_um"] = FIELD_SMOOTH_UM if smooth_um is None else float(smooth_um)
    summary["n_config"] = int(len(table))
    return summary, table


if os.environ.get("FIELD_SUBMISSION") and os.environ.get("FIELD_KEY"):
    summary, _ = score_field(os.environ["FIELD_SUBMISSION"], os.environ["FIELD_KEY"])
    for key_name, value in summary.items():
        print(f"{key_name:22s}  {value:.4f}" if isinstance(value, float)
              else f"{key_name:22s}  {value}")

if os.environ.get("PARTICLES_SUBMISSION") and os.environ.get("PARTICLES_KEY"):
    summary, _ = score_particles(os.environ["PARTICLES_SUBMISSION"], os.environ["PARTICLES_KEY"])
    for key_name, value in summary.items():
        print(f"{key_name:13s}  {value:.4f}" if isinstance(value, float)
              else f"{key_name:13s}  {value}")

if os.environ.get("SUBMISSION") and os.environ.get("KEY"):
    zone = os.environ.get("ZONE")      # unset: inside the plasma when the key has positions
    result = score_submission(os.environ["SUBMISSION"], os.environ["KEY"], zone=zone)
    print(f"zone: {zone or 'default (in_plasma for position-dependent keys)'}")
    width = max(len(k) for k in result)
    for key_name, value in result.items():
        print(f"{key_name:{width}s}  {value:.4f}" if isinstance(value, float)
              else f"{key_name:{width}s}  {value}")
