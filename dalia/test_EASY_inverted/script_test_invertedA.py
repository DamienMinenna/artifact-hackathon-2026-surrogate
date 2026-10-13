import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# The full pack is not in this folder: download it and unpack it under data/.
# Set PALLAS_PACK if you keep it anywhere else. Without it, the 10 % sample in
# data/sample/ is used, so the notebook runs straight from a clone of the repository.
PATH_DATA = "/workspace/data/surrogate-models/data"

CANDIDATES = [
    os.environ.get("PALLAS_PACK"),
    os.path.join(PATH_DATA, "pallas_hackathon_data"),
    os.path.join("..", "pallas_hackathon_data"),
    "pallas_hackathon_data",
    os.path.join(PATH_DATA, "sample"),
]

PACK = None
for candidate in CANDIDATES:  # the first folder that holds the pack wins
    if candidate and os.path.exists(os.path.join(candidate, "campaign_A.parquet")):
        PACK = candidate
        break
if PACK is None:
    raise SystemExit(
        "no data pack found -- see data/README.md for the download "
        "and unpack it as data/pallas_hackathon_data/"
    )

# An old or partial download stops here, not three cells later with a KeyError:
# every file and column listed in the schema must be in the pack.
with open(
    os.path.join(PATH_DATA, "ontologies", "pack_schema.json"), encoding="utf-8"
) as fh:
    SCHEMA = json.load(fh)
for name, spec in SCHEMA["files"].items():
    path = os.path.join(PACK, name)
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing: {PACK} is not the current pack (data/README.md)"
        )
    if name.endswith(".parquet"):
        absent = set(spec["columns"]) - set(pq.ParquetFile(path).schema_arrow.names)
        if absent:
            raise SystemExit(
                f"{name} lacks {sorted(absent)}: {PACK} is not the current pack"
            )
if PACK == os.path.join(PATH_DATA, "sample"):
    print(
        "using the 10 % SAMPLE in data/sample/ -- every number below will differ from "
        "the full pack's; download the full pack before training anything (data/README.md)"
    )

n_files = sum(1 for f in os.listdir(PACK) if os.path.isfile(os.path.join(PACK, f)))
size_mb = (
    sum(
        os.path.getsize(os.path.join(PACK, f))
        for f in os.listdir(PACK)
        if os.path.isfile(os.path.join(PACK, f))
    )
    / 1e6
)
print(f"pack at {PACK}: {n_files} files, {size_mb:.0f} MB\n")

# The array columns (spectra, density profiles) are most of the file size, so ask
# for the scalars unless you need them. Both campaigns are loaded once, here.
SCALARS = [
    "scan",
    "config",
    "p_1",
    "a_0",
    "c_N2",
    "x_of",
    "P_max",
    "cN2_max",
    "L_inj",
    "dip_frac",
    "E_med_MeV",
    "E_mean_MeV",
    "dE_mad",
    "dE_std",
    "q_end",
    "i_peak",
    "sigma_z",
    "sigma_y",
    "n_emit_x",
    "n_emit_y",
    "div_rms",
    "beta_x",
    "alpha_x",
    "z_avg",
    "injected",
]


def load(name):
    path = f"{PACK}/{name}.parquet"
    cols = pq.ParquetFile(path).schema.names  # the footer, not the data
    return pd.read_parquet(path, columns=[c for c in SCALARS if c in cols])


# Two campaigns, two files. Campaign A holds both of its scans, `scan` says which.
A = load("campaign_A")
B = load("campaign_B")
grid = np.load(f"{PACK}/common_energy_grid_MeV.npy")  # shared energy axis, MeV


def summarise(label, frame):
    print(
        f"{label:22s} {len(frame):6,} rows   {frame.injected.sum():5,} injected"
        f"   ({frame.injected.mean():.1%})"
    )


summarise("campaign_A", A)
for scan, rows in A.groupby("scan"):
    summarise(f"   scan = {scan}", rows)
summarise("campaign_B", B)


# The task below is the random scan's: the hidden test set is drawn from it.
inj_a = A[(A.scan == "random") & A.injected]
train, test = train_test_split(inj_a, test_size=0.2, random_state=0)
print(
    f"train {len(train):,} rows / {train.config.nunique():,} configs"
    f"   test {len(test):,} rows / {test.config.nunique():,} configs"
)
print(f"configs in both sides: {len(set(train.config) & set(test.config))}")

# Eight beam features plus the known focus in, three settings out -- the same inputs and
# targets as our reference ensemble, so the comparison below is like for like.
TARGETS = ["p_1", "a_0", "c_N2"]
FEATURES = [
    "E_med_MeV",
    "dE_mad",
    "q_end",
    "i_peak",
    "n_emit_x",
    "sigma_z",
    "sigma_y",
    "div_rms",
    "x_of",
]

data_tr = train.dropna(subset=FEATURES + TARGETS)
data_te = test.dropna(subset=FEATURES + TARGETS)

LOG_COLUMNS = ["q_end", "i_peak", "n_emit_x", "sigma_y"]  # these span several decades


def inputs_with_logs(frame):
    """The model's input table: FEATURES, with LOG_COLUMNS on log10 (all > 0 on injected rows)."""
    features = frame[FEATURES].copy()
    for col in LOG_COLUMNS:
        features[col] = np.log10(features[col])
    return features.to_numpy()


# model = make_pipeline(StandardScaler(), LinearRegression())

model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    # LinearRegression(),
    # RandomForestRegressor(n_estimators=200, random_state=42),
    #     # ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    # MultiOutputRegressor(
    #     GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    #     n_jobs=-1,
    # ),
    # RegressorChain(GradientBoostingRegressor(random_state=42), order=[0, 1, 2]),
    # RegressorChain(HistGradientBoostingRegressor(max_iter=1800, learning_rate=0.05, max_leaf_nodes=13, min_samples_leaf=8, random_state=0),order=[0, 1, 2],)
    MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=1800,
            learning_rate=0.05,
            max_leaf_nodes=13,
            min_samples_leaf=8,
            random_state=0,
        ),
        n_jobs=-1,
    ),
    # RegressorChain(SVR(kernel="rbf", C=1.0, epsilon=0.1) , order=[0, 1, 2]),
    #
    #     # Ridge(alpha=1.0)                      # L2: shrinks coefficients, stable with many correlated features
    #     # Lasso(alpha=0.01, max_iter=10000)     # L1: zeroes out useless polynomial terms (feature selection)
    #     # ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)  # mix of both
    # KNeighborsRegressor(n_neighbors=5)             # simple, scaling required
    # MLPRegressor(hidden_layer_sizes=(512, 512, 64, 32, 16), max_iter=100, random_state=42)  # small neural net
)

# model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=1800, learning_rate=0.05, max_leaf_nodes=13, min_samples_leaf=8, random_state=0),n_jobs=-1,)

model.fit(inputs_with_logs(data_tr), data_tr[TARGETS].to_numpy())


# a linear model can leave the range it was trained on (a negative c_N2, say): keep it inside
TRAIN_MIN = data_tr[TARGETS].min().to_numpy()
TRAIN_MAX = data_tr[TARGETS].max().to_numpy()
pred = np.clip(model.predict(inputs_with_logs(data_te)), TRAIN_MIN, TRAIN_MAX)

dumb = (
    DummyRegressor(strategy="mean")
    .fit(inputs_with_logs(data_tr), data_tr[TARGETS].to_numpy())
    .predict(inputs_with_logs(data_te))
)

for i, t in enumerate(TARGETS):
    print(
        f"  {t:6s}  linear R2 {r2_score(data_te[t], pred[:, i]):+.4f}"
        f"   mean-predictor {r2_score(data_te[t], dumb[:, i]):+.4f}"
    )
print(
    f"\n  mean linear R2 "
    f"{r2_score(data_te[TARGETS], pred, multioutput='uniform_average'):+.4f}"
)
