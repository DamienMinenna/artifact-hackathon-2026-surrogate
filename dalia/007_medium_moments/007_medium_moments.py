import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from pallas_score import (
    normalised_to_physical,
    physical_to_normalised,
    score_submission,
)

# 1. Hardware setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)  # Prevent PyTorch OpenMP from hogging all 36 CPU cores
print(
    f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
)

PACK = "/workspace/data/surrogate-models/data/pallas_hackathon_data"


TARGETS = [
    "mean_zeta_um",
    "mean_y_um",
    "mean_z_um",
    "mean_ux",
    "mean_uy",
    "mean_uz",
]
FEATURES = ["P_max", "cN2_max", "L_inj", "dip_frac", "x_of", "plane_mm"]
KEY = ["row_id"]

# 2. Data loading & cleaning
data = pd.read_parquet(f"{PACK}/campaign_B_moments.parquet")
data = data[(data.group == "all") & (data.q_tot_pC > 0)].reset_index(drop=True)
data["row_id"] = np.arange(len(data))
test_inputs = pd.read_parquet(f"{PACK}/test_moments_planes.parquet")
n_rows = len(data)


def features(frame):
    return frame[FEATURES].to_numpy()


def targets(frame):
    return np.column_stack([physical_to_normalised(t, frame[t]) for t in TARGETS])


def physical(pred):
    return np.column_stack(
        [normalised_to_physical(t, pred[:, i]) for i, t in enumerate(TARGETS)]
    )


def score(pred, frame, zone=None):
    pred = pd.DataFrame(physical(pred), columns=TARGETS)
    pred.insert(0, "row_id", frame.row_id.to_numpy())
    return score_submission(pred, frame[KEY + TARGETS], zone=zone)


# Train / test split using direct indexing
train_idx, test_idx = train_test_split(np.arange(n_rows), test_size=0.2, random_state=0)
train, test = data.iloc[train_idx], data.iloc[test_idx]

print(f"{len(data):,} total rows, {len(train):,} train / {len(test):,} test")


# 3. Linear baseline
lin = make_pipeline(StandardScaler(), LinearRegression()).fit(
    features(train), targets(train)
)
print("Linear Regression score:")
print(score(lin.predict(features(test)), test))

# 4. Random Forest baseline
rf = RandomForestRegressor(
    n_estimators=100,
    n_jobs=16,  # Cap worker threads to prevent full CPU thrashing
    random_state=0,
).fit(features(train), targets(train))
print("Random Forest score:")
print(score(rf.predict(features(test)), test))


# 5. MLP on GB200 GPU
def fit_mlp(train_df, val_df, epochs=5, batch=512, lr=1e-3):
    X_tr_np = features(train_df)
    y_tr_np = targets(train_df)
    X_va_np = features(val_df)
    y_va_np = targets(val_df)

    scaler = StandardScaler().fit(X_tr_np)
    y_mean = torch.tensor(y_tr_np.mean(0), dtype=torch.float32, device=device)
    y_std = torch.tensor(y_tr_np.std(0), dtype=torch.float32, device=device)

    # Pre-load full feature tensors directly onto GPU VRAM
    X = torch.tensor(scaler.transform(X_tr_np), dtype=torch.float32, device=device)
    y = (torch.tensor(y_tr_np, dtype=torch.float32, device=device) - y_mean) / y_std
    Xv = torch.tensor(scaler.transform(X_va_np), dtype=torch.float32, device=device)
    yv = (torch.tensor(y_va_np, dtype=torch.float32, device=device) - y_mean) / y_std

    net = nn.Sequential(
        nn.Linear(X.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, len(TARGETS)),
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    val_loss = []
    n_samples = len(X)

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(net(X[idx]), y[idx])
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            val_loss.append(loss_fn(net(Xv), yv).item())

    def predict(frame):
        net.eval()
        with torch.no_grad():
            X_eval = torch.tensor(
                scaler.transform(features(frame)),
                dtype=torch.float32,
                device=device,
            )
            return (net(X_eval) * y_std + y_mean).cpu().numpy()

    return predict, val_loss


torch.manual_seed(0)
mlp, val_loss = fit_mlp(train, test)
print("MLP val loss (end of run):", round(val_loss[-1], 4))
print(score(mlp(test), test))

# 6. Cross validation (5-fold)
rows, curves = [], []
kf = KFold(5, shuffle=True, random_state=0)

for k, (tr_idx, va_idx) in enumerate(kf.split(data)):
    # O(1) row slicing using iloc instead of isin()
    tr, va = data.iloc[tr_idx], data.iloc[va_idx]

    lin_k = make_pipeline(StandardScaler(), LinearRegression()).fit(
        features(tr), targets(tr)
    )
    rf_k = RandomForestRegressor(n_estimators=100, n_jobs=16, random_state=0).fit(
        features(tr), targets(tr)
    )

    torch.manual_seed(k)
    mlp_k, val_loss = fit_mlp(tr, va)
    curves.append(val_loss)
    rows.append(
        {
            "linear": score(lin_k.predict(features(va)), va)["mean"],
            "rf": score(rf_k.predict(features(va)), va)["mean"],
            "mlp": score(mlp_k(va), va)["mean"],
        }
    )
    print(
        f"fold {k}  mlp val loss {val_loss[-1]:.4f}  min {min(val_loss):.4f} at epoch {np.argmin(val_loss) + 1}"
    )

cv = pd.DataFrame(rows)
cv.loc["mean"] = cv.mean()
cv.loc["std"] = cv.iloc[:-1].std()
print("\nCV Summary:")
print(cv.round(4))


# 7. Submission file generation
submission = pd.DataFrame(physical(mlp(test_inputs)), columns=TARGETS)
submission.insert(0, "z_mm", test_inputs.z_mm.to_numpy())
submission.insert(0, "config", test_inputs.config.to_numpy())
submission.to_csv("submission_trajectory.csv", index=False)
print(f"\nWrote submission_trajectory.csv ({len(submission):,} rows)")

# 8. Benchmark comparison table
EMULATOR = {
    "E_MeV": 0.9633,
    "q_pC": 0.9331,
    "dE_pct": 0.8092,
    "emit_um": 0.8610,
    "div_mrad": 0.9469,
    "sigz_um": 0.8623,
    "mean": 0.8960,
}

print(
    pd.DataFrame(
        {
            "linear": score(lin.predict(features(test)), test),
            "rf": score(rf.predict(features(test)), test),
            "mlp": score(mlp(test), test),
            "mlp drift": score(mlp(test), test, zone="drift"),
            "internal emulator (medium bar)": EMULATOR,
        }
    )
    .T[TARGETS + ["mean"]]
    .round(3)
)
