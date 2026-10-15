import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

def cure_dataset(df: pd.DataFrame):
    rename_data = df[["P_max", "cN2_max", "L_inj", "dip_frac", "x_of", "plane_mm",
                        "mean_ux", "mean_uy", "mean_uz", "q_tot_pC"]].copy()

    rename_data["mean_zeta"] = df["mean_zeta_um"]
    rename_data["mean_y"] = df["mean_y_um"]
    rename_data["mean_z"] = df["mean_z_um"]
    rename_data["sigma_zeta"] = np.sqrt(df["cov_00"])
    rename_data["sigma_y"] = np.sqrt(df["cov_11"])
    rename_data["sigma_z"] = np.sqrt(df["cov_22"])
    rename_data["sigma_ux"] = np.sqrt(df["cov_33"])
    rename_data["sigma_uy"] = np.sqrt(df["cov_44"])
    rename_data["sigma_uz"] = np.sqrt(df["cov_55"])

    rename_data.dropna(inplace=True)

    return rename_data

def normalize_dataset(df: pd.DataFrame):
    df_mean = df.mean()
    df_std = df.std()
    norm_df = (df - df_mean) / (df_std)

    return norm_df, df_mean, df_std

def transform_dataset(df: pd.DataFrame):
    k1 = 0.02
    k2 = 0.1
    df["t_mean_y"] = np.tanh(df["mean_y"]) + k1 * df["mean_y"]
    df["t_mean_z"] = np.tanh(df["mean_z"]) + k1 * df["mean_z"]
    df["t_mean_uy"] = np.tanh(df["mean_uy"]) + k2 * df["mean_uy"]
    df["t_mean_uz"] = np.tanh(df["mean_uz"]) + k2 * df["mean_uz"]

    return df

# Load and cure data
PACK = "data/sample"
data = cure_dataset(pd.read_parquet(f"{PACK}/campaign_B_moments.parquet"))
norm_data, data_mean, data_std = normalize_dataset(data)
# norm_data = transform_dataset(norm_data)

# Find GPU devices to train on if any else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Set device to {device}")

# Split data between FEATURES and TARGETS
FEATURES = ["P_max", "cN2_max", "L_inj", "dip_frac", "x_of", "plane_mm"]
TARGETS = ["mean_zeta", "mean_y", "mean_z",
           "mean_ux", "mean_uy", "mean_uz",
           "sigma_zeta", "sigma_y", "sigma_z",
           "sigma_ux", "sigma_uy", "sigma_uz",
           "q_tot_pC"]

# Convert dataframe to tensor
torch_features = torch.tensor(norm_data[FEATURES].to_numpy(), device=device, dtype=torch.float32)
torch_targets = torch.tensor(norm_data[TARGETS].to_numpy(), device=device, dtype=torch.float32)

# Split train / test
X_train, X_test, Y_train, Y_test = train_test_split(torch_features, torch_targets, test_size=0.15, random_state=0)
# Create Dataloader for the training loop
loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=512, shuffle=True, pin_memory=True)

class TRM(nn.Module):
    def __init__(self, n_in, n_out, d=128, n_latent=4, n_cycles=2):
        super().__init__()
        self.n_latent, self.n_cycles = n_latent, n_cycles
        self.x_emb = nn.Linear(n_in, d)
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d)
        )
        self.head = nn.Linear(d, n_out)
        self.y0 = nn.Parameter(torch.randn(d) * 0.02)
        self.z0 = nn.Parameter(torch.randn(d) * 0.02)

    def init_state(self, B):
        return self.y0.expand(B, -1), self.z0.expand(B, -1)

    def cycle(self, xe, y, z):
        for _ in range(self.n_latent):
            z = z + self.net(xe + y + z)
        y = y + self.net(y + z)
        return y, z

    def forward(self, x, y, z):
        xe = self.x_emb(x)
        with torch.no_grad():
            for _ in range(self.n_cycles - 1):
                y, z = self.cycle(xe, y, z)
        y, z = self.cycle(xe, y, z)
        return self.head(y), y.detach(), z.detach()

    @torch.no_grad()
    def predict(X, n_sup):
        model.eval()
        y, z = model.init_state(len(X))
        for _ in range(n_sup):
            pred, y, z = model(X, y, z)
        return pred


N_SUP = 4
EPOCHS = 100
model = TRM(len(FEATURES), len(TARGETS)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

print("Start training TRM")
pbar = tqdm(range(EPOCHS), desc="VAE")
loss_arr = {"train": [], "test": []}

for epoch in pbar:
    model.train()
    tot, n = 0.0, 0
    for x, yt in loader:
        y, z = model.init_state(len(x))
        for _ in range(N_SUP):
            pred, y, z = model(x, y, z)
            loss = nn.functional.mse_loss(pred, yt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += loss.item() * len(x); n += len(x)
    sched.step()

    if epoch % 10 == 0:
        mse = ((model.predict(X_test) - Y_test) ** 2).mean(0)
        pbar.set_postfix(train=f"{tot / n / len(TARGETS):.4f}", test=f"{mse.mean():.3f}")
        loss_arr["train"].append(tot / n / len(TARGETS))
        loss_arr["test"].append(mse.mean())

# Plot training loss
x = np.arange(EPOCHS)
fig, ax = plt.subplots()
ax.semilogy(x,loss_arr["train"], label="Train loss")
ax.semilogy(x,loss_arr["test"], label="Test loss")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlabel("Training steps")
ax.set_ylabel("Loss value")
ax.set_title("Training of VAE model")
fig.tight_layout()
plt.show()

# Compute prediction of the trained model
model.eval()
pred = model.predict(X_test)
y_true = Y_test.cpu().numpy()
y_pred = pred.cpu().numpy()

# Compute R² score
r2 = r2_score(y_true, y_pred, multioutput="raw_values")
for col, s in zip(TARGETS, r2):
    print(f"{col:12s} R² = {s:.3f}")
print(f"{'mean':12s} R² = {r2.mean():.3f}")