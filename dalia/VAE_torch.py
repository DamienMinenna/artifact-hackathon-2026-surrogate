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

# Load and cure data
PACK = "data/sample"
data = cure_dataset(pd.read_parquet(f"{PACK}/campaign_B_moments.parquet"))
norm_data, data_mean, data_std = normalize_dataset(data)

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
loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=256, shuffle=True)


# Create VAE model
def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)

class VAE(nn.Module):
    def __init__(self, n_in, n_out, hidden=(256, 64, 16), latent=2):
        super(VAE, self).__init__()

        self.enc = mlp([n_in, *hidden])
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.dec = mlp([latent, *hidden[::-1], n_out])


    def forward(self, x):
        e = self.enc(x)
        mu, logvar = self.mu(e), self.logvar(e)
        z = mu + torch.randn_like(mu) * torch.exp(0.5*logvar)
        return self.dec(z), mu, logvar

    @torch.no_grad()
    def predict(self, X):
        return model.dec(model.mu(model.enc(X)))

# Setup VAE loss function with KL loss
def loss_fn(y_pred, y_true, mu, logvar):
    c_loss = torch.square(y_pred - y_true).sum(-1).mean()
    kl_loss = (-0.5 * (1 + logvar - torch.square(mu) - torch.exp(logvar)).sum(-1)).mean()

    return c_loss + 1e-3*kl_loss, c_loss, kl_loss

# Setup Hyperparameters
EPOCHS = 1_000
LR = 1e-3

# Setup VAE
model = VAE(n_in=len(FEATURES), n_out=len(TARGETS), latent=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Start training VAE
print("Start training VAE")
pbar = tqdm(range(EPOCHS), desc="VAE")
loss_arr = {"train": [], "test": []}

for epoch in pbar:
    tot, n = 0., 0

    model.train()
    # Training loop
    for x, y in loader:
        optimizer.zero_grad()
        y_pred, mu, logvar = model(x)
        loss, recon, kl = loss_fn(y_pred, y, mu, logvar)
        loss.backward()
        optimizer.step()
        tot += recon.item() * len(x)
        n += len(x)

    # Eval loop
    if epoch%5==0:
        model.eval()
        with torch.no_grad():
            pred = model.dec(model.mu(model.enc(X_test)))
            per_target = torch.square(pred - Y_test).mean(0)

    # Print info
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