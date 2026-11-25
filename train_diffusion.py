import datetime
import json
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
from pathlib import Path

project_dir = Path.cwd()
new_folder = project_dir / "images"
new_folder.mkdir(parents=True, exist_ok=True)
new_folder = project_dir / "data"
new_folder.mkdir(parents=True, exist_ok=True)
new_folder = project_dir / "models"
new_folder.mkdir(parents=True, exist_ok=True)
new_folder = project_dir / "scalers"
new_folder.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

DEFAULT_HYPERPARAMETERS = {
    "random_seed": 42,
    "data_split": {"train_val_size": 0.3, "random_state": 42},
    "diffusion": {
        "timesteps": 100,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "beta_schedule": "linear",  # 'linear' or 'cosine'
        "time_embed_dim": 32,
        "hidden_features": 256,
        "num_layers": 4,
        "dropout_probability": 0.10,
        "activation": "silu",
    },
    "training": {
        "learning_rate": 3e-4,
        "epochs": 200,
        "train_batch_size": 512,
        "validation_batch_size": 1024,
        "test_batch_size": 512,
        "train_shuffle": True,
        "validation_shuffle": False,
        "test_shuffle": False,
        "weight_decay": 1e-4,
        "max_grad_norm": 5.0,
    },
    "scheduler": {"type": "cosine", "eta_min": 1e-6},
}

# 25 features
# FEATURES = [
#     "ChargedLep_Charge",
#     "ChargedLep_E",
#     "ChargedLep_Eta",
#     "ChargedLep_pT",
#     "HT",
#     "Higgs_E",
#     "Higgs_eta",
#     "Higgs_pT",
#     "LeadBJet_E",
#     "LeadBJet_Eta",
#     "LeadBJet_pT",
#     "MET",
#     "MET_Sig",
#     "SubLeadBJet_E",
#     "SubLeadBJet_Eta",
#     "SubLeadBJet_pT",
#     "NLightJets",
#     "SumLightJets_E",
#     "SumLightJets_Pt",
#     "SumLightJets_eta",
#     "Relative_Phi_Lep_Higgs",
#     "Relative_Phi_Lep_MET",
#     "Relative_Phi_Lep_LeadBJet",
#     "Relative_Phi_Lep_SubleadBJet",
#     "Relative_Phi_Lep_SumLightJet",
# ]

# 21 features
FEATURES = [
    "ChargedLep_Charge",
    "ChargedLep_E",
    "ChargedLep_Eta",
    "ChargedLep_pT",
    "HT",
    # "Higgs_E",
    # "Higgs_eta",
    # "Higgs_pT",
    "LeadBJet_E",
    "LeadBJet_Eta",
    "LeadBJet_pT",
    "MET",
    "MET_Sig",
    "SubLeadBJet_E",
    "SubLeadBJet_Eta",
    "SubLeadBJet_pT",
    "NLightJets",
    "SumLightJets_E",
    "SumLightJets_Pt",
    "SumLightJets_eta",
    # "Relative_Phi_Lep_Higgs",
    "Relative_Phi_Lep_MET",
    "Relative_Phi_Lep_LeadBJet",
    "Relative_Phi_Lep_SubleadBJet",
    "Relative_Phi_Lep_SumLightJet",
]

# 13 features
# FEATURES = [
#     "ChargedLep_Charge",
#     "ChargedLep_Eta",
#     "ChargedLep_pT",
#     "HT",
#     "LeadBJet_Eta",
#     "LeadBJet_pT",
#     "MET",
#     "MET_Sig",
#     "SubLeadBJet_Eta",
#     "SubLeadBJet_pT",
#     "Relative_Phi_Lep_MET",
#     "Relative_Phi_Lep_LeadBJet",
#     "Relative_Phi_Lep_SubleadBJet",
# ]

TARGET = ["True_nu_eta"]


def load_hyperparameters(filename=None):
    if filename is None:
        return DEFAULT_HYPERPARAMETERS.copy()
    with open(filename, "r") as f:
        return json.load(f)


def save_hyperparameters(hyperparameters, filename="hyperparameters_diff"):
    output_filename = f"models/{filename}_{time_str}.json"
    with open(output_filename, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    return output_filename


def read_file(filename, FEATURES, TARGET):
    # df = pd.read_csv(filename, encoding_errors="ignore")
    with h5py.File(filename, "r") as f:
        data = f["NuFindingStudies"][:]
        df = pd.DataFrame(data)


    if len(FEATURES) == 21:
        df["Relative_Phi_Lep_MET"] = df['MET_Phi'] - df['ChargedLep_Phi']
        df["Relative_Phi_Lep_LeadBJet"] = df['LeadBJet_Phi'] - df['ChargedLep_Phi']
        df["Relative_Phi_Lep_SubleadBJet"] = df['SubLeadBJet_Phi'] - df['ChargedLep_Phi']
        df["Relative_Phi_Lep_SumLightJet"] = df['SumLightJets_phi'] - df['ChargedLep_Phi']
        for col in FEATURES[-4:]:
            df[col] = np.arctan2(np.sin(df[col]), np.cos(df[col]))

    elif len(FEATURES) == 25:
        df["Relative_Phi_Lep_Higgs"] = df['Higgs_phi'] - df['ChargedLep_Phi']
        df["Relative_Phi_Lep_MET"] = df['MET_Phi'] - df['ChargedLep_Phi']
        df["Relative_Phi_Lep_LeadBJet"] = df['LeadBJet_Phi'] - df['ChargedLep_Phi']
        df["Relative_Phi_Lep_SubleadBJet"] = df['SubLeadBJet_Phi'] - df['ChargedLep_Phi']
        df["Relative_Phi_Lep_SumLightJet"] = df['SumLightJets_phi'] - df['ChargedLep_Phi']
        for col in FEATURES[-5:]:
            df[col] = np.arctan2(np.sin(df[col]), np.cos(df[col]))

    elif len(FEATURES) == 13:
        df["Relative_Phi_Lep_MET"] = df['MET_Phi'] - df['ChargedLep_Phi']
        df["Relative_Phi_Lep_LeadBJet"] = df['LeadBJet_Phi'] - df['ChargedLep_Phi']
        df["Relative_Phi_Lep_SubleadBJet"] = df['SubLeadBJet_Phi'] - df['ChargedLep_Phi']
        for col in FEATURES[-3:]:
            df[col] = np.arctan2(np.sin(df[col]), np.cos(df[col]))

    features_data = df[FEATURES].values
    target_data = df[TARGET].values
    return features_data, target_data


def split_data(features, target, hyperparameters):
    features_train, features_val, target_train, target_val = train_test_split(
        features,
        target,
        test_size=hyperparameters["data_split"]["train_val_size"],
        random_state=hyperparameters["data_split"]["random_state"],
    )

    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    features_train = scaler_features.fit_transform(features_train)
    features_val = scaler_features.transform(features_val)
    target_train = scaler_target.fit_transform(target_train)
    target_val = scaler_target.transform(target_val)
    joblib.dump(
        scaler_features, f"scalers/scaler_features_diff_{time_str}.joblib"
    )
    joblib.dump(
        scaler_target, f"scalers/scaler_target_diff_{time_str}.joblib"
    )

    features_train_tensor = torch.tensor(features_train, dtype=torch.float32)
    features_val_tensor = torch.tensor(features_val, dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train, dtype=torch.float32)
    target_val_tensor = torch.tensor(target_val, dtype=torch.float32)

    train_dataset = TensorDataset(features_train_tensor, target_train_tensor)
    val_dataset = TensorDataset(features_val_tensor, target_val_tensor)
    return train_dataset, val_dataset


# -------------------------------------------------------------------------
# Neural Network: Denoising MLP with Sinusoidal Time Embeddings
# -------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoisingMLP(nn.Module):

    def __init__(self, target_dim, context_dim, hyperparameters):
        super().__init__()
        hparams = hyperparameters["diffusion"]
        time_dim = hparams["time_embed_dim"]
        hidden_dim = hparams["hidden_features"]
        dropout = hparams["dropout_probability"]

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        in_features = target_dim + context_dim + time_dim

        layers = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(hparams["num_layers"] - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, target_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, y_t, t, context):
        # y_t: [Batch, 1], t: [Batch], context: [Batch, 25]
        t_embed = self.time_mlp(t)
        x = torch.cat([y_t, t_embed, context], dim=-1)
        return self.network(x)


# -------------------------------------------------------------------------
# Diffusion Process Container (DDPM)
# -------------------------------------------------------------------------
class ConditionalDDPM(nn.Module):

    def __init__(self, target_dim, context_dim, hyperparameters):
        super().__init__()
        self.hparams = hyperparameters["diffusion"]
        self.T = self.hparams["timesteps"]

        # Variance schedule
        if self.hparams["beta_schedule"] == "linear":
            betas = torch.linspace(
                self.hparams["beta_start"], self.hparams["beta_end"], self.T
            )
        elif self.hparams["beta_schedule"] == "cosine":
            steps = self.T + 1
            x = torch.linspace(0, self.T, steps)
            alphas_cumprod = (
                torch.cos(((x / self.T) + 0.008) / (1 + 0.008) * torch.pi * 0.5)
                ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers so they are saved with model and placed on GPU automatically
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # Posterior variance: \tilde{\beta}_t
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        self.model = DenoisingMLP(target_dim, context_dim, hyperparameters)

    def forward_q(self, y_0, t, noise=None):
        """Forward diffusion step: compute y_t in closed form."""
        if noise is None:
            noise = torch.randn_like(y_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = (
            self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        )
        return (
            sqrt_alphas_cumprod_t * y_0
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def compute_loss(self, y_0, context):
        """Train step: sample t and noise, predict noise, and compute MSE loss."""
        batch_size = y_0.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=y_0.device).long()
        noise = torch.randn_like(y_0)

        y_t = self.forward_q(y_0, t, noise)
        predicted_noise = self.model(y_t, t, context)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample_ddpm(self, context, eta=1.0):
        batch_size = context.shape[0]
        y_t = torch.randn((batch_size, 1), device=context.device)

        for step in reversed(range(self.T)):
            t = torch.full(
                (batch_size,), step, device=context.device, dtype=torch.long
            )
            predicted_noise = self.model(y_t, t, context)

            alpha_bar_t = self.alphas_cumprod[step]
            alpha_bar_prev = self.alphas_cumprod_prev[step]

            # 1. Estimate clean y0
            pred_y0 = (y_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

            # 2. DDIM step (deterministic when eta = 0.0)
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            direction_pointing_to_yt = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=1e-8)
            ) * predicted_noise

            y_t = torch.sqrt(alpha_bar_prev) * pred_y0 + direction_pointing_to_yt
            if eta > 0.0 and step > 0:
                y_t += sigma * torch.randn_like(y_t)

        return y_t

    @torch.no_grad()
    def sample_ddim(self, context, eta=0.0, sampling_steps=50):
        batch_size = context.shape[0]
        y_t = torch.randn((batch_size, 1), device=context.device)

        timesteps = (
            torch.linspace(0, self.T - 1, steps=sampling_steps).long().to(context.device)
        )

        for i in reversed(range(len(timesteps))):
            step = timesteps[i]
            prev_step = timesteps[i - 1] if i > 0 else None

            t = torch.full(
                (batch_size,), step, device=context.device, dtype=torch.long
            )
            predicted_noise = self.model(y_t, t, context)

            alpha_bar_t = self.alphas_cumprod[step]

            alpha_bar_prev = (
                self.alphas_cumprod[prev_step]
                if prev_step is not None
                else torch.tensor(1.0, device=context.device)
            )

            pred_y0 = (
                              y_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
                      ) / torch.sqrt(alpha_bar_t)
            pred_y0 = torch.clamp(pred_y0, -4.0, 4.0)

            sigma = eta * torch.sqrt(
                (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_prev)
            )
            direction = (
                    torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=1e-8))
                    * predicted_noise
            )

            y_t = torch.sqrt(alpha_bar_prev) * pred_y0 + direction
            if eta > 0.0 and prev_step is not None:
                y_t += sigma * torch.randn_like(y_t)

        return y_t


# -------------------------------------------------------------------------
# Training Routine
# -------------------------------------------------------------------------
def train_model(diffusion, train_dataset, val_dataset, hyperparameters):
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["training"]["train_batch_size"],
        shuffle=hyperparameters["training"]["train_shuffle"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters["training"]["validation_batch_size"],
        shuffle=hyperparameters["training"]["validation_shuffle"],
    )

    best_val_loss = float("inf")
    optimizer = optim.AdamW(
        diffusion.parameters(),
        lr=hyperparameters["training"]["learning_rate"],
        weight_decay=hyperparameters["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=hyperparameters["training"]["epochs"],
        eta_min=hyperparameters["scheduler"]["eta_min"],
    )

    print(f"Starting Diffusion training on {device}...")
    for epoch in range(hyperparameters["training"]["epochs"]):
        diffusion.train()
        total_train_loss = 0.0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            loss = diffusion.compute_loss(batch_targets, batch_features)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                diffusion.parameters(),
                max_norm=hyperparameters["training"]["max_grad_norm"],
            )
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        diffusion.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                loss = diffusion.compute_loss(batch_targets, batch_features)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()

        print(
            f"Epoch [{epoch + 1:03d}/{hyperparameters['training']['epochs']}] | "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                diffusion.state_dict(), f"models/model_diff_{time_str}.pt"
            )
            print(
                f"  --> Saved new best model with Val MSE: {best_val_loss:.5f}"
            )


if __name__ == "__main__":
    hyperparameter_file = 'config_diffusion.json'  # Set to "config.json" if using external file
    hyperparameters = load_hyperparameters(hyperparameter_file)

    torch.manual_seed(hyperparameters["random_seed"])
    save_hyperparameters(hyperparameters)
    print("Hyperparameters saved.")

    n_features = len(FEATURES)
    n_target = len(TARGET)

    features, target = read_file(
        "data/new_Input_WH_NeutrinoEtaFindingForTraining_4th_October2026.h5",
        FEATURES,
        TARGET,
    )
    train_dataset, val_dataset = split_data(features, target, hyperparameters)

    diffusion_model = ConditionalDDPM(
        target_dim=n_target,
        context_dim=n_features,
        hyperparameters=hyperparameters,
    ).to(device)

    train_model(
        diffusion=diffusion_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        hyperparameters=hyperparameters,
    )