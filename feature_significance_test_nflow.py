from train_nflow import (
    device, FEATURES, TARGET, read_file, normalizing_flow, load_hyperparameters)
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
from sklearn.model_selection import train_test_split
from scipy.stats import kstest,pearsonr, spearmanr
from matplotlib.colors import LogNorm
import torch.nn.functional as F
import pandas as pd


import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset



def compute_permutation_importance(
    flow,
    dataset,
    features_test_scaled,
    target_test,
    model_name,
    batch_size=2048
):
    flow.load_state_dict(torch.load(model_name, map_location=device))
    flow.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    
    # 1. Compute Baseline Loss (NLL)
    total_loss = 0.0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            total_loss += -flow.log_prob(inputs=by, context=bx).sum().item()
    baseline_nll = total_loss / len(test_dataset)
    print(f"Baseline Validation NLL: {baseline_nll:.4f}\n")

    importances = {}

    # 2. Permute Each Feature One by One
    print(f"Evaluating Permutation Importance across {len(FEATURES)} features...")
    for j, name in enumerate(FEATURES):
        # Deep copy the features
        features_perm = np.copy(features_test_scaled)

        # Randomly shuffle ONLY column j
        np.random.seed(42)
        features_perm[:, j] = np.random.permutation(features_perm[:, j])

        perm_dataset = TensorDataset(
            torch.tensor(features_perm, dtype=torch.float32),
            torch.tensor(target_test, dtype=torch.float32),
        )
        perm_loader = DataLoader(
            perm_dataset, batch_size=batch_size, shuffle=False
        )

        perm_loss = 0.0
        with torch.no_grad():
            for bx, by in perm_loader:
                bx, by = bx.to(device), by.to(device)
                perm_loss += -flow.log_prob(inputs=by, context=bx).sum().item()

        shuffled_nll = perm_loss / len(test_dataset)
        delta_nll = shuffled_nll - baseline_nll
        importances[name] = delta_nll

        print(
            f"  Feature [{j+1:02d}/{len(FEATURES)}] {name:<30} | Delta NLL = {delta_nll:+.4f}"
        )

    return importances, baseline_nll


# ----------------------------------------------------
# Plotting Function: Horizontal Ranking Bar Chart
# ----------------------------------------------------
def plot_feature_importance(importances):
    # Sort features from least to most important
    sorted_features = sorted(importances.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_features]
    scores = [x[1] for x in sorted_features]

    plt.figure(figsize=(12, 7.5))
    y_pos = np.arange(len(names))

    # Color gradient (more important = darker color)
    norm_scores = (np.array(scores) - min(scores)) / (
        max(scores) - min(scores) + 1e-6
    )
    colors = plt.cm.Blues(0.35 + 0.65 * norm_scores)

    bars = plt.barh(
        y_pos, scores, align="center", color=colors, edgecolor="#1E3A8A", lw=1.2
    )
    plt.yticks(y_pos, names, fontsize=10)
    plt.xlabel(
        r"Permutation Feature Importance",
        fontsize=11,
    )
    plt.title(
        r"Input Feature Significance Ranking for Neutrino $\eta_\nu$ Reconstruction",
        fontsize=13,
        pad=12,
    )
    plt.axvline(0, color="black", linestyle="--", lw=1.0, alpha=0.7)

    # Annotate value on each bar
    for bar in bars:
        width = bar.get_width()
        offset = 0.005 if width >= 0 else -0.02
        plt.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=8.5,
            color="#1F2937",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    time = '20261026-205717'
    data_filename = 'data/new_Input_WLH_NeutrinoEtaFindingEvaluation_4th_October2026.csv'
    hyperparameter_file = f"models/hyperparameters_nflow_{time}.json"
    hyperparameters = load_hyperparameters(hyperparameter_file)

    model_name = f"models/model_nflow_{time}.pt"
    n_features = len(FEATURES)
    n_target = len(TARGET)
    features_test, target_test = read_file(data_filename, FEATURES, TARGET)
    # Load Scalers
    scaler_features = joblib.load(f"scalers/scaler_features_{time}.joblib")

    features_test = scaler_features.transform(features_test)

    features_test_tensor = torch.tensor(features_test, dtype=torch.float32).to(device)
    target_test_tensor = torch.tensor(target_test, dtype=torch.float32).to(device)
    test_dataset = TensorDataset(features_test_tensor, target_test_tensor)

    flow = normalizing_flow(
        features=n_target,
        context_features=n_features,
        hyperparameters=hyperparameters)
    
    
    importances, baseline_nll = compute_permutation_importance(
        flow=flow,
        model_name=model_name,
        features_test_scaled=features_test,
        target_test=target_test,
        dataset=test_dataset,
    )

    plot_feature_importance(importances)
