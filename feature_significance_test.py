from train_diffusion import (
    device, FEATURES, TARGET, read_file, ConditionalDDPM, load_hyperparameters)
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def compute_permutation_importance(
        diffusion_model,
        dataset,
        features_test_scaled,
        target_test,
        model_name,
        batch_size=2048
):
    diffusion_model.load_state_dict(torch.load(model_name, map_location=device))
    diffusion_model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    # 1. Compute Baseline Loss
    total_loss = 0.0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = diffusion_model.compute_loss(by, bx)
            total_loss += loss.item()
    baseline_loss = total_loss / len(loader)
    print(f"Baseline Validation Loss: {baseline_loss:.4f}\n")

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
                loss = diffusion_model.compute_loss(by,bx)
                perm_loss += loss.item()

        shuffled_loss = perm_loss / len(loader)
        delta_loss = shuffled_loss - baseline_loss
        importances[name] = delta_loss

        print(
            f"  Feature [{j + 1:02d}/{len(FEATURES)}] {name:<30} | Delta Loss = {delta_loss:+.4f}"
        )

    return importances, baseline_loss


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
    plt.xlim(-0.05,0.35)
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
    time = "20261016-211547"
    data_filename = (
        "data/new_Input_WTH_NeutrinoEtaFindingEvaluation_4th_October2026.h5"
    )

    hyperparameters = load_hyperparameters(
        f"models/hyperparameters_diff_{time}.json"
    )
    model_name = f"models/model_diff_{time}.pt"

    features_test, target_test = read_file(data_filename, FEATURES, TARGET)

    scaler_features = joblib.load(f"scalers/scaler_features_diff_{time}.joblib")
    scaler_target = joblib.load(f"scalers/scaler_target_diff_{time}.joblib")

    features_test = scaler_features.transform(features_test)
    target_test = scaler_target.transform(target_test)

    # Build PyTorch Dataset
    dataset = TensorDataset(
        torch.tensor(features_test, dtype=torch.float32),
        torch.tensor(target_test, dtype=torch.float32),
    )

    diffusion_model = ConditionalDDPM(
        target_dim=len(TARGET),
        context_dim=len(FEATURES),
        hyperparameters=hyperparameters,
    ).to(device)

    importances, baseline_loss = compute_permutation_importance(
        diffusion_model=diffusion_model,
        model_name=model_name,
        features_test_scaled=features_test,
        target_test=target_test,
        dataset=dataset,
    )

    plot_feature_importance(importances)