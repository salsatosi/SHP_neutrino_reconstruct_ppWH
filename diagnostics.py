from train import (
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


time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def interval_coverage(
    flow,
    dataset,
    levels,
    model_name,
    batch_size=512
):
    flow.load_state_dict(
        torch.load(model_name, map_location=device)
    )
    flow.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    covered_counts = {
        level: 0
        for level in levels
    }

    total_events = 0

    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device)
    )

    with torch.no_grad():

        for features, targets in loader:

            features = features.to(device)
            targets = targets.to(device)

            batch_size_actual = targets.shape[0]

            for level in levels:

                alpha = (1.0 - level) / 2.0

                # --------------------------------
                # Gaussian latent quantiles
                # --------------------------------

                lower_prob = torch.tensor(
                    alpha,
                    device=device
                )

                upper_prob = torch.tensor(
                    1.0 - alpha,
                    device=device
                )

                z_lower_value = normal.icdf(lower_prob)
                z_upper_value = normal.icdf(upper_prob)

                z_lower = torch.full(
                    (batch_size_actual, 1),
                    z_lower_value.item(),
                    device=device
                )

                z_upper = torch.full(
                    (batch_size_actual, 1),
                    z_upper_value.item(),
                    device=device
                )

                # --------------------------------
                # Transform latent -> eta
                # --------------------------------

                lower, _ = flow._transform.inverse(
                    z_lower,
                    context=features
                )

                upper, _ = flow._transform.inverse(
                    z_upper,
                    context=features
                )

                inside = (
                    (targets >= lower) &
                    (targets <= upper)
                )

                covered_counts[level] += (
                    inside.sum().item()
                )

            total_events += targets.numel()

    print("\nCalibration / Coverage Test")
    print("-" * 55)

    results = {}

    for level in levels:

        observed = (
            covered_counts[level]
            / total_events
        )

        difference = observed - level

        results[level] = observed

        # Approx binomial uncertainty
        standard_error = np.sqrt(
            level * (1.0 - level)
            / total_events
        )

        print(
            f"Nominal {level*100:5.1f}% | "
            f"Observed {observed*100:6.3f}% | "
            f"Difference {difference*100:+6.3f}% | "
            f"SE {standard_error*100:.3f}%"
        )

    return results


def shuffled_context_nll(flow, dataset, batch_size,model_name):
    flow.load_state_dict(torch.load(model_name))
    flow.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    total_normal_nll = 0.0
    total_shuffled_nll = 0.0
    total_events = 0

    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            batch_size_actual = batch_targets.size(0)

            # -------------------------------------------------
            # 1. Correct context
            # -------------------------------------------------
            log_prob_normal = flow.log_prob(
                inputs=batch_targets,
                context=batch_features
            )

            nll_normal = -log_prob_normal

            # -------------------------------------------------
            # 2. Randomly shuffled context
            # -------------------------------------------------
            permutation = torch.randperm(
                batch_size_actual,
                device=device
            )

            shuffled_features = batch_features[permutation]

            log_prob_shuffled = flow.log_prob(
                inputs=batch_targets,
                context=shuffled_features
            )

            nll_shuffled = -log_prob_shuffled

            # Sum over events
            total_normal_nll += nll_normal.sum().item()
            total_shuffled_nll += nll_shuffled.sum().item()
            total_events += batch_size_actual

    avg_normal_nll = total_normal_nll / total_events
    avg_shuffled_nll = total_shuffled_nll / total_events

    print("\nShuffled-context test")
    print("-" * 45)
    print(f"Normal context NLL   : {avg_normal_nll:.4f}")
    print(f"Shuffled context NLL : {avg_shuffled_nll:.4f}")
    print(
        f"NLL increase         : "
        f"{avg_shuffled_nll - avg_normal_nll:.4f}"
    )

    return avg_normal_nll, avg_shuffled_nll


def binned_coverage(
    flow,
    dataset,
    feature_index,
    model_name,
    levels=(0.68, 0.95),
    num_bins=10,
    num_samples=1000
):
    flow.load_state_dict(torch.load(model_name))
    flow.eval()

    all_features = []
    all_targets = []
    all_samples = []

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False
    )

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            samples = flow.sample(
                num_samples=num_samples,
                context=features
            ).squeeze(-1)

            all_features.append(features.cpu())
            all_targets.append(targets.squeeze(-1).cpu())
            all_samples.append(samples.cpu())

    all_features = torch.cat(all_features)
    all_targets = torch.cat(all_targets)
    all_samples = torch.cat(all_samples)

    # Variable to bin
    x = all_features[:, feature_index].numpy()

    # Equal-population bins
    bin_edges = np.quantile(
        x,
        np.linspace(0, 1, num_bins + 1)
    )

    print("\nBinned calibration")
    print("-" * 75)

    for i in range(num_bins):

        # Include upper edge in final bin
        if i == num_bins - 1:
            mask = (
                (x >= bin_edges[i]) &
                (x <= bin_edges[i + 1])
            )
        else:
            mask = (
                (x >= bin_edges[i]) &
                (x < bin_edges[i + 1])
            )

        targets_bin = all_targets[mask]
        samples_bin = all_samples[mask]

        print(
            f"\nBin {i+1}: "
            f"[{bin_edges[i]:.3f}, "
            f"{bin_edges[i+1]:.3f}] "
            f"N={mask.sum()}"
        )

        for level in levels:
            alpha = (1.0 - level) / 2.0

            lower = torch.quantile(
                samples_bin,
                alpha,
                dim=1
            )

            upper = torch.quantile(
                samples_bin,
                1.0 - alpha,
                dim=1
            )

            inside = (
                (targets_bin >= lower) &
                (targets_bin <= upper)
            )

            observed = inside.float().mean().item()

            print(
                f"  {level*100:5.1f}% nominal -> "
                f"{observed*100:5.2f}% observed "
                f"({(observed-level)*100:+.2f}%)"
            )


def pit_test(
    flow,
    dataset,
    model_name,
    batch_size=512,
    num_bins=20
):
    flow.load_state_dict(
        torch.load(model_name, map_location=device)
    )
    flow.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    pit_values = []

    # Standard normal CDF
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device)
    )

    with torch.no_grad():

        for features, targets in loader:

            features = features.to(device)
            targets = targets.to(device)

            # Transform data -> latent Gaussian
            #
            # z has shape [batch_size, 1]
            z, _ = flow._transform(
                targets,
                context=features
            )

            # Exact PIT:
            # u = Phi(z)
            pit_batch = normal.cdf(z).squeeze(-1)

            pit_values.append(
                pit_batch.cpu().numpy()
            )

    pit_values = np.concatenate(pit_values)

    # --------------------------------
    # PIT histogram
    # --------------------------------


    plt.hist(
        pit_values,
        bins=num_bins,
        range=(0, 1),
        density=True,
        alpha=0.7
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="Ideal uniform"
    )

    plt.xlabel("PIT value")
    plt.ylabel("Density")
    plt.title("PIT Histogram")
    # plt.legend()
    plt.show()

    # --------------------------------
    # Statistics
    # --------------------------------

    print("\nPIT test")
    print("-" * 45)

    print(f"Number of events : {len(pit_values)}")
    print(f"Mean PIT         : {pit_values.mean():.6f}")
    print(f"Std PIT          : {pit_values.std():.6f}")

    print("Ideal mean       : 0.500000")
    print(
        f"Ideal std        : "
        f"{np.sqrt(1.0 / 12.0):.6f}"
    )

    ks_stat, p_value = kstest(
        pit_values,
        "uniform"
    )

    print(f"KS statistic     : {ks_stat:.6f}")
    print(f"KS p-value       : {p_value:.6e}")

    return pit_values


def point_reconstruction_test(
    flow,
    dataset,
    model_name,
    num_samples=2000,
    batch_size=512,
    estimator="mode",  # Options: "mode", "median", "mean"
    grid_range=(-5.0, 5.0),
    num_grid_points=1000,
):
    flow.load_state_dict(torch.load(model_name))
    flow.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_targets = []

    # Pre-build grid for mode evaluation (1D across the physical domain)
    grid = torch.linspace(
        grid_range[0], grid_range[1], num_grid_points, device=device
    )

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            batch_sz = features.size(0)

            # ---------------------------------------------------------
            # MODE (MAP): Evaluate exact log_prob across a fine grid
            # ---------------------------------------------------------
            if estimator == "mode":
                # Expand grid for batch: [B * num_grid_points, 1]
                grid_expanded = (
                    grid.view(1, num_grid_points, 1)
                    .expand(batch_sz, num_grid_points, 1)
                    .reshape(batch_sz * num_grid_points, 1)
                )

                # Expand context features to match: [B * num_grid_points, context_dim]
                context_expanded = (
                    features.unsqueeze(1)
                    .expand(batch_sz, num_grid_points, -1)
                    .reshape(batch_sz * num_grid_points, -1)
                )

                # Exact log likelihood evaluation from the flow
                log_probs = flow.log_prob(
                    inputs=grid_expanded, context=context_expanded
                )
                log_probs = log_probs.view(batch_sz, num_grid_points)

                # Find argmax (peak of the PDF) for each event
                best_indices = torch.argmax(log_probs, dim=1)
                prediction = grid[best_indices]

            # ---------------------------------------------------------
            # MEDIAN / MEAN: Sample-based estimates
            # ---------------------------------------------------------
            elif estimator in ["median", "mean"]:
                # Shape: [batch_size, num_samples, 1]
                samples = flow.sample(num_samples=num_samples, context=features)
                samples = samples.squeeze(-1)  # [batch_size, num_samples]

                if estimator == "median":
                    prediction = torch.median(samples, dim=1).values
                else:
                    prediction = torch.mean(samples, dim=1)

            else:
                raise ValueError(
                    "estimator must be 'mode', 'median', or 'mean'"
                )

            all_predictions.append(prediction.cpu().numpy())
            all_targets.append(targets.squeeze(-1).cpu().numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Calculate residuals
    residuals = predictions - targets

    # Bias
    bias = np.mean(residuals)

    # RMS resolution
    rms = np.sqrt(np.mean(residuals**2))

    # Standard deviation of residuals
    std = np.std(residuals)

    # Robust 68% central interval
    q16 = np.percentile(residuals, 16)
    q84 = np.percentile(residuals, 84)

    central_68_width = q84 - q16
    resolution_68 = 0.5 * central_68_width

    print("\nPoint reconstruction test")
    print("-" * 45)
    print(f"Estimator              : {estimator}")
    print(f"Number of events       : {len(targets)}")
    print(f"Bias                   : {bias:.5f}")
    print(f"RMS residual           : {rms:.5f}")
    print(f"Residual std           : {std:.5f}")
    print(f"16th percentile        : {q16:.5f}")
    print(f"84th percentile        : {q84:.5f}")
    print(f"68% central width      : {central_68_width:.5f}")
    print(f"68% half-width         : {resolution_68:.5f}")

    return {
        "prediction": predictions,
        "target": targets,
        "residual": residuals,
        "bias": bias,
        "rms": rms,
        "std": std,
        "q16": q16,
        "q84": q84,
        "resolution_68": resolution_68,
    }


def plot_event_posteriors(
    flow,
    dataset,
    model_name,
    nb1_nu_eta=None,  # 1D numpy array of first quadratic solutions
    nb2_nu_eta=None,  # 1D numpy array of second quadratic solutions
    event_indices=None,
    num_random_events=6,
    grid_range=(-5.0, 5.0),
    num_grid_points=1000,
):
    """Plot the conditional PDF p(eta_nu | x) for individual test events and

    compare: - Predicted PDF - Mode (MAP estimate) - True eta_nu - Two
    analytical W-mass solutions (NB1 and NB2)
    """
    flow.load_state_dict(torch.load(model_name, map_location=device))
    flow.eval()

    # Select which event indices to visualize
    if event_indices is None:
        event_indices = np.random.choice(
            len(dataset), size=num_random_events, replace=False
        )

    num_events = len(event_indices)
    cols = 3
    rows = int(np.ceil(num_events / cols))

    # Grid over eta_nu
    grid = torch.linspace(
        grid_range[0], grid_range[1], num_grid_points, device=device
    )
    grid_inputs = grid.view(-1, 1)  # [num_grid_points, 1]

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows))
    axes = np.array(axes).reshape(-1)

    with torch.no_grad():
        for i, idx in enumerate(event_indices):
            features, target = dataset[idx]

            # Replicate context for all grid points: [num_grid_points, n_features]
            context = features.unsqueeze(0).expand(num_grid_points, -1)

            # Exact p(eta | x) = exp(log_prob)
            log_prob = flow.log_prob(inputs=grid_inputs, context=context)
            prob_density = torch.exp(log_prob).squeeze(-1).cpu().numpy()

            grid_np = grid.cpu().numpy()
            true_eta = target.item()

            # Find Mode
            mode_eta = grid_np[np.argmax(prob_density)]

            ax = axes[i]

            # 1. Predicted PDF
            ax.plot(
                grid_np,
                prob_density,
                color="royalblue",
                lw=2,
                label=r"Predicted $p(\eta_\nu | x)$",
            )

            # 2. True eta
            ax.axvline(
                true_eta,
                color="red",
                linestyle="--",
                lw=1.8,
                label=f"True $\\eta_\\nu$ ({true_eta:.2f})",
            )

            # 3. Predicted Mode
            ax.axvline(
                mode_eta,
                color="green",
                linestyle=":",
                lw=1.6,
                label=f"Mode ({mode_eta:.2f})",
            )

            # 4. Numerical solution 1 (NB1)
            if nb1_nu_eta is not None:
                val_nb1 = nb1_nu_eta[idx]
                if not np.isnan(val_nb1):
                    ax.axvline(
                        val_nb1,
                        color="darkorange",
                        linestyle="-.",
                        lw=1.4,
                        label=f"NB1 ({val_nb1:.2f})",
                    )
                else:
                    ax.plot([], [], " ", label="NB1: Complex (NaN)")

            # 5. Numerical solution 2 (NB2)
            if nb2_nu_eta is not None:
                val_nb2 = nb2_nu_eta[idx]
                if not np.isnan(val_nb2):
                    ax.axvline(
                        val_nb2,
                        color="purple",
                        linestyle="-.",
                        lw=1.4,
                        label=f"NB2 ({val_nb2:.2f})",
                    )
                else:
                    ax.plot([], [], " ", label="NB2: Complex (NaN)")

            ax.set_title(f"Event #{idx}", fontsize=11)
            ax.set_xlabel(r"$\eta_\nu$", fontsize=10)
            ax.set_ylabel("Probability Density", fontsize=10)
            ax.set_xlim(grid_range)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7.5, loc="upper right")

    # Hide unused subplots if num_events is not a multiple of cols
    for j in range(num_events, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        r"Posterior Density $p(\eta_\nu | x)$ vs. Analytical $W$ Solutions",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    plt.show()


def compute_mode(data, bins=100, data_range=None):
    """Calculates the mode (peak) of a continuous distribution using histogram binning."""
    counts, bin_edges = np.histogram(data, bins=bins, range=data_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers[np.argmax(counts)]


def invariant_mass_reconstruction(
    flow,
    features_test_scaled,
    target_test,
    scaler_features,
    model_name,
    num_samples=50,
    batch_size=512,
    num_bins = 120
):
    flow.load_state_dict(torch.load(model_name, map_location=device))
    flow.eval()

    # 1. Unscale features to physical units (GeV, radians)
    features_unscaled = scaler_features.inverse_transform(features_test_scaled)

    # 2. Extract kinematics
    lep_pt = features_unscaled[:, FEATURES.index("ChargedLep_pT")]
    lep_eta = features_unscaled[:, FEATURES.index("ChargedLep_Eta")]
    lep_e = features_unscaled[:, FEATURES.index("ChargedLep_E")]

    met = features_unscaled[:, FEATURES.index("MET")]
    phi_nu = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_MET")]

    higgs_pt = features_unscaled[:, FEATURES.index("Higgs_pT")]
    higgs_eta = features_unscaled[:, FEATURES.index("Higgs_eta")]
    higgs_e = features_unscaled[:, FEATURES.index("Higgs_E")]
    phi_h = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_Higgs")]

    true_eta = target_test.flatten()

    # 3. Base 4-momenta for Lepton (phi_lep = 0) and Higgs
    p4_lep = np.stack(
        [lep_pt, np.zeros_like(lep_pt), lep_pt * np.sinh(lep_eta), lep_e],
        axis=-1,
    )

    p4_h = np.stack(
        [
            higgs_pt * np.cos(phi_h),
            higgs_pt * np.sin(phi_h),
            higgs_pt * np.sinh(higgs_eta),
            higgs_e,
        ],
        axis=-1,
    )

    # Helper function to compute invariant mass from eta_nu
    def compute_masses(eta_nu):
        if eta_nu.ndim == 1:
            px_nu = met * np.cos(phi_nu)
            py_nu = met * np.sin(phi_nu)
            pz_nu = met * np.sinh(eta_nu)
            e_nu = met * np.cosh(eta_nu)

            p4_l = p4_lep
            p4_higgs = p4_h

        elif eta_nu.ndim == 2:
            n_s = eta_nu.shape[1]
            px_nu = np.repeat((met * np.cos(phi_nu))[:, None], n_s, axis=1)
            py_nu = np.repeat((met * np.sin(phi_nu))[:, None], n_s, axis=1)
            pz_nu = met[:, None] * np.sinh(eta_nu)
            e_nu = met[:, None] * np.cosh(eta_nu)

            p4_l = p4_lep[:, None, :]
            p4_higgs = p4_h[:, None, :]

        else:
            raise ValueError("eta_nu must be 1D or 2D array")

        p4_nu = np.stack([px_nu, py_nu, pz_nu, e_nu], axis=-1)

        # W Boson invariant mass
        p4_w = p4_l + p4_nu
        m_w_sq = (
                p4_w[..., 3] ** 2
                - (p4_w[..., 0] ** 2 + p4_w[..., 1] ** 2 + p4_w[..., 2] ** 2)
        )
        m_w = np.sqrt(np.maximum(0, m_w_sq))

        # Total WH invariant mass
        p4_wh = p4_w + p4_higgs
        m_wh_sq = (
                p4_wh[..., 3] ** 2
                - (p4_wh[..., 0] ** 2 + p4_wh[..., 1] ** 2 + p4_wh[..., 2] ** 2)
        )
        m_wh = np.sqrt(np.maximum(0, m_wh_sq))

        return m_w, m_wh

    # 4. Compute True Masses
    m_w_true, m_wh_true = compute_masses(true_eta)

    # 5. Compute Flow-Sampled Masses
    flow_samples = []
    features_tensor = torch.tensor(features_test_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            s = flow.sample(num_samples=num_samples, context=batch_x)
            flow_samples.append(s.squeeze(-1).cpu().numpy())

    eta_samples = np.concatenate(flow_samples, axis=0)
    m_w_sample, m_wh_sample = compute_masses(eta_samples)
    m_w_sampled_flat = m_w_sample.flatten()
    m_wh_sampled_flat = m_wh_sample.flatten()

    # --- Compute Modes ---
    mw_range = (0, 160)
    min_wh, max_wh = np.percentile(m_wh_true, [1, 99])
    wh_range = (min_wh, max_wh)

    mode_mw_true = compute_mode(m_w_true, bins=num_bins, data_range=mw_range)
    mode_mw_flow = compute_mode(m_w_sampled_flat, bins=num_bins, data_range=mw_range)

    mode_mwh_true = compute_mode(m_wh_true, bins=num_bins, data_range=wh_range)
    mode_mwh_flow = compute_mode(m_wh_sampled_flat, bins=num_bins, data_range=wh_range)

    # 6. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot mW ---
    axes[0].hist(
        m_w_true,
        bins=num_bins,
        range=mw_range,
        density=True,
        histtype="step",
        linewidth=2,
        color="black",
        label=rf"True $\eta_\nu$ (Mode: {mode_mw_true:.1f} GeV)",
    )
    axes[0].hist(
        m_w_sampled_flat,
        bins=num_bins,
        range=mw_range,
        density=True,
        alpha=0.5,
        color="royalblue",
        label=rf"Flow Samples (Mode: {mode_mw_flow:.1f} GeV)",
    )
    axes[0].axvline(
        80.379,
        color="red",
        linestyle="--",
        label=r"Nominal $m_W$ (80.4 GeV)",
    )
    axes[0].set_xlabel(r"$m_W$ [GeV]", fontsize=11)
    axes[0].set_ylabel("Normalized Density [GeV$^{-1}$]", fontsize=11)
    axes[0].set_title(r"$W$ Boson Invariant Mass Reconstruction", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # --- Plot mWH ---
    axes[1].hist(
        m_wh_true,
        bins=num_bins,
        range=wh_range,
        density=True,
        histtype="step",
        linewidth=2,
        color="black",
        label=rf"True $\eta_\nu$ (Mode: {mode_mwh_true:.1f} GeV)",
    )
    axes[1].hist(
        m_wh_sampled_flat,
        bins=num_bins,
        range=wh_range,
        density=True,
        alpha=0.5,
        color="darkorange",
        label=rf"Flow Samples (Mode: {mode_mwh_flow:.1f} GeV)",
    )
    axes[1].set_xlabel(r"$m_{WH}$ [GeV]", fontsize=11)
    axes[1].set_ylabel("Normalized Density [GeV$^{-1}$]", fontsize=11)
    axes[1].set_title(r"Total $WH$ Invariant Mass Reconstruction", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nInvariant Mass Reconstruction")
    print("-" * 50)
    print(f"True mW Peak (Mode)        : {mode_mw_true:.2f} GeV")
    print(f"Flow Reconstructed mW Peak : {mode_mw_flow:.2f} GeV")
    print(
        f"mW Resolution (IQR / 2)     : {0.5 * (np.percentile(m_w_sampled_flat, 75) - np.percentile(m_w_sampled_flat, 25)):.2f} GeV"
    )
    print(f"True mWH Peak (Mode)       : {mode_mwh_true:.2f} GeV")
    print(f"Flow Reconstructed mWH Peak: {mode_mwh_flow:.2f} GeV")


def mode_disambiguated_correlation(
    flow,
    dataset,
    model_name,
    batch_size=512,
    num_samples=10,  # Number of random samples per event
    num_grid=250,
    range_eta=(-4.5, 4.5),
):
    flow.load_state_dict(torch.load(model_name, map_location=device))
    flow.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 1D evaluation grid across eta_nu
    grid = torch.linspace(
        range_eta[0], range_eta[1], num_grid, device=device
    )  # [G]
    grid_expanded = grid.view(1, num_grid, 1)  # [1, G, 1]

    all_targets = []
    all_mode1 = []
    all_mode2 = []
    all_has_two_modes = []
    all_samples = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            B = features.size(0)

            # -------------------------------------------------
            # 1. Random Posterior Sampling
            # -------------------------------------------------
            samples = flow.sample(num_samples=num_samples, context=features)
            samples = samples.squeeze(-1)  # [B, num_samples]
            all_samples.append(samples.cpu().numpy())

            # -------------------------------------------------
            # 2. Grid Scan for Peak / Mode Detection
            # -------------------------------------------------
            g_in = grid_expanded.expand(B, num_grid, 1).reshape(B * num_grid, 1)
            c_in = (
                features.unsqueeze(1)
                .expand(B, num_grid, -1)
                .reshape(B * num_grid, -1)
            )

            # Exact p(eta | x)
            log_prob = flow.log_prob(inputs=g_in, context=c_in).view(B, num_grid)
            probs = torch.exp(log_prob)

            # Local maxima detection: probs[i] > probs[i-1] and probs[i] > probs[i+1]
            is_peak = (probs[:, 1:-1] > probs[:, :-2]) & (
                probs[:, 1:-1] > probs[:, 2:]
            )
            is_peak = F.pad(is_peak, (1, 1), value=False)  # [B, G]

            grid_cpu = grid.cpu().numpy()
            probs_cpu = probs.cpu().numpy()
            is_peak_cpu = is_peak.cpu().numpy()

            # Extract top 2 modes per event
            for b in range(B):
                peak_indices = np.where(is_peak_cpu[b])[0]

                if len(peak_indices) >= 2:
                    sorted_peaks = peak_indices[
                        np.argsort(-probs_cpu[b, peak_indices])
                    ]
                    m1 = grid_cpu[sorted_peaks[0]]
                    m2 = grid_cpu[sorted_peaks[1]]
                    has_two = True
                elif len(peak_indices) == 1:
                    m1 = grid_cpu[peak_indices[0]]
                    m2 = m1
                    has_two = False
                else:
                    m1 = grid_cpu[np.argmax(probs_cpu[b])]
                    m2 = m1
                    has_two = False

                all_mode1.append(m1)
                all_mode2.append(m2)
                all_has_two_modes.append(has_two)

            all_targets.append(targets.squeeze(-1).cpu().numpy())

    targets = np.concatenate(all_targets)
    mode1 = np.array(all_mode1)
    mode2 = np.array(all_mode2)
    has_two_modes = np.array(all_has_two_modes)

    # Random samples arrays
    samples_arr = np.concatenate(all_samples, axis=0)  # [N, num_samples]
    targets_repeated = np.repeat(targets, num_samples)
    samples_flat = samples_arr.flatten()

    # ----------------------------------------------------
    # Disambiguation Logic
    # ----------------------------------------------------
    diff1 = np.abs(mode1 - targets)
    diff2 = np.abs(mode2 - targets)

    # Nearest mode (oracle selection)
    nearest_mode = np.where(diff1 <= diff2, mode1, mode2)

    # Disambiguation Accuracy
    correct_branch = diff1 <= diff2
    accuracy_all = np.mean(correct_branch) * 100
    accuracy_bimodal = np.mean(correct_branch[has_two_modes]) * 100

    # ----------------------------------------------------
    # Statistical Metrics & Linear Fits
    # ----------------------------------------------------
    # 1. Random Sampling
    r_samples, _ = pearsonr(targets_repeated, samples_flat)
    slope_samp, intercept_samp = np.polyfit(targets_repeated, samples_flat, 1)
    rms_samples = np.sqrt(np.mean((samples_flat - targets_repeated) ** 2))

    # 2. Primary Mode (MAP)
    r_primary, _ = pearsonr(targets, mode1)
    slope_prim, intercept_prim = np.polyfit(targets, mode1, 1)
    rms_primary = np.sqrt(np.mean((mode1 - targets) ** 2))

    # 3. Nearest Mode (Disambiguated)
    r_nearest, _ = pearsonr(targets, nearest_mode)
    slope_near, intercept_near = np.polyfit(targets, nearest_mode, 1)
    rms_nearest = np.sqrt(np.mean((nearest_mode - targets) ** 2))

    # ----------------------------------------------------
    # Print Comprehensive Summary
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print("MODE-DISAMBIGUATED CORRELATION TEST (COMPLETE COMPARISON)")
    print("=" * 70)
    print(
        f"Events with explicit bimodal peaks : {np.sum(has_two_modes)} ({np.mean(has_two_modes)*100:.1f}%)"
    )
    print(f"Primary Mode Disambiguation Accuracy: {accuracy_all:.2f}% (All)")
    print(f"                                   : {accuracy_bimodal:.2f}% (Bimodal)")
    print("-" * 70)
    print(f"1. Random Posterior Sampling:")
    print(f"   Pearson Correlation (r)         : {r_samples:.4f}")
    print(f"   RMS Error                       : {rms_samples:.4f}")
    print(
        f"   Linear Fit                      : y = {slope_samp:.3f}x {intercept_samp:+.3f}"
    )
    print("-" * 70)
    print(f"2. Primary Mode (Global MAP):")
    print(f"   Pearson Correlation (r)         : {r_primary:.4f}")
    print(f"   RMS Error                       : {rms_primary:.4f}")
    print(
        f"   Linear Fit                      : y = {slope_prim:.3f}x {intercept_prim:+.3f}"
    )
    print("-" * 70)
    print(f"3. Nearest Mode (Kinematic Resolution - Disambiguated):")
    print(f"   Pearson Correlation (r)         : {r_nearest:.4f}")
    print(f"   RMS Error                       : {rms_nearest:.4f}")
    print(
        f"   Linear Fit                      : y = {slope_near:.3f}x {intercept_near:+.3f}"
    )
    print("=" * 70)

    # ----------------------------------------------------
    # Diagnostic Plots (3-Panel Progression)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    diag = np.linspace(range_eta[0], range_eta[1], 100)

    # Panel 1: Random Sampling
    h1 = axes[0].hist2d(
        targets_repeated,
        samples_flat,
        bins=80,
        range=[range_eta, range_eta],
        cmap="Purples",
        norm=LogNorm(),
    )
    fig.colorbar(h1[3], ax=axes[0], label="Counts (log scale)")
    axes[0].plot(diag, diag, "r--", lw=2, label="Ideal Identity ($y = x$)")
    axes[0].plot(
        diag,
        slope_samp * diag + intercept_samp,
        "k-",
        lw=1.5,
        label=f"Fit: $y = {slope_samp:.2f}x {intercept_samp:+.2f}$",
    )
    axes[0].set_xlabel(r"True $\eta_\nu$", fontsize=11)
    axes[0].set_ylabel(r"Sampled $\hat{\eta}_\nu$", fontsize=11)
    axes[0].set_title(
        rf"(a) Random Sampling ($r = {r_samples:.3f}$)", fontsize=12
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8.5)

    # Panel 2: Primary Mode (MAP)
    h2 = axes[1].hist2d(
        targets,
        mode1,
        bins=80,
        range=[range_eta, range_eta],
        cmap="Blues",
        norm=LogNorm(),
    )
    fig.colorbar(h2[3], ax=axes[1], label="Counts (log scale)")
    axes[1].plot(diag, diag, "r--", lw=2, label="Ideal Identity ($y = x$)")
    axes[1].plot(
        diag,
        slope_prim * diag + intercept_prim,
        "k-",
        lw=1.5,
        label=f"Fit: $y = {slope_prim:.2f}x {intercept_prim:+.2f}$",
    )
    axes[1].set_xlabel(r"True $\eta_\nu$", fontsize=11)
    axes[1].set_ylabel(r"Primary Mode $\hat{\eta}_\nu$ (MAP)", fontsize=11)
    axes[1].set_title(
        rf"(b) Primary Mode MAP ($r = {r_primary:.3f}$)", fontsize=12
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8.5)

    # Panel 3: Nearest Mode (Disambiguated)
    h3 = axes[2].hist2d(
        targets,
        nearest_mode,
        bins=80,
        range=[range_eta, range_eta],
        cmap="Greens",
        norm=LogNorm(),
    )
    fig.colorbar(h3[3], ax=axes[2], label="Counts (log scale)")
    axes[2].plot(diag, diag, "r--", lw=2, label="Ideal Identity ($y = x$)")
    axes[2].plot(
        diag,
        slope_near * diag + intercept_near,
        "k-",
        lw=1.5,
        label=f"Fit: $y = {slope_near:.2f}x {intercept_near:+.2f}$",
    )
    axes[2].set_xlabel(r"True $\eta_\nu$", fontsize=11)
    axes[2].set_ylabel(r"Nearest Mode $\hat{\eta}_\nu$", fontsize=11)
    axes[2].set_title(
        rf"(c) Disambiguated Mode ($r = {r_nearest:.3f}$)", fontsize=12
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left", fontsize=8.5)

    plt.suptitle(
        r"Reconstruction Progression: Generative Spread $\to$ MAP $\to$ Disambiguated Resolution",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    return {
        "r_samples": r_samples,
        "rms_samples": rms_samples,
        "r_primary": r_primary,
        "rms_primary": rms_primary,
        "r_nearest": r_nearest,
        "rms_nearest": rms_nearest,
        "accuracy_all": accuracy_all,
        "accuracy_bimodal": accuracy_bimodal,
    }


def analytical_solutions_correlation(
    true_eta,
    nb1_eta,
    nb2_eta,
    met=None,  # Optional: MET array (if None, |sinh(eta)| is used)
    seed=42,
    bins=80,
    range_eta=(-4.5, 4.5),
):
    """Performs correlation tests on the analytical quadratic solutions NB1 and

    NB2: 1. Random 50/50 selection between NB1 and NB2. 2. Minimum |p_z|
    heuristic (the paper's method from Sec. 2.3). 3. Closest solution to the
    ground truth (ideal oracle).
    """
    # ----------------------------------------------------
    # 1. Filter out complex / NaN solutions
    # ----------------------------------------------------
    valid_mask = (
        ~np.isnan(nb1_eta)
        & ~np.isnan(nb2_eta)
        & ~np.isinf(nb1_eta)
        & ~np.isinf(nb2_eta)
    )

    t = true_eta[valid_mask]
    s1 = nb1_eta[valid_mask]
    s2 = nb2_eta[valid_mask]
    m = met[valid_mask] if met is not None else None

    n_total = len(true_eta)
    n_valid = len(t)
    n_complex = n_total - n_valid

    # ----------------------------------------------------
    # Method 1: Equal Chance (Random 50/50 Choice)
    # ----------------------------------------------------
    rng = np.random.default_rng(seed=seed)
    coin_flip = rng.choice([True, False], size=n_valid)
    random_pick = np.where(coin_flip, s1, s2)

    r_rand, _ = pearsonr(t, random_pick)
    rho_rand, _ = spearmanr(t, random_pick)
    slope_rand, intercept_rand = np.polyfit(t, random_pick, 1)
    rms_rand = np.sqrt(np.mean((random_pick - t) ** 2))

    # ----------------------------------------------------
    # Method 2: Minimum |p_z| (The Paper's Heuristic, Sec 2.3)
    # ----------------------------------------------------
    if m is not None:
        pz1 = m * np.abs(np.sinh(s1))
        pz2 = m * np.abs(np.sinh(s2))
    else:
        pz1 = np.abs(np.sinh(s1))
        pz2 = np.abs(np.sinh(s2))

    minpz_pick = np.where(pz1 <= pz2, s1, s2)

    r_minpz, _ = pearsonr(t, minpz_pick)
    rho_minpz, _ = spearmanr(t, minpz_pick)
    slope_minpz, intercept_minpz = np.polyfit(t, minpz_pick, 1)
    rms_minpz = np.sqrt(np.mean((minpz_pick - t) ** 2))

    # ----------------------------------------------------
    # Method 3: Closest Solution (Oracle Upper Bound)
    # ----------------------------------------------------
    diff1 = np.abs(s1 - t)
    diff2 = np.abs(s2 - t)
    closest_pick = np.where(diff1 <= diff2, s1, s2)

    r_close, _ = pearsonr(t, closest_pick)
    rho_close, _ = spearmanr(t, closest_pick)
    slope_close, intercept_close = np.polyfit(t, closest_pick, 1)
    rms_close = np.sqrt(np.mean((closest_pick - t) ** 2))

    # Accuracy of min-|pz| vs. Oracle choice
    minpz_accuracy = np.mean(minpz_pick == closest_pick) * 100

    # ----------------------------------------------------
    # Print Summary Table
    # ----------------------------------------------------
    print("\n" + "=" * 75)
    print("ANALYTICAL SOLUTIONS (NB1 & NB2) CORRELATION TEST")
    print("=" * 75)
    print(
        f"Total events analyzed     : {n_total:,} (Valid Real: {n_valid:,} / {n_valid/n_total*100:.1f}%)"
    )
    print(
        f"Complex / NaN events      : {n_complex:,} ({n_complex/n_total*100:.1f}%)"
    )
    print("-" * 75)
    print(f"1. Method 1: Equal Chance (Random 50/50 Coin Flip):")
    print(f"   Pearson Correlation (r)  : {r_rand:.4f}")
    print(f"   RMS Error                : {rms_rand:.4f}")
    print(
        f"   Linear Fit               : y = {slope_rand:.3f}x {intercept_rand:+.3f}"
    )
    print("-" * 75)
    print(f"2. Method 2: Minimum |p_z| (Paper Heuristic, Sec 2.3):")
    print(f"   Branch Accuracy          : {minpz_accuracy:.2f}% (Matches Oracle)")
    print(f"   Pearson Correlation (r)  : {r_minpz:.4f}")
    print(f"   RMS Error                : {rms_minpz:.4f}")
    print(
        f"   Linear Fit               : y = {slope_minpz:.3f}x {intercept_minpz:+.3f}"
    )
    print("-" * 75)
    print(f"3. Method 3: Closest Solution (Theoretical Limit / Oracle):")
    print(f"   Pearson Correlation (r)  : {r_close:.4f}")
    print(f"   RMS Error                : {rms_close:.4f}")
    print(
        f"   Linear Fit               : y = {slope_close:.3f}x {intercept_close:+.3f}"
    )
    print("=" * 75)

    # ----------------------------------------------------
    # Diagnostic Plots (3-Panel Figure)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    diag = np.linspace(range_eta[0], range_eta[1], 100)

    # Panel 1: Random 50/50 Selection
    h1 = axes[0].hist2d(
        t,
        random_pick,
        bins=bins,
        range=[range_eta, range_eta],
        cmap="Blues",
        norm=LogNorm(),
    )
    fig.colorbar(h1[3], ax=axes[0], label="Counts (log scale)")
    axes[0].plot(diag, diag, "r--", lw=2, label="Ideal Identity ($y = x$)")
    axes[0].plot(
        diag,
        slope_rand * diag + intercept_rand,
        "k-",
        lw=1.5,
        label=f"Fit: $y = {slope_rand:.2f}x {intercept_rand:+.2f}$",
    )
    axes[0].set_xlabel(r"True $\eta_\nu$", fontsize=11)
    axes[0].set_ylabel(r"Random Analytical $\hat{\eta}_\nu$", fontsize=11)
    axes[0].set_title(
        rf"(a) Equal Chance 50/50 ($r = {r_rand:.3f}$)", fontsize=12
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8.5)

    # Panel 2: Minimum |p_z| (Paper Heuristic)
    h2 = axes[1].hist2d(
        t,
        minpz_pick,
        bins=bins,
        range=[range_eta, range_eta],
        cmap="Oranges",
        norm=LogNorm(),
    )
    fig.colorbar(h2[3], ax=axes[1], label="Counts (log scale)")
    axes[1].plot(diag, diag, "r--", lw=2, label="Ideal Identity ($y = x$)")
    axes[1].plot(
        diag,
        slope_minpz * diag + intercept_minpz,
        "k-",
        lw=1.5,
        label=f"Fit: $y = {slope_minpz:.2f}x {intercept_minpz:+.2f}$",
    )
    axes[1].set_xlabel(r"True $\eta_\nu$", fontsize=11)
    axes[1].set_ylabel(r"Min-$|p_z|$ Analytical $\hat{\eta}_\nu$", fontsize=11)
    axes[1].set_title(
        rf"(b) Min-$|p_z|$ Heuristic ($r = {r_minpz:.3f}$)", fontsize=12
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8.5)

    # Panel 3: Closest Solution (Oracle)
    h3 = axes[2].hist2d(
        t,
        closest_pick,
        bins=bins,
        range=[range_eta, range_eta],
        cmap="Greens",
        norm=LogNorm(),
    )
    fig.colorbar(h3[3], ax=axes[2], label="Counts (log scale)")
    axes[2].plot(diag, diag, "r--", lw=2, label="Ideal Identity ($y = x$)")
    axes[2].plot(
        diag,
        slope_close * diag + intercept_close,
        "k-",
        lw=1.5,
        label=f"Fit: $y = {slope_close:.2f}x {intercept_close:+.2f}$",
    )
    axes[2].set_xlabel(r"True $\eta_\nu$", fontsize=11)
    axes[2].set_ylabel(r"Closest Analytical $\hat{\eta}_\nu$", fontsize=11)
    axes[2].set_title(
        rf"(c) Closest Solution / Oracle ($r = {r_close:.3f}$)", fontsize=12
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left", fontsize=8.5)

    plt.suptitle(
        r"Analytical Solutions (NB1 & NB2): Random $\to$ Min-$|p_z|$ $\to$ Oracle",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    return {
        "r_random": r_rand,
        "rms_random": rms_rand,
        "r_minpz": r_minpz,
        "rms_minpz": rms_minpz,
        "r_closest": r_close,
        "rms_closest": rms_close,
        "minpz_accuracy": minpz_accuracy,
        "n_valid": n_valid,
        "n_complex": n_complex,
    }


if __name__ == "__main__":
    time = '20261001-224739'
    data_filename = 'new_Input_WH_NeutrinoEtaFinding_27th_September2026.csv'
    hyperparameter_file = f"models/hyperparameters_{time}.json"
    hyperparameters = load_hyperparameters(hyperparameter_file)
    # torch.manual_seed(hyperparameters["random_seed"])

    model_name = f"models/model_{time}.pt"
    n_features = len(FEATURES)
    n_target = len(TARGET)
    features, target = read_file(data_filename, FEATURES, TARGET)
    # features_test, target_test = read_file(data_filename, FEATURES, TARGET)

    features_train, features_temp, target_train, target_temp = train_test_split(
        features, target, test_size=hyperparameters["data_split"]["train_val_size"],
        random_state=hyperparameters["data_split"]["random_state"])

    features_val, features_test, target_val, target_test = train_test_split(
        features_temp, target_temp, test_size=hyperparameters["data_split"]["val_test_size"],
        random_state=hyperparameters["data_split"]["random_state"])

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


    # ----------------------------
    # Coverage test

    coverage = interval_coverage(
        flow=flow,
        dataset=test_dataset,
        levels=(0.50,0.68,0.90,0.95),
        model_name=model_name
    )

    # ----------------------------
    # Shuffle test

    normal_nll, shuffled_nll = shuffled_context_nll(
        flow=flow,
        dataset=test_dataset,
        batch_size=512,
        model_name=model_name
    )

    # ----------------------------
    # Conditional Calibration

    # index = 'ChargedLep_E'
    # binned_coverage(
    #     flow=flow,
    #     dataset=test_dataset,
    #     feature_index=FEATURES.index(index),
    #     model_name=model_name,
    # )

    # ----------------------------
    # PIT Value test

    pit_values = pit_test(
        flow=flow,
        dataset=test_dataset,
        model_name=model_name,
        batch_size=512,
        num_bins=20
    )

    # ----------------------------
    # Point Reconstruction test

    results = point_reconstruction_test(
        flow=flow,
        dataset=test_dataset,
        model_name=model_name,
        num_samples=2000,
        estimator="mode"
    )

    residuals = results["residual"]

    plt.figure(figsize=(8, 5))

    plt.hist(
        residuals,
        bins=100,
        density=True
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Zero residual"
    )

    plt.xlabel(
        r"$\Delta \hat{\eta}_\nu^{mode}$"
    )
    plt.ylabel("Probability Density")
    plt.title(r"$\eta_\nu$ Reconstruction Residual")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Multimodality / Posterior Shape Visualization

    df = pd.read_csv(data_filename,encoding_errors='ignore')
    nb1_nu_eta = df['NB1_nu_eta'].values
    nb2_nu_eta = df['NB2_nu_eta'].values

    plot_event_posteriors(
        flow=flow,
        dataset=test_dataset,
        model_name=model_name,
        nb1_nu_eta=nb1_nu_eta,
        nb2_nu_eta=nb2_nu_eta,
        num_random_events=6,  # Or pass specific events e.g. event_indices=[12, 45, 99]
        grid_range=(-5.0, 5.0),
        num_grid_points=1000,
    )


    # ----------------------------
    # Invariant Mass Reconstruction Test

    invariant_mass_reconstruction(
        flow=flow,
        features_test_scaled=features_test,
        target_test=target_test,
        scaler_features=scaler_features,
        model_name=model_name,
        num_samples=500,
        batch_size=512,
        num_bins=120
    )

    # # ----------------------------
    # # Correlation Test
    #
    # disambig_results = mode_disambiguated_correlation(
    #     flow=flow,
    #     dataset=test_dataset,
    #     model_name=model_name,
    #     batch_size=512,
    #     num_grid=250,
    #     range_eta=(-4.5, 4.5),
    # )
    #
    # # ----------------------------
    # # Analytical Solutions Correlation Test
    #
    # df_raw = pd.read_csv(data_filename, encoding_errors="ignore")
    #
    # # Match test set indices
    # df_test_raw = df_raw.iloc[-len(target_test):]
    # true_eta_test = df_test_raw["True_nu_eta"].values
    # nb1_test = df_test_raw["NB1_nu_eta"].values
    # nb2_test = df_test_raw["NB2_nu_eta"].values
    # met_test = df_test_raw["MET"].values
    #
    # analytical_results = analytical_solutions_correlation(
    #     true_eta=true_eta_test,
    #     nb1_eta=nb1_test,
    #     nb2_eta=nb2_test,
    #     met=met_test,
    #     seed=42,
    #     bins=80,
    #     range_eta=(-4.5, 4.5),
    # )