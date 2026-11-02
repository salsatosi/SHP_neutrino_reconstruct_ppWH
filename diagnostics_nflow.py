from train_nflow import (
    device, FEATURES, TARGET, read_file, normalizing_flow, load_hyperparameters)
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
import h5py
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
    plt.title("PIT Histogram [Normalising Flow]")
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
            ax.legend(fontsize=7.5, loc="upper right")

    # Hide unused subplots if num_events is not a multiple of cols
    for j in range(num_events, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        r"Posterior Density $p(\eta_\nu | x)$ vs. Analytical $W$ Solutions [Normalising Flow]",
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
    # lep_e = features_unscaled[:, FEATURES.index("ChargedLep_E")]
    lep_e = lep_pt * np.cosh(lep_eta)

    met = features_unscaled[:, FEATURES.index("MET")]
    phi_nu = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_MET")]

    leadb_pt = features_unscaled[:, FEATURES.index("LeadBJet_pT")]
    leadb_eta = features_unscaled[:, FEATURES.index("LeadBJet_Eta")]
    # leadb_e = features_unscaled[:, FEATURES.index("LeadBJet_E")]
    leadb_e = leadb_pt * np.cosh(leadb_eta)
    leadb_phi = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_LeadBJet")]
    p4_leadb = np.stack(
        [
            leadb_pt * np.cos(leadb_phi),
            leadb_pt * np.sin(leadb_phi),
            leadb_pt * np.sinh(leadb_eta),
            leadb_e,
        ],
        axis=-1,
    )

    subleadb_pt = features_unscaled[:, FEATURES.index("SubLeadBJet_pT")]
    subleadb_eta = features_unscaled[:, FEATURES.index("SubLeadBJet_Eta")]
    # subleadb_e = features_unscaled[:, FEATURES.index("SubLeadBJet_E")]
    subleadb_e = subleadb_pt * np.cosh(subleadb_eta)
    subleadb_phi = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_SubleadBJet")]

    p4_subleadb = np.stack(
        [
            subleadb_pt * np.cos(subleadb_phi),
            subleadb_pt * np.sin(subleadb_phi),
            subleadb_pt * np.sinh(subleadb_eta),
            subleadb_e,
        ],
        axis=-1,
    )

    p4_h = p4_leadb + p4_subleadb

    true_eta = target_test.flatten()

    # 3. Base 4-momenta for Lepton (phi_lep = 0) and Higgs
    p4_lep = np.stack(
        [lep_pt, np.zeros_like(lep_pt), lep_pt * np.sinh(lep_eta), lep_e],
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
        label=rf"True $\eta_\nu$ (Mode: {mode_mw_true:.2f} GeV)",
    )
    axes[0].hist(
        m_w_sampled_flat,
        bins=num_bins,
        range=mw_range,
        density=True,
        alpha=0.5,
        color="royalblue",
        label=rf"Flow Samples (Mode: {mode_mw_flow:.2f} GeV)",
    )
    axes[0].axvline(
        80.36,
        color="red",
        linestyle="--",
        label=r"Nominal $m_W$ (80.36 GeV)",
    )
    axes[0].set_xlabel(r"$m_W$ [GeV]", fontsize=11)
    axes[0].set_ylabel("Normalized Density [GeV$^{-1}$]", fontsize=11)
    axes[0].set_title(r"$W$ Boson Invariant Mass Reconstruction [Normalising Flow]", fontsize=12)
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
        label=rf"True $\eta_\nu$ (Mode: {mode_mwh_true:.2f} GeV)",
    )
    axes[1].hist(
        m_wh_sampled_flat,
        bins=num_bins,
        range=wh_range,
        density=True,
        alpha=0.5,
        color="darkorange",
        label=rf"Flow Samples (Mode: {mode_mwh_flow:.2f} GeV)",
    )
    axes[1].set_xlabel(r"$m_{WH}$ [GeV]", fontsize=11)
    axes[1].set_ylabel("Normalized Density [GeV$^{-1}$]", fontsize=11)
    axes[1].set_title(r"Total $WH$ Invariant Mass Reconstruction [Normalising Flow]", fontsize=12)
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


def correlation_test(
    flow,
    dataset,
    model_name,
    nleadb_eta,
    nsubleadb_eta,
    met=None,
    device=None,
    batch_size=512,
    num_samples=10,
    num_grid=250,
    range_eta=(-5, 5),
    bins=80,
    seed=42,
):
    """Evaluates and compares both Normalizing Flow reconstruction and Analytical

    quadratic solutions (NB1/NB2) against ground-truth neutrino eta, plotting all
    diagnostics in a single 2x3 figure.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # PART 1: Normalizing Flow Grid-Scan & Sampling
    # =========================================================================
    flow.load_state_dict(torch.load(model_name, map_location=device))
    flow.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    grid = torch.linspace(
        range_eta[0], range_eta[1], num_grid, device=device
    )  # [G]
    grid_expanded = grid.view(1, num_grid, 1)

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

            # 1. Random Posterior Sampling
            samples = flow.sample(num_samples=num_samples, context=features)
            samples = samples.squeeze(-1)  # [B, num_samples]
            all_samples.append(samples.cpu().numpy())

            # 2. Grid Scan for Peak / Mode Detection
            g_in = grid_expanded.expand(B, num_grid, 1).reshape(B * num_grid, 1)
            c_in = (
                features.unsqueeze(1)
                .expand(B, num_grid, -1)
                .reshape(B * num_grid, -1)
            )

            log_prob = flow.log_prob(inputs=g_in, context=c_in).view(B, num_grid)
            probs = torch.exp(log_prob)

            # Local peak detection
            is_peak = (probs[:, 1:-1] > probs[:, :-2]) & (
                probs[:, 1:-1] > probs[:, 2:]
            )
            is_peak = F.pad(is_peak, (1, 1), value=False)

            grid_cpu = grid.cpu().numpy()
            probs_cpu = probs.cpu().numpy()
            is_peak_cpu = is_peak.cpu().numpy()

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

    # Flatten random posterior samples
    samples_arr = np.concatenate(all_samples, axis=0)
    targets_repeated = np.repeat(targets, num_samples)
    samples_flat = samples_arr.flatten()

    # Flow mode disambiguation
    diff1_flow = np.abs(mode1 - targets)
    diff2_flow = np.abs(mode2 - targets)
    nearest_mode = np.where(diff1_flow <= diff2_flow, mode1, mode2)
    accuracy_flow_all = np.mean(diff1_flow <= diff2_flow) * 100
    accuracy_flow_bimodal = (
        np.mean((diff1_flow <= diff2_flow)[has_two_modes]) * 100
    )

    # Flow Metrics
    r_samples, _ = pearsonr(targets_repeated, samples_flat)
    slope_samp, intercept_samp = np.polyfit(targets_repeated, samples_flat, 1)
    rms_samples = np.sqrt(np.mean((samples_flat - targets_repeated) ** 2))

    r_primary, _ = pearsonr(targets, mode1)
    slope_prim, intercept_prim = np.polyfit(targets, mode1, 1)
    rms_primary = np.sqrt(np.mean((mode1 - targets) ** 2))

    r_nearest, _ = pearsonr(targets, nearest_mode)
    slope_near, intercept_near = np.polyfit(targets, nearest_mode, 1)
    rms_nearest = np.sqrt(np.mean((nearest_mode - targets) ** 2))

    ratio = np.std(samples_flat) / np.std(targets)
    print(f"Variance ratio (sigma_pred / sigma_true): {ratio:.3f}")

    # =========================================================================
    # PART 2: Analytical Quadratic Solutions (NB1 & NB2)
    # =========================================================================
    valid_mask = (
        ~np.isnan(nleadb_eta)
        & ~np.isnan(nsubleadb_eta)
        & ~np.isinf(nleadb_eta)
        & ~np.isinf(nsubleadb_eta)
    )

    t_ana = targets[valid_mask]
    s1 = nleadb_eta[valid_mask]
    s2 = nsubleadb_eta[valid_mask]
    m_met = met[valid_mask] if met is not None else None

    n_total = len(targets)
    n_valid = len(t_ana)
    n_complex = n_total - n_valid

    # 1. Random 50/50 Selection
    rng = np.random.default_rng(seed=seed)
    coin_flip = rng.choice([True, False], size=n_valid)
    random_pick = np.where(coin_flip, s1, s2)
    r_rand, _ = pearsonr(t_ana, random_pick)
    slope_rand, intercept_rand = np.polyfit(t_ana, random_pick, 1)
    rms_rand = np.sqrt(np.mean((random_pick - t_ana) ** 2))

    # 2. Min-|p_z| Selection Heuristic
    if m_met is not None:
        pz1 = m_met * np.abs(np.sinh(s1))
        pz2 = m_met * np.abs(np.sinh(s2))
    else:
        pz1 = np.abs(np.sinh(s1))
        pz2 = np.abs(np.sinh(s2))
    minpz_pick = np.where(pz1 <= pz2, s1, s2)
    r_minpz, _ = pearsonr(t_ana, minpz_pick)
    slope_minpz, intercept_minpz = np.polyfit(t_ana, minpz_pick, 1)
    rms_minpz = np.sqrt(np.mean((minpz_pick - t_ana) ** 2))

    # 3. Closest Analytical Selection
    diff1_ana = np.abs(s1 - t_ana)
    diff2_ana = np.abs(s2 - t_ana)
    closest_pick = np.where(diff1_ana <= diff2_ana, s1, s2)
    r_close, _ = pearsonr(t_ana, closest_pick)
    slope_close, intercept_close = np.polyfit(t_ana, closest_pick, 1)
    rms_close = np.sqrt(np.mean((closest_pick - t_ana) ** 2))
    minpz_accuracy = np.mean(minpz_pick == closest_pick) * 100

    # =========================================================================
    # PART 3: Print Unified Summary Report
    # =========================================================================
    print("\n" + "=" * 78)
    print("RECONSTRUCTION PERFORMANCE BENCHMARK (FLOW vs. ANALYTICAL)")
    print("=" * 78)
    print(
        f"Analytical Solvability : {n_valid:,}/{n_total:,} valid ({n_valid/n_total*100:.1f}%), {n_complex:,} complex ({n_complex/n_total*100:.1f}%)"
    )
    print(
        f"Flow Bimodality Rate   : {np.sum(has_two_modes):,}/{n_total:,} ({np.mean(has_two_modes)*100:.1f}%)"
    )
    print(
        f"Disambiguation Acc.    : Flow = {accuracy_flow_all:.1f}% | Min-|pz| Heuristic = {minpz_accuracy:.1f}%"
    )
    print("-" * 78)
    print("--- NORMALIZING FLOW ---")
    print(
        f"  Random Posterior     : r = {r_samples:.4f} | RMS = {rms_samples:.4f} | y = {slope_samp:.2f}x {intercept_samp:+.2f}"
    )
    print(
        f"  Primary Mode (MAP)   : r = {r_primary:.4f} | RMS = {rms_primary:.4f} | y = {slope_prim:.2f}x {intercept_prim:+.2f}"
    )
    print(
        f"  Disambiguated Mode   : r = {r_nearest:.4f} | RMS = {rms_nearest:.4f} | y = {slope_near:.2f}x {intercept_near:+.2f}"
    )
    print("--- ANALYTICAL RECONSTRUCTION (NB1 / NB2) ---")
    print(
        f"  Random 50/50 Choice  : r = {r_rand:.4f} | RMS = {rms_rand:.4f} | y = {slope_rand:.2f}x {intercept_rand:+.2f}"
    )
    print(
        f"  Min-|pz| Heuristic   : r = {r_minpz:.4f} | RMS = {rms_minpz:.4f} | y = {slope_minpz:.2f}x {intercept_minpz:+.2f}"
    )
    print(
        f"  Closest Analytical Selection: r = {r_close:.4f} | RMS = {rms_close:.4f} | y = {slope_close:.2f}x {intercept_close:+.2f}"
    )
    print("=" * 78)

    # =========================================================================
    # PART 4: Unified 2x3 Plotting
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    diag = np.linspace(range_eta[0], range_eta[1], 100)

    # Helper function for uniform panel styling
    def plot_panel(ax, x, y, slope, intercept, r_val, cmap, title, ylabel):
        h = ax.hist2d(
            x,
            y,
            bins=bins,
            range=[range_eta, range_eta],
            cmap=cmap,
            norm=LogNorm(),
        )
        fig.colorbar(h[3], ax=ax, label="Counts (log scale)")
        ax.plot(diag, diag, "r--", lw=2, label="Ideal ($y = x$)")
        ax.plot(
            diag,
            slope * diag + intercept,
            "k-",
            lw=1.5,
            label=f"Fit: $y = {slope:.2f}x {intercept:+.2f}$",
        )
        ax.set_xlabel(r"True $\eta_\nu$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(rf"{title} ($r = {r_val:.3f}$)", fontsize=12)

        ax.legend(loc="upper left", fontsize=8.5)

    # Row 1: Flow Progression
    plot_panel(
        axes[0, 0],
        targets_repeated,
        samples_flat,
        slope_samp,
        intercept_samp,
        r_samples,
        "Purples",
        "(a) Flow: Random Sampling",
        r"Sampled $\hat{\eta}_\nu$",
    )
    plot_panel(
        axes[0, 1],
        targets,
        mode1,
        slope_prim,
        intercept_prim,
        r_primary,
        "Blues",
        "(b) Flow: Primary Mode (MAP)",
        r"Primary Mode $\hat{\eta}_\nu$ (MAP)",
    )
    plot_panel(
        axes[0, 2],
        targets,
        nearest_mode,
        slope_near,
        intercept_near,
        r_nearest,
        "Greens",
        "(c) Flow: Disambiguated Mode",
        r"Nearest Mode $\hat{\eta}_\nu$",
    )

    # Row 2: Analytical Progression
    plot_panel(
        axes[1, 0],
        t_ana,
        random_pick,
        slope_rand,
        intercept_rand,
        r_rand,
        "Blues",
        "(d) Analytical: Random 50/50",
        r"Random Analytical $\hat{\eta}_\nu$",
    )
    plot_panel(
        axes[1, 1],
        t_ana,
        minpz_pick,
        slope_minpz,
        intercept_minpz,
        r_minpz,
        "Oranges",
        r"(e) Analytical: Min-$|p_z|$",
        r"Min-$|p_z|$ $\hat{\eta}_\nu$",
    )
    plot_panel(
        axes[1, 2],
        t_ana,
        closest_pick,
        slope_close,
        intercept_close,
        r_close,
        "Greens",
        "(f) Closest Analytical Selection",
        r"Closest Analytical $\hat{\eta}_\nu$",
    )

    plt.suptitle(
        r"Comparison of Neutrino $\eta_\nu$ Reconstruction: Normalizing Flow vs. Analytical Quadratic Solution [Normalising Flow]",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()
    plt.show()

    return {
        # Flow metrics
        "flow_r_samples": r_samples,
        "flow_rms_samples": rms_samples,
        "flow_r_primary": r_primary,
        "flow_rms_primary": rms_primary,
        "flow_r_nearest": r_nearest,
        "flow_rms_nearest": rms_nearest,
        "flow_accuracy_all": accuracy_flow_all,
        "flow_accuracy_bimodal": accuracy_flow_bimodal,
        # Analytical metrics
        "ana_r_random": r_rand,
        "ana_rms_random": rms_rand,
        "ana_r_minpz": r_minpz,
        "ana_rms_minpz": rms_minpz,
        "ana_r_closest": r_close,
        "ana_rms_closest": rms_close,
        "ana_minpz_accuracy": minpz_accuracy,
        "n_valid": n_valid,
        "n_complex": n_complex,
    }


def lorentz_boost(p4, beta):
    """Boost a 4-momentum p4 = (px, py, pz, E) by velocity vector beta = (bx,

    by, bz).

    p4 shape: (N, 4) or (N, S, 4) beta shape: (N, 3) or (N, S, 3)
    """
    beta_sq = np.sum(beta**2, axis=-1, keepdims=True)  # beta^2
    gamma = 1.0 / np.sqrt(np.maximum(1e-10, 1.0 - beta_sq))

    p3 = p4[..., :3]
    E = p4[..., 3:4]

    # Lorentz boost formula along arbitrary 3-vector beta
    beta_dot_p = np.sum(beta * p3, axis=-1, keepdims=True)

    factor = np.where(
        beta_sq > 1e-10, (gamma - 1.0) / np.maximum(beta_sq, 1e-10), 0.0
    )
    p3_boosted = p3 + factor * beta_dot_p * beta - gamma * E * beta
    E_boosted = gamma * (E - beta_dot_p)

    return np.concatenate([p3_boosted, E_boosted], axis=-1)


def unit_vector(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-9)


def compute_angles_vectorized(p4_lep, p4_nu, p4_b1, p4_b2):
    """Correctly computes angular observables in the WH Center-of-Mass frame

    and W/H helicity rest frames according to the paper's definition.
    """
    # ----------------------------------------------------
    # 1. Composite 4-momenta in Lab Frame (px, py, pz, E)
    # ----------------------------------------------------
    p4_w = p4_lep + p4_nu
    p4_h = p4_b1 + p4_b2
    p4_wh = p4_w + p4_h

    # ----------------------------------------------------
    # 2. Boost ALL particles to the WH Center-of-Mass Frame (X)
    # (Note: beta is POSITIVE because lorentz_boost subtracts gamma*E*beta!)
    # ----------------------------------------------------
    beta_wh = p4_wh[:, :3] / p4_wh[:, 3:4]

    p4_w_X = lorentz_boost(p4_w, beta_wh)
    p4_h_X = lorentz_boost(p4_h, beta_wh)
    p4_lep_X = lorentz_boost(p4_lep, beta_wh)
    p4_nu_X = lorentz_boost(p4_nu, beta_wh)
    p4_b1_X = lorentz_boost(p4_b1, beta_wh)
    p4_b2_X = lorentz_boost(p4_b2, beta_wh)

    p3_w_X = p4_w_X[:, :3]
    p3_h_X = p4_h_X[:, :3]
    u_w_X = unit_vector(p3_w_X)
    u_h_X = unit_vector(p3_h_X)

    # ----------------------------------------------------
    # costhetastar: Direction of W in WH CoM frame vs. Beam Axis (z)
    # ----------------------------------------------------
    costhetastar = p3_w_X[:, 2] / np.maximum(
        1e-9, np.linalg.norm(p3_w_X, axis=-1)
    )
    costhetastar = np.clip(costhetastar, -1.0, 1.0)

    # ----------------------------------------------------
    # costheta1: Lepton in W rest frame vs. W flight direction in WH frame
    # ----------------------------------------------------
    beta_w = p4_w[:, :3] / p4_w[:, 3:4]
    p4_lep_in_W = lorentz_boost(p4_lep, beta_w)
    u_lep_in_W = unit_vector(p4_lep_in_W[:, :3])

    # Angle relative to W's direction of flight in the WH frame
    costheta1 = np.sum(-u_lep_in_W * u_h_X, axis=-1)
    costheta1 = np.clip(costheta1, -1.0, 1.0)

    # ----------------------------------------------------
    # Planes and Azimuthal Angles (All in WH CoM frame)
    # ----------------------------------------------------
    # Normal to W decay plane
    normal1_X = unit_vector(np.cross(p4_lep_X[:, :3], p4_nu_X[:, :3]))

    # Normal to H decay plane
    normal2_X = unit_vector(np.cross(p4_b1_X[:, :3], p4_b2_X[:, :3]))

    # Normal to production / scattering plane (Beam axis = [0, 0, 1])
    beam_axis = np.zeros_like(u_w_X)
    beam_axis[:, 2] = 1.0
    normalSC_X = unit_vector(np.cross(beam_axis, u_w_X))

    # --- Phi (Angle between W and H decay planes) ---
    tmp_sgn_phi = np.sum(u_w_X * np.cross(normal1_X, normal2_X), axis=-1)
    sgn_phi = np.sign(tmp_sgn_phi)
    dot_12 = np.clip(np.sum(normal1_X * normal2_X, axis=-1), -1.0, 1.0)
    Phi = sgn_phi * np.arccos(-dot_12)

    # --- Phi1 (Angle between W decay plane and scattering plane) ---
    tmp_sgn_phi1 = np.sum(u_w_X * np.cross(normal1_X, normalSC_X), axis=-1)
    sgn_phi1 = np.sign(tmp_sgn_phi1)
    dot_1sc = np.clip(np.sum(normal1_X * normalSC_X, axis=-1), -1.0, 1.0)
    Phi1 = sgn_phi1 * np.arccos(dot_1sc)

    # --- dphi_ll (Azimuthal separation between lepton and neutrino in WH frame) ---
    phi_lep_X = np.arctan2(p4_lep_X[:, 1], p4_lep_X[:, 0])
    phi_nu_X = np.arctan2(p4_nu_X[:, 1], p4_nu_X[:, 0])
    dphi_ll = np.abs(
        np.arctan2(np.sin(phi_lep_X - phi_nu_X), np.cos(phi_lep_X - phi_nu_X))
    )

    return {
        "costhetastar": costhetastar,
        "costheta1": costheta1,
        "Phi": Phi,
        "Phi1": Phi1,
        "dphi_ll": dphi_ll,
    }


def angle_comparison(
    flow,
    features_test_scaled,
    target_test,
    scaler_features,
    model_name,
    nb1_nu_eta=None,
    nb2_nu_eta=None,
    batch_size=512,
    num_grid=500,  # Grid resolution for exact Mode (MAP) search
    range_eta=(-4.5, 4.5),
    bins=40,
    seed=42,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow.load_state_dict(torch.load(model_name, map_location=device))
    flow.eval()

    # ----------------------------------------------------
    # 1. Unscale features to physical units
    # ----------------------------------------------------
    feat_unscaled = scaler_features.inverse_transform(features_test_scaled)
    n_events = len(target_test)

    # Lepton kinematics (aligned along x-axis: phi_l = 0)
    lep_pt = feat_unscaled[:, FEATURES.index("ChargedLep_pT")]
    lep_eta = feat_unscaled[:, FEATURES.index("ChargedLep_Eta")]
    # lep_e = feat_unscaled[:, FEATURES.index("ChargedLep_E")]
    lep_e = lep_pt * np.cosh(lep_eta)
    p4_lep = np.stack(
        [lep_pt, np.zeros_like(lep_pt), lep_pt * np.sinh(lep_eta), lep_e],
        axis=-1,
    )

    # MET kinematics
    met = feat_unscaled[:, FEATURES.index("MET")]
    phi_nu = feat_unscaled[:, FEATURES.index("Relative_Phi_Lep_MET")]

    # b-Jets kinematics
    leadb_pt = feat_unscaled[:, FEATURES.index("LeadBJet_pT")]
    leadb_eta = feat_unscaled[:, FEATURES.index("LeadBJet_Eta")]
    leadb_phi = feat_unscaled[:, FEATURES.index("Relative_Phi_Lep_LeadBJet")]
    # leadb_e = feat_unscaled[:, FEATURES.index("LeadBJet_E")]
    leadb_e = leadb_pt * np.cosh(leadb_eta)
    p4_b1 = np.stack(
        [
            leadb_pt * np.cos(leadb_phi),
            leadb_pt * np.sin(leadb_phi),
            leadb_pt * np.sinh(leadb_eta),
            leadb_e,
        ],
        axis=-1,
    )

    subleadb_pt = feat_unscaled[:, FEATURES.index("SubLeadBJet_pT")]
    subleadb_eta = feat_unscaled[:, FEATURES.index("SubLeadBJet_Eta")]
    subleadb_phi = feat_unscaled[:, FEATURES.index("Relative_Phi_Lep_SubleadBJet")]
    # subleadb_e = feat_unscaled[:, FEATURES.index("SubLeadBJet_E")]
    subleadb_e = subleadb_pt * np.cosh(subleadb_eta)
    p4_b2 = np.stack(
        [
            subleadb_pt * np.cos(subleadb_phi),
            subleadb_pt * np.sin(subleadb_phi),
            subleadb_pt * np.sinh(subleadb_eta),
            subleadb_e,
        ],
        axis=-1,
    )

    true_eta = target_test.flatten()

    def build_p4_nu(eta):
        px = met * np.cos(phi_nu)
        py = met * np.sin(phi_nu)
        pz = met * np.sinh(eta)
        e = met * np.cosh(eta)
        return np.stack([px, py, pz, e], axis=-1)

    # ----------------------------------------------------
    # 2. Reconstruct Neutrino 4-Vectors
    # ----------------------------------------------------
    # (a) True MC
    print("Computing True MC Angles...")
    p4_nu_true = build_p4_nu(true_eta)
    obs_true = compute_angles_vectorized(p4_lep, p4_nu_true, p4_b1, p4_b2)

    # (b) Model prediction
    features_tensor = torch.tensor(features_test_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    all_flow_samples = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            # Sample 10 draws per event to correctly populate the continuous quantum phase space
            s = flow.sample(num_samples=10, context=bx)
            all_flow_samples.append(s.squeeze(-1).cpu().numpy())

    eta_flow_samples = np.concatenate(all_flow_samples, axis=0)

    # Repeat lepton and b-jets 10 times to match the 10 neutrino draws
    p4_lep_rep = np.repeat(p4_lep, 10, axis=0)
    p4_b1_rep = np.repeat(p4_b1, 10, axis=0)
    p4_b2_rep = np.repeat(p4_b2, 10, axis=0)
    met_rep = np.repeat(met, 10, axis=0)
    phi_nu_rep = np.repeat(phi_nu, 10, axis=0)

    # Build neutrino and compute angles
    px = met_rep * np.cos(phi_nu_rep)
    py = met_rep * np.sin(phi_nu_rep)
    pz = met_rep * np.sinh(eta_flow_samples.flatten())
    e = met_rep * np.cosh(eta_flow_samples.flatten())
    p4_nu_flow = np.stack([px, py, pz, e], axis=-1)

    obs_flow = compute_angles_vectorized(
        p4_lep_rep, p4_nu_flow, p4_b1_rep, p4_b2_rep
    )


    # (c) Baselines: Random 50/50 and min-|pz|
    obs_minpz = None
    obs_5050 = None

    if nb1_nu_eta is not None and nb2_nu_eta is not None:
        # Min-|pz|
        pz1 = np.abs(met * np.sinh(nb1_nu_eta))
        pz2 = np.abs(met * np.sinh(nb2_nu_eta))
        eta_minpz = np.where(pz1 <= pz2, nb1_nu_eta, nb2_nu_eta)
        p4_nu_minpz = build_p4_nu(eta_minpz)
        obs_minpz = compute_angles_vectorized(p4_lep, p4_nu_minpz, p4_b1, p4_b2)

        # 50/50
        rng = np.random.default_rng(seed=seed)
        coin = rng.choice([True, False], size=n_events)
        eta_5050 = np.where(coin, nb1_nu_eta, nb2_nu_eta)
        p4_nu_5050 = build_p4_nu(eta_5050)
        obs_5050 = compute_angles_vectorized(p4_lep, p4_nu_5050, p4_b1, p4_b2)

    # ----------------------------------------------------
    # Helper: Two-Sample Poisson Chi2
    # ----------------------------------------------------
    def compute_chi2_ndf(counts_test, counts_ref):
        counts_test = np.asarray(counts_test, dtype=float)
        counts_ref = np.asarray(counts_ref, dtype=float)

        n_ref = np.sum(counts_ref)
        n_test = np.sum(counts_test)
        if n_ref == 0 or n_test == 0:
            return np.nan, 0, np.nan

        p_pooled = (counts_ref + counts_test) / (n_ref + n_test)
        exp_ref = n_ref * p_pooled
        exp_test = n_test * p_pooled

        valid = (exp_ref >= 5.0) & (exp_test >= 5.0)
        if np.sum(valid) <= 1:
            return np.nan, 0, np.nan

        chi2_ref = (counts_ref[valid] - exp_ref[valid]) ** 2 / exp_ref[valid]
        chi2_test = (
            counts_test[valid] - exp_test[valid]
        ) ** 2 / exp_test[valid]

        chi2 = np.sum(chi2_ref + chi2_test)
        ndf = np.sum(valid) - 1
        return chi2, ndf, chi2 / ndf

    # ----------------------------------------------------
    # 3. Calculate Chi2 and Plot in a 2x2 Grid
    # ----------------------------------------------------
    plot_vars = [
        ("costhetastar", r"$\cos\theta^*_{WH}$", (-1.0, 1.0)),
        ("costheta1", r"$\cos\theta_1$", (-1.0, 1.0)),
        ("Phi1", r"$|\Phi_1|$ [rad]", (0.0, np.pi)),
        ("dphi_ll", r"$\Delta\Phi(L, \bar{L})$ [rad]", (0.0, np.pi)),
    ]

    print("\n" + "=" * 90)
    print(
        f"{'Observable':<22} | {'Model Samples χ²/ndf':<20} | {'50/50 χ²/ndf':<18} | {'Min-|pz| χ²/ndf':<18}"
    )
    print("=" * 90)

    # 2x2 Clean Subplot Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    axes = axes.flatten()

    for i, (key, label, rng) in enumerate(plot_vars):
        # Extract data (take absolute value for Phi1 to avoid negative truncation!)
        if key == "Phi1":
            data_true = np.abs(obs_true[key])
            data_flow = np.abs(obs_flow[key])
            data_5050 = np.abs(obs_5050[key]) if obs_5050 else None
            data_minpz = np.abs(obs_minpz[key]) if obs_minpz else None
        else:
            data_true = obs_true[key]
            data_flow = obs_flow[key]
            data_5050 = obs_5050[key] if obs_5050 else None
            data_minpz = obs_minpz[key] if obs_minpz else None

        # Filter NaNs from complex analytical solutions
        data_true = data_true[~np.isnan(data_true)]
        data_flow = data_flow[~np.isnan(data_flow)]

        # Raw counts for chi2
        raw_true, edges = np.histogram(data_true, bins=bins, range=rng)
        raw_flow, _ = np.histogram(data_flow, bins=bins, range=rng)
        centers = 0.5 * (edges[:-1] + edges[1:])

        _, _, r_chi2_f = compute_chi2_ndf(raw_flow, raw_true)

        # Baseline Chi2
        r_chi2_50 = np.nan
        if data_5050 is not None:
            raw_5050, _ = np.histogram(
                data_5050[~np.isnan(data_5050)], bins=bins, range=rng
            )
            _, _, r_chi2_50 = compute_chi2_ndf(raw_5050, raw_true)

        r_chi2_pz = np.nan
        if data_minpz is not None:
            raw_minpz, _ = np.histogram(
                data_minpz[~np.isnan(data_minpz)], bins=bins, range=rng
            )
            _, _, r_chi2_pz = compute_chi2_ndf(raw_minpz, raw_true)

        print(
            f"{key:<22} | {r_chi2_f:<20.2f} | {r_chi2_50:<18.2f} | {r_chi2_pz:<18.2f}"
        )

        # Normalized densities for plotting
        norm_true, _ = np.histogram(
            data_true, bins=bins, range=rng, density=True
        )
        norm_flow, _ = np.histogram(
            data_flow, bins=bins, range=rng, density=True
        )

        ax = axes[i]
        ax.hist(
            centers,
            bins=edges,
            weights=norm_true,
            histtype="step",
            lw=2.2,
            color="black",
            label="MC Truth",
        )
        ax.hist(
            centers,
            bins=edges,
            weights=norm_flow,
            histtype="stepfilled",
            alpha=0.45,
            color="royalblue",
            label=rf"Model Samples ($\chi^2/\mathrm{{ndf}} = {r_chi2_f:.2f}$)",
        )

        if data_5050 is not None:
            norm_5050, _ = np.histogram(
                data_5050[~np.isnan(data_5050)],
                bins=bins,
                range=rng,
                density=True,
            )
            ax.hist(
                centers,
                bins=edges,
                weights=norm_5050,
                histtype="step",
                ls="--",
                lw=1.6,
                color="crimson",
                label=rf"50/50 ($\chi^2/\mathrm{{ndf}} = {r_chi2_50:.2f}$)",
            )

        if data_minpz is not None:
            norm_minpz, _ = np.histogram(
                data_minpz[~np.isnan(data_minpz)],
                bins=bins,
                range=rng,
                density=True,
            )
            ax.hist(
                centers,
                bins=edges,
                weights=norm_minpz,
                histtype="step",
                ls=":",
                lw=1.8,
                color="darkorange",
                label=rf"Min-$|p_z|$ ($\chi^2/\mathrm{{ndf}} = {r_chi2_pz:.2f}$)",
            )

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Normalized Density", fontsize=11)
        ax.set_xlim(rng)
        ax.legend(fontsize=8.5, loc="upper right")

    print("=" * 90)

    plt.suptitle(
        r"Reconstruction of Angular Observables",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()
    plt.show()

    return {
        "obs_true": obs_true,
        "obs_flow": obs_flow,
        "obs_5050": obs_5050,
        "obs_minpz": obs_minpz,
    }


def compare_eta_nu_distribution(
    flow,
    features_test_scaled,
    target_test,
    scaler_features,
    model_name,
    nb1_nu_eta=None,  # 1st quadratic solution
    nb2_nu_eta=None,  # 2nd quadratic solution
    num_samples=10,  # Posterior draws per event (1 to 10)
    batch_size=512,
    bins=60,
    range_eta=(-5.0, 5.0),
    seed=42,
    device=None,
):
    """Directly compares reconstructed neutrino pseudorapidity (eta_nu)

    distributions: - MC Truth vs. Normalizing Flow vs. 50/50 vs. Min-|pz| -
    Calculates chi2/ndf against MC Truth - Plots 2-panel Normalized Density +
    Ratio-to-Truth
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow.load_state_dict(torch.load(model_name, map_location=device))
    flow.eval()

    true_eta = target_test.flatten()
    n_events = len(true_eta)

    # 1. Unscale features to extract MET for min-|pz| calculation
    features_unscaled = scaler_features.inverse_transform(features_test_scaled)
    met = features_unscaled[:, FEATURES.index("MET")]

    # ----------------------------------------------------
    # 2. Sample from Normalizing Flow
    # ----------------------------------------------------
    print(
        f"Sampling eta_nu from Normalizing Flow (num_samples={num_samples})..."
    )
    flow_samples = []
    features_tensor = torch.tensor(features_test_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            s = flow.sample(num_samples=num_samples, context=bx)
            flow_samples.append(s.squeeze(-1).cpu().numpy())

    eta_flow = np.concatenate(flow_samples, axis=0).flatten()

    # ----------------------------------------------------
    # 3. Construct Analytical Baselines (50/50 & min-|pz|)
    # ----------------------------------------------------
    eta_5050 = None
    eta_minpz = None

    if nb1_nu_eta is not None and nb2_nu_eta is not None:
        # Baseline 1: Random 50/50 Choice
        rng = np.random.default_rng(seed=seed)
        coin_flip = rng.choice([True, False], size=n_events)
        eta_5050_raw = np.where(coin_flip, nb1_nu_eta, nb2_nu_eta)
        valid_5050 = ~(np.isnan(eta_5050_raw) | np.isinf(eta_5050_raw))
        eta_5050 = eta_5050_raw[valid_5050]

        # Baseline 2: Minimum |pz|
        pz1 = np.abs(met * np.sinh(nb1_nu_eta))
        pz2 = np.abs(met * np.sinh(nb2_nu_eta))
        eta_minpz_raw = np.where(pz1 <= pz2, nb1_nu_eta, nb2_nu_eta)
        valid_minpz = ~(np.isnan(eta_minpz_raw) | np.isinf(eta_minpz_raw))
        eta_minpz = eta_minpz_raw[valid_minpz]

    # ----------------------------------------------------
    # 4. Two-Sample Poisson Chi-Squared Function
    # ----------------------------------------------------
    def compute_chi2_ndf(counts_test, counts_ref):
        counts_test = np.asarray(counts_test, dtype=float)
        counts_ref = np.asarray(counts_ref, dtype=float)

        n_ref = np.sum(counts_ref)
        n_test = np.sum(counts_test)

        if n_ref == 0 or n_test == 0:
            return np.nan, 0, np.nan

        # 1. Best estimate of true bin probabilities under Null Hypothesis (pooled)
        p_pooled = (counts_ref + counts_test) / (n_ref + n_test)

        # 2. Expected counts in each bin
        exp_ref = n_ref * p_pooled
        exp_test = n_test * p_pooled

        # 3. Filter bins: require at least 5 expected events in both distributions
        valid = (exp_ref >= 5.0) & (exp_test >= 5.0)

        if np.sum(valid) <= 1:
            return np.nan, 0, np.nan

        # 4. Standard Pearson Two-Sample Chi2
        # Sum of (Obs - Exp)^2 / Exp for both histograms
        chi2_ref = (counts_ref[valid] - exp_ref[valid]) ** 2 / exp_ref[valid]
        chi2_test = (counts_test[valid] - exp_test[valid]) ** 2 / exp_test[valid]

        chi2 = np.sum(chi2_ref + chi2_test)

        # 5. Degrees of Freedom
        # K valid bins - 1 constraint from fixing total normalization
        ndf = np.sum(valid) - 1
        chi2_ndf = chi2 / ndf

        return chi2, ndf, chi2_ndf

    # ----------------------------------------------------
    # 5. Compute Histograms and Chi2
    # ----------------------------------------------------
    raw_truth, edges = np.histogram(true_eta, bins=bins, range=range_eta)
    raw_flow, _ = np.histogram(eta_flow, bins=bins, range=range_eta)
    centers = 0.5 * (edges[:-1] + edges[1:])

    _, _, r_chi2_flow = compute_chi2_ndf(raw_flow, raw_truth)

    r_chi2_5050 = None
    r_chi2_minpz = None

    if eta_5050 is not None:
        raw_5050, _ = np.histogram(eta_5050, bins=bins, range=range_eta)
        _, _, r_chi2_5050 = compute_chi2_ndf(raw_5050, raw_truth)

    if eta_minpz is not None:
        raw_minpz, _ = np.histogram(eta_minpz, bins=bins, range=range_eta)
        _, _, r_chi2_minpz = compute_chi2_ndf(raw_minpz, raw_truth)

    # Normalized Densities for plotting
    norm_truth, _ = np.histogram(
        true_eta, bins=bins, range=range_eta, density=True
    )
    norm_flow, _ = np.histogram(
        eta_flow, bins=bins, range=range_eta, density=True
    )

    # ----------------------------------------------------
    # 6. Print Summary Table
    # ----------------------------------------------------
    print("\n" + "=" * 75)
    print(
        f"{'Method':<24} | {'Mean':<8} | {'Std (Width)':<12} | {'chi2 / ndf':<12}"
    )
    print("=" * 75)
    print(
        f"{'1. MC Truth (Ref)':<24} | {np.mean(true_eta):<8.3f} | {np.std(true_eta):<12.3f} | {'—':<12}"
    )
    print(
        f"{'2. Flow Samples':<24} | {np.mean(eta_flow):<8.3f} | {np.std(eta_flow):<12.3f} | {r_chi2_flow:<12.2f}"
    )
    if eta_5050 is not None:
        print(
            f"{'3. Baseline (50/50)':<24} | {np.mean(eta_5050):<8.3f} | {np.std(eta_5050):<12.3f} | {r_chi2_5050:<12.2f}"
        )
    if eta_minpz is not None:
        print(
            f"{'4. Baseline (min-|pz|)':<24} | {np.mean(eta_minpz):<8.3f} | {np.std(eta_minpz):<12.3f} | {r_chi2_minpz:<12.2f}"
        )
    print("=" * 75)

    # ----------------------------------------------------
    # 7. Plotting: Distribution & Ratio-to-Truth
    # ----------------------------------------------------
    fig, (ax_main, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(9, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.2]},
    )

    # --- Main Panel ---
    ax_main.hist(
        centers,
        bins=edges,
        weights=norm_truth,
        histtype="step",
        linewidth=2.2,
        color="black",
        label=rf"MC Truth",
    )
    ax_main.hist(
        centers,
        bins=edges,
        weights=norm_flow,
        histtype="stepfilled",
        alpha=0.45,
        color="royalblue",
        label=rf"Model Samples ($\chi^2/\mathrm{{ndf}} = {r_chi2_flow:.2f}$)",
    )

    if eta_5050 is not None:
        norm_5050, _ = np.histogram(
            eta_5050, bins=bins, range=range_eta, density=True
        )
        ax_main.hist(
            centers,
            bins=edges,
            weights=norm_5050,
            histtype="step",
            linestyle="--",
            linewidth=1.8,
            color="crimson",
            label=rf"Baseline (50/50) ($\chi^2/\mathrm{{ndf}} = {r_chi2_5050:.2f}$)",
        )

    if eta_minpz is not None:
        norm_minpz, _ = np.histogram(
            eta_minpz, bins=bins, range=range_eta, density=True
        )
        ax_main.hist(
            centers,
            bins=edges,
            weights=norm_minpz,
            histtype="step",
            linestyle=":",
            linewidth=2.0,
            color="darkorange",
            label=rf"Baseline min-$|p_z|$ ($\chi^2/\mathrm{{ndf}} = {r_chi2_minpz:.2f}$)",
        )

    ax_main.set_ylabel(r"Normalized Density", fontsize=12)
    ax_main.set_title(
        r"Neutrino Pseudorapidity ($\eta_\nu$) Reconstruction [Normalising Flow]", fontsize=13
    )
    ax_main.legend(fontsize=9.0, loc="upper right")

    # --- Ratio Panel (Model / Truth) ---
    ratio_flow = norm_flow / np.maximum(norm_truth, 1e-6)

    # Poisson uncertainties on Flow ratio (no ugly caps)
    valid_bins = (raw_flow > 0) & (raw_truth > 0)
    ratio_err_flow = np.zeros_like(ratio_flow)
    ratio_err_flow[valid_bins] = ratio_flow[valid_bins] * np.sqrt(
        1.0 / raw_flow[valid_bins] + 1.0 / raw_truth[valid_bins]
    )

    ax_ratio.errorbar(
        centers[valid_bins],
        ratio_flow[valid_bins],
        yerr=ratio_err_flow[valid_bins],
        fmt="o-",
        color="royalblue",
        ecolor="royalblue",
        ms=3.0,
        elinewidth=1.0,
        capsize=0,
        label="Flow / Truth",
        zorder=3,
    )

    if eta_5050 is not None:
        ratio_5050 = norm_5050 / np.maximum(norm_truth, 1e-6)
        ax_ratio.plot(
            centers,
            ratio_5050,
            "--",
            color="crimson",
            lw=1.5,
            label="50/50 / Truth",
            zorder=2,
        )

    if eta_minpz is not None:
        ratio_minpz = norm_minpz / np.maximum(norm_truth, 1e-6)
        ax_ratio.plot(
            centers,
            ratio_minpz,
            ":",
            color="darkorange",
            lw=1.8,
            label="min-$|p_z|$ / Truth",
            zorder=2,
        )

    ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.8)
    ax_ratio.set_ylim(0.6, 1.4)
    ax_ratio.set_xlabel(r"Neutrino Pseudorapidity $\eta_\nu$", fontsize=11)
    ax_ratio.set_ylabel("Ratio to Truth", fontsize=10)
    ax_ratio.legend(fontsize=8.0, loc="lower right", ncol=3)

    plt.tight_layout()
    plt.show()

    return {
        "r_chi2_flow": r_chi2_flow,
        "r_chi2_5050": r_chi2_5050,
        "r_chi2_minpz": r_chi2_minpz,
    }



if __name__ == "__main__":
    time = '20261016-201026'
    data_filename = 'data/new_Input_WTH_NeutrinoEtaFindingEvaluation_4th_October2026.h5'
    hyperparameter_file = f"models/hyperparameters_nflow_{time}.json"
    hyperparameters = load_hyperparameters(hyperparameter_file)
    # torch.manual_seed(hyperparameters["random_seed"])

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


    # ----------------------------
    # Coverage test

    # interval_coverage(
    #     flow=flow,
    #     dataset=test_dataset,
    #     levels=(0.50,0.68,0.90,0.95),
    #     model_name=model_name
    # )

    # ----------------------------
    # Shuffle test

    # shuffled_context_nll(
    #     flow=flow,
    #     dataset=test_dataset,
    #     batch_size=512,
    #     model_name=model_name
    # )

    # ----------------------------
    # Conditional Calibration

    # index = 'ChargedLep_pT'
    # binned_coverage(
    #     flow=flow,
    #     dataset=test_dataset,
    #     feature_index=FEATURES.index(index),
    #     model_name=model_name,
    # )

    # ----------------------------
    # PIT Value test

    # pit_test(
    #     flow=flow,
    #     dataset=test_dataset,
    #     model_name=model_name,
    #     batch_size=512,
    #     num_bins=20
    # )

    # ----------------------------
    # Point Reconstruction test

    # results = point_reconstruction_test(
    #     flow=flow,
    #     dataset=test_dataset,
    #     model_name=model_name,
    #     num_samples=2000,
    #     estimator="mode"
    # )
    #
    # residuals = results["residual"]
    #
    # plt.figure(figsize=(8, 5))
    #
    # plt.hist(
    #     residuals,
    #     bins=100,
    #     density=True
    # )
    #
    # plt.axvline(
    #     0,
    #     linestyle="--",
    #     label="Zero residual"
    # )
    #
    # plt.xlabel(
    #     r"$\Delta \hat{\eta}_\nu^{mode}$"
    # )
    # plt.ylabel("Probability Density")
    # plt.title(r"$\eta_\nu$ Reconstruction Residual")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # ----------------------------
    # Multimodality / Posterior Shape Visualization

    # with h5py.File(data_filename, "r") as f:
    #     data = f["NuFindingStudies"][:]
    #     df = pd.DataFrame(data)
    # nb1_nu_eta = df['NB1_nu_eta'].values
    # nb2_nu_eta = df['NB2_nu_eta'].values
    #
    # plot_event_posteriors(
    #     flow=flow,
    #     dataset=test_dataset,
    #     model_name=model_name,
    #     nb1_nu_eta=nb1_nu_eta,
    #     nb2_nu_eta=nb2_nu_eta,
    #     num_random_events=6,  # Or pass specific events e.g. event_indices=[12, 45, 99]
    #     grid_range=(-5.0, 5.0),
    #     num_grid_points=1000,
    # )


    # ----------------------------
    # Invariant Mass Reconstruction Test

    # invariant_mass_reconstruction(
    #     flow=flow,
    #     features_test_scaled=features_test,
    #     target_test=target_test,
    #     scaler_features=scaler_features,
    #     model_name=model_name,
    #     num_samples=500,
    #     batch_size=512,
    #     num_bins=120
    # )

    # ----------------------------
    # Correlation Test

    # with h5py.File(data_filename, "r") as f:
    #     data = f["NuFindingStudies"][:]
    #     df = pd.DataFrame(data)
    #
    # # Match test set indices
    # df_test_raw = df.iloc[-len(target_test):]
    # true_eta_test = df_test_raw["True_nu_eta"].values
    # nb1_test = df_test_raw["NB1_nu_eta"].values
    # nb2_test = df_test_raw["NB2_nu_eta"].values
    # met_test = df_test_raw["MET"].values
    #
    # correlation_test(
    #     flow=flow,
    #     dataset=test_dataset,
    #     model_name=model_name,
    #     nleadb_eta=nb1_test,
    #     nsubleadb_eta=nb2_test,
    #     met=met_test,
    #     seed=42,
    #     bins=80,
    #     range_eta=(-5,5),
    # )

    # ----------------------------
    # Angle Comparison

    with h5py.File(data_filename, "r") as f:
        data = f["NuFindingStudies"][:]
        df = pd.DataFrame(data)
    nb1_nu_eta = df['NB1_nu_eta'].values
    nb2_nu_eta = df['NB2_nu_eta'].values

    angle_comparison(
        flow=flow,
        features_test_scaled=features_test,
        target_test=target_test,
        scaler_features=scaler_features,
        model_name=model_name,
        nb1_nu_eta=nb1_nu_eta,
        nb2_nu_eta=nb2_nu_eta,
        batch_size=512,
        num_grid=500,
        range_eta=(-5, 5),
        bins=40,
    )

    # ----------------------------
    # Eta Comparison

    # with h5py.File(data_filename, "r") as f:
    #     data = f["NuFindingStudies"][:]
    #     df = pd.DataFrame(data)
    # nb1_nu_eta = df['NB1_nu_eta'].values
    # nb2_nu_eta = df['NB2_nu_eta'].values
    #
    # compare_eta_nu_distribution(
    #     flow=flow,
    #     features_test_scaled=features_test,
    #     target_test=target_test,
    #     scaler_features=scaler_features,
    #     model_name=model_name,
    #     nb1_nu_eta=nb1_nu_eta,
    #     nb2_nu_eta=nb2_nu_eta,
    #     num_samples=10,  # 10 posterior draws per event gives clean statistics
    #     bins=60,
    #     range_eta=(-5.0, 5.0),
    # )