from train import (
    device, FEATURES, TARGET, read_file, normalizing_flow, load_hyperparameters)
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
from sklearn.model_selection import train_test_split
from scipy.stats import kstest, kurtosis, norm, probplot, skew
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

def latent_space_test(
        flow, dataset, model_name, batch_size=512, num_bins=60, range_z=(-4.0, 4.0)
):
    flow.load_state_dict(torch.load(model_name, map_location=device))
    flow.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    z_values = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            # Forward map: target (eta) -> latent (z)
            # z has shape [batch_size, 1]
            z, _ = flow._transform(targets, context=features)

            z_values.append(z.squeeze(-1).cpu().numpy())

    z_values = np.concatenate(z_values)

    # --------------------------------
    # Statistical Metrics
    # --------------------------------
    mean_z = np.mean(z_values)
    std_z = np.std(z_values)
    skew_z = skew(z_values)
    kurt_z = kurtosis(z_values)  # Fisher kurtosis (0.0 for standard normal)

    ks_stat, p_value = kstest(z_values, "norm")

    print("\nLatent Space Mapping Test (z = f(eta; x))")
    print("-" * 50)
    print(f"Number of events    : {len(z_values)}")
    print(f"Mean (ideal: 0.0)   : {mean_z:+.6f}")
    print(f"Std  (ideal: 1.0)   : {std_z:.6f}")
    print(f"Skew (ideal: 0.0)   : {skew_z:+.6f}")
    print(f"Excess Kurtosis (0) : {kurt_z:+.6f}")
    print(f"KS statistic        : {ks_stat:.6f}")
    print(f"KS p-value          : {p_value:.6e}")

    # --------------------------------
    # Diagnostic Plots
    # --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Histogram vs Unit Gaussian
    x_axis = np.linspace(range_z[0], range_z[1], 400)

    axes[0].hist(
        z_values,
        bins=num_bins,
        range=range_z,
        density=True,
        alpha=0.6,
        color="royalblue",
        label=r"Encoded $z = f(\eta_\nu; x)$",
    )
    axes[0].plot(
        x_axis,
        norm.pdf(x_axis),
        "r--",
        linewidth=2,
        label=r"Target $\mathcal{N}(0, 1)$",
    )
    axes[0].set_xlabel(r"Latent variable $z$", fontsize=11)
    axes[0].set_ylabel("Probability Density", fontsize=11)
    axes[0].set_title("Latent Space Distribution", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Panel 2: Quantile-Quantile (Q-Q) Plot
    # Shows deviations in the tails (crucial for physical bounds)
    res = probplot(z_values, dist="norm", plot=axes[1])

    # Style points and regression line
    axes[1].get_lines()[0].set_markerfacecolor("royalblue")
    axes[1].get_lines()[0].set_markeredgecolor("none")
    axes[1].get_lines()[0].set_markersize(3.0)
    axes[1].get_lines()[0].set_alpha(0.4)
    axes[1].get_lines()[1].set_color("red")
    axes[1].get_lines()[1].set_linewidth(1.8)

    axes[1].set_title("Normal Q-Q Plot", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Flow Base Distribution Verification", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    return {
        "z": z_values,
        "mean": mean_z,
        "std": std_z,
        "skew": skew_z,
        "kurtosis": kurt_z,
        "ks_stat": ks_stat,
        "p_value": p_value,
    }


def invariant_mass_reconstruction(
    flow,
    features_test_scaled,
    target_test,
    scaler_features,
    model_name,
    num_samples=50,
    batch_size=512,
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
        # 1D array: [n_events]
        if eta_nu.ndim == 1:
            px_nu = met * np.cos(phi_nu)
            py_nu = met * np.sin(phi_nu)
            pz_nu = met * np.sinh(eta_nu)
            e_nu = met * np.cosh(eta_nu)

            p4_l = p4_lep
            p4_higgs = p4_h

        # 2D array: [n_events, num_samples]
        elif eta_nu.ndim == 2:
            n_s = eta_nu.shape[1]

            # Broadcast px and py across all samples
            px_nu = np.repeat((met * np.cos(phi_nu))[:, None], n_s, axis=1)
            py_nu = np.repeat((met * np.sin(phi_nu))[:, None], n_s, axis=1)
            pz_nu = met[:, None] * np.sinh(eta_nu)
            e_nu = met[:, None] * np.cosh(eta_nu)

            p4_l = p4_lep[:, None, :]
            p4_higgs = p4_h[:, None, :]

        else:
            raise ValueError("eta_nu must be 1D or 2D array")

        # Now all components have the exact same shape
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

    # 5. Compute Flow-Sampled Masses (Posterior Marginalization)
    flow_samples = []
    features_tensor = torch.tensor(features_test_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            # Sample shape: [batch_size, num_samples, 1]
            s = flow.sample(num_samples=num_samples, context=batch_x)
            flow_samples.append(s.squeeze(-1).cpu().numpy())

    eta_samples = np.concatenate(flow_samples, axis=0)  # [n_events, num_samples]
    m_w_sample, m_wh_sample = compute_masses(eta_samples)
    m_w_sampled_flat = m_w_sample.flatten()
    m_wh_sampled_flat = m_wh_sample.flatten()

    # 6. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot mW ---
    axes[0].hist(
        m_w_true,
        bins=80,
        range=(40, 120),
        density=True,
        histtype="step",
        linewidth=2,
        color="black",
        label=r"True $\eta_\nu$",
    )
    axes[0].hist(
        m_w_sampled_flat,
        bins=80,
        range=(40, 120),
        density=True,
        alpha=0.5,
        color="royalblue",
        label=r"Flow $p(\eta_\nu|x)$ Samples",
    )
    axes[0].axvline(
        80.379,
        color="red",
        linestyle="--",
        label=r"Nominal $m_W$ (80.4 GeV)",
    )
    axes[0].set_xlabel(r"$m_W$ [GeV]", fontsize=11)
    axes[0].set_ylabel("Normalized Density", fontsize=11)
    axes[0].set_title(r"$W$ Boson Invariant Mass Reconstruction", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # --- Plot mWH ---
    min_wh, max_wh = np.percentile(m_wh_true, [1, 99])
    axes[1].hist(
        m_wh_true,
        bins=80,
        range=(min_wh, max_wh),
        density=True,
        histtype="step",
        linewidth=2,
        color="black",
        label=r"True $\eta_\nu$",
    )
    axes[1].hist(
        m_wh_sampled_flat,
        bins=80,
        range=(min_wh, max_wh),
        density=True,
        alpha=0.5,
        color="darkorange",
        label=r"Flow $p(\eta_\nu|x)$ Samples",
    )
    axes[1].set_xlabel(r"$m_{WH}$ [GeV]", fontsize=11)
    axes[1].set_ylabel("Normalized Density", fontsize=11)
    axes[1].set_title(r"Total $WH$ Invariant Mass Reconstruction", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nInvariant Mass Reconstruction Metrics")
    print("-" * 50)
    print(f"True mW Peak (Median)      : {np.median(m_w_true):.2f} GeV")
    print(f"Flow Reconstructed mW Peak : {np.median(m_w_sampled_flat):.2f} GeV")
    print(
        f"mW Resolution (IQR / 2)     : {0.5 * (np.percentile(m_w_sampled_flat, 75) - np.percentile(m_w_sampled_flat, 25)):.2f} GeV"
    )
    print(f"True mWH Peak (Median)     : {np.median(m_wh_true):.2f} GeV")
    print(f"Flow Reconstructed mWH Peak: {np.median(m_wh_sampled_flat):.2f} GeV")


if __name__ == "__main__":
    time = '20261001-003138'
    hyperparameter_file = f"models/hyperparameters_{time}.json"
    hyperparameters = load_hyperparameters(hyperparameter_file)
    # torch.manual_seed(hyperparameters["random_seed"])

    model_name = f"models/model_{time}.pt"
    n_features = len(FEATURES)
    n_target = len(TARGET)
    features, target = read_file("data.csv", FEATURES, TARGET)

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
        r"$\hat{\eta}_\nu - \eta_{\nu,\mathrm{true}}$"
    )
    plt.ylabel("Density")
    plt.title("Neutrino eta reconstruction residual")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Multimodality / Posterior Shape Visualization

    df = pd.read_csv('data.csv',encoding_errors='ignore')
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
    # Latent Space Mapping Test

    latent_results = latent_space_test(
        flow=flow,
        dataset=test_dataset,
        model_name=model_name,
        batch_size=512,
        num_bins=60,
    )

    # ----------------------------
    # Invariant Mass Reconstruction Test

    invariant_mass_reconstruction(
        flow=flow,
        features_test_scaled=features_test,
        target_test=target_test,
        scaler_features=scaler_features,
        model_name=model_name,
        num_samples=50,
        batch_size=512,
    )