'''
Diagnostics tests including statistical tests and physics reconstruction tests
Uncomment corresponding function to perform the test
'''

import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import kstest,pearsonr, spearmanr
from matplotlib.colors import LogNorm
import joblib
import pandas as pd
from train_diffusion import (
    device,
    FEATURES,
    TARGET,
    read_file,
    ConditionalDDPM,
    load_hyperparameters,
)
from scipy.stats import gaussian_kde


def interval_coverage_diffusion(
    diffusion_model,
    dataset,
    levels,
    model_name,
    scaler_target=None,  # Joblib-loaded scaler
    num_samples_per_event=100,  # K draws to build empirical quantiles
    max_events=5000,  # 5,000 events finishes in ~6 mins with SE ~ 0.6%
    batch_size=50,  # 50 events * 100 samples = 5000 elements per forward pass
    device=None,
    seed=42,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_model.load_state_dict(torch.load(model_name, map_location=device))
    diffusion_model.eval()

    # 1. Subsample dataset if max_events is specified
    total_available = len(dataset)
    if max_events is not None and max_events < total_available:
        rng = np.random.default_rng(seed)
        sub_indices = rng.choice(total_available, size=max_events, replace=False)
        eval_dataset = Subset(dataset, sub_indices)
        total_eval_events = max_events
    else:
        eval_dataset = dataset
        total_eval_events = total_available

    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    covered_counts = {level: 0 for level in levels}
    total_events = 0

    print(
        f"\nRunning Diffusion Coverage Test on {total_eval_events:,} events ({num_samples_per_event} samples/event)..."
    )

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(loader):
            features = features.to(device)
            B = targets.shape[0]

            # True targets in physical coordinates
            t_phys = targets.squeeze(-1).numpy()
            if (
                scaler_target is not None
                and np.std(t_phys) < 1.2
                and scaler_target.scale_[0] > 1.2
            ):
                t_phys = scaler_target.inverse_transform(
                    t_phys.reshape(-1, 1)
                ).flatten()

            # Repeat context K times for each event in the batch
            feat_rep = features.repeat_interleave(num_samples_per_event, dim=0)

            # Draw K posterior samples
            raw_samples = diffusion_model.sample_ddpm(feat_rep)

            raw_samples = raw_samples.cpu().numpy()

            # Unscale samples to physical coordinates
            if scaler_target is not None:
                samples_phys = scaler_target.inverse_transform(
                    raw_samples
                ).flatten()
            else:
                samples_phys = raw_samples.flatten()

            # Reshape into matrix: [B events, K posterior draws]
            samples_matrix = samples_phys.reshape(B, num_samples_per_event)

            # ----------------------------------------------------
            # 2. Check Empirical Quantile Coverage for Each Level
            # ----------------------------------------------------
            for level in levels:
                alpha = (1.0 - level) / 2.0
                pct_lower = alpha * 100.0
                pct_upper = (1.0 - alpha) * 100.0

                # Compute empirical lower and upper bounds for each event
                q_lower = np.percentile(samples_matrix, pct_lower, axis=1)
                q_upper = np.percentile(samples_matrix, pct_upper, axis=1)

                inside = (t_phys >= q_lower) & (t_phys <= q_upper)
                covered_counts[level] += np.sum(inside)

            total_events += B

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
                print(
                    f"  Processed {total_events:,}/{total_eval_events:,} events..."
                )

    # ----------------------------------------------------
    # 3. Print Results Summary Table
    # ----------------------------------------------------
    print("\nCalibration / Coverage Test (Diffusion Model)")
    print("-" * 58)

    results = {}
    for level in levels:
        observed = covered_counts[level] / total_events
        difference = observed - level
        results[level] = observed

        # Exact binomial standard error
        standard_error = np.sqrt(level * (1.0 - level) / total_events)

        print(
            f"Nominal {level*100:5.1f}% | "
            f"Observed {observed*100:6.3f}% | "
            f"Difference {difference*100:+6.3f}% | "
            f"SE {standard_error*100:.3f}%"
        )

    return results

def correlation_test_diffusion(
    diffusion_model,
    dataset,
    model_name,
    nb1_eta,
    nb2_eta,
    met=None,
    scaler_target=None,  # Pass if target was scaled during training
    device=None,
    batch_size=512,
    num_samples=10,  # Number of stochastic samples for panels (a) and (c)
    range_eta=(-5,5),
    bins=80,
    seed=42,
):
    """Evaluates and compares both Diffusion Model reconstruction and Analytical

    quadratic solutions (NB1/NB2) against ground-truth neutrino eta, plotting all
    diagnostics in a unified 2x3 figure.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # PART 1: Diffusion Model Sampling (Deterministic & Stochastic)
    # =========================================================================
    diffusion_model.load_state_dict(torch.load(model_name, map_location=device))
    diffusion_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_targets = []
    all_stochastic_samples = []
    all_deterministic_preds = []

    print(
        f"Running Diffusion Evaluation on {device} (num_samples = {num_samples} per event)..."
    )

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(loader):
            features = features.to(device)
            B = features.size(0)

            # -----------------------------------------------------------------
            # 1. Deterministic DDIM Prediction (eta = 0.0) -> Replaces Flow MAP
            # -----------------------------------------------------------------
            # Starts from y_T and follows the ODE trajectory deterministically

            det_pred = diffusion_model.sample_ddim(features)

            det_np = det_pred.cpu().numpy()  # [B, 1]
            all_deterministic_preds.append(det_np)

            # -----------------------------------------------------------------
            # 2. Stochastic Posterior Draws (num_samples per event)
            # -----------------------------------------------------------------
            # Repeat context to draw num_samples independent random trajectories
            feat_rep = features.repeat_interleave(num_samples, dim=0)


            stoch_pred = diffusion_model.sample_ddpm(feat_rep,eta=1.0)

            stoch_np = stoch_pred.view(B, num_samples).cpu().numpy()
            all_stochastic_samples.append(stoch_np)

            all_targets.append(targets.squeeze(-1).cpu().numpy())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")

    targets = np.concatenate(all_targets)
    det_preds = np.concatenate(all_deterministic_preds, axis=0).flatten()
    stoch_samples = np.concatenate(all_stochastic_samples, axis=0)  # [N, S]


    # Unscale predictions if a target scaler was used during training
    if scaler_target is not None:
        det_preds = scaler_target.inverse_transform(
            det_preds.reshape(-1, 1)
        ).flatten()
        stoch_samples = scaler_target.inverse_transform(
            stoch_samples.reshape(-1, 1)
        ).reshape(-1, num_samples)
        # If dataset targets were also in scaled space, unscale them too:
        if (
            np.std(targets) < 1.2
            and scaler_target.scale_[0] > 1.2
            and np.mean(targets) < 0.1
        ):
            targets = scaler_target.inverse_transform(
                targets.reshape(-1, 1)
            ).flatten()

    ratio = np.std(stoch_samples) / np.std(targets)
    print(f"Variance ratio (sigma_pred / sigma_true): {ratio:.3f}")

    # Flatten stochastic posterior samples for Panel (a)
    targets_repeated = np.repeat(targets, num_samples)
    stoch_flat = stoch_samples.flatten()

    # Disambiguated / Oracle Best Sample out of S draws for Panel (c)
    # Picks the stochastic sample closest to the true value
    diffs = np.abs(stoch_samples - targets[:, None])  # [N, S]
    best_sample_idx = np.argmin(diffs, axis=1)
    nearest_sample = stoch_samples[np.arange(len(targets)), best_sample_idx]

    # Metrics: (a) Stochastic Random Sampling
    r_samples, _ = pearsonr(targets_repeated, stoch_flat)
    slope_samp, intercept_samp = np.polyfit(targets_repeated, stoch_flat, 1)
    rms_samples = np.sqrt(np.mean((stoch_flat - targets_repeated) ** 2))

    # Metrics: (b) Deterministic DDIM (Analogous to MAP)
    r_det, _ = pearsonr(targets, det_preds)
    slope_det, intercept_det = np.polyfit(targets, det_preds, 1)
    rms_det = np.sqrt(np.mean((det_preds - targets) ** 2))

    # Metrics: (c) Nearest Sample (Oracle Disambiguation)
    r_nearest, _ = pearsonr(targets, nearest_sample)
    slope_near, intercept_near = np.polyfit(targets, nearest_sample, 1)
    rms_nearest = np.sqrt(np.mean((nearest_sample - targets) ** 2))

    # =========================================================================
    # PART 2: Analytical Quadratic Solutions (NB1 & NB2)
    # =========================================================================
    valid_mask = (
        ~np.isnan(nb1_eta)
        & ~np.isnan(nb2_eta)
        & ~np.isinf(nb1_eta)
        & ~np.isinf(nb2_eta)
    )

    t_ana = targets[valid_mask]
    s1 = nb1_eta[valid_mask]
    s2 = nb2_eta[valid_mask]
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
    print("RECONSTRUCTION PERFORMANCE BENCHMARK (DIFFUSION vs. ANALYTICAL)")
    print("=" * 78)
    print(
        f"Analytical Solvability : {n_valid:,}/{n_total:,} valid ({n_valid/n_total*100:.1f}%), {n_complex:,} complex ({n_complex/n_total*100:.1f}%)"
    )
    print(f"Min-|pz| Accuracy      : {minpz_accuracy:.1f}%")
    print("-" * 78)
    print("--- CONDITIONAL DIFFUSION MODEL ---")
    print(
        f"  Random Posterior     : r = {r_samples:.4f} | RMS = {rms_samples:.4f} | y = {slope_samp:.2f}x {intercept_samp:+.2f}"
    )
    print(
        f"  Deterministic DDIM   : r = {r_det:.4f} | RMS = {rms_det:.4f} | y = {slope_det:.2f}x {intercept_det:+.2f}"
    )
    print(
        f"  Closest Sample : r = {r_nearest:.4f} | RMS = {rms_nearest:.4f} | y = {slope_near:.2f}x {intercept_near:+.2f}"
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

    # Row 1: Diffusion Progression
    plot_panel(
        axes[0, 0],
        targets_repeated,
        stoch_flat,
        slope_samp,
        intercept_samp,
        r_samples,
        "Purples",
        "(a) Diffusion: DDPM ($\eta = 1$)",
        r"Sampled $\hat{\eta}_\nu$",
    )
    plot_panel(
        axes[0, 1],
        targets,
        det_preds,
        slope_det,
        intercept_det,
        r_det,
        "Blues",
        r"(b) Diffusion: DDIM ($\eta = 0$)",
        r"Deterministic DDIM $\hat{\eta}_\nu$",
    )
    plot_panel(
        axes[0, 2],
        targets,
        nearest_sample,
        slope_near,
        intercept_near,
        r_nearest,
        "Greens",
        "(c) Diffusion: Closest Sample",
        r"Nearest Sample $\hat{\eta}_\nu$",
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
        r"Comparison of Neutrino $\eta_\nu$ Reconstruction: Diffusion Model vs. Analytical Quadratic Solution",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()
    plt.show()

    return {
        "diff_r_samples": r_samples,
        "diff_rms_samples": rms_samples,
        "diff_r_det": r_det,
        "diff_rms_det": rms_det,
        "diff_r_nearest": r_nearest,
        "diff_rms_nearest": rms_nearest,
        "ana_r_random": r_rand,
        "ana_rms_random": rms_rand,
        "ana_r_minpz": r_minpz,
        "ana_rms_minpz": rms_minpz,
        "ana_r_closest": r_close,
        "ana_rms_closest": rms_close,
    }

def compute_mode(data, bins=120, data_range=None):
    """Calculates the empirical mode (peak location) of a distribution."""
    counts, edges = np.histogram(data, bins=bins, range=data_range)
    max_idx = np.argmax(counts)
    return 0.5 * (edges[max_idx] + edges[max_idx + 1])


def invariant_mass_reconstruction_diffusion(
    diffusion_model,
    features_test,
    target_test,
    scaler_features,
    model_name,
    scaler_target=None,  # Pass scaler_target used in training
    batch_size=1024,
    num_bins=120,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_model.load_state_dict(torch.load(model_name, map_location=device))
    diffusion_model.eval()

    # 1. Unscale features to physical units (GeV, radians)
    features_unscaled = scaler_features.inverse_transform(features_test)

    # 2. Extract kinematics
    lep_pt = features_unscaled[:, FEATURES.index("ChargedLep_pT")]
    lep_eta = features_unscaled[:, FEATURES.index("ChargedLep_Eta")]
    lep_e = features_unscaled[:, FEATURES.index("ChargedLep_E")]

    met = features_unscaled[:, FEATURES.index("MET")]
    phi_nu = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_MET")]

    leadb_pt = features_unscaled[:, FEATURES.index("LeadBJet_pT")]
    leadb_eta = features_unscaled[:, FEATURES.index("LeadBJet_Eta")]
    leadb_e = features_unscaled[:, FEATURES.index("LeadBJet_E")]
    leadb_phi = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_LeadBJet")]
    p4_leadb = np.stack(
        [leadb_pt*np.cos(leadb_phi), leadb_pt*np.sin(leadb_phi), leadb_pt * np.sinh(leadb_eta), leadb_e],
        axis=-1,
    )

    subleadb_pt = features_unscaled[:, FEATURES.index("SubLeadBJet_pT")]
    subleadb_eta = features_unscaled[:, FEATURES.index("SubLeadBJet_Eta")]
    subleadb_e = features_unscaled[:, FEATURES.index("SubLeadBJet_E")]
    subleadb_phi = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_SubleadBJet")]
    p4_subleadb = np.stack(
        [subleadb_pt * np.cos(subleadb_phi), subleadb_pt * np.sin(subleadb_phi), subleadb_pt * np.sinh(subleadb_eta), subleadb_e],
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
        px_nu = met * np.cos(phi_nu)
        py_nu = met * np.sin(phi_nu)
        pz_nu = met * np.sinh(eta_nu)
        e_nu = met * np.cosh(eta_nu)

        p4_nu = np.stack([px_nu, py_nu, pz_nu, e_nu], axis=-1)

        # W Boson invariant mass: m^2 = E^2 - |p|^2
        p4_w = p4_lep + p4_nu
        m_w_sq = (
            p4_w[..., 3] ** 2
            - (p4_w[..., 0] ** 2 + p4_w[..., 1] ** 2 + p4_w[..., 2] ** 2)
        )
        m_w = np.sqrt(np.maximum(0, m_w_sq))

        # Total WH invariant mass
        p4_wh = p4_w + p4_h
        m_wh_sq = (
            p4_wh[..., 3] ** 2
            - (p4_wh[..., 0] ** 2 + p4_wh[..., 1] ** 2 + p4_wh[..., 2] ** 2)
        )
        m_wh = np.sqrt(np.maximum(0, m_wh_sq))

        return m_w, m_wh

    # 4. Compute True Masses
    m_w_true, m_wh_true = compute_masses(true_eta)

    # 5. Compute Diffusion Deterministic (DDIM eta=0) Predictions
    # This acts as the clean point estimator (equivalent to MAP)
    print(
        f"Reconstructing masses using Deterministic DDIM (eta=0.0) on {device}..."
    )
    diff_preds = []
    features_tensor = torch.tensor(features_test, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_idx, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(device)

            pred = diffusion_model.sample_ddim(batch_x)

            diff_preds.append(pred.cpu().numpy())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")

    eta_diff = np.concatenate(diff_preds, axis=0).flatten()

    # Invert target scaling back to physical pseudorapidity
    if scaler_target is not None:
        eta_diff = scaler_target.inverse_transform(
            eta_diff.reshape(-1, 1)
        ).flatten()

    # Compute reconstructed masses
    m_w_diff, m_wh_diff = compute_masses(eta_diff)

    # 6. Mode & Range Calculations
    mw_range = (0, 160)
    min_wh, max_wh = np.percentile(m_wh_true, [1, 99])
    wh_range = (min_wh, max_wh)

    mode_mw_true = compute_mode(m_w_true, bins=num_bins, data_range=mw_range)
    mode_mw_diff = compute_mode(m_w_diff, bins=num_bins, data_range=mw_range)

    mode_mwh_true = compute_mode(m_wh_true, bins=num_bins, data_range=wh_range)
    mode_mwh_diff = compute_mode(m_wh_diff, bins=num_bins, data_range=wh_range)

    # 7. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Plot mW ---
    axes[0].hist(
        m_w_true,
        bins=num_bins,
        range=mw_range,
        density=True,
        histtype="step",
        linewidth=2.2,
        color="black",
        label=rf"True $\eta_\nu$ (Peak: {mode_mw_true:.2f} GeV)",
    )
    axes[0].hist(
        m_w_diff,
        bins=num_bins,
        range=mw_range,
        density=True,
        alpha=0.5,
        color="royalblue",
        label=rf"Diffusion DDIM ($\eta=0$) (Peak: {mode_mw_diff:.2f} GeV)",
    )
    axes[0].axvline(
        80.36,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=r"Nominal $m_W$ (80.36 GeV)",
    )
    axes[0].set_xlabel(r"$m_W$ [GeV]", fontsize=12)
    axes[0].set_ylabel("Normalized Density [GeV$^{-1}$]", fontsize=11)
    axes[0].set_title(
        r"$W$ Boson Invariant Mass Reconstruction [Diffusion Model]",
        fontsize=13,
    )
    axes[0].legend(fontsize=8, loc="upper right")

    # --- Plot mWH ---
    axes[1].hist(
        m_wh_true,
        bins=num_bins,
        range=wh_range,
        density=True,
        histtype="step",
        linewidth=2.2,
        color="black",
        label=rf"True $\eta_\nu$ (Peak: {mode_mwh_true:.2f} GeV)",
    )
    axes[1].hist(
        m_wh_diff,
        bins=num_bins,
        range=wh_range,
        density=True,
        alpha=0.5,
        color="darkorange",
        label=rf"Diffusion DDIM ($\eta=0$) (Peak: {mode_mwh_diff:.2f} GeV)",
    )
    axes[1].set_xlabel(r"$m_{WH}$ [GeV]", fontsize=12)
    axes[1].set_ylabel("Normalized Density [GeV$^{-1}$]", fontsize=11)
    axes[1].set_title(
        r"Total $WH$ Invariant Mass Reconstruction [Diffusion Model]",
        fontsize=13,
    )
    axes[1].legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.show()

    # Print Summary Report
    res_true = 0.5 * (
        np.percentile(m_w_true, 75) - np.percentile(m_w_true, 25)
    )
    res_diff = 0.5 * (
        np.percentile(m_w_diff, 75) - np.percentile(m_w_diff, 25)
    )

    print("\n" + "=" * 55)
    print("INVARIANT MASS RECONSTRUCTION BENCHMARK (DIFFUSION)")
    print("=" * 55)
    print(f"True mW Peak (Mode)              : {mode_mw_true:.2f} GeV")
    print(f"Diffusion Reconstructed mW Peak  : {mode_mw_diff:.2f} GeV")
    print(f"True mW Resolution (IQR / 2)     : {res_true:.2f} GeV")
    print(f"Diffusion mW Resolution (IQR / 2): {res_diff:.2f} GeV")
    print("-" * 55)
    print(f"True mWH Peak (Mode)             : {mode_mwh_true:.2f} GeV")
    print(f"Diffusion Reconstructed mWH Peak : {mode_mwh_diff:.2f} GeV")
    print("=" * 55)

    return {
        "m_w_true": m_w_true,
        "m_w_diff": m_w_diff,
        "m_wh_true": m_wh_true,
        "m_wh_diff": m_wh_diff,
        "mode_mw_diff": mode_mw_diff,
        "mode_mwh_diff": mode_mwh_diff,
    }


def WH_mass_reconstruction_diffusion(
    diffusion_model,
    features_test,
    target_test,
    scaler_features,
    model_name,
    scaler_target=None,  # Pass scaler_target used in training
    batch_size=1024,
    num_bins=120,
    device=None,
    nb1_nu_eta=None,
    nb2_nu_eta=None,
    seed=42
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_model.load_state_dict(torch.load(model_name, map_location=device))
    diffusion_model.eval()

    # 1. Unscale features to physical units (GeV, radians)
    features_unscaled = scaler_features.inverse_transform(features_test)

    # 2. Extract kinematics
    lep_pt = features_unscaled[:, FEATURES.index("ChargedLep_pT")]
    lep_eta = features_unscaled[:, FEATURES.index("ChargedLep_Eta")]
    lep_e = features_unscaled[:, FEATURES.index("ChargedLep_E")]

    met = features_unscaled[:, FEATURES.index("MET")]
    phi_nu = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_MET")]

    leadb_pt = features_unscaled[:, FEATURES.index("LeadBJet_pT")]
    leadb_eta = features_unscaled[:, FEATURES.index("LeadBJet_Eta")]
    leadb_e = features_unscaled[:, FEATURES.index("LeadBJet_E")]
    leadb_phi = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_LeadBJet")]
    p4_leadb = np.stack(
        [leadb_pt*np.cos(leadb_phi), leadb_pt*np.sin(leadb_phi), leadb_pt * np.sinh(leadb_eta), leadb_e],
        axis=-1,
    )

    subleadb_pt = features_unscaled[:, FEATURES.index("SubLeadBJet_pT")]
    subleadb_eta = features_unscaled[:, FEATURES.index("SubLeadBJet_Eta")]
    subleadb_e = features_unscaled[:, FEATURES.index("SubLeadBJet_E")]
    subleadb_phi = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_SubleadBJet")]
    p4_subleadb = np.stack(
        [subleadb_pt * np.cos(subleadb_phi), subleadb_pt * np.sin(subleadb_phi), subleadb_pt * np.sinh(subleadb_eta), subleadb_e],
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
        px_nu = met * np.cos(phi_nu)
        py_nu = met * np.sin(phi_nu)
        pz_nu = met * np.sinh(eta_nu)
        e_nu = met * np.cosh(eta_nu)

        p4_nu = np.stack([px_nu, py_nu, pz_nu, e_nu], axis=-1)

        # W Boson invariant mass: m^2 = E^2 - |p|^2
        p4_w = p4_lep + p4_nu


        # Total WH invariant mass
        p4_wh = p4_w + p4_h
        m_wh_sq = (
            p4_wh[..., 3] ** 2
            - (p4_wh[..., 0] ** 2 + p4_wh[..., 1] ** 2 + p4_wh[..., 2] ** 2)
        )
        m_wh = np.sqrt(np.maximum(0, m_wh_sq))

        return m_wh

    # 4. Compute True Masses
    m_wh_true = compute_masses(true_eta)

    # 5. Compute Diffusion Deterministic (DDIM eta=0) Predictions
    # This acts as the clean point estimator (equivalent to MAP)
    print(
        f"Reconstructing masses using Deterministic DDIM (eta=0.0) on {device}..."
    )
    diff_preds = []
    features_tensor = torch.tensor(features_test, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_idx, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(device)

            pred = diffusion_model.sample_ddim(batch_x)

            diff_preds.append(pred.cpu().numpy())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")

    eta_diff = np.concatenate(diff_preds, axis=0).flatten()

    # Invert target scaling back to physical pseudorapidity
    if scaler_target is not None:
        eta_diff = scaler_target.inverse_transform(
            eta_diff.reshape(-1, 1)
        ).flatten()

    # Compute reconstructed masses
    m_wh_diff = compute_masses(eta_diff)

    eta_5050 = None
    eta_minpz = None
    n_events = len(true_eta)

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
    m_wh_5050 = compute_masses(eta_5050)
    m_wh_minpz = compute_masses(eta_minpz)


    # 6. Mode & Range Calculations
    min_wh, max_wh = np.percentile(m_wh_true, [1, 99])
    wh_range = (min_wh, max_wh)
    print(wh_range)
    print(f"Bin width: {(max_wh-min_wh)/num_bins}")


    mode_mwh_true = compute_mode(m_wh_true, bins=num_bins, data_range=wh_range)
    mode_mwh_diff = compute_mode(m_wh_diff, bins=num_bins, data_range=wh_range)
    mode_mwh_5050 = compute_mode(m_wh_5050, bins=num_bins, data_range=wh_range)
    mode_mwh_minpz = compute_mode(m_wh_minpz, bins=num_bins, data_range=wh_range)

    # 7. Plotting

    # --- Plot mWH ---
    plt.hist(
        m_wh_true,
        bins=num_bins,
        range=wh_range,
        density=True,
        histtype="step",
        linewidth=2.2,
        color="black",
        label=rf"True $\eta_\nu$ (Peak: {mode_mwh_true:.2f} GeV)",
    )
    plt.hist(
        m_wh_diff,
        bins=num_bins,
        range=wh_range,
        density=True,
        alpha=0.5,
        color="royalblue",
        label=rf"Diffusion DDIM ($\eta=0$) (Peak: {mode_mwh_diff:.2f} GeV)",
    )
    if eta_5050 is not None:
        plt.hist(
            m_wh_5050,
            bins=num_bins,
            range=wh_range,
            histtype="step",
            density=True,
            linestyle="--",
            linewidth=1.8,
            color="crimson",
            label=rf"Baseline 50/50 (Peak: {mode_mwh_5050:.2f} GeV)",
        )

    if eta_minpz is not None:
        plt.hist(
            m_wh_minpz,
            bins=num_bins,
            range=wh_range,
            histtype="step",
            density=True,
            linestyle=":",
            linewidth=2.0,
            color="darkorange",
            label=rf"Baseline min-$|p_z|$ (Peak: {mode_mwh_minpz:.2f} GeV)",
        )
    plt.xlabel(r"$m_{WH}$ [GeV]", fontsize=12)
    plt.ylabel("Normalized Density [GeV$^{-1}$]", fontsize=11)
    plt.title(
        r"Total $WH$ Invariant Mass Reconstruction [Diffusion Model]",
        fontsize=13,
    )
    plt.legend(fontsize=8,frameon=True)

    plt.tight_layout()
    plt.show()


    print("\n" + "=" * 55)
    print("INVARIANT MASS RECONSTRUCTION BENCHMARK (DIFFUSION)")
    print("=" * 55)
    print(f"True mWH Peak (Mode)             : {mode_mwh_true:.2f} GeV")
    print(f"Diffusion Reconstructed mWH Peak : {mode_mwh_diff:.2f} GeV")
    print(f"Baseline 50/50 mWH Peak : {mode_mwh_5050:.2f} GeV")
    print(f"Baseline min-$|p_z|$ mWH Peak : {mode_mwh_minpz:.2f} GeV")
    print("=" * 55)

    return {
        "m_wh_true": m_wh_true,
        "m_wh_diff": m_wh_diff,
        "mode_mwh_diff": mode_mwh_diff,
    }


def plot_event_posteriors_diffusion(
    diffusion_model,
    dataset,
    model_name,
    scaler_target=None,  # Pass scaler_target used in training
    nb1_nu_eta=None,  # 1D numpy array of first quadratic solutions
    nb2_nu_eta=None,  # 1D numpy array of second quadratic solutions
    event_indices=None,
    num_random_events=6,
    num_samples_per_event=500,  # Number of posterior draws per event
    grid_range=(-5.0, 5.0),
    device=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_model.load_state_dict(torch.load(model_name, map_location=device))
    diffusion_model.eval()

    # 1. Select which event indices to visualize
    if event_indices is None:
        rng = np.random.default_rng(42)
        event_indices = rng.choice(
            len(dataset), size=num_random_events, replace=False
        )

    num_events = len(event_indices)
    cols = 3
    rows = int(np.ceil(num_events / cols))

    grid_eval = np.linspace(grid_range[0], grid_range[1], 600)

    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.4 * rows))
    axes = np.array(axes).reshape(-1)

    print(
        f"Generating posterior distributions for {num_events} events using Diffusion Model..."
    )

    with torch.no_grad():
        for i, idx in enumerate(event_indices):
            features, target = dataset[idx]

            # True target value in physical coordinates
            true_eta = target.item()
            if (
                scaler_target is not None
                and np.std(dataset.tensors[1].numpy()) < 1.2
            ):
                true_eta = scaler_target.inverse_transform([[true_eta]])[0, 0]

            # Replicate context K times to draw K stochastic posterior samples
            # Shape: [num_samples_per_event, n_features]
            bx = (
                features.unsqueeze(0)
                .repeat(num_samples_per_event, 1)
                .to(device)
            )

            # Draw stochastic posterior samples (Langevin / DDPM dynamics)
            raw_samples = diffusion_model.sample_ddpm(bx).cpu().numpy()

            # Unscale to physical pseudorapidity
            if scaler_target is not None:
                samples_phys = scaler_target.inverse_transform(
                    raw_samples
                ).flatten()
            else:
                samples_phys = raw_samples.flatten()

            # Filter any rare unphysical outlier samples for smooth KDE
            valid_samples = samples_phys[
                (samples_phys >= grid_range[0])
                & (samples_phys <= grid_range[1])
            ]

            # Construct smooth Continuous Density via Gaussian KDE
            kde = gaussian_kde(valid_samples, bw_method=0.18)
            prob_density = kde(grid_eval)

            # Find the peak mode
            mode_eta = grid_eval[np.argmax(prob_density)]

            ax = axes[i]

            # ----------------------------------------------------
            # 1. Plot Diffusion Posterior (Smooth KDE + Light Histogram)
            # ----------------------------------------------------
            ax.hist(
                valid_samples,
                bins=40,
                range=grid_range,
                density=True,
                alpha=0.25,
                color="royalblue",
                edgecolor="none",
            )
            ax.plot(
                grid_eval,
                prob_density,
                color="royalblue",
                lw=2.2,
                label=r"Diffusion $p(\eta_\nu \mid \mathbf{x})$",
            )

            # ----------------------------------------------------
            # 2. Mark True eta_nu
            # ----------------------------------------------------
            ax.axvline(
                true_eta,
                color="red",
                linestyle="--",
                lw=2.0,
                label=rf"True $\eta_\nu$ ({true_eta:.2f})",
                zorder=4,
            )

            # ----------------------------------------------------
            # 3. Mark Diffusion Mode (Peak)
            # ----------------------------------------------------
            ax.axvline(
                mode_eta,
                color="forestgreen",
                linestyle=":",
                lw=1.8,
                label=rf"Mode ({mode_eta:.2f})",
                zorder=3,
            )

            # ----------------------------------------------------
            # 4. Mark Analytical Solution 1 (NB1)
            # ----------------------------------------------------
            if nb1_nu_eta is not None:
                val_nb1 = nb1_nu_eta[idx]
                if not (np.isnan(val_nb1) or np.isinf(val_nb1)):
                    ax.axvline(
                        val_nb1,
                        color="darkorange",
                        linestyle="-.",
                        lw=1.4,
                        label=rf"NB1 ({val_nb1:.2f})",
                        zorder=2,
                    )
                else:
                    ax.plot([], [], " ", label="NB1: Complex (NaN)")

            # ----------------------------------------------------
            # 5. Mark Analytical Solution 2 (NB2)
            # ----------------------------------------------------
            if nb2_nu_eta is not None:
                val_nb2 = nb2_nu_eta[idx]
                if not (np.isnan(val_nb2) or np.isinf(val_nb2)):
                    ax.axvline(
                        val_nb2,
                        color="purple",
                        linestyle="-.",
                        lw=1.4,
                        label=rf"NB2 ({val_nb2:.2f})",
                        zorder=2,
                    )
                else:
                    ax.plot([], [], " ", label="NB2: Complex (NaN)")

            ax.set_title(f"Event #{idx}", fontsize=11)
            ax.set_xlabel(r"$\eta_\nu$", fontsize=10)
            ax.set_ylabel("Probability Density", fontsize=10)
            ax.set_xlim(grid_range)
            ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    # Hide unused subplot panels
    for j in range(num_events, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        r"Posterior Density $p(\eta_\nu \mid \mathbf{x})$: Diffusion Model vs. Analytical Roots",
        fontsize=14,
        y=1,
    )
    plt.tight_layout()
    plt.show()


def compare_eta_nu_distribution_diffusion(
    diffusion_model,
    features_test,
    target_test,
    model_name,
    scaler_target=None,  # Joblib-loaded target scaler for inverse transform
    nb1_nu_eta=None,  # 1st quadratic solution
    nb2_nu_eta=None,  # 2nd quadratic solution
    num_samples=1,  # 1 sample/event gives full dataset statistics rapidly
    sampling_steps=50,
    batch_size=1024,
    bins=60,
    range_eta=(-5.0, 5.0),
    seed=42,
    device=None,
):
    """Directly compares reconstructed neutrino pseudorapidity (eta_nu)

    distributions for the Diffusion Model:
      - MC Truth vs. Diffusion Model vs. 50/50 vs. Min-|pz|
      - Calculates chi2/ndf against MC Truth
      - Plots 2-panel Normalized Density + Ratio-to-Truth (No grid lines)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_model.load_state_dict(torch.load(model_name, map_location=device))
    diffusion_model.eval()

    true_eta = target_test.flatten()

    n_events = len(true_eta)

    # 1. Unscale features to extract MET for min-|pz| baseline
    met = features_test[:, FEATURES.index("MET")]

    # ----------------------------------------------------
    # 2. Sample from Diffusion Model
    # ----------------------------------------------------
    print(
        f"Sampling eta_nu from Diffusion Model on {device} (num_samples={num_samples})..."
    )
    diff_samples = []
    features_tensor = torch.tensor(features_test, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_idx, (bx,) in enumerate(loader):
            bx = bx.to(device)

            if num_samples == 1:
                # Fast deterministic single-sample inference via DDIM (eta=0.0)
                # sampling_steps=50 gives a 10x speedup with near-identical physics
                s = diffusion_model.sample_ddim(bx, eta=0.0, sampling_steps=sampling_steps)
                diff_samples.append(s.cpu().numpy())
            else:
                # Multiple stochastic posterior draws per event via DDPM (eta=1.0)
                bx_rep = bx.repeat_interleave(num_samples, dim=0)
                s = diffusion_model.sample_ddpm(bx_rep, eta=1.0)
                diff_samples.append(s.cpu().numpy())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")

    eta_diff_raw = np.concatenate(diff_samples, axis=0)
    orig_shape = eta_diff_raw.shape
    eta_diff = scaler_target.inverse_transform(
        eta_diff_raw.reshape(-1, 1)
    ).reshape(orig_shape)

    # ----------------------------------------------------
    # 4. Construct Analytical Baselines (50/50 & min-|pz|)
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
    # 5. Two-Sample Poisson Chi-Squared Calculation
    # ----------------------------------------------------
    def compute_chi2_ndf(counts_test, counts_ref):
        n_ref = np.sum(counts_ref)
        n_test = np.sum(counts_test)
        scale = n_ref / n_test

        scaled_test = counts_test * scale
        variance = counts_ref + (scale**2) * counts_test

        valid = (counts_ref > 5) & (variance > 0)
        chi2 = np.sum(
            (counts_ref[valid] - scaled_test[valid]) ** 2 / variance[valid]
        )
        ndf = np.sum(valid) - 1
        chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
        return chi2, ndf, chi2_ndf

    # Compute raw histograms for Chi2 and errors
    raw_truth, edges = np.histogram(true_eta, bins=bins, range=range_eta)
    raw_diff, _ = np.histogram(eta_diff, bins=bins, range=range_eta)
    centers = 0.5 * (edges[:-1] + edges[1:])

    _, _, r_chi2_diff = compute_chi2_ndf(raw_diff, raw_truth)

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
    norm_diff, _ = np.histogram(
        eta_diff, bins=bins, range=range_eta, density=True
    )

    # ----------------------------------------------------
    # 6. Print Summary Table
    # ----------------------------------------------------
    std_true = np.std(true_eta)
    std_diff = np.std(eta_diff)

    print("\n" + "=" * 75)
    print(
        f"{'Method':<24} | {'Mean':<8} | {'Std (Width)':<12} | {'chi2 / ndf':<12}"
    )
    print("=" * 75)
    print(
        f"{'1. MC Truth (Ref)':<24} | {np.mean(true_eta):<8.3f} | {std_true:<12.3f} | {'—':<12}"
    )
    print(
        f"{'2. Diffusion Samples':<24} | {np.mean(eta_diff):<8.3f} | {std_diff:<12.3f} | {r_chi2_diff:<12.2f}"
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
    # 7. Plotting: Distribution & Ratio (NO GRID LINES)
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
        weights=norm_diff,
        histtype="stepfilled",
        alpha=0.45,
        color="royalblue",
        label=rf"Model Samples ($\chi^2/\mathrm{{ndf}} = {r_chi2_diff:.2f}$)",
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
        r"Neutrino Pseudorapidity ($\eta_\nu$) Reconstruction [Diffusion Model]",
        fontsize=13,
    )
    ax_main.legend(fontsize=9.0, loc="upper right")
    # NO GRID ON ax_main

    # --- Ratio Panel (Model / Truth) ---
    ratio_diff = norm_diff / np.maximum(norm_truth, 1e-6)

    # Poisson uncertainties on Diffusion ratio (clean, capless)
    valid_bins = (raw_diff > 0) & (raw_truth > 0)
    ratio_err_diff = np.zeros_like(ratio_diff)
    ratio_err_diff[valid_bins] = ratio_diff[valid_bins] * np.sqrt(
        1.0 / raw_diff[valid_bins] + 1.0 / raw_truth[valid_bins]
    )

    ax_ratio.errorbar(
        centers[valid_bins],
        ratio_diff[valid_bins],
        yerr=ratio_err_diff[valid_bins],
        fmt="o-",
        color="royalblue",
        ecolor="royalblue",
        ms=3.0,
        elinewidth=1.0,
        capsize=0,
        label="Diffusion / Truth",
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
        "r_chi2_diff": r_chi2_diff,
        "r_chi2_5050": r_chi2_5050,
        "r_chi2_minpz": r_chi2_minpz
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
    p4_h_in_W = lorentz_boost(p4_h_X, beta_w)
    u_h_in_W = unit_vector(p4_h_in_W[:, :3])

    # Angle relative to W's direction of flight in the WH frame
    costheta1 = np.sum(-u_lep_in_W * u_h_in_W, axis=-1)
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
    diffusion_model,
    features_test,
    target_test,
    scaler_features,
    model_name,
    scaler_target=None,  # Joblib-loaded target scaler for inverse transform
    nb1_nu_eta=None,  # 1st quadratic solution
    nb2_nu_eta=None,  # 2nd quadratic solution
    batch_size=512,
    num_samples=1,
    bins=40,
    seed=42,
    sampling_steps=50,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_model.load_state_dict(torch.load(model_name, map_location=device))
    diffusion_model.eval()

    # ----------------------------------------------------
    # 1. Unscale features to physical units
    # ----------------------------------------------------
    feat_unscaled = scaler_features.inverse_transform(features_test)
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
    print(
        f"Sampling eta_nu from Diffusion Model on {device} (num_samples={num_samples})..."
    )
    diff_samples = []
    features_tensor = torch.tensor(features_test, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_idx, (bx,) in enumerate(loader):
            bx = bx.to(device)

            if num_samples == 1:
                # Fast deterministic single-sample inference via DDIM (eta=0.0)
                # sampling_steps=50 gives a 10x speedup with near-identical physics
                s = diffusion_model.sample_ddim(bx, eta=0.0, sampling_steps=sampling_steps)
                diff_samples.append(s.cpu().numpy())
            else:
                # Multiple stochastic posterior draws per event via DDPM (eta=1.0)
                bx_rep = bx.repeat_interleave(num_samples, dim=0)
                s = diffusion_model.sample_ddpm(bx_rep, eta=1.0)
                diff_samples.append(s.cpu().numpy())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")

    eta_diff_raw = np.concatenate(diff_samples, axis=0)
    orig_shape = eta_diff_raw.shape
    eta_diff = scaler_target.inverse_transform(
        eta_diff_raw.reshape(-1, 1)
    ).reshape(orig_shape)

    if num_samples >1:
        p4_lep_rep = np.repeat(p4_lep, num_samples, axis=0)
        p4_b1_rep = np.repeat(p4_b1, num_samples, axis=0)
        p4_b2_rep = np.repeat(p4_b2, num_samples, axis=0)
        met_rep = np.repeat(met, num_samples, axis=0)
        phi_nu_rep = np.repeat(phi_nu, num_samples, axis=0)
    else:
        p4_lep_rep = p4_lep
        p4_b1_rep = p4_b1
        p4_b2_rep = p4_b2
        met_rep = met
        phi_nu_rep = phi_nu

    # Build neutrino and compute angles
    px = met_rep * np.cos(phi_nu_rep)
    py = met_rep * np.sin(phi_nu_rep)
    pz = met_rep * np.sinh(eta_diff.flatten())
    e = met_rep * np.cosh(eta_diff.flatten())
    p4_nu_diff = np.stack([px, py, pz, e], axis=-1)

    obs_diff = compute_angles_vectorized(
        p4_lep_rep, p4_nu_diff, p4_b1_rep, p4_b2_rep
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
            data_diff = np.abs(obs_diff[key])
            data_5050 = np.abs(obs_5050[key]) if obs_5050 else None
            data_minpz = np.abs(obs_minpz[key]) if obs_minpz else None
        else:
            data_true = obs_true[key]
            data_diff = obs_diff[key]
            data_5050 = obs_5050[key] if obs_5050 else None
            data_minpz = obs_minpz[key] if obs_minpz else None

        # Filter NaNs from complex analytical solutions
        data_true = data_true[~np.isnan(data_true)]
        data_diff = data_diff[~np.isnan(data_diff)]

        # Raw counts for chi2
        raw_true, edges = np.histogram(data_true, bins=bins, range=rng)
        raw_diff, _ = np.histogram(data_diff, bins=bins, range=rng)
        centers = 0.5 * (edges[:-1] + edges[1:])

        _, _, r_chi2_f = compute_chi2_ndf(raw_diff, raw_true)

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
        norm_diff, _ = np.histogram(
            data_diff, bins=bins, range=rng, density=True
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
            weights=norm_diff,
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
        ax.set_ylim(0,0.9)
        ax.set_xlim(rng)
        ax.legend(fontsize=8, loc="upper left",frameon=False)

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
        "obs_diff": obs_diff,
        "obs_5050": obs_5050,
        "obs_minpz": obs_minpz,
    }


if __name__ == "__main__":

    time = "20261016-211547"
    data_filename = (
        "data/new_Input_WLH_NeutrinoEtaFindingEvaluation_4th_October2026.h5"
    )

    hyperparameters = load_hyperparameters(
        f"models/hyperparameters_diff_{time}.json"
    )
    model_name = f"models/model_diff_{time}.pt"

    features_test, target_test = read_file(data_filename, FEATURES, TARGET)

    scaler_features = joblib.load(f"scalers/scaler_features_diff_{time}.joblib")
    scaler_target = joblib.load(f"scalers/scaler_target_diff_{time}.joblib")

    features_test = scaler_features.transform(features_test)

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

    # Load analytical solutions
    with h5py.File(data_filename, "r") as f:
        data = f["NuFindingStudies"][:]
        df = pd.DataFrame(data)
    nb1_eta = df["NB1_nu_eta"].values
    nb2_eta = df["NB2_nu_eta"].values
    met = df["MET"].values

    # ------------------------------------
    # coverage test
    # levels = [0.50, 0.68, 0.90, 0.95]
    #
    # interval_coverage_diffusion(
    #     diffusion_model=diffusion_model,
    #     dataset=dataset,
    #     levels=levels,
    #     model_name=model_name,
    #     scaler_target=scaler_target,  # Joblib-loaded scaler
    #     num_samples_per_event=2000,  # 100 draws builds clean empirical percentiles
    #     max_events=5000,  # 5,000 events takes ~6 mins
    #     batch_size=50,
    # )

    # ------------------------------------
    # invariant mass reconstruction

    # invariant_mass_reconstruction_diffusion(
    #     diffusion_model=diffusion_model,
    #     features_test=features_test,
    #     target_test=target_test,
    #     scaler_features=scaler_features,
    #     scaler_target=scaler_target,  # Joblib-loaded scaler
    #     model_name=model_name,
    #     batch_size=1024,
    #     num_bins=120,
    # )


    # ------------------------------------
    # correlation test

    # correlation_test_diffusion(
    #     diffusion_model=diffusion_model,
    #     dataset=dataset,
    #     model_name=model_name,
    #     nb1_eta=nb1_eta,
    #     nb2_eta=nb2_eta,
    #     met=met,
    #     scaler_target=scaler_target,
    #     num_samples=10,
    #     batch_size=512,
    #     range_eta=(-5,5)
    # )

    # ------------------------------------
    # Plot event posterior

    # plot_event_posteriors_diffusion(
    #     diffusion_model=diffusion_model,
    #     dataset=dataset,
    #     model_name=model_name,
    #     scaler_target=scaler_target,  # Joblib-loaded scaler
    #     nb1_nu_eta=nb1_eta,  # Analytical roots
    #     nb2_nu_eta=nb2_eta,
    #     num_random_events=6,  # Plots 6 events (2x3 grid)
    #     num_samples_per_event=5000,  # 500 draws per event (finishes in ~10 seconds)
    #     grid_range=(-5.0, 5.0),
    # )

    # ------------------------------------
    # Eta comparison

    # compare_eta_nu_distribution_diffusion(
    #     diffusion_model=diffusion_model,
    #     features_test=features_test,
    #     target_test=target_test,
    #     scaler_features=scaler_features,
    #     scaler_target=scaler_target,  # Inverse transforms predictions automatically
    #     model_name=model_name,
    #     nb1_nu_eta=nb1_eta,
    #     nb2_nu_eta=nb2_eta,
    #     num_samples=1,  # Fast 1-sample evaluation across the dataset
    #     batch_size=1024,
    #     bins=60,
    #     range_eta=(-5.0, 5.0),
    # )

    # ------------------------------------
    # Angular observables

    angle_comparison(
        diffusion_model=diffusion_model,
        features_test=features_test,
        target_test=target_test,
        scaler_features=scaler_features,
        scaler_target=scaler_target,
        model_name=model_name,
        nb1_nu_eta=nb1_eta,
        nb2_nu_eta=nb2_eta,
        num_samples=1,
    )

    # ------------------------------------
    # WH mass reconstruction

#     WH_mass_reconstruction_diffusion(
#         diffusion_model=diffusion_model,
#         features_test=features_test,
#         target_test=target_test,
#         scaler_features=scaler_features,
#         scaler_target=scaler_target,  # Joblib-loaded scaler
#         model_name=model_name,
#         batch_size=1024,
#         num_bins=120,
#         nb1_nu_eta=nb1_eta,
#         nb2_nu_eta=nb2_eta,
# )
