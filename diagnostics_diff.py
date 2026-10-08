import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
from sklearn.model_selection import train_test_split
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
    range_eta=(-4.5, 4.5),
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
            if hasattr(diffusion_model, "sample_ddim"):
                det_pred = diffusion_model.sample_ddim(features, eta=0.0)
            else:
                det_pred = diffusion_model.sample(
                    features
                )  # Fallback to standard sample

            det_np = det_pred.cpu().numpy()  # [B, 1]
            all_deterministic_preds.append(det_np)

            # -----------------------------------------------------------------
            # 2. Stochastic Posterior Draws (num_samples per event)
            # -----------------------------------------------------------------
            # Repeat context to draw num_samples independent random trajectories
            feat_rep = features.repeat_interleave(num_samples, dim=0)

            if hasattr(diffusion_model, "sample_ddim"):
                stoch_pred = diffusion_model.sample_ddim(feat_rep, eta=1.0)
            else:
                stoch_pred = diffusion_model.sample(feat_rep)

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
        ax.grid(True, alpha=0.3)
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
        "(a) Diffusion: Random Sampling",
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
        r"(b) Diffusion: Deterministic DDIM ($\eta = 0$)",
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
    features_test_scaled,
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
    features_tensor = torch.tensor(features_test_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_idx, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(device)

            # Deterministic ODE trajectory
            if hasattr(diffusion_model, "sample_ddim"):
                pred = diffusion_model.sample_ddim(batch_x, eta=0.0)
            else:
                pred = diffusion_model.sample(batch_x)

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
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper right")

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
    axes[1].grid(True, alpha=0.3)
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


if __name__ == "__main__":

    time = "20261008-222146"
    data_filename = (
        "data/new_Input_WTH_NeutrinoEtaFindingEvaluation_4th_October2026.csv"
    )

    hyperparameters = load_hyperparameters(
        f"models/hyperparameters_diff_{time}.json"
    )
    model_name = f"models/model_diff_{time}.pt"

    features_test, target_test = read_file(data_filename, FEATURES, TARGET)

    scaler_features = joblib.load(f"scalers/scaler_features_diff_{time}.joblib")
    scaler_target = joblib.load(f"scalers/scaler_target_diff_{time}.joblib")

    features_test_scaled = scaler_features.transform(features_test)

    # Build PyTorch Dataset
    dataset = TensorDataset(
        torch.tensor(features_test_scaled, dtype=torch.float32),
        torch.tensor(target_test, dtype=torch.float32),
    )

    diffusion_model = ConditionalDDPM(
        target_dim=len(TARGET),
        context_dim=len(FEATURES),
        hyperparameters=hyperparameters,
    ).to(device)

    # Load analytical solutions
    df = pd.read_csv(data_filename, encoding_errors="ignore")
    nb1_eta = df["NB1_nu_eta"].values
    nb2_eta = df["NB2_nu_eta"].values
    met = df["MET"].values

    # ------------------------------------
    # invariant mass reconstruction

    # invariant_mass_reconstruction_diffusion(
    #     diffusion_model=diffusion_model,
    #     features_test_scaled=features_test_scaled,
    #     target_test=target_test,
    #     scaler_features=scaler_features,
    #     scaler_target=scaler_target,  # Joblib-loaded scaler
    #     model_name=model_name,
    #     batch_size=1024,
    #     num_bins=120,
    # )


    # ------------------------------------
    # correlation test

    correlation_test_diffusion(
        diffusion_model=diffusion_model,
        dataset=dataset,
        model_name=model_name,
        nb1_eta=nb1_eta,
        nb2_eta=nb2_eta,
        met=met,
        scaler_target=scaler_target,
        num_samples=10,
        batch_size=512,
    )