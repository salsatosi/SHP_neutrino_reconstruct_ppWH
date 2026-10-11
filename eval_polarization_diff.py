import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import from your diffusion training script
from train_diffusion import (
    device,
    FEATURES,
    TARGET,
    read_file,
    ConditionalDDPM,
    load_hyperparameters,
)


def lorentz_boost(p4, beta):
    """Boost a 4-momentum p4 = (px, py, pz, E) by velocity vector beta = (bx,

    by, bz).

    p4 shape: (N, 4) or (N, S, 4) beta shape: (N, 3) or (N, S, 3)
    """
    beta_sq = np.sum(beta**2, axis=-1, keepdims=True)
    gamma = 1.0 / np.sqrt(np.maximum(1e-10, 1.0 - beta_sq))

    p3 = p4[..., :3]
    E = p4[..., 3:4]

    beta_dot_p = np.sum(beta * p3, axis=-1, keepdims=True)
    factor = np.where(
        beta_sq > 1e-10, (gamma - 1.0) / np.maximum(beta_sq, 1e-10), 0.0
    )
    p3_boosted = p3 + factor * beta_dot_p * beta - gamma * E * beta
    E_boosted = gamma * (E - beta_dot_p)

    return np.concatenate([p3_boosted, E_boosted], axis=-1)


def reconstruct_diffusion_polarization(
    diffusion_model,
    features_test_scaled,
    target_test,
    scaler_features,
    scaler_target,
    model_name,
    nb1_nu_eta=None,  # 1st quadratic solution
    nb2_nu_eta=None,  # 2nd quadratic solution
    num_samples=1,  # Samples per event (1 is fast; >1 captures posterior width)
    batch_size=1024,
    bins=50,
    seed=42,  # Seed for reproducible 50/50 baseline
):
    diffusion_model.load_state_dict(
        torch.load(model_name, map_location=device)
    )
    diffusion_model.eval()

    # 1. Unscale features to physical units (GeV, radians)
    features_unscaled = scaler_features.inverse_transform(features_test_scaled)

    lep_pt = features_unscaled[:, FEATURES.index("ChargedLep_pT")]
    lep_eta = features_unscaled[:, FEATURES.index("ChargedLep_Eta")]
    lep_e = features_unscaled[:, FEATURES.index("ChargedLep_E")]

    met = features_unscaled[:, FEATURES.index("MET")]
    phi_nu = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_MET")]

    true_eta = scaler_target.inverse_transform(
        target_test.reshape(-1, 1)
    ).flatten()

    n_events = len(true_eta)

    # ----------------------------------------------------
    # Helper: Vectorized Lorentz Boost to W Rest Frame
    # ----------------------------------------------------
    def get_cos_theta_star(eta_nu):
        is_2d = eta_nu.ndim == 2
        if is_2d:
            n_s = eta_nu.shape[1]
            m = met[:, None]
            p = phi_nu[:, None]
            l_pt = lep_pt[:, None]
            l_eta = lep_eta[:, None]
            l_e = lep_e[:, None]
        else:
            m = met
            p = phi_nu
            l_pt = lep_pt
            l_eta = lep_eta
            l_e = lep_e

        p3_l = np.stack(
            [l_pt, np.zeros_like(l_pt), l_pt * np.sinh(l_eta)], axis=-1
        )
        if is_2d:
            p3_l = np.repeat(p3_l, n_s, axis=1)
            l_e = np.repeat(l_e, n_s, axis=1)

        px_nu = m * np.cos(p)
        py_nu = m * np.sin(p)
        if is_2d:
            px_nu = np.repeat(px_nu, n_s, axis=1)
            py_nu = np.repeat(py_nu, n_s, axis=1)
        pz_nu = m * np.sinh(eta_nu)
        e_nu = m * np.cosh(eta_nu)

        p3_nu = np.stack([px_nu, py_nu, pz_nu], axis=-1)
        p3_w = p3_l + p3_nu
        e_w = l_e + e_nu

        beta = p3_w / np.expand_dims(e_w, axis=-1)
        beta_sq = np.sum(beta**2, axis=-1, keepdims=True)
        gamma = 1.0 / np.sqrt(np.maximum(1e-10, 1.0 - beta_sq))

        beta_dot_p = np.sum(beta * p3_l, axis=-1, keepdims=True)
        factor = np.where(
            beta_sq > 1e-10, (gamma - 1.0) / np.maximum(beta_sq, 1e-10), 0.0
        )
        p3_l_star = (
            p3_l
            + factor * beta_dot_p * beta
            - gamma * np.expand_dims(l_e, -1) * beta
        )

        dot_prod = np.sum(p3_l_star * p3_w, axis=-1)
        norm_l = np.linalg.norm(p3_l_star, axis=-1)
        norm_w = np.linalg.norm(p3_w, axis=-1)

        cos_t = dot_prod / np.maximum(1e-7, norm_l * norm_w)
        return np.clip(cos_t, -1.0, 1.0)

    # ----------------------------------------------------
    # Helper: Template Fit for Mixed Sample (f0, f1, f2)
    # ----------------------------------------------------
    def fit_polarization(cos_array):
        # Unbinned moments
        m_cos = np.mean(cos_array)
        m_cos2 = np.mean(cos_array**2)
        f0_m = 2.0 - 5.0 * m_cos2
        f1_m = 0.5 * (1.0 - f0_m) - m_cos
        f2_m = 0.5 * (1.0 - f0_m) + m_cos

        # General spin-1 template:
        # P(cos_t) = 3/4*(1-cos_t^2)*f0 + 3/8*(1-cos_t)^2*f1 + 3/8*(1+cos_t)^2*f2
        def angular_pdf(cos_t, f0, f1):
            f2 = 1.0 - f0 - f1
            return (
                0.75 * (1.0 - cos_t**2) * f0
                + 0.375 * (1.0 - cos_t) ** 2 * f1
                + 0.375 * (1.0 + cos_t) ** 2 * f2
            )

        counts, edges = np.histogram(
            cos_array, bins=bins, range=(-1, 1), density=True
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        popt, pcov = curve_fit(
            angular_pdf, centers, counts, p0=[0.80, 0.05], bounds=(0, 1)
        )
        f0_fit, f1_fit = popt
        f2_fit = 1.0 - f0_fit - f1_fit
        errors = np.sqrt(np.diag(pcov))

        sigma_f0 = errors[0]
        sigma_f1 = errors[1]
        sigma_f2 = np.sqrt(sigma_f0**2 + sigma_f1**2)

        return {
            "moments": (f0_m, f1_m, f2_m),
            "fit": (f0_fit, f1_fit, f2_fit),
            "errors": (sigma_f0, sigma_f1, sigma_f2),
        }

    # ----------------------------------------------------
    # Helper: Reduced Chi-Squared (Two-Sample Poisson Test)
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

    # ----------------------------------------------------
    # 2. Compute Cos(theta*) for MC Truth and Diffusion
    # ----------------------------------------------------
    print("Computing MC Truth Polarization...")
    cos_truth = get_cos_theta_star(true_eta)
    res_truth = fit_polarization(cos_truth)

    print(
        f"Sampling from Diffusion Model (num_samples={num_samples} per event)..."
    )
    diff_samples = []
    features_tensor = torch.tensor(features_test_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(features_tensor), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_idx, (bx,) in enumerate(loader):
            bx = bx.to(device)

            if num_samples == 1:
                # Direct sampling [batch_size, 1]
                s = diffusion_model.sample_ddpm(bx)
                diff_samples.append(s.squeeze(-1).cpu().numpy())
            else:
                # Repeat each event context num_samples times
                bx_rep = bx.repeat_interleave(num_samples, dim=0)
                s = diffusion_model.sample_ddpm(bx_rep)  # [B * S, 1]
                s = s.view(bx.shape[0], num_samples)  # [B, S]
                diff_samples.append(s.cpu().numpy())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")


    eta_samples = np.concatenate(diff_samples, axis=0)
    orig_shape = eta_samples.shape
    eta_samples = scaler_target.inverse_transform(
        eta_samples.reshape(-1, 1)
    ).reshape(orig_shape)
    cos_diff = get_cos_theta_star(eta_samples).flatten()
    res_diff = fit_polarization(cos_diff)

    # ----------------------------------------------------
    # 3. Compute Baselines (50/50 and min-|pz|)
    # ----------------------------------------------------
    cos_5050, res_5050 = None, None
    cos_minpz, res_minpz = None, None

    if nb1_nu_eta is not None and nb2_nu_eta is not None:
        # Baseline 1: Random 50/50 Choice
        rng = np.random.default_rng(seed=seed)
        coin_flip = rng.choice([True, False], size=n_events)
        eta_5050 = np.where(coin_flip, nb1_nu_eta, nb2_nu_eta)

        cos_5050_raw = get_cos_theta_star(eta_5050)
        valid_5050 = ~np.isnan(cos_5050_raw)
        cos_5050 = cos_5050_raw[valid_5050]
        res_5050 = fit_polarization(cos_5050)

        # Baseline 2: Minimum |pz| (Paper Heuristic)
        pz1 = np.abs(met * np.sinh(nb1_nu_eta))
        pz2 = np.abs(met * np.sinh(nb2_nu_eta))
        eta_minpz = np.where(pz1 <= pz2, nb1_nu_eta, nb2_nu_eta)

        cos_minpz_raw = get_cos_theta_star(eta_minpz)
        valid_minpz = ~np.isnan(cos_minpz_raw)
        cos_minpz = cos_minpz_raw[valid_minpz]
        res_minpz = fit_polarization(cos_minpz)

    # ----------------------------------------------------
    # 4. Calculate Chi-Squared vs. MC Truth
    # ----------------------------------------------------
    raw_truth, edges = np.histogram(cos_truth, bins=bins, range=(-1, 1))
    raw_diff, _ = np.histogram(cos_diff, bins=bins, range=(-1, 1))

    chi2_diff, ndf_diff, r_chi2_diff = compute_chi2_ndf(raw_diff, raw_truth)

    r_chi2_5050 = None
    r_chi2_minpz = None

    if cos_5050 is not None:
        raw_5050, _ = np.histogram(cos_5050, bins=bins, range=(-1, 1))
        _, _, r_chi2_5050 = compute_chi2_ndf(raw_5050, raw_truth)

    if cos_minpz is not None:
        raw_minpz, _ = np.histogram(cos_minpz, bins=bins, range=(-1, 1))
        _, _, r_chi2_minpz = compute_chi2_ndf(raw_minpz, raw_truth)

    # ----------------------------------------------------
    # 5. Print Summary Table
    # ----------------------------------------------------
    print("\n" + "=" * 85)
    print(
        f"{'Method':<22} | {'f0 (Longitudinal)':<18} | {'f1':<8} | {'f2':<8} | {'chi2 / ndf':<12}"
    )
    print("=" * 85)
    print(
        f"{'1. MC Truth (Ref)':<22} | {res_truth['fit'][0]:.4f} ± {res_truth['errors'][0]:.4f}   | {res_truth['fit'][1]:.4f} ± {res_truth['errors'][1]:.4f} | {res_truth['fit'][2]:.4f} ± {res_truth['errors'][2]:.4f} | {'—':<12}"
    )
    print(
        f"{'2. Diffusion Samples':<22} | {res_diff['fit'][0]:.4f} ± {res_diff['errors'][0]:.4f}   | {res_diff['fit'][1]:.4f} ± {res_diff['errors'][1]:.4f} | {res_diff['fit'][2]:.4f} ± {res_diff['errors'][2]:.4f} | {r_chi2_diff:<12.3f}"
    )
    if res_5050:
        print(
            f"{'3. Baseline (50/50)':<22} | {res_5050['fit'][0]:.4f} ± {res_5050['errors'][0]:.4f}   | {res_5050['fit'][1]:.4f} ± {res_5050['errors'][1]:.4f} | {res_5050['fit'][2]:.4f} ± {res_5050['errors'][2]:.4f} | {r_chi2_5050:<12.3f}"
        )
    if res_minpz:
        print(
            f"{'4. Baseline (min-|pz|)':<22} | {res_minpz['fit'][0]:.4f} ± {res_minpz['errors'][0]:.4f}   | {res_minpz['fit'][1]:.4f} ± {res_minpz['errors'][1]:.4f} | {res_minpz['fit'][2]:.4f} ± {res_minpz['errors'][2]:.4f} | {r_chi2_minpz:<12.3f}"
        )
    print("=" * 85)

    # ----------------------------------------------------
    # 6. Plotting: Distribution & Ratio
    # ----------------------------------------------------
    fig, (ax_main, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(9, 7.5),
        sharex=True,

    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    norm_truth, _ = np.histogram(
        cos_truth, bins=bins, range=(-1, 1), density=True
    )
    norm_diff, _ = np.histogram(
        cos_diff, bins=bins, range=(-1, 1), density=True
    )

    # Main Panel
    ax_main.hist(
        centers,
        bins=edges,
        weights=norm_truth,
        histtype="step",
        linewidth=2.2,
        color="black",
        label=rf"MC Truth ($f_0 = {res_truth['fit'][0]:.3f}$)",
    )
    ax_main.hist(
        centers,
        bins=edges,
        weights=norm_diff,
        histtype="stepfilled",
        alpha=0.45,
        color="royalblue",
        label=rf"Diffusion $p(\eta_\nu|x)$ ($f_0 = {res_diff['fit'][0]:.3f}$, $\chi^2/\mathrm{{ndf}} = {r_chi2_diff:.2f}$)",
    )

    if cos_5050 is not None:
        norm_5050, _ = np.histogram(
            cos_5050, bins=bins, range=(-1, 1), density=True
        )
        ax_main.hist(
            centers,
            bins=edges,
            weights=norm_5050,
            histtype="step",
            linestyle="--",
            linewidth=1.8,
            color="crimson",
            label=rf"Baseline (50/50) ($f_0 = {res_5050['fit'][0]:.3f}$, $\chi^2/\mathrm{{ndf}} = {r_chi2_5050:.2f}$)",
        )

    if cos_minpz is not None:
        norm_minpz, _ = np.histogram(
            cos_minpz, bins=bins, range=(-1, 1), density=True
        )
        ax_main.hist(
            centers,
            bins=edges,
            weights=norm_minpz,
            histtype="step",
            linestyle=":",
            linewidth=2.0,
            color="darkorange",
            label=rf"Baseline min-$|p_z|$ ($f_0 = {res_minpz['fit'][0]:.3f}$, $\chi^2/\mathrm{{ndf}} = {r_chi2_minpz:.2f}$)",
        )

    ax_main.set_ylabel(
        r"$\frac{1}{\sigma}\frac{d\sigma}{d\cos\theta^*}$", fontsize=12
    )
    ax_main.set_title(
        r"$W$ Boson Polarization Reconstruction ($\cos\theta^*$) [Diffusion Model]",
        fontsize=13,
    )
    ax_main.legend(fontsize=9.0, loc="upper right")

    centers = 0.5 * (edges[:-1] + edges[1:])

    # 2. Compute Ratio
    ratio_diff = norm_diff / np.maximum(norm_truth, 1e-6)

    # 3. Propagate Poisson Statistical Uncertainties
    # Avoid division by zero in empty bins
    valid_bins = (raw_diff > 0) & (raw_truth > 0)
    ratio_err_diff = np.zeros_like(ratio_diff)
    ratio_err_diff[valid_bins] = ratio_diff[valid_bins] * np.sqrt(
        1.0 / raw_diff[valid_bins] + 1.0 / raw_truth[valid_bins]
    )


    # Ratio Panel (Method / Truth)
    ax_ratio.errorbar(
        centers[valid_bins],
        ratio_diff[valid_bins],
        yerr=ratio_err_diff[valid_bins],
        fmt="o-",
        color="royalblue",
        ms=2,
        elinewidth=1.0,
        capsize=0,
        capthick=1.0,
        label="Model / Truth",
    )

    if cos_5050 is not None:
        ratio_5050 = norm_5050 / np.maximum(norm_truth, 1e-6)
        ax_ratio.plot(
            centers,
            ratio_5050,
            "s--",
            color="crimson",
            ms=2.0,
            label="50/50 / Truth",
        )

    if cos_minpz is not None:
        ratio_minpz = norm_minpz / np.maximum(norm_truth, 1e-6)
        ax_ratio.plot(
            centers,
            ratio_minpz,
            "^:",
            color="darkorange",
            ms=2.0,
            label="min-$|p_z|$ / Truth",
        )

    ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.8)
    ax_ratio.set_ylim(0.7, 1.3)
    ax_ratio.set_xlabel(r"$\cos\theta^*$ (Helicity frame)", fontsize=11)
    ax_ratio.set_ylabel("Ratio to Truth", fontsize=10)

    ax_ratio.legend(fontsize=7.5, loc="lower right", ncol=3)

    plt.tight_layout()
    plt.show()

    return {
        "res_truth": res_truth,
        "res_diff": res_diff,
        "res_5050": res_5050,
        "res_minpz": res_minpz,
        "r_chi2_diff": r_chi2_diff,
        "r_chi2_5050": r_chi2_5050,
        "r_chi2_minpz": r_chi2_minpz,
    }


if __name__ == "__main__":
    time = "20261008-222146"
    data_filename = (
        "data/new_Input_WTH_NeutrinoEtaFindingEvaluation_4th_October2026.csv"
    )

    hyperparameter_file = f"models/hyperparameters_diff_{time}.json"
    hyperparameters = load_hyperparameters(hyperparameter_file)

    model_name = f"models/model_diff_{time}.pt"
    n_features = len(FEATURES)
    n_target = len(TARGET)

    features_test, target_test = read_file(data_filename, FEATURES, TARGET)

    # Load Scalers
    scaler_features = joblib.load(f"scalers/scaler_features_diff_{time}.joblib")
    scaler_target = joblib.load(f"scalers/scaler_target_diff_{time}.joblib")
    features_test = scaler_features.transform(features_test)
    target_test = scaler_target.transform(target_test)

    # Instantiate Diffusion Model
    diffusion_model = ConditionalDDPM(
        target_dim=n_target,
        context_dim=n_features,
        hyperparameters=hyperparameters,
    ).to(device)

    # Load analytical solutions from DataFrame if available
    df = pd.read_csv(data_filename, encoding_errors="ignore")
    df_test_raw = df.iloc[-len(target_test) :]
    nb1_nu_eta_test = df_test_raw["NB1_nu_eta"].values
    nb2_nu_eta_test = df_test_raw["NB2_nu_eta"].values

    # Run polarization evaluation
    # Note: num_samples=1 draws 1 sample per event, which is fast and gives the full population histogram.
    res = reconstruct_diffusion_polarization(
        diffusion_model=diffusion_model,
        features_test_scaled=features_test,
        target_test=target_test,
        scaler_features=scaler_features,
        scaler_target=scaler_target,
        model_name=model_name,
        nb1_nu_eta=nb1_nu_eta_test,
        nb2_nu_eta=nb2_nu_eta_test,
        num_samples=1,
        batch_size=1024,
        bins=50,
    )