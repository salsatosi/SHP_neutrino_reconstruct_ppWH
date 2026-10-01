from scipy.optimize import curve_fit
from train import (
    device, FEATURES, TARGET, read_file, normalizing_flow, load_hyperparameters)
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd


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


def reconstruct_flow_polarization(
    flow,
    features_test_scaled,
    target_test,
    scaler_features,
    model_name,
    nb1_nu_eta=None,  # 1st quadratic solution
    nb2_nu_eta=None,  # 2nd quadratic solution
    num_samples=50,
    batch_size=512,
    is_w_plus=True,
    bins=50,
):
    flow.load_state_dict(torch.load(model_name, map_location=device))
    flow.eval()

    # 1. Unscale features to physical units (GeV, radians)
    features_unscaled = scaler_features.inverse_transform(features_test_scaled)

    lep_pt = features_unscaled[:, FEATURES.index("ChargedLep_pT")]
    lep_eta = features_unscaled[:, FEATURES.index("ChargedLep_Eta")]
    lep_e = features_unscaled[:, FEATURES.index("ChargedLep_E")]

    met = features_unscaled[:, FEATURES.index("MET")]
    phi_nu = features_unscaled[:, FEATURES.index("Relative_Phi_Lep_MET")]

    true_eta = target_test.flatten()

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
            p3_l + factor * beta_dot_p * beta - gamma * np.expand_dims(l_e, -1) * beta
        )

        dot_prod = np.sum(p3_l_star * p3_w, axis=-1)
        norm_l = np.linalg.norm(p3_l_star, axis=-1)
        norm_w = np.linalg.norm(p3_w, axis=-1)

        cos_t = dot_prod / np.maximum(1e-7, norm_l * norm_w)
        return np.clip(cos_t, -1.0, 1.0)

    # ----------------------------------------------------
    # Helper: Template Fit for Polarization Fractions
    # ----------------------------------------------------
    sign = -1.0 if is_w_plus else 1.0

    def fit_polarization(cos_array):
        m_cos = np.mean(cos_array)
        m_cos2 = np.mean(cos_array**2)
        f0_m = 2.0 - 5.0 * m_cos2
        fL_m = 0.5 * (1.0 - f0_m) + sign * m_cos
        fR_m = 0.5 * (1.0 - f0_m) - sign * m_cos

        def angular_pdf(cos_t, f0, fL):
            fR = 1.0 - f0 - fL
            return (
                0.75 * (1.0 - cos_t**2) * f0
                + 0.375 * (1.0 + sign * cos_t) ** 2 * fL
                + 0.375 * (1.0 - sign * cos_t) ** 2 * fR
            )

        counts, edges = np.histogram(
            cos_array, bins=bins, range=(-1, 1), density=True
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        popt, pcov = curve_fit(
            angular_pdf, centers, counts, p0=[0.68, 0.10], bounds=(0, 1)
        )
        f0_fit, fL_fit = popt
        fR_fit = 1.0 - f0_fit - fL_fit
        errors = np.sqrt(np.diag(pcov))

        return {
            "moments": (f0_m, fL_m, fR_m),
            "fit": (f0_fit, fL_fit, fR_fit),
            "errors": errors,
        }

    # ----------------------------------------------------
    # Helper: Reduced Chi-Squared (Two-Sample Poisson Test)
    # ----------------------------------------------------
    def compute_chi2_ndf(counts_test, counts_ref):
        """
        Computes chi2 / ndf comparing test distribution to reference (truth).
        Uses Poisson variance: Var(diff) = Var(ref) + Var(scaled_test).
        """
        n_ref = np.sum(counts_ref)
        n_test = np.sum(counts_test)
        scale = n_ref / n_test

        scaled_test = counts_test * scale

        # Combined variance in each bin
        variance = counts_ref + (scale**2) * counts_test

        # Consider non-empty bins (counts_ref > 5 for valid asymptotic normality)
        valid = (counts_ref > 5) & (variance > 0)

        chi2 = np.sum((counts_ref[valid] - scaled_test[valid]) ** 2 / variance[valid])
        ndf = np.sum(valid) - 1  # -1 for total normalization constraint
        chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
        return chi2, ndf, chi2_ndf

    # ----------------------------------------------------
    # 2. Compute Cos(theta*) Arrays
    # ----------------------------------------------------
    # MC Truth
    cos_truth = get_cos_theta_star(true_eta)
    res_truth = fit_polarization(cos_truth)

    # Flow Samples
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

    eta_samples = np.concatenate(flow_samples, axis=0)
    cos_flow = get_cos_theta_star(eta_samples).flatten()
    res_flow = fit_polarization(cos_flow)

    # Baseline (min-|pz|)
    cos_heuristic = None
    res_heuristic = None
    if nb1_nu_eta is not None and nb2_nu_eta is not None:
        pz1 = np.abs(met * np.sinh(nb1_nu_eta))
        pz2 = np.abs(met * np.sinh(nb2_nu_eta))
        heuristic_eta = np.where(pz1 <= pz2, nb1_nu_eta, nb2_nu_eta)
        valid = ~np.isnan(heuristic_eta)
        cos_heuristic = get_cos_theta_star(heuristic_eta[valid])
        res_heuristic = fit_polarization(cos_heuristic)

    # ----------------------------------------------------
    # 3. Calculate Chi-Squared vs. MC Truth
    # ----------------------------------------------------
    raw_truth, edges = np.histogram(cos_truth, bins=bins, range=(-1, 1))
    raw_flow, _ = np.histogram(cos_flow, bins=bins, range=(-1, 1))

    chi2_flow, ndf_flow, r_chi2_flow = compute_chi2_ndf(raw_flow, raw_truth)

    r_chi2_heur = None
    if cos_heuristic is not None:
        raw_heur, _ = np.histogram(cos_heuristic, bins=bins, range=(-1, 1))
        chi2_heur, ndf_heur, r_chi2_heur = compute_chi2_ndf(raw_heur, raw_truth)

    # ----------------------------------------------------
    # 4. Print Summary Table
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Method':<22} | {'f0':<16} | {'fL':<8} | {'fR':<8} | {'chi2 / ndf':<12}")
    print("=" * 80)
    print(
        f"{'1. MC Truth (Ref)':<22} | {res_truth['fit'][0]:.4f} ± {res_truth['errors'][0]:.4f}   | {res_truth['fit'][1]:.4f} | {res_truth['fit'][2]:.4f} | {'—':<12}"
    )
    print(
        f"{'2. Flow Samples':<22} | {res_flow['fit'][0]:.4f} ± {res_flow['errors'][0]:.4f}   | {res_flow['fit'][1]:.4f} | {res_flow['fit'][2]:.4f} | {r_chi2_flow:<12.3f}"
    )
    if res_heuristic:
        print(
            f"{'3. Baseline (min|pz|)':<22} | {res_heuristic['fit'][0]:.4f} ± {res_heuristic['errors'][0]:.4f}   | {res_heuristic['fit'][1]:.4f} | {res_heuristic['fit'][2]:.4f} | {r_chi2_heur:<12.3f}"
        )
    print("=" * 80)
    print(f"Flow Compatibility: chi2 = {chi2_flow:.2f}, ndf = {ndf_flow}, reduced chi2 = {r_chi2_flow:.3f}")

    # ----------------------------------------------------
    # 5. Plotting: Distribution & Ratio
    # ----------------------------------------------------
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3.5, 1.2]}
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    norm_truth, _ = np.histogram(cos_truth, bins=bins, range=(-1, 1), density=True)
    norm_flow, _ = np.histogram(cos_flow, bins=bins, range=(-1, 1), density=True)

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
        weights=norm_flow,
        histtype="stepfilled",
        alpha=0.45,
        color="royalblue",
        label=rf"Flow $p(\eta_\nu|x)$ ($f_0 = {res_flow['fit'][0]:.3f}$, $\chi^2/\mathrm{{ndf}} = {r_chi2_flow:.2f}$)",
    )

    if cos_heuristic is not None:
        norm_heur, _ = np.histogram(cos_heuristic, bins=bins, range=(-1, 1), density=True)
        ax_main.hist(
            centers,
            bins=edges,
            weights=norm_heur,
            histtype="step",
            linestyle="--",
            linewidth=1.8,
            color="crimson",
            label=rf"Baseline min-$|p_z|$ ($f_0 = {res_heuristic['fit'][0]:.3f}$, $\chi^2/\mathrm{{ndf}} = {r_chi2_heur:.1f}$)",
        )

    ax_main.set_ylabel(r"$\frac{1}{\sigma}\frac{d\sigma}{d\cos\theta^*}$", fontsize=12)
    ax_main.set_title(r"$W$ Boson Polarization Reconstruction ($\cos\theta^*$)", fontsize=13)
    ax_main.legend(fontsize=9.5, loc="upper right")
    ax_main.grid(True, alpha=0.3)

    # Ratio Panel (Method / Truth)
    ratio_flow = norm_flow / np.maximum(norm_truth, 1e-6)
    ax_ratio.plot(centers, ratio_flow, "o-", color="royalblue", ms=3.5, label="Flow / Truth")
    if cos_heuristic is not None:
        ratio_heur = norm_heur / np.maximum(norm_truth, 1e-6)
        ax_ratio.plot(
            centers, ratio_heur, "s--", color="crimson", ms=3.0, label="Baseline / Truth"
        )

    ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.8)
    ax_ratio.set_ylim(0.7, 1.3)
    ax_ratio.set_xlabel(r"$\cos\theta^*$ (Helicity frame)", fontsize=11)
    ax_ratio.set_ylabel("Ratio to Truth", fontsize=10)
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.show()

    return {
        "res_truth": res_truth,
        "res_flow": res_flow,
        "chi2_flow": chi2_flow,
        "ndf_flow": ndf_flow,
        "r_chi2_flow": r_chi2_flow,
        "r_chi2_heur": r_chi2_heur,
    }



if __name__ == "__main__":
    time = '20261001-003138'
    hyperparameter_file = f"models/hyperparameters_{time}.json"
    hyperparameters = load_hyperparameters(hyperparameter_file)
    # torch.manual_seed(hyperparameters["random_seed"])

    model_name = f"models/model_{time}.pt"
    n_features = len(FEATURES)
    n_target = len(TARGET)

    df = pd.read_csv('data.csv', encoding_errors='ignore')
    features_data = df[FEATURES].values
    target_data = df[TARGET].values
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

    flow = normalizing_flow(
        features=n_target,
        context_features=n_features,
        hyperparameters=hyperparameters)

    # ----------------------------
    # Polarization Reconstruction Test

    # Extract analytical solutions from DataFrame if available

    df_test_raw = df.iloc[-len(target_test):]  # or matching test slice
    nb1_nu_eta_test = df_test_raw['NB1_nu_eta'].values
    nb2_nu_eta_test = df_test_raw['NB2_nu_eta'].values

    res = reconstruct_flow_polarization(
        flow=flow,
        features_test_scaled=features_test,
        target_test=target_test,
        scaler_features=scaler_features,
        model_name=model_name,
        nb1_nu_eta=nb1_nu_eta_test,
        nb2_nu_eta=nb2_nu_eta_test,
        num_samples=50,
        batch_size=512,
        is_w_plus=True,
        bins=50
    )