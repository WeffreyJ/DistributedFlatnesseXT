"""Structured Tier-2 residual interaction scaffold."""

from __future__ import annotations

import numpy as np

from src.model.coupling_tier1 import active_edges_tier1
from src.model.wake_geometry import pairwise_geometry, transverse_metric_squared


def _tier2_cfg(params):
    tier2 = getattr(params, "tier2", None)
    if tier2 is None:
        raise ValueError("system.tier2 configuration is required for plant_family='tier2'")
    return tier2


def residual_enabled(params) -> bool:
    cfg = _tier2_cfg(params)
    residual = getattr(cfg, "residual", None)
    if residual is None:
        return False
    return bool(getattr(residual, "enabled", False)) and float(getattr(residual, "amplitude", 0.0)) != 0.0


def _sigmoid(x: float) -> float:
    x_clip = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x_clip)))


def support_transition_pair_terms_tier2(
    x: np.ndarray,
    leader: int,
    follower: int,
    params,
    *,
    s: np.ndarray | None = None,
) -> dict[str, float]:
    """Return pairwise support-transition diagnostics for one ordered pair."""
    cfg = _tier2_cfg(params)
    residual = getattr(cfg, "residual", None)
    if residual is None:
        raise ValueError("system.tier2.residual configuration is required for support_transition_bias")

    gamma_edge = float(getattr(params, "gamma_edge", getattr(params, "gamma", 0.0)))
    wake_rx = float(getattr(params, "wake_Rx", np.inf))
    rho_max = float(getattr(params.tier1.edge, "transverse_radius", np.inf))

    sigma_ell = max(float(getattr(residual, "support_transition_sigma_ell", 0.10)), 1.0e-9)
    sigma_rho = max(float(getattr(residual, "support_transition_sigma_rho", 0.20)), 1.0e-9)
    m_ell = float(getattr(residual, "support_transition_m_ell", 0.03))
    m_rho = float(getattr(residual, "support_transition_m_rho", 0.15))
    alpha_ell = float(getattr(residual, "support_transition_alpha_ell", 1.0))
    alpha_rho = float(getattr(residual, "support_transition_alpha_rho", 0.35))
    k_v = float(getattr(residual, "support_transition_k_v", 18.0))
    k_shell = float(getattr(residual, "support_transition_k_shell", 20.0))
    rho_core = max(float(getattr(residual, "support_transition_rho_core", getattr(params.tier1.kernel, "r_core", 0.5))), 0.0)
    rho_floor = max(float(getattr(residual, "support_transition_rho_floor", 0.05)), 1.0e-9)
    gain = float(getattr(residual, "support_transition_gain", 0.30))

    geom = pairwise_geometry(x, leader, follower, params, s=s)
    ds = float(geom["ds"])
    dl = float(geom["dl"])
    dh = float(geom["dh"])
    rho_plain = float(geom["transverse_radius"])
    rho_reg = float(np.sqrt(rho_plain * rho_plain + rho_core * rho_core))

    d_ell_left = ds - gamma_edge
    d_ell_right = wake_rx - ds
    d_ell = float(min(d_ell_left, d_ell_right))
    d_rho = float(rho_max - rho_plain)

    n = int(params.N)
    vel = np.asarray(x[n : 2 * n], dtype=float)
    ell_dot = float(vel[int(leader)] - vel[int(follower)])
    if d_ell_left <= d_ell_right:
        d_ell_dot = ell_dot
        nearest_longitudinal_boundary = "ell_min"
    else:
        d_ell_dot = -ell_dot
        nearest_longitudinal_boundary = "ell_max"

    # The current scaffold uses static lateral/vertical offsets, so radial
    # crossing rate is zero in the first implementation. Keep the exact proxy
    # explicit rather than inventing controller-aware motion.
    y_dot = 0.0
    z_dot = 0.0
    rho_dot = float((dl * y_dot + dh * z_dot) / max(rho_reg, rho_floor))
    d_rho_dot = -rho_dot

    g_ell = _sigmoid(k_shell * (d_ell + m_ell))
    g_rho = _sigmoid(k_shell * (d_rho + m_rho))
    w_ell = float(np.exp(-0.5 * (d_ell / sigma_ell) ** 2))
    w_rho = float(np.exp(-0.5 * (d_rho / sigma_rho) ** 2))
    phi_ell = float(np.tanh(k_v * d_ell_dot))
    phi_rho = float(np.tanh(k_v * d_rho_dot))

    long_decay = max(float(getattr(params.tier1.kernel, "longitudinal_decay", 2.0)), 1.0e-9)
    power_p = float(getattr(params.tier1.kernel, "power_p", 1.0))
    r_core = float(getattr(params.tier1.kernel, "r_core", 0.5))
    transverse_sq = transverse_metric_squared(dl, dh, params)
    regularized = max(r_core * r_core + transverse_sq, 1.0e-9)
    activity_scale = float(np.exp(-max(d_ell_left, 0.0) / long_decay) / (regularized ** (0.5 * power_p)))

    term_ell = alpha_ell * g_rho * w_ell * phi_ell
    term_rho = alpha_rho * g_ell * w_rho * phi_rho
    pair_residual = float(gain * activity_scale * (term_ell + term_rho))
    shell_strength = float(max(g_rho * w_ell, g_ell * w_rho))

    return {
        "ds": ds,
        "dl": dl,
        "dh": dh,
        "rho_plain": rho_plain,
        "rho_reg": rho_reg,
        "d_ell": d_ell,
        "d_rho": d_rho,
        "d_ell_dot": float(d_ell_dot),
        "d_rho_dot": float(d_rho_dot),
        "nearest_longitudinal_boundary": nearest_longitudinal_boundary,
        "g_ell": float(g_ell),
        "g_rho": float(g_rho),
        "w_ell": float(w_ell),
        "w_rho": float(w_rho),
        "phi_ell": float(phi_ell),
        "phi_rho": float(phi_rho),
        "term_ell": float(term_ell),
        "term_rho": float(term_rho),
        "activity_scale": float(activity_scale),
        "shell_strength": float(shell_strength),
        "pair_residual": pair_residual,
    }


def residual_delta_accel_tier2(
    x: np.ndarray,
    params,
    *,
    s: np.ndarray | None = None,
) -> np.ndarray:
    """Return the Tier-2 plant-side residual interaction.

    Residual modes share the Tier-1 active-edge support and stay plant-side only.
    This keeps Tier-2 interpretable, bounded, and easy to disable.
    """
    n = int(params.N)
    out = np.zeros(n, dtype=float)
    if not residual_enabled(params):
        return out

    cfg = _tier2_cfg(params)
    residual = getattr(cfg, "residual", None)
    if residual is None:
        return out

    mode = str(getattr(residual, "mode", "zero")).strip().lower()
    amplitude = float(getattr(residual, "amplitude", 0.0))
    if amplitude == 0.0 or mode == "zero":
        return out
    gamma_edge = float(getattr(params, "gamma_edge", getattr(params, "gamma", 0.0)))
    wake_Rx = float(getattr(params, "wake_Rx", np.inf))
    tier1 = getattr(params, "tier1", None)
    if tier1 is None:
        raise ValueError("system.tier1 configuration is required for Tier-2 residual evaluation")
    longitudinal_decay = max(float(getattr(tier1.kernel, "longitudinal_decay", 2.0)), 1.0e-9)

    for leader, follower in active_edges_tier1(x, params, s=s):
        geom = pairwise_geometry(x, leader, follower, params, s=s)
        ds_eff = max(float(geom["ds"]) - gamma_edge, 0.0)
        if ds_eff >= wake_Rx:
            continue
        longitudinal = float(np.exp(-ds_eff / longitudinal_decay))
        if mode == "transverse_skew":
            transverse_scale = max(float(getattr(residual, "transverse_scale", 1.0)), 1.0e-9)
            bias_gain = float(getattr(residual, "bias_gain", 0.25))
            dl = float(geom["dl"])
            skew = float(np.tanh(dl / transverse_scale))
            out[int(follower)] += -amplitude * bias_gain * longitudinal * skew
        elif mode == "longitudinal_bias":
            longitudinal_scale = max(float(getattr(residual, "longitudinal_scale", 0.35)), 1.0e-9)
            bias_center = max(float(getattr(residual, "longitudinal_bias_center", 0.35)), 0.0)
            longitudinal_bias_gain = float(getattr(residual, "longitudinal_bias_gain", 0.35))
            # Concentrate the residual inside a longitudinal band where marginal
            # switch timing is most likely to be sensitive.
            window_bias = float(np.exp(-0.5 * ((ds_eff - bias_center) / longitudinal_scale) ** 2))
            out[int(follower)] += -amplitude * longitudinal_bias_gain * longitudinal * window_bias
        elif mode == "edge_band_bias":
            longitudinal_center = max(float(getattr(residual, "edge_band_longitudinal_center", 0.08)), 0.0)
            longitudinal_scale = max(float(getattr(residual, "edge_band_longitudinal_scale", 0.10)), 1.0e-9)
            transverse_center = max(float(getattr(residual, "edge_band_transverse_center", 2.30)), 0.0)
            transverse_scale = max(float(getattr(residual, "edge_band_transverse_scale", 0.25)), 1.0e-9)
            lateral_scale = max(float(getattr(residual, "edge_band_lateral_scale", 1.0)), 1.0e-9)
            edge_band_gain = float(getattr(residual, "edge_band_gain", 0.40))
            rho = float(geom["transverse_radius"])
            dl = float(geom["dl"])
            longitudinal_window = float(np.exp(-0.5 * ((ds_eff - longitudinal_center) / longitudinal_scale) ** 2))
            transverse_window = float(np.exp(-0.5 * ((rho - transverse_center) / transverse_scale) ** 2))
            lateral_factor = float(np.tanh(dl / lateral_scale))
            out[int(follower)] += -amplitude * edge_band_gain * longitudinal_window * transverse_window * lateral_factor
        elif mode == "support_transition_bias":
            terms = support_transition_pair_terms_tier2(x, leader, follower, params, s=s)
            out[int(follower)] += amplitude * float(terms["pair_residual"])
        else:
            raise ValueError(f"Unsupported Tier-2 residual mode: {mode!r}")

    if mode == "support_transition_bias":
        shell_margin_ell = float(getattr(residual, "support_transition_m_ell", 0.03))
        shell_margin_rho = float(getattr(residual, "support_transition_m_rho", 0.15))
        for leader in range(n):
            for follower in range(n):
                if leader == follower:
                    continue
                terms = support_transition_pair_terms_tier2(x, leader, follower, params, s=s)
                if terms["d_ell"] < -shell_margin_ell or terms["d_rho"] < -shell_margin_rho:
                    continue
                if (leader, follower) in active_edges_tier1(x, params, s=s):
                    continue
                out[int(follower)] += amplitude * float(terms["pair_residual"])
    return out
