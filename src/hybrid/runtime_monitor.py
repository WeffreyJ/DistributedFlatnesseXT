"""Runtime admissibility monitor and fallback scaffolding."""

from __future__ import annotations

from typing import Any


def compute_switch_rate_recent(history: list[float], now_t: float, window_sec: float) -> float:
    """Compute recent switch-event rate over a sliding time window."""
    if window_sec <= 0.0:
        return float(len(history))
    recent = [float(t) for t in history if now_t - float(t) <= window_sec]
    return float(len(recent) / window_sec)


def compute_edge_churn_recent(history: list[dict[str, Any]], now_t: float, window_sec: float) -> float:
    """Compute recent edge-churn rate over a sliding time window."""
    if window_sec <= 0.0:
        return float(sum(float(item.get("edge_delta", 0.0)) for item in history))
    recent = [item for item in history if now_t - float(item.get("t", 0.0)) <= window_sec]
    total = sum(float(item.get("edge_delta", 0.0)) for item in recent)
    return float(total / window_sec)


def evaluate_candidate_risk(
    x,
    current_pi: list[int],
    candidate_pi: list[int],
    params,
    selector_meta: dict[str, Any],
    history: dict[str, Any],
    now_t: float,
) -> dict[str, Any]:
    """Evaluate online admissibility and churn-related risk for one proposed candidate."""
    del x
    monitor_cfg = getattr(params, "monitor", None)
    window_switch = float(getattr(monitor_cfg, "switch_rate_window_sec", 1.0)) if monitor_cfg is not None else 1.0
    window_churn = float(getattr(monitor_cfg, "edge_churn_window_sec", 1.0)) if monitor_cfg is not None else 1.0
    tie_warn = float(getattr(monitor_cfg, "tie_margin_warn", 0.02)) if monitor_cfg is not None else 0.02
    gap_warn = float(getattr(monitor_cfg, "predicted_gap_warn", 0.25)) if monitor_cfg is not None else 0.25
    cond_warn = float(getattr(monitor_cfg, "conditioning_warn_threshold", 10.0)) if monitor_cfg is not None else 10.0
    max_switch_rate = float(getattr(monitor_cfg, "max_switches_per_sec", 5.0)) if monitor_cfg is not None else 5.0
    max_edge_churn = float(getattr(monitor_cfg, "max_edge_churn_per_sec", 8.0)) if monitor_cfg is not None else 8.0

    tie_margin = float(selector_meta.get("tie_margin", float("inf")))
    predicted_gap = float(selector_meta.get("predicted_gap", 0.0))
    conditioning = float(selector_meta.get("conditioning_proxy", 0.0))
    dag_ok = bool(selector_meta.get("dag_ok", selector_meta.get("admissible", True)))
    topo_ok = bool(selector_meta.get("topo_ok", selector_meta.get("admissible", True)))
    admissible = bool(selector_meta.get("admissible", dag_ok and topo_ok))
    proposed_switch = bool(candidate_pi != current_pi)
    switch_rate = compute_switch_rate_recent(list(history.get("switch_times", [])), now_t, window_switch)
    edge_churn = compute_edge_churn_recent(list(history.get("edge_churn", [])), now_t, window_churn)

    risk_reasons: list[str] = []
    risk_level = "low"
    if not admissible or not dag_ok or not topo_ok:
        risk_reasons.append("inadmissible_candidate")
        risk_level = "high"
    if proposed_switch and tie_margin < tie_warn:
        risk_reasons.append("low_tie_margin")
        if risk_level != "high":
            risk_level = "moderate"
    if proposed_switch and predicted_gap > gap_warn:
        risk_reasons.append("large_predicted_gap")
        if risk_level != "high":
            risk_level = "moderate"
    if conditioning > cond_warn:
        risk_reasons.append("conditioning_warning")
        if risk_level != "high":
            risk_level = "moderate"
    if proposed_switch and switch_rate > max_switch_rate:
        risk_reasons.append("high_switch_rate_recent")
        risk_level = "high" if switch_rate > 1.5 * max_switch_rate else max(risk_level, "moderate", key=_risk_rank)
    if proposed_switch and edge_churn > max_edge_churn:
        risk_reasons.append("high_edge_churn_recent")
        risk_level = "high" if edge_churn > 1.5 * max_edge_churn else max(risk_level, "moderate", key=_risk_rank)
    if not risk_reasons:
        risk_level = "low"

    return {
        "dag_ok": dag_ok,
        "topo_ok": topo_ok,
        "admissible": admissible,
        "tie_margin": tie_margin,
        "predicted_gap": predicted_gap,
        "conditioning_proxy": conditioning,
        "switch_rate_recent": switch_rate,
        "edge_churn_recent": edge_churn,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "proposed_switch": proposed_switch,
    }


def _risk_rank(level: str) -> int:
    return {"low": 0, "moderate": 1, "high": 2}.get(str(level), 0)


def decide_monitor_action(risk_info: dict[str, Any], cfg, high_risk_streak: int = 0) -> str:
    """Map risk information to an explicit monitor action."""
    monitor_cfg = getattr(cfg, "monitor", None)
    fallback_cfg = getattr(cfg, "fallback", None)
    moderate_action = str(getattr(monitor_cfg, "moderate_risk_action", "hold_current")) if monitor_cfg is not None else "hold_current"
    high_action = str(getattr(monitor_cfg, "high_risk_action", "fallback_fixed_order")) if monitor_cfg is not None else "fallback_fixed_order"
    fallback_enabled = bool(getattr(fallback_cfg, "enabled", False)) if fallback_cfg is not None else False
    fallback_mode = str(getattr(fallback_cfg, "mode", "hold_then_fixed")) if fallback_cfg is not None else "hold_then_fixed"
    consecutive_high = int(getattr(fallback_cfg, "consecutive_high_risk_steps", 10)) if fallback_cfg is not None else 10

    if not bool(risk_info.get("proposed_switch", False)):
        return "allow_switch"
    level = str(risk_info.get("risk_level", "low"))
    if level == "high":
        if high_action == "fallback_fixed_order" and fallback_enabled and fallback_mode == "hold_then_fixed":
            return "fallback_fixed_order" if high_risk_streak >= consecutive_high else "hold_current"
        return high_action
    if level == "moderate":
        return moderate_action
    return "allow_switch"


def runtime_monitor_step(
    *,
    x,
    current_pi: list[int],
    candidate_pi: list[int],
    params,
    selector_meta: dict[str, Any],
    history: dict[str, Any],
    now_t: float,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate runtime risk, update streak state, and choose a monitor action."""
    state = dict(state or {})
    risk_info = evaluate_candidate_risk(
        x=x,
        current_pi=current_pi,
        candidate_pi=candidate_pi,
        params=params,
        selector_meta=selector_meta,
        history=history,
        now_t=now_t,
    )
    high_risk_streak = int(state.get("high_risk_streak", 0))
    if str(risk_info["risk_level"]) == "high" and bool(risk_info["proposed_switch"]):
        high_risk_streak += 1
    else:
        high_risk_streak = 0

    action = decide_monitor_action(risk_info, params, high_risk_streak=high_risk_streak)
    fallback_cfg = getattr(params, "fallback", None)
    fallback_pi = (
        [int(v) for v in getattr(fallback_cfg, "fixed_order", [])]
        if fallback_cfg is not None and hasattr(fallback_cfg, "fixed_order")
        else []
    )

    state["high_risk_streak"] = high_risk_streak
    out = dict(risk_info)
    out["monitor_action"] = str(action)
    out["fallback_pi"] = fallback_pi
    out["high_risk_streak"] = high_risk_streak
    out["fallback_active"] = bool(action == "fallback_fixed_order")
    return out
