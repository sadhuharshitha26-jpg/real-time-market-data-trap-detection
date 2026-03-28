"""
Explainable trap risk inference.
Combines structure, volume, anomaly, and momentum signals into a 0-100 score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrapRiskWeights:
    structure_failure: float = 0.35
    volume_behavior: float = 0.25
    anomaly_score: float = 0.15
    momentum_exhaustion: float = 0.25


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_from_flags(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.tail(3).mean())


def _latest_value(series: pd.Series, default: float = 0.0) -> float:
    if len(series) == 0:
        return default
    return float(series.iloc[-1])


def compute_trap_components(
    df: pd.DataFrame,
    anomaly_score: float = 0.0,
) -> Dict[str, float]:
    """Compute normalized component scores from trap features."""
    structure_failure = _clip01(
        0.5 * _score_from_flags(df.get("breakout_failure", pd.Series(dtype=float)))
        + 0.3 * _score_from_flags(df.get("failed_retest", pd.Series(dtype=float)))
        + 0.2 * _latest_value(df.get("upper_wick_ratio", pd.Series(dtype=float)))
    )

    volume_behavior = _clip01(
        0.5 * _score_from_flags(df.get("volume_divergence", pd.Series(dtype=float)))
        + 0.2 * _score_from_flags(df.get("volume_spike_on_reversal", pd.Series(dtype=float)))
        + 0.3 * _latest_value(df.get("pv_correlation", pd.Series(dtype=float)) * -1)
    )

    momentum_exhaustion = _clip01(
        0.4 * _score_from_flags(df.get("fast_reversal", pd.Series(dtype=float)))
        + 0.3 * _score_from_flags(df.get("rsi_overbought_fall", pd.Series(dtype=float)))
        + 0.3 * _score_from_flags(df.get("momentum_fade", pd.Series(dtype=float)))
    )

    anomaly_component = _clip01(float(anomaly_score))

    return {
        "structure_failure": structure_failure,
        "volume_behavior": volume_behavior,
        "momentum_exhaustion": momentum_exhaustion,
        "anomaly_score": anomaly_component,
    }


def aggregate_trap_risk(
    components: Dict[str, float],
    weights: TrapRiskWeights = TrapRiskWeights(),
) -> float:
    score = (
        weights.structure_failure * components.get("structure_failure", 0.0)
        + weights.volume_behavior * components.get("volume_behavior", 0.0)
        + weights.anomaly_score * components.get("anomaly_score", 0.0)
        + weights.momentum_exhaustion * components.get("momentum_exhaustion", 0.0)
    )
    return float(round(score * 100, 1))


def risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def top_3_reasons(df: pd.DataFrame, components: Dict[str, float]) -> List[str]:
    reasons: List[Tuple[str, float]] = []

    if components.get("structure_failure", 0) > 0.4:
        reasons.append(("Breakout failed or rejected quickly", components["structure_failure"]))
    if components.get("volume_behavior", 0) > 0.4:
        reasons.append(("Price rose without volume support (divergence)", components["volume_behavior"]))
    if components.get("momentum_exhaustion", 0) > 0.4:
        reasons.append(("Momentum faded after a sharp move", components["momentum_exhaustion"]))
    if components.get("anomaly_score", 0) > 0.6:
        reasons.append(("Anomaly model flagged unusual behavior", components["anomaly_score"]))

    reasons = sorted(reasons, key=lambda item: item[1], reverse=True)
    return [reason for reason, _ in reasons[:3]]


def invalidation_conditions(df: pd.DataFrame) -> List[str]:
    conditions: List[str] = []
    if _latest_value(df.get("volume_change", pd.Series(dtype=float))) > 0:
        conditions.append("Rising volume confirms the move")
    if _latest_value(df.get("breakout_strength", pd.Series(dtype=float))) > 0:
        conditions.append("Price holds above the breakout zone")
    return conditions


def compute_trap_risk(
    df: pd.DataFrame,
    anomaly_score: float = 0.0,
    weights: TrapRiskWeights = TrapRiskWeights(),
) -> Dict[str, object]:
    components = compute_trap_components(df, anomaly_score=anomaly_score)
    score = aggregate_trap_risk(components, weights=weights)
    return {
        "trap_risk_score": score,
        "risk_level": risk_level(score),
        "top_3_reasons": top_3_reasons(df, components),
        "invalidated_by": invalidation_conditions(df),
        "components": components,
    }
