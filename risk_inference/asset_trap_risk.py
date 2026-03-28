"""
Asset-specific trap risk inference.
Applies per-crypto weights, spiky amplification, and plain-English reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from asset_registry import get_asset_params


@dataclass
class TrapRiskResult:
    trap_risk_score: float
    risk_level: str
    top_3_reasons: List[str]
    invalidated_by: List[str]
    components: Dict[str, float]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _latest(series: pd.Series, default: float = 0.0) -> float:
    if len(series) == 0:
        return default
    return float(series.iloc[-1])


def _spiky_amplification(components: Dict[str, float]) -> float:
    """Apply non-linear amplification when 2+ components spike together."""
    spike_count = sum(1 for v in components.values() if v > 0.6)
    if spike_count >= 2:
        base = sum(components.values())
        return min(1.0, base * 1.3)  # 30% boost when multiple spikes align
    return sum(components.values())


def compute_asset_trap_components(df: pd.DataFrame, symbol: str, anomaly_score: float = 0.0) -> Dict[str, float]:
    """Compute normalized component scores using asset-specific features."""
    params = get_asset_params(symbol)

    structure_failure = _clip01(
        0.5 * _latest(df.get("fake_breakout_prob", pd.Series(dtype=float)))
        + 0.3 * _latest(df.get("sharp_reversal", pd.Series(dtype=float)))
        + 0.2 * _latest(df.get("compression_to_expansion", pd.Series(dtype=float)))
    )

    volume_behavior = _clip01(
        0.6 * _latest(df.get("volume_absorption_prob", pd.Series(dtype=float)))
        + 0.4 * _latest(df.get("compression_ratio", pd.Series(dtype=float)))
    )

    momentum_exhaustion = _clip01(
        0.5 * _latest(df.get("momentum_exhaustion_prob", pd.Series(dtype=float)))
        + 0.3 * _latest(df.get("reversal_velocity", pd.Series(dtype=float)) * -1)
        + 0.2 * _latest(df.get("compression_to_expansion", pd.Series(dtype=float)))
    )

    anomaly_component = _clip01(float(anomaly_score))

    return {
        "structure_failure": structure_failure,
        "volume_behavior": volume_behavior,
        "momentum_exhaustion": momentum_exhaustion,
        "anomaly_score": anomaly_component,
    }


def aggregate_asset_trap_risk(components: Dict[str, float], symbol: str) -> float:
    """Apply asset-specific weights and spiky amplification."""
    params = get_asset_params(symbol)
    base_score = (
        params.w_structure * components.get("structure_failure", 0.0)
        + params.w_volume * components.get("volume_behavior", 0.0)
        + params.w_momentum * components.get("momentum_exhaustion", 0.0)
        + params.w_anomaly * components.get("anomaly_score", 0.0)
    )
    amplified = _spiky_amplification(components)
    final_score = max(base_score, amplified)
    return float(round(final_score * 100, 1))


def risk_level(score: float) -> str:
    if score >= 80:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def generate_asset_reasons(df: pd.DataFrame, components: Dict[str, float], symbol: str) -> List[str]:
    """Generate plain-English reasons tailored to the asset."""
    params = get_asset_params(symbol)
    reasons: List[str] = []

    if components.get("structure_failure", 0) > 0.5:
        reasons.append(
            params.breakout_reason.format(
                lookback=params.breakout_lookback,
                threshold=params.breakout_threshold,
                reversal=params.reversal_window,
                window=params.absorption_window,
            )
        )
    if components.get("volume_behavior", 0) > 0.5:
        reasons.append(
            params.volume_reason.format(
                window=params.absorption_window,
                lookback=params.breakout_lookback,
                threshold=params.breakout_threshold,
                reversal=params.reversal_window,
            )
        )
    if components.get("momentum_exhaustion", 0) > 0.5:
        reasons.append(
            params.momentum_reason.format(
                threshold=params.momentum_threshold,
                window=params.momentum_window,
                lookback=params.breakout_lookback,
                reversal=params.reversal_window,
            )
        )
    if components.get("anomaly_score", 0) > 0.6:
        reasons.append("Anomaly model flagged unusual market behavior.")
    return reasons[:3]


def invalidation_conditions(df: pd.DataFrame) -> List[str]:
    """What would invalidate the current trap assessment."""
    conditions: List[str] = []
    if _latest(df.get("volume_change", pd.Series(dtype=float))) > 0.2:
        conditions.append("Strong volume surge confirms the move.")
    if _latest(df.get("price_return", pd.Series(dtype=float))) > 0.02:
        conditions.append("Price continues rising with strength.")
    return conditions


def compute_asset_trap_risk(
    df: pd.DataFrame,
    symbol: str,
    anomaly_score: float = 0.0,
) -> TrapRiskResult:
    """Main entry point: compute asset-specific trap risk with explanations."""
    components = compute_asset_trap_components(df, symbol, anomaly_score)
    score = aggregate_asset_trap_risk(components, symbol)
    return TrapRiskResult(
        trap_risk_score=score,
        risk_level=risk_level(score),
        top_3_reasons=generate_asset_reasons(df, components, symbol),
        invalidated_by=invalidation_conditions(df),
        components=components,
    )
