"""Realtime trap intelligence helpers for MarketTrap dashboard."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_latest(series: pd.Series, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    return float(series.iloc[-1])


def _compute_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_component_scores(df_1m: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return normalized component scores and diagnostics for explainability."""
    if df_1m is None or len(df_1m) < 25:
        zero = {
            "structure_failure": 0.0,
            "volume_behavior": 0.0,
            "momentum_exhaustion": 0.0,
        }
        return zero, {}

    frame = df_1m.copy()
    frame["returns"] = frame["price"].pct_change().fillna(0.0)
    frame["volume_change"] = frame["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    rolling_high = frame["price"].rolling(20, min_periods=5).max().shift(1)
    breakout = frame["price"] > rolling_high
    breakout_failure = breakout.shift(1).fillna(False) & (frame["price"] < rolling_high)

    near_high = frame["price"] >= frame["price"].rolling(20, min_periods=5).max() * 0.995
    low_rel_volume = frame["volume"] < frame["volume"].rolling(20, min_periods=5).mean() * 0.9

    rsi = _compute_rsi(frame["price"], period=14)
    rsi_falling_from_high = (rsi.shift(1) > 68) & (rsi < rsi.shift(1) - 2)

    momentum_fade = (
        frame["returns"].rolling(3, min_periods=3).mean()
        < frame["returns"].rolling(8, min_periods=5).mean()
    )

    price_up_volume_down = (frame["returns"] > 0) & (frame["volume_change"] < 0)

    structure_failure = _clip01(
        0.7 * float(breakout_failure.tail(5).mean())
        + 0.3 * float(((frame["returns"] < -0.002) & near_high).tail(5).mean())
    )
    volume_behavior = _clip01(
        0.65 * float(price_up_volume_down.tail(8).mean())
        + 0.35 * float((near_high & low_rel_volume).tail(8).mean())
    )
    momentum_exhaustion = _clip01(
        0.6 * float(rsi_falling_from_high.tail(8).mean())
        + 0.4 * float(momentum_fade.tail(8).mean())
    )

    diagnostics = {
        "breakout_failure_strength": float(breakout_failure.tail(8).mean()),
        "price_up_volume_down_strength": float(price_up_volume_down.tail(8).mean()),
        "rsi_fall_strength": float(rsi_falling_from_high.tail(8).mean()),
        "near_high_low_volume_strength": float((near_high & low_rel_volume).tail(8).mean()),
        "momentum_fade_strength": float(momentum_fade.tail(8).mean()),
        "latest_rsi": _safe_latest(rsi, 50.0),
        "latest_return": _safe_latest(frame["returns"], 0.0),
        "latest_volume_change": _safe_latest(frame["volume_change"], 0.0),
    }

    components = {
        "structure_failure": structure_failure,
        "volume_behavior": volume_behavior,
        "momentum_exhaustion": momentum_exhaustion,
    }
    return components, diagnostics


def classify_trap_type(components: Dict[str, float], anomaly_component: float) -> str:
    dominant = max(
        {
            "structure_failure": components.get("structure_failure", 0.0),
            "volume_behavior": components.get("volume_behavior", 0.0),
            "momentum_exhaustion": components.get("momentum_exhaustion", 0.0),
            "anomaly": anomaly_component,
        }.items(),
        key=lambda item: item[1],
    )[0]

    if dominant == "structure_failure":
        return "Breakout Failure Trap"
    if dominant == "volume_behavior":
        return "Distribution Trap"
    if dominant == "anomaly":
        return "Liquidity Sweep Trap"
    return "Fake Momentum Trap"


def extract_trap_reasons(
    components: Dict[str, float],
    diagnostics: Dict[str, float],
    anomaly_component: float,
    max_reasons: int = 3,
) -> List[Dict[str, float]]:
    candidates: List[Tuple[str, float]] = []

    breakout_conf = max(
        components.get("structure_failure", 0.0),
        diagnostics.get("breakout_failure_strength", 0.0),
    )
    candidates.append(("Price broke resistance but failed to hold", breakout_conf))

    volume_conf = max(
        components.get("volume_behavior", 0.0),
        diagnostics.get("price_up_volume_down_strength", 0.0),
    )
    candidates.append(("Price rising without volume support", volume_conf))

    momentum_conf = max(
        components.get("momentum_exhaustion", 0.0),
        diagnostics.get("rsi_fall_strength", 0.0),
    )
    candidates.append(("Momentum exhausted after sharp push", momentum_conf))

    candidates.append(("Anomalous behavior detected vs recent history", anomaly_component))

    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    filtered = [item for item in ranked if item[1] >= 0.20]
    if len(filtered) < 2:
        filtered = ranked[:2]
    candidates = filtered[:max_reasons]
    return [
        {"reason": reason, "confidence": round(_clip01(conf) * 100, 1)}
        for reason, conf in candidates
    ]


def buyer_seller_control(df_1m: pd.DataFrame) -> str:
    if df_1m is None or len(df_1m) < 10:
        return "Neutral"

    frame = df_1m.copy()
    frame["returns"] = frame["price"].pct_change().fillna(0.0)
    frame["volume_change"] = frame["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    price_trend = float(frame["returns"].tail(5).mean())
    vol_trend = float(frame["volume_change"].tail(5).mean())

    if price_trend > 0.0008 and vol_trend >= -0.01:
        return "Buyers in Control"
    if price_trend < -0.0008 and vol_trend >= -0.01:
        return "Sellers in Control"
    return "Neutral"
