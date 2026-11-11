"""PLACEHOLDER, no data behind thresholds.

The rule and the probability below were typed in by hand. There is no dataset,
no tuning and no source for 0.4 s or 15 px/s, and 15 px/s cannot transfer
between cameras because it is in pixels. The probability is a constant.
Do not read the output as a screening result.
"""

from __future__ import annotations


def placeholder_rule(features: dict) -> tuple[int, float]:
    """Return (prediction, probability). Abnormal if latency > 0.4 or velocity < 15."""
    is_abnormal = (
        features["plr_latency"] > 0.4 or features["constriction_velocity"] < 15
    )
    prediction = 1 if is_abnormal else 0
    probability = 0.7 if is_abnormal else 0.3
    return prediction, probability
