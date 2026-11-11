"""Feature extraction ported from the original notebook's backend.

Names in the returned dict are kept from the original (plr_latency and so on).
What they measure, in plain terms:

  plr_latency            time to minimum, from the start of the video
  constriction_velocity  mean rate of decline, (baseline - minimum) / time to minimum
  hippus_irregularity    post-minimum variability
  rmssd                  RMS of frame-to-frame differences over the whole signal

None of these is validated against real pupil measurements.
"""

from __future__ import annotations

import numpy as np

from .filtering import safe_lowpass


def compute_features(times, sizes, fps: float) -> dict:
    """Compute the four feature values from a size-over-time series."""
    if len(sizes) < 10:
        raise ValueError("Not enough pupil data detected")

    filt = safe_lowpass(sizes, fps)

    # Baseline: mean of the first int(0.5 * fps) samples of the same video.
    baseline = np.mean(filt[: min(int(0.5 * fps), len(filt))])
    # Minimum over the whole video, so a blink can become the minimum.
    min_idx = np.argmin(filt)
    # Time from video start, not from stimulus onset.
    latency = times[min_idx] if min_idx < len(times) else 0.3
    velocity = (baseline - filt[min_idx]) / max(latency, 0.01)

    # Post-minimum variability: std of first differences, 0.5 s to 2.5 s after the minimum.
    start_idx = min(min_idx + int(0.5 * fps), len(filt) - 1)
    end_idx = min(min_idx + int(2.5 * fps), len(filt))
    segment = filt[start_idx:end_idx]
    hippus_std = np.std(np.diff(segment)) if len(segment) > 5 else 0.5

    diffs = np.diff(filt)
    rmssd = np.sqrt(np.mean(diffs**2)) if len(diffs) > 0 else 0.03

    return {
        "plr_latency": float(latency),
        "constriction_velocity": float(velocity),
        "hippus_irregularity": float(hippus_std),
        "rmssd": float(rmssd),
    }
