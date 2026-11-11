"""Low-pass filter ported from the original notebook."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def safe_lowpass(data, fs: float, cutoff: float = 4.0):
    """2nd-order Butterworth low-pass applied with filtfilt (zero phase).

    Returns the input unchanged if fs <= 0 or there are fewer than 10 samples.
    Known limitation: samples are treated as evenly spaced, so dropped frames
    (no face found) are not accounted for.
    """
    if fs <= 0 or len(data) < 10:
        return data
    b, a = butter(2, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, np.asarray(data, dtype=float))
