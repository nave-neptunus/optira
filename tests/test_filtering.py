import numpy as np

from optira.filtering import safe_lowpass

FS = 30


def _sine(freq, seconds=10):
    t = np.arange(seconds * FS) / FS
    return np.sin(2 * np.pi * freq * t)


def _rms_middle(x):
    mid = x[len(x) // 4 : -len(x) // 4]
    return float(np.sqrt(np.mean(mid**2)))


def test_short_input_returned_unchanged():
    data = np.arange(9, dtype=float)
    out = safe_lowpass(data, FS)
    assert out is data


def test_nonpositive_fs_returned_unchanged():
    data = np.arange(50, dtype=float)
    assert safe_lowpass(data, 0) is data


def test_one_hz_sine_passes():
    x = _sine(1.0)
    ratio = _rms_middle(safe_lowpass(x, FS)) / _rms_middle(x)
    assert ratio > 0.95


def test_twelve_hz_sine_attenuated_over_90_percent():
    x = _sine(12.0)
    ratio = _rms_middle(safe_lowpass(x, FS)) / _rms_middle(x)
    assert ratio < 0.10
