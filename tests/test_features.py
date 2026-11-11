import numpy as np
import pytest

from optira.features import compute_features
from optira.filtering import safe_lowpass
from conftest import FPS, dip_signal


def test_minimum_at_two_seconds_within_one_sample(dip):
    t, y = dip
    f = compute_features(t, y, FPS)
    assert abs(f["plr_latency"] - 2.0) <= 1 / FPS


def test_velocity_is_baseline_minus_min_over_latency(dip):
    t, y = dip
    f = compute_features(t, y, FPS)
    filt = safe_lowpass(y, FPS)
    baseline = filt[: int(0.5 * FPS)].mean()
    expected = (baseline - filt.min()) / f["plr_latency"]
    assert f["constriction_velocity"] == pytest.approx(expected)


def test_short_post_minimum_slice_gives_0_5_fallback():
    # Monotonic decline: the minimum is the last sample, so the slice has 1 sample.
    y = np.linspace(30, 20, 60)
    t = np.arange(60) / FPS
    f = compute_features(t, y, FPS)
    assert f["hippus_irregularity"] == 0.5


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        compute_features(np.arange(9) / FPS, np.ones(9), FPS)


def test_returns_original_json_keys(dip):
    t, y = dip
    assert set(compute_features(t, y, FPS)) == {
        "plr_latency",
        "constriction_velocity",
        "hippus_irregularity",
        "rmssd",
    }


def test_rmssd_is_rms_of_filtered_first_differences(dip):
    t, y = dip
    f = compute_features(t, y, FPS)
    d = np.diff(safe_lowpass(y, FPS))
    assert f["rmssd"] == pytest.approx(float(np.sqrt(np.mean(d**2))))
