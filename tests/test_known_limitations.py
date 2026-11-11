"""Tests that pin known flaws of the ported code.

These pass today because the flaws are present. If one of them starts failing,
someone fixed the behavior: update the test and the README at the same time.
"""

import base64
from types import SimpleNamespace

import numpy as np

from optira.features import compute_features
from optira.screening import placeholder_rule
from optira.server import create_app
from optira.signal import iris_distance_px
from conftest import FPS, dip_signal


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def test_flat_noise_is_flagged_abnormal():
    rng = np.random.default_rng(0)
    t = np.arange(6 * FPS) / FPS
    y = 30 + rng.normal(0, 0.3, t.size)  # no reflex at all
    f = compute_features(t, y, FPS)
    prediction, probability = placeholder_rule(f)
    assert prediction == 1
    assert probability == 0.7


def test_reflex_flagged_abnormal_since_latency_counts_from_video_start():
    # A reflex whose minimum falls at 2.0 s: the latency counts from video start, so it exceeds 0.4 s.
    t, y = dip_signal(center=2.0)
    f = compute_features(t, y, FPS)
    assert f["plr_latency"] > 0.4
    assert placeholder_rule(f)[0] == 1


def test_probability_is_a_constant_not_a_probability():
    assert placeholder_rule({"plr_latency": 0.1, "constriction_velocity": 99.0}) == (0, 0.3)
    assert placeholder_rule({"plr_latency": 9.0, "constriction_velocity": 99.0}) == (1, 0.7)


def test_argmin_catches_a_late_blink():
    t, y = dip_signal(center=2.0, depth=4.0)
    y = y.copy()
    blink = int(5.0 * FPS)
    y[blink - 2 : blink + 3] -= 8.0  # a late, deeper eyelid-closure dip
    f = compute_features(t, y, FPS)
    assert f["plr_latency"] > 4.5  # the blink, not the reflex, sets the minimum


def test_baseline_video_is_ignored():
    t, y = dip_signal()
    other_t, other_y = dip_signal(center=4.0, depth=2.0)

    def run(baseline_series):
        def extractor(path):
            with open(path, "rb") as f:
                kind = f.read()
            return (t, y, FPS) if kind == b"light" else (*baseline_series, FPS)

        app = create_app(extractor=extractor)
        return app.test_client().post(
            "/analyze",
            json={"baseline_video": _b64(b"base"), "light_test_video": _b64(b"light")},
        ).get_json()

    assert run((t, y)) == run((other_t, other_y))


def test_landmark_coordinates_are_truncated_to_whole_pixels():
    def lm(x470, y470):
        pts = [SimpleNamespace(x=0.0, y=0.0) for _ in range(478)]
        pts[468] = SimpleNamespace(x=0.5, y=0.5)
        pts[470] = SimpleNamespace(x=x470, y=y470)
        return pts

    # 0.5599 and 0.5501 differ by about 1 px in a 100 px frame; int() collapses both to 55.
    assert iris_distance_px(lm(0.5, 0.5599), 100, 100) == iris_distance_px(
        lm(0.5, 0.5501), 100, 100
    )


def test_dropped_frames_are_filtered_as_if_evenly_spaced():
    # Same values, one series with a 1.5 s gap in the time axis: the filter cannot tell.
    t, y = dip_signal()
    gap_t = t.copy()
    gap_t[60:] += 1.5
    a = compute_features(t, y, FPS)
    b = compute_features(gap_t, y, FPS)
    assert a["rmssd"] == b["rmssd"]
    assert a["hippus_irregularity"] == b["hippus_irregularity"]
