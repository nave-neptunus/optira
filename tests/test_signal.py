from types import SimpleNamespace

import pytest

from optira.signal import effective_fps, iris_distance_px


def _landmarks(p468, p470, n=478):
    lm = [SimpleNamespace(x=0.0, y=0.0) for _ in range(n)]
    lm[468] = SimpleNamespace(x=p468[0], y=p468[1])
    lm[470] = SimpleNamespace(x=p470[0], y=p470[1])
    return lm


def test_iris_distance_is_468_to_470():
    lm = _landmarks((0.5, 0.5), (0.5, 0.6))
    # 100 x 100 frame: (50, 50) to (50, 60) is 10 px.
    assert iris_distance_px(lm, 100, 100) == pytest.approx(10.0)


def test_other_landmarks_do_not_matter():
    lm = _landmarks((0.5, 0.5), (0.5, 0.6))
    lm[469] = SimpleNamespace(x=0.9, y=0.9)
    lm[471] = SimpleNamespace(x=0.1, y=0.1)
    assert iris_distance_px(lm, 100, 100) == pytest.approx(10.0)


def test_importing_package_does_not_need_cv2_or_mediapipe():
    import sys

    import optira.signal  # noqa: F401

    assert "cv2" not in sys.modules
    assert "mediapipe" not in sys.modules


def test_effective_fps_fallbacks():
    assert effective_fps(0) == 30
    assert effective_fps(30.0) == 30.0
    assert effective_fps(60.0) == 60.0
