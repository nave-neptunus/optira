import numpy as np
import pytest

FPS = 30


def dip_signal(center=2.0, width=0.6, depth=6.0, baseline=30.0, duration=6.0, fps=FPS):
    """Noiseless synthetic dip with its minimum at `center` seconds."""
    t = np.arange(int(duration * fps)) / fps
    return t, baseline - depth * np.exp(-(((t - center) / width) ** 2))


@pytest.fixture
def dip():
    return dip_signal()
