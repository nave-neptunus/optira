import base64
import os

import pytest

from optira.server import create_app
from conftest import FPS, dip_signal


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


@pytest.fixture
def seen_paths():
    return []


@pytest.fixture
def client(seen_paths):
    t, y = dip_signal()

    def fake_extractor(path):
        seen_paths.append(path)
        assert os.path.exists(path)
        return t, y, FPS

    app = create_app(extractor=fake_extractor)
    app.testing = True
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


@pytest.mark.parametrize("missing", ["baseline_video", "light_test_video"])
def test_analyze_missing_key_is_400(client, missing):
    body = {"baseline_video": _b64(b"a"), "light_test_video": _b64(b"b")}
    del body[missing]
    r = client.post("/analyze", json=body)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_analyze_returns_expected_keys(client):
    r = client.post(
        "/analyze",
        json={"baseline_video": _b64(b"base"), "light_test_video": _b64(b"light")},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) == {"prediction", "probability", "features", "graphs"}
    assert set(body["features"]) == {
        "plr_latency",
        "constriction_velocity",
        "hippus_irregularity",
        "rmssd",
    }


def test_temp_files_are_deleted(client, seen_paths):
    client.post(
        "/analyze",
        json={"baseline_video": _b64(b"base"), "light_test_video": _b64(b"light")},
    )
    assert len(seen_paths) == 2
    assert not any(os.path.exists(p) for p in seen_paths)


def test_temp_files_are_deleted_when_extractor_fails():
    seen = []

    def boom(path):
        seen.append(path)
        raise ValueError("Cannot open video")

    app = create_app(extractor=boom)
    r = app.test_client().post(
        "/analyze",
        json={"baseline_video": _b64(b"x"), "light_test_video": _b64(b"y")},
    )
    assert r.status_code == 500
    assert r.get_json() == {"error": "Cannot open video"}
    assert seen and not any(os.path.exists(p) for p in seen)
