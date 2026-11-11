"""Flask API: POST /analyze and GET /health.

Ported from the notebook backend. There is no authentication. Do not expose it
publicly with real face video.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile

from flask import Flask, jsonify, request

from .features import compute_features
from .screening import placeholder_rule
from .signal import extract_series

log = logging.getLogger("optira")


def create_app(extractor=extract_series) -> Flask:
    """Build the app. `extractor(path) -> (times, sizes, fps)` can be replaced in tests."""
    app = Flask(__name__)

    @app.route("/analyze", methods=["POST"])
    def analyze():
        baseline_path = None
        light_test_path = None
        try:
            data = request.json
            if not data or "baseline_video" not in data or "light_test_video" not in data:
                return jsonify({"error": "Missing video data"}), 400

            baseline_bytes = base64.b64decode(data["baseline_video"])
            light_test_bytes = base64.b64decode(data["light_test_video"])

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(baseline_bytes)
                baseline_path = f.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                f.write(light_test_bytes)
                light_test_path = f.name

            # The baseline video is processed and then not used (known limitation).
            extractor(baseline_path)
            light_times, light_sizes, light_fps = extractor(light_test_path)

            features = compute_features(light_times, light_sizes, light_fps)
            prediction, probability = placeholder_rule(features)

            return jsonify(
                {
                    "prediction": prediction,
                    "probability": probability,
                    "features": features,
                    # Placeholder strings: no graph image is generated.
                    "graphs": {
                        "pupil_size_graph": "base64_graph_data",
                        "plr_graph": "base64_graph_data",
                    },
                }
            )
        except Exception as e:  # same behavior as the original: 500 with the message
            log.exception("analysis failed")
            return jsonify({"error": str(e)}), 500
        finally:
            for p in (baseline_path, light_test_path):
                if p and os.path.exists(p):
                    os.unlink(p)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "message": "Backend is running"})

    return app


def main() -> None:
    """Run the Flask development server on 127.0.0.1:5000. Pass --tunnel to open the optional tunnel (see tunnel.py)."""
    import sys

    logging.basicConfig(level=logging.INFO)
    if "--tunnel" in sys.argv:
        from .tunnel import start_tunnel

        start_tunnel(5000)
    create_app().run(port=5000)


if __name__ == "__main__":
    main()
