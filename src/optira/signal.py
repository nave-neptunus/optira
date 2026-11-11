"""Landmark distance and video reading.

iris_distance_px is pure and testable. extract_series needs OpenCV and MediaPipe,
which are optional: they are imported inside the function so the rest of the
package (and the tests) work without them.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_detector = None  # one shared detector, no lock (as in the original)


def iris_distance_px(landmarks, w: int, h: int) -> float:
    """Distance in pixels between face-mesh landmarks 468 and 470.

    Coordinates are normalized x/y times the frame size, truncated with int().
    This is an iris landmark distance, not a pupil diameter, and it is quantized
    to whole pixels.
    """
    pts = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in (468, 470)]
    )
    return float(np.linalg.norm(pts[0] - pts[1]))


def effective_fps(fps: float) -> float:
    """Frame rate used for the time axis: 30 if the file reports 0 or more than 120."""
    if fps == 0 or fps > 120:
        return 30
    return fps


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector
    import mediapipe as mp  # noqa: F401  (lazy, optional)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = os.environ.get("OPTIRA_MODEL_PATH") or "face_landmarker.task"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_faces=1,
    )
    _detector = vision.FaceLandmarker.create_from_options(options)
    return _detector


def extract_series(video_path: str):
    """Read a video and return (times, sizes, fps).

    Each frame is detected independently. Frames with no face are skipped, so
    the returned samples are not evenly spaced. Requires opencv and mediapipe
    (see requirements-video.txt). Not covered by the tests in this repo.
    """
    import cv2  # lazy, optional
    import mediapipe as mp  # lazy, optional

    detector = _get_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = effective_fps(cap.get(cv2.CAP_PROP_FPS))

    times, sizes = [], []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.face_landmarks:
            h, w, _ = frame.shape
            sizes.append(iris_distance_px(result.face_landmarks[0], w, h))
            times.append(frame_idx / fps)
        frame_idx += 1

    cap.release()
    return np.array(times), np.array(sizes), fps
