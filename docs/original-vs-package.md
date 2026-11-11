# Original notebook vs this package

The project started as a single Colab notebook export (`diabetes.py`, 484 lines, not included in this repo). This table maps each part of it to a module here and says what behavior was kept. Line numbers refer to that notebook.

Rule used for the port: keep the logic as it was, including its flaws. Nothing was silently fixed. The flaws are listed in the README and pinned by `tests/test_known_limitations.py`.

## Ported

| Notebook part | Lines | Module | Behavior kept |
|---|---|---|---|
| Cell 1: read video, detect face landmarks, size per frame | 43-77 | `src/optira/signal.py` `extract_series` | Frame by frame, no tracking. `num_faces=1`. Frames with no face are skipped. Time is `frame_idx / fps`. FPS fallback to 30 if the file reports 0. |
| Backend `extract_pupil_data` | 320-356 | `src/optira/signal.py` | Same as Cell 1, plus the fallback to 30 when the file reports more than 120 fps (`effective_fps`). |
| Landmark distance | 68-71, 346-349 | `src/optira/signal.py` `iris_distance_px` | Landmarks 468 and 470, normalized coordinates times frame size, `int()` truncation, Euclidean distance in px. Landmarks 469 and 471 are read in the notebook but do not affect the result. |
| Low-pass filter | 104-110, 358-363 | `src/optira/filtering.py` `safe_lowpass` | Order-2 Butterworth, 4 Hz cutoff, `filtfilt`. Input returned unchanged if fewer than 10 samples. |
| Backend `compute_features` | 365-394 | `src/optira/features.py` | Baseline from the first `int(0.5 * fps)` samples. `argmin` over the whole video. Time to minimum from video start. Mean rate of decline is `(baseline - min) / max(latency, 0.01)`. Post-minimum variability over 0.5 to 2.5 s after the minimum, 0.5 fallback for 5 or fewer samples. `rmssd` over the whole filtered signal. Dead-code fallbacks (0.3 and 0.03) kept. JSON keys unchanged. |
| Decision rule | 436-440 | `src/optira/screening.py` `placeholder_rule` | `latency > 0.4 or velocity < 15`. Probability is the constant 0.7 or 0.3. |
| Flask app: `/analyze`, `/health` | 396-471 | `src/optira/server.py` `create_app` | Two base64 videos in JSON. 400 on a missing key. 500 with the message on any error. Temp files deleted in `finally`. Both videos are extracted, only the light test video is used. `graphs` holds the literal string `base64_graph_data`. |
| Tunnel | 473-482 | `src/optira/tunnel.py` | Optional. Reads `NGROK_AUTHTOKEN` from the environment and exits with a message if it is unset. The notebook set the token in code. |

## Changed on purpose (structure only)

| Change | Why |
|---|---|
| Split into modules under `src/optira/` | So the pure parts can be tested without a video or a GPU. |
| `create_app(extractor=...)` | Tests pass a fake extractor, so no video file or MediaPipe is needed. |
| `cv2` and `mediapipe` are imported inside functions | They are optional installs. |
| Model path from `OPTIRA_MODEL_PATH`, default `face_landmarker.task` in the working directory | The notebook hard-coded the path. The model is still downloaded from the same public URL if the file is missing. |
| `print` calls with emoji replaced by `logging` | Output only. |
| Cell 2 plots replaced by `scripts/demo_synthetic.py` | The demo runs on a synthetic signal. |

## Not ported (described only)

| Notebook part | Lines | What it was |
|---|---|---|
| Cell 2 feature code | 112-138 | An earlier version of the feature code with small differences: time-based baseline (`t <= 0.5`), no `rmssd`, returns 0 for a short post-minimum slice. The backend version was ported instead. |
| Cell 3: logistic regression | 140-184 | Trained on 4 hand-typed rows with hand-written labels. Saved a `.pkl` that the backend never loads. |
| Cell 4: inference | 186-219 | Scores 4 hard-coded numbers, not a video. |
| Cell 5: ROC curve | 221-248 | Uses one test sample, so the ROC is undefined. |
| Cell 6: random forest | 250-282 | Trained on 2 hand-typed rows labeled healthy and diabetic. The input columns are typed numbers that nothing in the code computes. |

The scikit-learn cells have no data behind them and are not part of this package.
