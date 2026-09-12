"""Central configuration for the face recognition system."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

DB_PATH = PROJECT_ROOT / "data" / "gallery.db"

# ---------------------------------------------------------------------------
# Matching policy — calibrated on LFW (see reports/metrics.json)
#
# Two regimes, one knob:
#   * 1:1 / small gallery (a handful of enrolled people):
#       DEFAULT_MATCH_THRESHOLD = 0.36 — the accuracy-maximizing point of the
#       verification sweep (FAR 0.10%, FRR 3.8% on LFW). Welcoming default.
#   * 1:N with a large gallery: impostor max-similarity compounds, so raise
#       the threshold. Measured open-set FPIR on a ~1.4k-embedding gallery:
#       0.36 -> 69%, 0.40 -> 27%, 0.45 -> 3.4%, 0.50 -> 0.25%.
#       OPEN_SET_MATCH_THRESHOLD = 0.45 is the security-oriented default.
# ---------------------------------------------------------------------------
DEFAULT_MATCH_THRESHOLD = 0.36
OPEN_SET_MATCH_THRESHOLD = 0.45

# Detection settings
DETECTION_SCORE_THRESHOLD = 0.6   # YuNet confidence cut-off
DETECTION_NMS_THRESHOLD = 0.3
# A probe face narrower than this (pixels) is flagged as low quality.
MIN_FACE_WIDTH_PX = 60

# Enrollment settings
MAX_GALLERY_EMBEDDINGS_PER_PERSON = 8
DUPLICATE_SIMILARITY_WARN = 0.45  # warn when a new face nearly matches an existing person
