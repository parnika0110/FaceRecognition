"""Face detection with YuNet (cv2.FaceDetectorYN)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Face:
    """One detected face and its landmarks."""

    box: tuple[int, int, int, int]  # x, y, w, h
    score: float
    landmarks: np.ndarray  # (5, 2) right eye, left eye, nose, right mouth, left mouth
    raw: np.ndarray  # full YuNet row (1, 15); FaceRecognizerSF.alignCrop consumes this

    @property
    def width(self) -> int:
        return self.box[2]


class FaceDetector:
    """Thin wrapper around cv2.FaceDetectorYN with per-image input resizing."""

    def __init__(self, model_path=None, score_threshold=None):
        self.model_path = str(model_path or config.DETECTOR_MODEL_PATH)
        self.score_threshold = (
            config.DETECTION_SCORE_THRESHOLD
            if score_threshold is None
            else score_threshold
        )
        self._nets: dict[tuple[int, int], cv2.FaceDetectorYN] = {}

    def _net_for_size(self, width: int, height: int) -> cv2.FaceDetectorYN:
        key = (width, height)
        net = self._nets.get(key)
        if net is None:
            net = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (width, height),
                self.score_threshold,
                config.DETECTION_NMS_THRESHOLD,
            )
            self._nets[key] = net
        else:
            net.setInputSize((width, height))
        return net

    def detect(self, bgr_image: np.ndarray) -> list[Face]:
        """Detect faces in a BGR image, strongest score first."""
        if bgr_image is None or bgr_image.size == 0:
            return []
        h, w = bgr_image.shape[:2]
        net = self._net_for_size(w, h)
        ok, faces = net.detect(bgr_image)
        if not ok or faces is None:
            return []

        out: list[Face] = []
        for row in faces:
            x, y, bw, bh = (int(v) for v in row[:4])
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            score = float(row[14])
            out.append(Face((x, y, bw, bh), score, landmarks,
                            row.reshape(1, -1).astype(np.float32).copy()))
        out.sort(key=lambda f: f.score, reverse=True)
        return out

    def detect_largest(self, bgr_image: np.ndarray) -> Face | None:
        faces = self.detect(bgr_image)
        return faces[0] if faces else None
