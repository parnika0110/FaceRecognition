"""SFace embedding extraction with 5-point alignment."""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .detector import Face, FaceDetector


class FaceEmbedder:
    """Wraps cv2.FaceRecognizerSF: align → 112x112 crop → 128-D embedding.

    Embeddings are L2-normalized so cosine similarity is a plain dot product.
    """

    EMBEDDING_DIM = 128

    def __init__(self, model_path=None):
        self.model_path = str(model_path or config.RECOGNIZER_MODEL_PATH)
        self._recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")

    def aligned(self, image: np.ndarray, face: Face) -> np.ndarray:
        """Return the aligned 112x112 BGR crop for a detected face."""
        return self._recognizer.alignCrop(image, face.raw.astype(np.float32))

    def embed_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        vec = self._recognizer.feature(aligned_bgr).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            return vec
        return vec / norm

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        return self.embed_aligned(self.aligned(image, face))

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def quality_flags(face: Face) -> dict[str, bool]:
        """Cheap heuristics that predict embedding failure."""
        x, y, w, h = face.box
        return {
            "too_small": w < config.MIN_FACE_WIDTH_PX,
            "low_confidence": face.score < config.DETECTION_SCORE_THRESHOLD,
        }


class FacePipeline:
    """Detector + embedder convenience facade."""

    def __init__(self, detector: FaceDetector | None = None, embedder: FaceEmbedder | None = None):
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()

    def detect_faces(self, image: np.ndarray) -> list[Face]:
        return self.detector.detect(image)

    def embed_face(self, image: np.ndarray, face: Face) -> np.ndarray:
        return self.embedder.embed(image, face)

    def best_face(self, image: np.ndarray) -> Face | None:
        return self.detector.detect_largest(image)

    def match(self, probe: np.ndarray, gallery: list[tuple[str, np.ndarray]], threshold: float):
        """Return (best_person, best_score, per_person_scores).

        gallery entries are (person_name, embedding). Matching is max-similarity
        over all of a person's gallery embeddings.
        """
        probe_face = self.best_face(probe)
        if probe_face is None:
            return None, -1.0, {}
        probe_vec = self.embedder.embed(probe, probe_face)

        per_person: dict[str, float] = {}
        for name, vec in gallery:
            sim = self.embedder.cosine(probe_vec, vec)
            if name not in per_person or sim > per_person[name]:
                per_person[name] = sim

        if not per_person:
            return None, -1.0, {}
        best_person = max(per_person, key=per_person.get)  # type: ignore[arg-type]
        best_score = per_person[best_person]
        identity = best_person if best_score >= threshold else None
        return identity, best_score, per_person
