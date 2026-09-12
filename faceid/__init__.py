"""Face recognition identification system (YuNet + SFace, CPU-only)."""

from .config import DEFAULT_MATCH_THRESHOLD
from .database import GalleryDB
from .detector import Face, FaceDetector
from .embedder import FaceEmbedder, FacePipeline

__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "Face",
    "FaceDetector",
    "FaceEmbedder",
    "FacePipeline",
    "GalleryDB",
]
