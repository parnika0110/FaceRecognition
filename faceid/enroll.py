"""Enrollment: turn photos into gallery embeddings, with guards."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config
from .database import GalleryDB
from .detector import FaceDetector
from .embedder import FaceEmbedder


@dataclass
class EnrollReport:
    person: str
    added: int = 0
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def enroll_person(
    db: GalleryDB,
    person: str,
    image_paths: list[str],
    detector: FaceDetector | None = None,
    embedder: FaceEmbedder | None = None,
    allow_duplicates: bool = False,
) -> EnrollReport:
    """Enroll one or more images for `person`.

    Guards:
      * refuses images with no / multiple faces (picks the strongest if you pass
        allow_duplicates=False it still refuses ambiguous multi-face images),
      * caps embeddings per person,
      * warns when a new face is suspiciously similar to another identity.
    """
    detector = detector or FaceDetector()
    embedder = embedder or FaceEmbedder()
    report = EnrollReport(person=person)

    existing = dict(db.people())
    if existing.get(person, 0) >= config.MAX_GALLERY_EMBEDDINGS_PER_PERSON:
        report.warnings.append(
            f"{person} already has {existing[person]} embeddings "
            f"(cap {config.MAX_GALLERY_EMBEDDINGS_PER_PERSON}); not adding more."
        )
        return report

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            report.skipped.append(f"{path}: unreadable image")
            continue

        faces = detector.detect(image)
        if not faces:
            report.skipped.append(f"{path}: no face detected")
            continue
        if len(faces) > 1:
            report.skipped.append(
                f"{path}: {len(faces)} faces detected; use a photo with a single face"
            )
            continue

        face = faces[0]
        flags = FaceEmbedder.quality_flags(face)
        if flags["too_small"]:
            report.warnings.append(f"{path}: face is small ({face.width}px), embedding may be weak")

        vec = embedder.embed(image, face)

        if not allow_duplicates:
            best = db.max_similarity_to(vec)
            if best and best[0] != person and best[1] >= config.DUPLICATE_SIMILARITY_WARN:
                report.warnings.append(
                    f"{path}: similarity {best[1]:.3f} to enrolled person '{best[0]}' "
                    f"— possible duplicate enrollment"
                )

        db.add(person, vec, source=path)
        report.added += 1

    return report
