"""Prepare LFW for evaluation.

Triggers the scikit-learn LFW (funneled) download/extract (~200 MB), then
processes the extracted JPEGs straight from disk (avoiding sklearn's large
in-memory image array): YuNet detection -> SFace alignment -> 112x112 crop ->
128-D embedding. Outputs land in data/lfw_crops/<person>/NNN.jpg with the
embedding cached next to it as NNN.npy, so evaluate.py never re-embeds.

People with fewer than 2 images on disk are skipped (genuine pairs and
closed-set probes need at least 2-3 images per identity). Re-running resumes
where it left off.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people, get_data_home

from faceid.detector import FaceDetector
from faceid.embedder import FaceEmbedder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "lfw_crops"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading LFW (funneled) via scikit-learn — first run fetches ~200 MB ...")
    # Tiny resize keeps the in-memory array small; we only want download+extract.
    fetch_lfw_people(min_faces_per_person=2, color=True, funneled=True, resize=0.125)
    lfw_dir = Path(get_data_home()) / "lfw_home" / "lfw_funneled"
    if not lfw_dir.is_dir():
        print(f"error: expected extracted LFW at {lfw_dir}")
        return 1

    person_dirs = [
        d for d in sorted(lfw_dir.iterdir())
        if d.is_dir() and len(list(d.glob("*.jpg"))) >= 2
    ]
    print(f"{len(person_dirs)} people with >= 2 images on disk")

    detector = FaceDetector(score_threshold=0.5)
    embedder = FaceEmbedder()
    saved, no_face, reused = 0, 0, 0

    for i, person_dir in enumerate(person_dirs):
        out_person = OUT_DIR / person_dir.name
        out_person.mkdir(exist_ok=True)
        for j, img_path in enumerate(sorted(person_dir.glob("*.jpg"))):
            out_jpg = out_person / f"{j:03d}.jpg"
            out_npy = out_person / f"{j:03d}.npy"
            if out_jpg.exists() and out_npy.exists():
                reused += 1
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                no_face += 1
                continue
            face = detector.detect_largest(bgr)
            if face is None:
                no_face += 1
                continue
            aligned = embedder.aligned(bgr, face)
            if aligned is None or aligned.size == 0:
                no_face += 1
                continue
            vec = embedder.embed_aligned(aligned)
            cv2.imwrite(str(out_jpg), aligned)
            np.save(out_npy, vec)
            saved += 1
        if (i + 1) % 100 == 0:
            print(f"  people processed: {i + 1}/{len(person_dirs)} (crops saved: {saved})", flush=True)

    print(f"Saved {saved} aligned crops + embeddings ({no_face} images skipped: no face).")
    print(f"Reused {reused} already-processed images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
