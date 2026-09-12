"""Command-line interface for the face recognition system."""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from faceid import config
from faceid.database import GalleryDB
from faceid.detector import FaceDetector
from faceid.embedder import FaceEmbedder, FacePipeline
from faceid.enroll import enroll_person


def _load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"error: could not read image: {path}")
    return img


def cmd_enroll(args) -> None:
    with GalleryDB(args.db) as db:
        report = enroll_person(
            db,
            args.name,
            args.images,
            allow_duplicates=args.allow_duplicates,
        )
    print(f"Enrolled '{report.person}': {report.added} new embedding(s)")
    for s in report.skipped:
        print(f"  skipped: {s}")
    for w in report.warnings:
        print(f"  WARNING: {w}")
    if report.added == 0:
        sys.exit(1)


def cmd_identify(args) -> None:
    pipeline = FacePipeline()
    with GalleryDB(args.db) as db:
        gallery = db.all_vectors()
    if not gallery:
        raise SystemExit("error: gallery is empty — enroll someone first")

    results = []
    for path in args.images:
        img = _load_image(path)
        faces = pipeline.detect_faces(img)
        if not faces:
            entry = {"image": path, "status": "no_face"}
            print(f"{path}: no face detected")
        else:
            face = faces[0]
            identity, score, per_person = pipeline.match(img, gallery, args.threshold)
            entry = {
                "image": path,
                "status": "ok",
                "identity": identity if identity else "Unknown",
                "score": round(score, 4),
                "threshold": args.threshold,
                "faces_detected": len(faces),
                "low_quality": any(FaceEmbedder.quality_flags(f)["too_small"] for f in faces),
                "top_matches": sorted(per_person.items(), key=lambda kv: -kv[1])[: args.top_k],
            }
            label = identity if identity else "Unknown"
            print(f"{path}: {label}  (cosine={score:.3f}, threshold={args.threshold:.3f})")
            if not identity:
                ranked = ", ".join(f"{n}={s:.3f}" for n, s in entry["top_matches"])
                if ranked:
                    print(f"    best candidates below threshold: {ranked}")
        results.append(entry)

    if args.json:
        print(json.dumps(results, indent=2))


def cmd_list(args) -> None:
    with GalleryDB(args.db) as db:
        rows = db.people()
    if not rows:
        print("Gallery is empty.")
        return
    print(f"{len(rows)} person(s), {sum(c for _, c in rows)} embeddings total:")
    for name, count in rows:
        print(f"  {name}: {count} embedding(s)")


def cmd_remove(args) -> None:
    with GalleryDB(args.db) as db:
        removed = db.remove_person(args.name)
    print(f"Removed {removed} embedding(s) for '{args.name}'.")


def cmd_clear(args) -> None:
    with GalleryDB(args.db) as db:
        removed = db.clear()
    print(f"Removed {removed} embedding(s). Gallery is now empty.")


def cmd_live(args) -> None:
    """Webcam loop: green box + name for matches, red box + 'Unknown' otherwise."""
    pipeline = FacePipeline()
    with GalleryDB(args.db) as db:
        gallery = db.all_vectors()
    if not gallery:
        print("Gallery is empty — enroll someone first (python cli.py enroll ...).")
        print("Starting anyway so you can verify detection.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"error: cannot open camera {args.camera}")
    print("Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = pipeline.detect_faces(frame)
        for face in faces:
            vec = pipeline.embedder.embed(frame, face)
            best_name, best_score = None, -1.0
            for name, gvec in gallery:
                sim = pipeline.embedder.cosine(vec, gvec)
                if sim > best_score:
                    best_name, best_score = name, sim

            identity = best_name if (best_name and best_score >= args.threshold) else None
            label = f"{identity} {best_score:.2f}" if identity else f"Unknown {best_score:.2f}"
            color = (0, 200, 0) if identity else (0, 0, 220)

            x, y, w, h = face.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face ID (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faceid", description="Face enrollment & identification")
    p.add_argument("--db", default=str(config.DB_PATH), help="gallery database path")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("enroll", help="enroll a person from one or more images")
    e.add_argument("--name", required=True)
    e.add_argument("--images", nargs="+", required=True)
    e.add_argument("--allow-duplicates", action="store_true",
                   help="skip the duplicate-enrollment similarity warning")
    e.set_defaults(func=cmd_enroll)

    i = sub.add_parser("identify", help="identify people in images")
    i.add_argument("--images", nargs="+", required=True)
    i.add_argument("--threshold", type=float, default=config.DEFAULT_MATCH_THRESHOLD)
    i.add_argument("--top-k", type=int, default=3)
    i.add_argument("--json", action="store_true", help="also print JSON results")
    i.set_defaults(func=cmd_identify)

    sub.add_parser("list", help="list enrolled people").set_defaults(func=cmd_list)

    r = sub.add_parser("remove", help="remove a person's embeddings")
    r.add_argument("--name", required=True)
    r.set_defaults(func=cmd_remove)

    c = sub.add_parser("clear", help="delete all embeddings").set_defaults(func=cmd_clear)

    l = sub.add_parser("live", help="webcam demo")
    l.add_argument("--camera", type=int, default=0)
    l.add_argument("--threshold", type=float, default=config.DEFAULT_MATCH_THRESHOLD)
    l.set_defaults(func=cmd_live)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
