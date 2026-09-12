"""Evaluate the recognition system on LFW and calibrate the match threshold.

Writes reports/metrics.json, reports/roc_curve.png, reports/threshold_sweep.png,
reports/score_distribution.png.

Protocol
--------
Verification: all within-person (genuine) and cross-person (impostor) pairs,
capped for tractability. Score = cosine similarity of SFace embeddings of the
aligned crops from scripts/prepare_lfw.py.

Closed-set identification: for each person with >= 3 usable images, 2 gallery
images + the rest as probes; rank-1 accuracy with max-similarity matching.

Open-set: probe with images of people NOT in the gallery; a probe is correctly
rejected when its best gallery similarity is below the operating threshold.
"""

from __future__ import annotations

import json
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from faceid import config
from faceid.embedder import FaceEmbedder

PROJECT_ROOT = Path(__file__).resolve().parent  # this file lives at the repo root
CROPS_DIR = PROJECT_ROOT / "data" / "lfw_crops"
REPORTS = PROJECT_ROOT / "reports"
MAX_PAIRS_PER_PERSON = 25          # genuine pairs per person
MAX_IMPOSTOR_PAIRS = 60_000        # impostor pairs total
RNG_SEED = 42

embedder = FaceEmbedder()


def load_crops() -> dict[str, list[Path]]:
    people: dict[str, list[Path]] = {}
    for person_dir in sorted(CROPS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted(person_dir.glob("*.jpg"))
        if len(imgs) >= 2:
            people[person_dir.name] = imgs
    return people


def get_embedding(path: Path, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = str(path)
    if key not in cache:
        npy = path.with_suffix(".npy")
        if npy.exists():  # cached by scripts/prepare_lfw.py
            vec = np.load(npy)
        else:
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"cannot read crop: {path}")
            vec = embedder.embed_aligned(img)
            np.save(npy, vec)
        cache[key] = vec
    return cache[key]


def build_pairs(people: dict[str, list[Path]]):
    rng = random.Random(RNG_SEED)

    genuine: list[tuple[str, str, int]] = []
    for person, imgs in people.items():
        pairs = list(combinations(imgs, 2))
        if len(pairs) > MAX_PAIRS_PER_PERSON:
            pairs = rng.sample(pairs, MAX_PAIRS_PER_PERSON)
        genuine.extend((str(a), str(b), 1) for a, b in pairs)

    names = list(people)
    impostor: list[tuple[str, str, int]] = []
    tries = 0
    while len(impostor) < MAX_IMPOSTOR_PAIRS and tries < 500_000:
        a_name, b_name = rng.sample(names, 2)
        a, b = rng.choice(people[a_name]), rng.choice(people[b_name])
        impostor.append((str(a), str(b), 0))
        tries += 1
    return genuine, impostor


def score_pairs(pairs, cache) -> np.ndarray:
    sims = []
    for a, b, _ in pairs:
        va, vb = get_embedding(Path(a), cache), get_embedding(Path(b), cache)
        sims.append(embedder.cosine(va, vb))
    return np.asarray(sims)


def sweep_thresholds(y_true, sims):
    thresholds = np.round(np.arange(0.0, 0.9001, 0.01), 3)
    rows = []
    for t in thresholds:
        pred = (sims >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        far = fp / max(fp + tn, 1)
        frr = fn / max(fn + tp, 1)
        acc = (tp + tn) / max(len(y_true), 1)
        rows.append({
            "threshold": float(t), "far": round(far, 4), "frr": round(frr, 4),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(0.5 * ((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1))), 4),
        })
    return rows


def eer_from_roc(fpr, tpr, thresholds):
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(thresholds[i]), float(fpr[i])


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    people = load_crops()
    if not people:
        print("No crops found. Run scripts/prepare_lfw.py first.")
        return 1
    print(f"{len(people)} people loaded. Building pairs ...")

    cache: dict[str, np.ndarray] = {}
    genuine, impostor = build_pairs(people)
    y = np.array([1] * len(genuine) + [0] * len(impostor))
    print(f"Scoring {len(genuine)} genuine + {len(impostor)} impostor pairs ...")
    sims = np.concatenate([score_pairs(genuine, cache), score_pairs(impostor, cache)])

    auc = float(roc_auc_score(y, sims))
    fpr, tpr, thr = roc_curve(y, sims)
    eer_threshold, eer = eer_from_roc(fpr, tpr, thr)

    sweep = sweep_thresholds(y, sims)
    acc_point = max(sweep, key=lambda r: r["accuracy"])
    youden_point = max(sweep, key=lambda r: r["balanced_accuracy"])
    far1_points = [r for r in sweep if r["far"] <= 0.01]
    far1_point = max(far1_points, key=lambda r: r["balanced_accuracy"]) if far1_points else None

    # ---- closed-set identification + open-set rejection ------------------
    # 20% of identities (and all 2-image identities) are held OUT of the
    # gallery: their probes must be rejected as Unknown (open-set).
    print("Closed-set identification + open-set rejection ...")
    people3 = {n: imgs for n, imgs in people.items() if len(imgs) >= 3}
    people2 = {n: imgs for n, imgs in people.items() if len(imgs) == 2}
    rng = random.Random(RNG_SEED + 1)
    names3 = sorted(people3)
    rng.shuffle(names3)
    n_unseen = max(1, int(round(0.2 * len(names3))))
    unseen_names = set(names3[:n_unseen])
    enrolled_names = names3[n_unseen:]

    gallery_vecs: list[tuple[str, np.ndarray]] = []
    known_probes: list[tuple[str, np.ndarray]] = []
    for name in enrolled_names:
        chosen = rng.sample(people3[name], 3)
        gallery_vecs += [(name, get_embedding(p, cache)) for p in chosen[:2]]
        known_probes += [(name, get_embedding(p, cache)) for p in chosen[2:]]

    gallery_names = [n for n, _ in gallery_vecs]
    gallery_matrix = np.stack([v for _, v in gallery_vecs])
    rank1_hits, rank1_total = 0, 0
    for person, vec in known_probes:
        best = gallery_names[int(np.argmax(gallery_matrix @ vec))]
        rank1_total += 1
        rank1_hits += int(best == person)

    unknown_vecs = [get_embedding(p, cache) for n in unseen_names for p in people3[n]]
    unknown_vecs += [get_embedding(p, cache) for imgs in people2.values() for p in imgs]
    if len(unknown_vecs) > 800:
        unknown_vecs = rng.sample(unknown_vecs, 800)
    unknown_total = len(unknown_vecs)
    t = acc_point["threshold"]
    unknown_hits = sum(1 for v in unknown_vecs if float(np.max(gallery_matrix @ v)) < t)

    # FPIR (false-positive identification rate) of unseen-identity probes vs
    # threshold. With a large gallery, max-similarity over N impostor
    # candidates inflates scores: P(max > tau) = 1 - (1 - FAR)^N. This curve
    # quantifies why a fixed pairwise threshold does not transfer to big
    # galleries (see README "Matching threshold").
    unknown_matrix = np.stack(unknown_vecs)
    unknown_best = np.max(unknown_matrix @ gallery_matrix.T, axis=1)
    fpir_curve = {
        f"{tv:.2f}": round(float((unknown_best >= tv).mean()), 4)
        for tv in (0.30, 0.36, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    }

    metrics = {
        "dataset": "LFW (funneled), people with >= 2 usable images",
        "n_people": len(people),
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "verification": {
            "auc": round(auc, 4),
            "eer": round(eer, 4),
            "eer_threshold": round(eer_threshold, 3),
        },
        "threshold_sweep": sweep,
        "operating_points": {
            "accuracy_max": acc_point,
            "youden_j": youden_point,
            "far_le_1pct": far1_point,
            "chosen_default": acc_point,
        },
        "closed_set_identification": {
            "gallery_size": len(gallery_vecs),
            "n_probes": rank1_total,
            "rank1_accuracy": round(rank1_hits / max(rank1_total, 1), 4),
        },
        "open_set_rejection": {
            "operating_threshold": t,
            "n_unseen_identities": len(unseen_names),
            "n_unknown_probes": unknown_total,
            "correctly_rejected_rate": round(unknown_hits / max(unknown_total, 1), 4),
            "fpir_by_threshold": fpir_curve,
            "gallery_embeddings": len(gallery_vecs),
            "note": "pairwise FAR compounds over the gallery: P(accept unknown) = 1 - (1-FAR)^N",
        },
    }

    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"AUC={auc:.4f}  EER={eer:.4f} @ threshold={eer_threshold:.3f}")
    print(f"Accuracy-max threshold={acc_point['threshold']:.2f} "
          f"(acc={acc_point['accuracy']:.4f}, FAR={acc_point['far']:.4f}, FRR={acc_point['frr']:.4f})")
    print(f"Closed-set rank-1: {rank1_hits}/{rank1_total} = "
          f"{rank1_hits / max(rank1_total, 1):.4f}")
    print(f"Open-set rejection @ threshold={t:.2f}: {unknown_hits}/{unknown_total} rejected")
    print("Open-set FPIR by threshold: " +
          ", ".join(f"{k}:{v}" for k, v in fpir_curve.items()))

    # ---- plots -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.scatter([eer], [1 - eer], color="red", zorder=3, label=f"EER={eer:.3f}")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("LFW verification ROC (SFace)"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS / "roc_curve.png", dpi=140); plt.close(fig)

    accs = [r["accuracy"] for r in sweep]
    fars = [r["far"] for r in sweep]
    frrs = [r["frr"] for r in sweep]
    ts = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, accs, label="accuracy")
    ax.plot(ts, fars, label="FAR")
    ax.plot(ts, frrs, label="FRR")
    ax.axvline(acc_point["threshold"], color="black", ls="--", lw=0.8,
               label=f"chosen={acc_point['threshold']:.2f}")
    ax.set_xlabel("cosine threshold"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Threshold sweep on LFW")
    fig.tight_layout(); fig.savefig(REPORTS / "threshold_sweep.png", dpi=140); plt.close(fig)

    g_sims = score_pairs(genuine, cache)
    i_sims = score_pairs(impostor, cache)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(i_sims, bins=60, alpha=0.6, density=True, label="impostor")
    ax.hist(g_sims, bins=60, alpha=0.6, density=True, label="genuine")
    ax.axvline(acc_point["threshold"], color="black", ls="--", lw=1, label=f"threshold={acc_point['threshold']:.2f}")
    ax.set_xlabel("cosine similarity"); ax.legend()
    ax.set_title("Score distribution (LFW)")
    fig.tight_layout(); fig.savefig(REPORTS / "score_distribution.png", dpi=140); plt.close(fig)

    print(f"Reports written to {REPORTS}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
