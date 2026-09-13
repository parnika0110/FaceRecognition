# Face Recognition Identification System (YuNet + SFace)

Face enrollment and identification that runs **100% locally on CPU** — no cloud, no API keys, ₹0/$0 spend.

Built for the Code Nimbus Solutions AI/ML Intern assignment.

## What it does

- **Enroll** people from photos: detects the face, extracts a 128-D embedding, stores it in a local database.
- **Identify** new faces: detects faces, embeds them, matches against the enrolled gallery by cosine similarity.
- **Unknown rejection**: if the best similarity is below the calibrated threshold, the face is reported as `Unknown` instead of forcing a wrong match.
- **Evaluation**: verification ROC/AUC/EER, threshold sweep, closed-set rank-1 accuracy and open-set (unseen-person) false-accept rate on the LFW dataset.

## Pipeline

```
image ──► face detection ──► 5-point alignment ──► SFace embedding ──► cosine match ──► threshold gate
          (YuNet ONNX)        (112x112, ArcFace     (128-D, L2         (max over a        │
                               landmark template)    normalized)        person's gallery)  ├─≥ τ ──► identity
                                                                                            └─< τ ──► "Unknown"
```

## Models used

| Stage     | Model | Notes |
|-----------|-------|-------|
| Detection | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (`face_detection_yunet_2023mar.onnx`, 232 KB) | Single-stage CNN, returns box + 5 landmarks + score. Fast on CPU, robust to small/rotated faces. |
| Embedding | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (`face_recognition_sface_2021dec.onnx`, 38 MB) | SOTA face recognizer distilled to an 128-D embedding, shipped as ONNX with OpenCV. |

Both are Apache-2.0 licensed and run through `cv2.dnn` — no GPU, no account, no spend.

## Quick start

```bash
pip install -r requirements.txt
python download_models.py          # fetches the two ONNX models into ./models

# Enroll (one or more photos per person; run again to add more)
python cli.py enroll --name "Alice" --images alice1.jpg alice2.jpg

# Identify
python cli.py identify --images probe.jpg
python cli.py identify --images probe.jpg --json --top-k 3

# Live webcam demo (green = recognized, red = Unknown)
python cli.py live
```

## Evaluation on LFW

```bash
python scripts/prepare_lfw.py     # downloads LFW via scikit-learn, saves crops to data/lfw_crops
python evaluate.py                # writes reports/metrics.json + plots
```

The first `prepare_lfw.py` run downloads ~200 MB of LFW and takes ~20-30 minutes on CPU
(resumable); crops and embeddings are cached to disk, so later runs and `evaluate.py`
finish in minutes.

Protocol (details in `evaluate.py`):

- **Verification pairs** — genuine (same person, different photos) vs impostor (different people) cosine scores → ROC/AUC/EER, FAR/FRR, balanced accuracy sweep over thresholds.
- **Closed-set identification** — enroll 2 gallery photos per person, probe the rest → rank-1 accuracy.
- **Open-set rejection** — probe with photos of people never enrolled → fraction correctly rejected as `Unknown` at the operating threshold.

Results are committed under `reports/` (metrics + ROC curve + score distribution + threshold sweep).

## Results (measured on this machine, reports/)

LFW (funneled), 1,680 people, 9,164 images, cached SFace embeddings, 70,963 verification pairs:

| Metric | Value |
|---|---|
| Verification ROC-AUC | **0.9828** |
| EER | **2.95%** @ τ = 0.256 |
| Accuracy-max operating point | τ = 0.36 → accuracy **99.3%**, FAR 0.10%, FRR 3.8% |
| Closed-set identification (1,443-embedding gallery) | rank-1 **98.5%** (710/721) |
| Open-set rejection (800 probes from 180 unseen identities, τ = 0.36) | 30.6% rejected |

`reports/roc_curve.png`, `reports/threshold_sweep.png`, `reports/score_distribution.png` and the full
sweep/FPIR table in `reports/metrics.json` are committed so every number above is reproducible.

### The 1:N threshold story (the most important evaluation finding)

A pairwise threshold does **not** transfer unchanged to a big gallery. When an unknown face is
matched against N embeddings by max-similarity, the per-pair FAR compounds:
`P(unknown accepted) = 1 − (1 − FAR)^N`. With N ≈ 1,400 and FAR = 0.1%, that predicts ~75% of
unknowns wrongly accepted — exactly what we measure (FPIR = 69% at τ = 0.36):

| Threshold | Open-set FPIR (unknowns accepted) |
|---|---|
| 0.36 | 69.4% |
| 0.40 | 26.5% |
| **0.45** | **3.4%** |
| 0.50 | 0.25% |
| 0.65 | 0.25% |
| 0.70 | 0.13% (residual = LFW lookalikes/mislabeled twins) |

So the system ships **two selectable operating points** (`faceid/config.py`, exposed as `--policy` on `identify` and `live`):

- `--policy default` → `DEFAULT_MATCH_THRESHOLD = 0.36` — 1:1 / friendly small-gallery use.
- `--policy open-set` → `OPEN_SET_MATCH_THRESHOLD = 0.45` — untrusted-gallery / security use.
- An explicit `--threshold` value overrides the policy, e.g. `python cli.py identify --images probe.jpg --threshold 0.50`.

All reported metrics come from `python evaluate.py`, run end-to-end on CPU in minutes — no cloud, no spend.

## Matching threshold (policy)

Calibrated on LFW by `evaluate.py`, never hand-picked: the sweep reports the accuracy-max point
(chosen for the default), a Youden's-J point, and an EER point; the full table is in
`reports/metrics.json`. Select per call with `--policy default|open-set`, or override with an
explicit `--threshold` value.

**Disclosure:** thresholds are selected on the same LFW evaluation set (no held-out calibration
split); a production system would calibrate on a separate split.

## Design decisions

- **YuNet + SFace via OpenCV Zoo** instead of cloud APIs (AWS Rekognition, Face++) or heavy frameworks (dlib/insightface): zero cost, fully local, ~40 MB of models, CPU-only, pip-installable in one line. Trade-off: slightly lower accuracy than Ensemble-of-Losses ArcFace models.
- **Cosine similarity** on L2-normalized embeddings — standard for ArcFace-style models; SFace's native `match()` uses cosine too.
- **Per-person gallery of multiple embeddings, matched by max similarity.** A single enrollment photo over-constrains the gallery (pose/lighting variance); max over several embeddings gives robustness. Mean would blur distinct poses together.
- **5-point similarity alignment** to the standard ArcFace 112×112 template before embedding (done by `FaceRecognizerSF.alignCrop`). Embedding unaligned crops measurably hurts accuracy.
- **SQLite for the gallery** — zero-dependency, atomic, easy to inspect; vectors stored as BLOBs. Swappable for FAISS/pgvector at scale.
- **Unknown rejection is a first-class outcome**, not an afterthought: the matcher returns `(None, best_score)` and every caller renders `Unknown`. The open-set eval quantifies it.
- **Duplicate-enrollment guard**: enrolling a face that already strongly matches another person in the gallery warns you (unless `--allow-duplicates`), preventing accidental identity shadowing.

## Known failure cases (observed, with evidence)

1. **Large-gallery 1:N acceptance of unknowns** — the compounding effect above: at the friendly τ = 0.36, ~69% of unseen identities get accepted on a 1.4k-embedding gallery. Mitigation: raise the threshold for open-set use (documented in `config.py`); per-probe score normalization (T-norm/Z-norm) is the next step.
2. **Extreme pose / profile faces** — 5-point alignment degrades past ~45° yaw; the largest genuine-pair error source visible in the FRR tail of `reports/score_distribution.png`.
3. **Very low resolution** — probe faces below ~60 px wide lose discriminative detail (CLI flags these via `low_quality`).
4. **Heavy occlusion** — masks/sunglasses/hair over landmarks shift the embedding substantially.
5. **Lookalikes & label noise** — the residual FPIR ≈ 0.13–0.25% that persists even at τ = 0.65 comes from LFW lookalikes and mislabeled twins; the same effect explains residual rank-1 errors.
6. **Cross-age & lighting drift** — big gaps between enrollment and probe photos reduce similarity; multiple gallery photos per person mitigate it.
7. **Multi-face images** — enrollment deliberately refuses them (returns `skipped`) rather than guessing which face you meant.

## Improvements (next steps)

- **Better embeddings**: swap SFace for AdaFace / ArcFace-R100 via ONNX Runtime (+2–4% on hard sets).
- **Score normalization for open-set**: T-norm/Z-norm against a cohort of distractor embeddings, or a gallery-size-aware threshold — directly attacks failure case #1.
- **Quality gating**: reject blurred / extreme-pose / too-small faces before embedding (landmark geometry + variance-of-Laplacian).
- **Gallery consolidation**: cluster per-person embeddings and keep centroids + outliers, with per-embedding quality weights.
- **Re-ranking**: top-k shortlist with a second (stronger) model for borderline scores.
- **Liveness / anti-spoofing**: blink + texture analysis for webcam enrollment to block photo attacks.
- **Scale**: FAISS index (IVF-PQ) for 10⁵–10⁷ identities, with `sqlite` as the metadata sidecar.
- **Cross-age evaluation**: AgeDB/CFP-FP benchmarks to quantify pose/age robustness.
- **Privacy**: consent-gated enrollment, on-device-only storage, and an easy `remove` path (already in CLI) as a GDPR-style baseline.

## Repository layout

```
faceid/            core package (detector, embedder, database, config)
cli.py             enroll / identify / live / list / remove
evaluate.py        LFW evaluation → reports/
scripts/           LFW preparation
tests/             unit tests (gallery math, DB, threshold gating)
download_models.py fetch the two ONNX models
reports/           committed evaluation artifacts
models/            downloaded at setup (gitignored)
```

## Run the tests

```bash
pytest -q                # or: python -m pytest -q
```

## License

Code: [MIT](LICENSE). Models: Apache-2.0 (OpenCV Zoo).
