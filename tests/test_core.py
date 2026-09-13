"""Unit tests that do not need the ONNX models."""

from __future__ import annotations

import numpy as np
import pytest

from faceid import config
from faceid.database import GalleryDB
from faceid.embedder import FaceEmbedder, FacePipeline


@pytest.fixture()
def tmp_db(tmp_path):
    db = GalleryDB(tmp_path / "gallery.db")
    yield db
    db.close()


def test_add_and_count(tmp_db):
    vec = np.ones(128, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    tmp_db.add("alice", vec, source="a.jpg")
    tmp_db.add("alice", vec * 0.5, source="b.jpg")
    assert tmp_db.count() == 2
    assert tmp_db.count("alice") == 2
    assert tmp_db.people() == [("alice", 2)]


def test_all_vectors_roundtrip(tmp_db):
    rng = np.random.default_rng(0)
    vec = rng.normal(size=128).astype(np.float32)
    tmp_db.add("bob", vec)
    people, stored = tmp_db.all_vectors()[0]
    assert people == "bob"
    np.testing.assert_allclose(stored, vec, rtol=1e-6)


def test_remove_and_clear(tmp_db):
    tmp_db.add("alice", np.ones(4, dtype=np.float32))
    tmp_db.add("bob", np.ones(4, dtype=np.float32))
    assert tmp_db.remove_person("alice") == 1
    assert tmp_db.count("alice") == 0
    assert tmp_db.clear() == 1


def test_cosine_identical_and_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([-1.0, 0.0])
    assert FaceEmbedder.cosine(a, b) == pytest.approx(1.0)
    assert FaceEmbedder.cosine(a, c) == pytest.approx(-1.0)


class _FakeDetector:
    """Stands in for FaceDetector so FacePipeline can be tested model-free."""

    def __init__(self, face):
        self._face = face

    def detect(self, image):
        return [self._face] if self._face else []

    def detect_largest(self, image):
        return self._face


class _FakeFace:
    box = (0, 0, 100, 100)
    score = 0.9
    width = 100
    landmarks = None


def test_match_threshold_gating(tmp_path, monkeypatch):
    probe = np.zeros((10, 10, 3), dtype=np.uint8)
    alice = np.zeros(8, dtype=np.float32); alice[0] = 1.0
    bob = np.zeros(8, dtype=np.float32); bob[1] = 1.0
    near = np.zeros(8, dtype=np.float32); near[0] = 0.2; near[1] = 1.0

    monkeypatch.setattr(FaceEmbedder, "embed",
                        staticmethod(lambda image, face: np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0])))

    # Matches alice (sim=1.0) well above threshold
    pipe = FacePipeline(detector=_FakeDetector(_FakeFace()))
    identity, score, per_person = pipe.match(probe, [("alice", alice), ("bob", bob)], 0.363)
    assert identity == "alice" and score == pytest.approx(1.0)

    # Below threshold → Unknown
    identity2, score2, _ = pipe.match(probe, [("almost", near)], 0.363)
    assert identity2 is None and score2 < 0.363

    # No faces at all
    pipe_empty = FacePipeline(detector=_FakeDetector(None))
    identity3, score3, _ = pipe_empty.match(probe, [("alice", alice)], 0.363)
    assert identity3 is None and score3 == -1.0


def test_default_threshold_is_calibrated_value():
    assert 0.0 < config.DEFAULT_MATCH_THRESHOLD < 1.0


def test_match_uses_max_similarity_across_gallery(tmp_path, monkeypatch):
    """A person with several gallery embeddings is matched by their best one."""
    probe = np.zeros((10, 10, 3), dtype=np.uint8)
    probe_vec = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    close = probe_vec.copy()          # cosine 1.0 to the probe
    far = np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0])  # cosine 0.0 to the probe

    monkeypatch.setattr(FaceEmbedder, "embed", staticmethod(lambda image, face: probe_vec))
    pipe = FacePipeline(detector=_FakeDetector(_FakeFace()))

    # 'far' listed first: the max over the person's embeddings must win.
    identity, score, per_person = pipe.match(
        probe, [("multi", far), ("multi", close)], config.DEFAULT_MATCH_THRESHOLD
    )
    assert identity == "multi"
    assert score == pytest.approx(1.0)
    assert per_person["multi"] == pytest.approx(1.0)


def test_threshold_boundary_is_inclusive(tmp_path, monkeypatch):
    """Gating uses >=: a score exactly equal to the threshold is accepted."""
    probe = np.zeros((10, 10, 3), dtype=np.uint8)
    probe_vec = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    # Unit-norm vector at 60 degrees -> cosine exactly 0.5 in exact math.
    edge = np.zeros(8, dtype=np.float32)
    edge[0] = np.float32(0.5)
    edge[1] = np.float32(np.sqrt(0.75))

    monkeypatch.setattr(FaceEmbedder, "embed", staticmethod(lambda image, face: probe_vec))
    pipe = FacePipeline(detector=_FakeDetector(_FakeFace()))

    # Same inputs, same cosine code path -> bit-identical score, so the
    # boundary comparison is exact regardless of float rounding.
    sim = FaceEmbedder.cosine(probe_vec, edge)
    identity, score, _ = pipe.match(probe, [("edge", edge)], sim)
    assert score == sim
    assert identity == "edge"

    # The smallest representable step above the score must reject.
    just_above = float(np.nextafter(sim, 1.0))
    identity2, score2, _ = pipe.match(probe, [("edge", edge)], just_above)
    assert identity2 is None and score2 < just_above


def test_match_empty_gallery_returns_sentinel(tmp_path, monkeypatch):
    """Documented empty-gallery result: (None, -1.0, {})."""
    probe = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(
        FaceEmbedder, "embed", staticmethod(lambda image, face: np.ones(8, dtype=np.float32))
    )
    pipe = FacePipeline(detector=_FakeDetector(_FakeFace()))
    identity, score, per_person = pipe.match(probe, [], config.DEFAULT_MATCH_THRESHOLD)
    assert identity is None
    assert score == -1.0
    assert per_person == {}
