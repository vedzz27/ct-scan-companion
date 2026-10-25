"""Face recognition on top of body re-ID.  Run: python test_face.py
Part 1 needs no model. Part 2 runs the real InsightFace model on its sample photo if installed
(first run downloads ~280 MB)."""
import time

import numpy as np

import config
import face as face_mod
import matching
from store import Store
from worker import CameraWorker


def vec(seed, n=64):
    r = np.random.default_rng(seed).normal(size=n).astype(np.float32)
    return r / np.linalg.norm(r)


def near(v, seed, noise=0.15):
    r = v + noise * vec(seed, len(v))
    return (r / np.linalg.norm(r)).astype(np.float32)


BIND, KEEP, _ = config.thresholds("osnet")


def test_fuse_rules():
    f = matching.fuse_face
    assert f(0.55, None, BIND, KEEP) == 0.55                      # no face: body only
    assert f(0.30, 0.70, BIND, KEEP) >= BIND + 0.1                # same face, other clothes
    assert f(0.90, 0.05, BIND, KEEP) < KEEP                       # same clothes, other face
    assert f(0.55, 0.32, BIND, KEEP) == 0.55                      # unclear face: body decides
    print("ok  fuse rules")


def test_same_clothes_different_faces():
    s = Store()
    uniform = vec(1)                                              # e.g. two nurses in the same scrubs
    anna = s.check_in("Anna", [uniform], faces=[vec(100, 512)])
    ben = s.check_in("Ben", [uniform], faces=[vec(200, 512)])
    sc = s.scores(near(uniform, 3, 0.05), face=near(vec(200, 512), 4))
    assert sc[ben.id] >= BIND and sc[anna.id] < KEEP, sc
    pid, _, margin, _ = s.match(near(uniform, 3, 0.05), face=near(vec(200, 512), 4))
    assert pid == ben.id and margin > 0.08
    print("ok  same clothes, told apart by face")


def test_jacket_off():
    s = Store()
    p = s.check_in("Ravi", [vec(10)], faces=[vec(300, 512)])
    other_clothes = vec(11)
    assert s.scores(other_clothes)[p.id] < KEEP                   # body alone: lost
    assert s.scores(other_clothes, face=near(vec(300, 512), 5))[p.id] >= BIND   # face: found
    print("ok  recognised by face after changing clothes")


class FakeEmbedder:
    name = "osnet"


class FakeFaces:
    """Returns a fixed face for any head region; counts calls."""
    name = "arcface"

    def __init__(self, emb):
        self.emb, self.calls = emb, 0

    def from_person(self, frame, box):
        self.calls += 1
        return face_mod.Face(self.emb, 80, 0.9, box)


def test_newcomer_face_collected_while_talking():
    s = Store()
    w = CameraWorker("waiting", {"source": 0, "role": "tracking"}, s, FakeEmbedder())
    ravi_face = vec(400, 512)
    fake = FakeFaces(ravi_face)
    orig = face_mod.get_face_embedder
    import worker as worker_mod
    worker_mod.get_face_embedder = lambda: fake
    try:
        frame = np.zeros((720, 1280, 3), np.uint8)
        box = (100, 100, 220, 500)
        t = time.time()
        ts = w.tracks.setdefault(7, matching.TrackState(t))
        ts.quality = "edge"                         # moving / cut off: body crop unusable, face still OK
        for i in range(5):                          # ~0.75 s while the person speaks
            got = w.collect_faces(frame, [(*box, 7, 0.9)], t + i * (config.FACE_EVERY_S_NEW + 0.01))
            assert got == {7}
        assert len(w.track_faces(7)) == 5
        # enroll as the newcomer endpoint does
        for _ in range(config.NEW_PERSON_MIN_VIEWS):
            ts.embs.append(vec(20))
        p = s.check_in("Ravi", w.track_views(7), faces=w.track_faces(7))
        w.enroll_track(7, p.id)
        assert s.location(p.id)["face_enrolled"]
        # later, same person, other clothes, on a new track: found through the face
        assert s.match(vec(21), face=near(ravi_face, 6))[0] == p.id
        # hidden behind someone: no face is taken from that box
        ts.quality, ts.last_face = "occluded", float("-inf")
        assert w.collect_faces(frame, [(*box, 7, 0.9)], t + 10) == set()
    finally:
        worker_mod.get_face_embedder = orig
    print("ok  newcomer face collected while moving/talking")


def test_async_faces_dont_block_and_drop_stale():
    import threading
    import worker as worker_mod
    s = Store()
    w = CameraWorker("waiting", {"source": 0, "role": "tracking"}, s, FakeEmbedder())

    class SlowFaces(FakeFaces):
        def from_region(self, crop):
            time.sleep(0.3)                          # a slow CPU
            return face_mod.Face(self.emb, 80, 0.9, (0, 0, 1, 1))
    fake = SlowFaces(vec(500, 512))
    orig = worker_mod.get_face_embedder
    worker_mod.get_face_embedder = lambda: fake
    try:
        w.face_async = True
        threading.Thread(target=w.face_loop, daemon=True).start()
        frame = np.zeros((720, 1280, 3), np.uint8)
        box = (100, 100, 220, 500)
        t = time.time()
        ts = w.tracks.setdefault(3, matching.TrackState(t))
        ts.quality = "ok"
        t0 = time.time()
        assert w.collect_faces(frame, [(*box, 3, 0.9)], t) == set()     # queued, not waited for
        assert time.time() - t0 < 0.05, "tracking must not wait for the face model"
        time.sleep(0.5)
        assert w.collect_faces(frame, [(*box, 3, 0.9)], t + 0.01) == {3}  # result arrives later
        # a face from a picture taken before two people touched is thrown away
        ts.last_face = float("-inf")
        w.collect_faces(frame, [(*box, 3, 0.9)], t + 1)
        ts.mark_close(t + 1.1)
        time.sleep(0.5)
        assert w.collect_faces(frame, [], t + 1.2) == set() and len(ts.faces) == 0
    finally:
        w.stop_flag = True
        worker_mod.get_face_embedder = orig
    print("ok  faces in background thread, stale results dropped")


def test_real_model():
    try:
        import insightface  # noqa: F401
        from insightface.data import get_image
    except ImportError:
        print("skip real model test (pip install insightface onnxruntime)")
        return
    import cv2
    fe = face_mod.FaceEmbedder()
    img = get_image("t1")                              # group photo shipped with insightface
    faces = fe.app.get(img)
    assert len(faces) >= 4, len(faces)
    good = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        pad = (x2 - x1)
        region = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        a = fe.from_region(region)
        if a is None:
            continue
        # the same face again: smaller, darker, slightly moved, light motion blur
        b_img = cv2.resize(region, None, fx=0.8, fy=0.8)
        b_img = cv2.convertScaleAbs(b_img, alpha=0.8, beta=-10)
        b_img = cv2.filter2D(b_img, -1, np.ones((1, 3), np.float32) / 3)
        b = fe.from_region(b_img)
        if b is not None:
            good.append((a.emb, b.emb, a.width))
    assert len(good) >= 3, f"only {len(good)} usable faces"
    same = [float(np.dot(a, b)) for a, b, _ in good]
    diff = [float(np.dot(good[i][0], good[j][0])) for i in range(len(good)) for j in range(len(good)) if i < j]
    print(f"    real faces: {len(good)} usable, widths {[w for *_, w in good]} px")
    print(f"    same face (changed photo): min {min(same):.2f}   different people: max {max(diff):.2f}")
    assert min(same) >= config.FACE_SAME_T, same
    assert max(diff) <= config.FACE_DIFF_T, diff
    # a heavily blurred (fast moving) face is rejected, not trusted
    x1, y1, x2, y2 = faces[0].bbox.astype(int)
    pad = x2 - x1
    region = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    smeared = cv2.filter2D(region, -1, np.ones((1, 25), np.float32) / 25)
    r = fe.from_region(smeared)
    assert r is None, "a smeared (fast moving) face must be rejected, not trusted"
    print("    strongly smeared face: rejected")
    print("ok  real model")


test_fuse_rules()
test_same_clothes_different_faces()
test_jacket_off()
test_newcomer_face_collected_while_talking()
test_async_faces_dont_block_and_drop_stale()
test_real_model()
print("all face tests passed")
