"""Face recognition next to the body re-ID (OSNet). Faces are found in the head region of each
tracked person, so a face always belongs to the person box it came from.

Model: InsightFace "buffalo_l" (SCRFD face detector + ArcFace ResNet-50 embeddings), CPU or GPU
via onnxruntime. Downloads ~280 MB to ~/.insightface on first use.
NOTE: InsightFace's pretrained weights are for non-commercial research use only.

Works on moving people: every frame is checked independently, and only sharp-enough, fairly
frontal faces above FACE_MIN_PX are used. If insightface is not installed, faces are simply
off and everything runs on body appearance as before.
"""
import threading

import numpy as np

import config


class Face:
    __slots__ = ("emb", "width", "score", "box")

    def __init__(self, emb, width, score, box):
        self.emb, self.width, self.score, self.box = emb, width, score, box


def frontal_enough(kps, max_yaw=None):
    """5 landmarks (eyes, nose, mouth corners): the nose must sit near the middle of the eyes.
    Side views give poor ArcFace vectors, so they are skipped."""
    max_yaw = config.FACE_MAX_YAW if max_yaw is None else max_yaw
    le, re, nose = kps[0], kps[1], kps[2]
    eye_d = abs(re[0] - le[0])
    if eye_d < 1:
        return False
    return abs(nose[0] - (le[0] + re[0]) / 2) / eye_d <= max_yaw


def sharp_enough(gray_face, min_var=None):
    """Variance of the Laplacian: low = motion-blurred. Blurry faces are skipped, not trusted."""
    import cv2
    min_var = config.FACE_MIN_SHARPNESS if min_var is None else min_var
    return cv2.Laplacian(gray_face, cv2.CV_64F).var() >= min_var


def head_region(box, frame_shape):
    """Upper part of a person box (a bit wider), where the face must be."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    bh, bw = y2 - y1, x2 - x1
    pad = int(0.15 * bw)
    return (max(0, x1 - pad), max(0, y1 - int(0.05 * bh)), min(w, x2 + pad), min(h, y1 + int(0.45 * bh)))


class FaceEmbedder:
    name = "arcface"

    def __init__(self, model=None):
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if p in avail]
        providers.append("CPUExecutionProvider")
        self.app = FaceAnalysis(name=model or config.FACE_MODEL, allowed_modules=["detection", "recognition"],
                                providers=providers)
        self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                         det_thresh=config.FACE_DET_SCORE, det_size=(config.FACE_DET_SIZE, config.FACE_DET_SIZE))
        self.lock = threading.Lock()
        self.providers = providers

    def from_region(self, img):
        """Best usable face in an image region (head region or a check-in crop), or None."""
        import cv2
        from insightface.utils import face_align
        if img is None or img.size == 0 or min(img.shape[:2]) < config.FACE_MIN_PX:
            return None
        from insightface.app.common import Face as InsightFace
        size = (config.FACE_DET_SIZE, config.FACE_DET_SIZE)
        with self.lock:
            boxes, kpss = self.app.models["detection"].detect(img, input_size=size)
        # pick the best candidate first (cheap checks), then run the expensive recognition once
        best = None
        for b, k in zip(boxes if boxes is not None else [], kpss if kpss is not None else []):
            x1, y1, x2, y2 = b[:4].astype(int)
            width = x2 - x1
            if width < config.FACE_MIN_PX or b[4] < config.FACE_DET_SCORE or not frontal_enough(k):
                continue
            # sharpness on the aligned 112x112 face ArcFace actually sees, so the limit doesn't
            # depend on face size; blurred faces would look like "a different person"
            aligned = face_align.norm_crop(img, k)
            if not sharp_enough(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)):
                continue
            if best is None or width > best[0]:
                best = (width, b, k)
        if best is None:
            return None
        width, b, k = best
        f = InsightFace(bbox=b[:4], kps=k, det_score=b[4])
        with self.lock:
            self.app.models["recognition"].get(img, f)
        emb = (f.embedding / (np.linalg.norm(f.embedding) + 1e-9)).astype(np.float32)
        x1, y1, x2, y2 = b[:4].astype(int)
        return Face(emb, int(width), float(b[4]), (int(x1), int(y1), int(x2), int(y2)))

    def from_person(self, frame, box):
        x1, y1, x2, y2 = head_region(box, frame.shape)
        return self.from_region(frame[y1:y2, x1:x2])


_cached, _tried = None, False
_lock = threading.Lock()


def get_face_embedder():
    """Shared instance, or None when faces are off or insightface isn't installed."""
    global _cached, _tried
    with _lock:
        if _tried:
            return _cached
        _tried = True
        if not config.FACE_ENABLED:
            return None
        try:
            _cached = FaceEmbedder()
            print(f"[face] {config.FACE_MODEL} loaded ({_cached.providers[0]})", flush=True)
        except Exception as e:   # not installed, no internet for the first download, ...
            print(f"[face] face recognition OFF: {e!r}. Install with: pip install insightface onnxruntime",
                  flush=True)
            _cached = None
        return _cached


def mean_face(faces):
    if not faces:
        return None
    m = np.mean(np.stack(faces), axis=0)
    return m / (np.linalg.norm(m) + 1e-9)
