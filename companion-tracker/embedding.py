"""Turns person crops into anonymous appearance vectors. Raw images are never stored.

Embedders (pick with EMBEDDER):
  osnet  person re-ID network trained on MSMT17 (default, needs weights: python download_reid.py)
  fused  osnet + a small share of colour histogram (can help when the two cameras' colours match well)
  hsv    colour histogram only (no extra deps, weak: similar clothes look identical)
If the OSNet weights are missing, the app falls back to hsv and says so in the log and /api/health.
"""
import os
import threading

import cv2
import numpy as np

import config


def colour_gains(frame):
    """Grey-world white balance gains for one frame, so two cameras report similar colours."""
    small = cv2.resize(frame, (64, 40), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    means = small.mean(axis=0) + 1e-6
    return (means.mean() / means).clip(0.6, 1.6).astype(np.float32)


def apply_gains(crop, gains):
    if gains is None:
        return crop
    return np.clip(crop.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def _hsv(bgr):
    img = cv2.resize(bgr, (64, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats = []
    for part in (hsv[20:60], hsv[60:120]):  # torso, legs (skip head and feet)
        h = cv2.calcHist([part], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        h /= h.sum() + 1e-9
        feats.append(np.sqrt(h))
    v = np.concatenate(feats)
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


class HsvEmbedder:
    name = "hsv"

    def batch(self, crops):
        return [None if c is None or c.size == 0 else _hsv(c) for c in crops]

    def __call__(self, bgr):
        return self.batch([bgr])[0]


class OsnetEmbedder:
    name = "osnet"
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, weights):
        import torch
        from osnet import load_osnet
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_osnet(weights).to(self.device)
        self.lock = threading.Lock()          # shared by all camera threads

    def _prep(self, bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (128, 256), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        return ((rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1)

    def batch(self, crops):
        idx = [i for i, c in enumerate(crops) if c is not None and c.size > 0]
        out = [None] * len(crops)
        if not idx:
            return out
        x = self.torch.from_numpy(np.stack([self._prep(crops[i]) for i in idx])).to(self.device)
        with self.lock, self.torch.inference_mode():
            f = self.model(x) + self.model(self.torch.flip(x, dims=[3]))   # flip averaging: steadier
        f = f.cpu().numpy()
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
        for j, i in enumerate(idx):
            out[i] = f[j].astype(np.float32)
        return out

    def __call__(self, bgr):
        return self.batch([bgr])[0]


class FusedEmbedder:
    """Concatenation weighted so cosine = w * osnet_cos + (1 - w) * hsv_cos."""
    name = "fused"

    def __init__(self, osnet, w=0.8):
        self.osnet, self.hsv, self.a, self.b = osnet, HsvEmbedder(), np.sqrt(w), np.sqrt(1 - w)

    def batch(self, crops):
        res = []
        for o, h in zip(self.osnet.batch(crops), self.hsv.batch(crops)):
            res.append(None if o is None or h is None else np.concatenate([self.a * o, self.b * h]).astype(np.float32))
        return res

    def __call__(self, bgr):
        return self.batch([bgr])[0]


_cached = None
_cache_lock = threading.Lock()


def get_embedder():
    """One shared instance for the whole app (the model is loaded once)."""
    global _cached
    with _cache_lock:
        if _cached is not None:
            return _cached
        want = config.EMBEDDER
        if want in ("osnet", "fused"):
            if not os.path.exists(config.OSNET_WEIGHTS):
                print(f"[embedding] WARNING: {config.OSNET_WEIGHTS} not found, using the weak 'hsv' "
                      f"embedder. Run: python download_reid.py")
                _cached = HsvEmbedder()
            else:
                os_emb = OsnetEmbedder(config.OSNET_WEIGHTS)
                _cached = os_emb if want == "osnet" else FusedEmbedder(os_emb, config.FUSED_OSNET_WEIGHT)
                print(f"[embedding] {_cached.name} loaded from {config.OSNET_WEIGHTS} on {os_emb.device}")
        else:
            _cached = HsvEmbedder()
        return _cached
