"""Camera-independent detection logic: crop quality, track-to-patient assignment, hysteresis, swaps.
No OpenCV windows, no threads, no model: everything here is unit-tested in test_matching.py."""
import time
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

import config


# ---- geometry ----------------------------------------------------------------
def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a, b):
    i = inter(a, b)
    return i / (area(a) + area(b) - i + 1e-9)


def crop_quality(box, det_conf, others, frame_shape):
    """Return None if the crop is clean enough to describe the person, else the reason it isn't.
    Bad crops (cut off, hidden behind someone, tiny) are what poison appearance matching."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    m = config.EDGE_MARGIN_PX
    if y2 - y1 < config.MIN_CROP_HEIGHT:
        return "small"
    if det_conf < config.MIN_DET_CONF:
        return "low_conf"
    if x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m:
        return "edge"
    ratio = (y2 - y1) / max(1, x2 - x1)
    if not 0.9 <= ratio <= 5.0:   # seated people ~1.2-2, standing ~2-4
        return "shape"
    a = area(box)
    for o in others:
        if inter(box, o) / (a + 1e-9) > config.MAX_OVERLAP:
            return "occluded"
    return None


def close_pairs(boxes_by_tid, moving=None):
    """Pairs of tracks close enough that the tracker could swap their IDs. Neighbours sitting
    still next to each other don't swap, so a pair only counts if at least one of them moves."""
    items = list(boxes_by_tid.items())
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ta, a), (tb, b) = items[i], items[j]
            if iou(a, b) > config.CLOSE_IOU and (moving is None or ta in moving or tb in moving):
                out.append((ta, tb))
    return out


# ---- per-track state ---------------------------------------------------------
class TrackState:
    def __init__(self, now=None):
        now = now or time.time()
        self.embs = deque(maxlen=6)
        self.faces = deque(maxlen=6)    # ArcFace vectors seen on this track (moving or not)
        self.face_px = 0                # width of the last usable face
        self.last_face = float("-inf")
        self.faces_reset_at = 0.0       # face results from pictures older than this are dropped
        self.pid = None            # bound patient
        self.score = 0.0           # latest score for the bound patient
        self.conf = 0.0
        self.cand, self.cand_streak = None, 0   # candidate waiting to be bound / switched to
        self.bad_streak = 0
        self.recheck = False       # set after a possible ID swap: decide on the next evaluation
        self.ambiguous_until = 0.0
        self.last_eval = 0.0
        self.last_embed = float("-inf")
        self.last_seen = now
        self.unbound_since = None  # when this track started being nobody (newcomer detection)
        self.quality = ""
        self.debug = {}
        self.path = deque(maxlen=12)   # (t, foot_x, foot_y, box_height) for movement detection

    def note_position(self, box, now):
        self.path.append((now, (box[0] + box[2]) / 2, box[3], box[3] - box[1]))

    def moving(self, now, window=1.5):
        """Feet moved more than MOVE_FRACTION of the body height within the last `window` s."""
        old = [p for p in self.path if now - p[0] <= window]
        if len(old) < 2:
            return False
        (_, x0, y0, h0), (_, x1, y1, _) = old[0], old[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > config.MOVE_FRACTION * max(h0, 1)

    def ambiguous(self, now):
        return now < self.ambiguous_until

    def mean_emb(self):
        m = np.mean(np.stack(self.embs), axis=0)
        return m / (np.linalg.norm(m) + 1e-9)

    def mark_close(self, now):
        """Another person is touching this one: the tracker may swap IDs, so forget the
        appearance collected so far and re-decide from clean crops after they separate."""
        self.ambiguous_until = now + config.AMBIGUOUS_HOLD_S
        self.embs.clear()
        self.faces.clear()
        self.faces_reset_at = now
        self.last_embed = float("-inf")
        if self.pid:
            self.recheck = True


def fuse_face(body, face_sim, bind, keep):
    """Combine the body score with a face comparison, on the body score's scale so all
    thresholds stay the same. No face on either side -> body score unchanged.
      face clearly the same      -> at least bind + 0.1 (recognised even after taking off a jacket)
      face clearly different     -> at most keep - 0.01 (same clothes can't fool it)
      in between (small/unclear) -> body score unchanged"""
    if face_sim is None:
        return body
    if face_sim >= config.FACE_SAME_T:
        return max(body, bind + 0.1)
    if face_sim <= config.FACE_DIFF_T:
        return min(body, keep - 0.01)
    return body


def assign(score_rows, keep):
    """score_rows: {tid: {pid: score}}. One patient per track and one track per patient
    (Hungarian on the score matrix). Pairs below `keep` are left unassigned."""
    tids = list(score_rows)
    pids = sorted({p for r in score_rows.values() for p in r})
    if not tids or not pids:
        return {}
    S = np.full((len(tids), len(pids)), -1.0)
    for i, t in enumerate(tids):
        for j, p in enumerate(pids):
            S[i, j] = score_rows[t].get(p, -1.0)
    rows, cols = linear_sum_assignment(-S)
    return {tids[i]: (pids[j], float(S[i, j])) for i, j in zip(rows, cols) if S[i, j] >= keep}


def decide(ts, assigned, row, bind, keep, margin, now):
    """Update one track's binding from this round's assignment, with hysteresis.
    Returns 'bind', 'switch', 'unbind', 'keep' or 'none' (for logging)."""
    pid, score = assigned if assigned else (None, 0.0)
    others = [s for p, s in row.items() if p != pid]
    m = score - (max(others) if others else 0.0)
    strong = pid is not None and score >= bind and m >= margin
    fast = ts.recheck                     # after a possible swap, don't wait for streaks
    ts.recheck = False
    ts.debug = {"assigned": pid, "score": round(score, 3), "margin": round(m, 3)}

    if ts.pid is None:
        if strong:
            ts.cand_streak = ts.cand_streak + 1 if ts.cand == pid else 1
            ts.cand = pid
            if fast or ts.cand_streak >= config.BIND_STREAK:
                ts.pid, ts.score, ts.bad_streak, ts.cand, ts.cand_streak = pid, score, 0, None, 0
                return "bind"
        else:
            ts.cand, ts.cand_streak = None, 0
        return "none"

    if pid == ts.pid and score >= keep:
        ts.score, ts.bad_streak, ts.cand, ts.cand_streak = score, 0, None, 0
        return "keep"

    if strong:   # someone else fits clearly better
        ts.cand_streak = ts.cand_streak + 1 if ts.cand == pid else 1
        ts.cand = pid
        if fast or ts.cand_streak >= config.BIND_STREAK:
            ts.pid, ts.score, ts.bad_streak, ts.cand, ts.cand_streak = pid, score, 0, None, 0
            return "switch"
    ts.bad_streak += 1
    if fast or ts.bad_streak >= config.UNBIND_STREAK:
        ts.pid, ts.score, ts.bad_streak = None, 0.0, 0
        return "unbind"
    return "keep"


def confidence(score, keep, ambiguous):
    c = 0.5 + 0.5 * (score - keep) / max(1e-6, 1 - keep)
    c = float(np.clip(c, 0.3, 1.0))
    return round(c * (0.5 if ambiguous else 1.0), 3)
