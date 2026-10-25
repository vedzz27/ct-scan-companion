"""Patient table + fusion logic. In memory only; embeddings and states, never images."""
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config
from matching import fuse_face


@dataclass
class Patient:
    id: str
    name: str
    created: float
    anchors: list = field(default_factory=list)                     # check-in views, never replaced
    emb: deque = field(default_factory=lambda: deque(maxlen=10))    # views learned while tracking
    faces: list = field(default_factory=list)                       # check-in face vectors (ArcFace)
    face_emb: deque = field(default_factory=lambda: deque(maxlen=10))  # faces learned while tracking
    zone: str = "unknown"
    camera: str = ""
    xy: tuple | None = None          # floor position in metres, if the camera is calibrated
    robot_target: str | None = None  # named stop point on the robot's map for this zone
    ambiguous: bool = False          # was close to someone else: tracker IDs may have swapped
    last_seen: float = 0.0
    raw_conf: float = 0.0
    attempts: int = 0
    blocked: bool = False
    confirmed: bool = False


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.patients: dict[str, Patient] = {}
        self.newcomers: dict[str, dict] = {}   # unknown people seen by tracking cameras
        self.pending: dict[str, float] = {}     # checked in at the front camera, body not seen yet -> deadline
        self.embedder_name = config.EMBEDDER      # set by the app to the loaded embedder (thresholds)

    # ---- newcomers (unknown people) ---------------------------------------
    def report_new(self, camera, track, zone, xy, robot_target, now=None, best_score=0.0):
        """A tracking camera sees someone who matches nobody. Keyed by camera + track id."""
        now = now or time.time()
        nid = f"N-{camera}-{track}"
        with self.lock:
            n = self.newcomers.get(nid)
            if n is None:
                n = self.newcomers[nid] = {"id": nid, "camera": camera, "track": track, "first_seen": now}
            n.update(zone=zone, xy=[round(v, 2) for v in xy] if xy else None, robot_target=robot_target,
                     last_seen=now, best_score=round(float(best_score), 3))
            return nid

    def clear_new(self, camera, track):
        with self.lock:
            self.newcomers.pop(f"N-{camera}-{track}", None)

    def list_new(self, now=None):
        """Newcomers still visible, oldest first."""
        now = now or time.time()
        with self.lock:
            for k in [k for k, n in self.newcomers.items() if now - n["last_seen"] > config.NEW_PERSON_STALE_S]:
                del self.newcomers[k]
            out = sorted((dict(n) for n in self.newcomers.values()), key=lambda n: n["first_seen"])
        for n in out:
            n["waiting_s"] = round(now - n["first_seen"], 1)
        return out

    def get_new(self, nid):
        with self.lock:
            n = self.newcomers.get(nid)
            return dict(n) if n else None

    # ---- lifecycle -------------------------------------------------------
    def check_in(self, name, embs, zone="check_in", faces=None):
        """embs: one vector or a list of vectors from the check-in burst. faces: optional
        ArcFace vectors collected at check-in (e.g. while the person talked to the robot)."""
        if isinstance(embs, np.ndarray) and embs.ndim == 1:
            embs = [embs]
        embs = [e for e in embs if e is not None]
        if not embs:
            raise ValueError("No usable picture of the person")
        with self.lock:
            if len(self.patients) >= config.MAX_ACTIVE_PATIENTS:
                raise ValueError(f"Gallery full ({config.MAX_ACTIVE_PATIENTS} active patients)")
            now = time.time()
            p = Patient(id="P-" + uuid.uuid4().hex[:4], name=name or "patient", created=now,
                        zone=zone, last_seen=now, raw_conf=1.0)
            p.anchors = list(embs)
            p.faces = [f for f in (faces or []) if f is not None]
            self.patients[p.id] = p
            return p

    def remove(self, pid):
        with self.lock:
            return self.patients.pop(pid, None) is not None

    def purge(self, now=None):
        """Delete everything older than the retention ceiling."""
        now = now or time.time()
        limit = config.RETENTION_HOURS * 3600
        with self.lock:
            old = [k for k, p in self.patients.items() if now - p.created > limit]
            for k in old:
                del self.patients[k]
            return len(old)

    # ---- confirmation ----------------------------------------------------
    def confirm(self, pid, accepted):
        with self.lock:
            p = self._get(pid)
            if p.blocked:
                return {"ok": False, "blocked": True, "attempts": p.attempts}
            if accepted:
                p.confirmed, p.attempts = True, 0
                p.ambiguous, p.raw_conf, p.last_seen = False, 1.0, time.time()   # robot saw them: proof
            else:
                p.attempts += 1
                if p.attempts >= config.MAX_CONFIRM_ATTEMPTS:
                    p.blocked = True
            return {"ok": accepted, "blocked": p.blocked, "attempts": p.attempts}

    # ---- vision side -----------------------------------------------------
    def scores(self, emb, face=None):
        """{patient_id: similarity}. Best of the check-in views and the learned views.
        With a face vector, the face comparison confirms or vetoes the body score (fuse_face)."""
        bind, keep, _ = config.thresholds(self.embedder_name)
        with self.lock:
            out = {}
            for p in self.patients.values():
                gallery = p.anchors + list(p.emb)
                if gallery:
                    out[p.id] = max(float(np.dot(emb, g)) for g in gallery)
                    fg = p.faces + list(p.face_emb)
                    if face is not None and fg:
                        out[p.id] = fuse_face(out[p.id], max(float(np.dot(face, g)) for g in fg), bind, keep)
            return out

    def face_scores(self, face):
        """{patient_id: face similarity} for patients with a known face."""
        with self.lock:
            return {p.id: max(float(np.dot(face, g)) for g in p.faces + list(p.face_emb))
                    for p in self.patients.values() if p.faces or p.face_emb}

    def add_pending(self, pid, now=None):
        with self.lock:
            self.pending[pid] = (now or time.time()) + config.HANDOFF_WAIT_S

    def pending_ids(self, now=None):
        now = now or time.time()
        with self.lock:
            for k in [k for k, t in self.pending.items() if t < now or k not in self.patients]:
                del self.pending[k]
            return list(self.pending)

    def link_body(self, pid, embs):
        """Back camera found the person checked in at the front camera: keep their whole-body views."""
        with self.lock:
            p = self.patients.get(pid)
            if p:
                p.anchors.extend(e for e in embs if e is not None)
            self.pending.pop(pid, None)

    def add_face(self, pid, face):
        with self.lock:
            p = self.patients.get(pid)
            if p:
                p.face_emb.append(face)

    def match(self, emb, face=None):
        """Best patient for one vector. Returns (pid, score, margin, all_scores)."""
        scores = self.scores(emb, face)
        if not scores:
            return None, 0.0, 0.0, {}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best = ranked[0]
        margin = best[1] - ranked[1][1] if len(ranked) > 1 else 1.0
        return best[0], best[1], margin, scores

    def add_embedding(self, pid, emb):
        with self.lock:
            p = self.patients.get(pid)
            if p:
                p.emb.append(emb)

    def observe(self, pid, zone, camera, conf, now=None, xy=None, ambiguous=False, robot_target=None):
        now = now or time.time()
        with self.lock:
            p = self.patients.get(pid)
            if not p:
                return
            if camera == p.camera or now - p.last_seen > 1.5 or conf >= p.raw_conf - 0.05:
                p.zone, p.camera, p.last_seen, p.raw_conf = zone, camera, now, conf
                p.xy, p.ambiguous, p.robot_target = xy, ambiguous, robot_target

    # ---- robot-facing view ----------------------------------------------
    def location(self, pid, now=None):
        now = now or time.time()
        with self.lock:
            p = self._get(pid)
            age = now - p.last_seen
            conf = p.raw_conf * 0.5 ** (age / config.CONFIDENCE_HALF_LIFE_S)
            if age > config.GIVE_UP_S or p.zone == "unknown":
                action = "alert_staff"
            elif conf >= 0.6 and age <= config.FRESH_S and not p.ambiguous:
                action = "go_to_zone_and_confirm"
            else:
                action = "go_to_last_zone_and_scan"
            present = age <= config.ABSENT_AFTER_S
            return {"id": p.id, "name": p.name, "zone": p.zone, "camera": p.camera,
                    "status": "present" if present else "absent",
                    "face_enrolled": bool(p.faces or p.face_emb),
                    "waiting_for_room_camera": p.id in self.pending,
                    "absent_since": None if present else round(p.last_seen, 1),
                    "xy": [round(v, 2) for v in p.xy] if p.xy else None,
                    "robot_target": p.robot_target,
                    "confidence": round(conf, 2), "age_s": round(age, 1), "ambiguous": p.ambiguous,
                    "may_use_name": action == "go_to_zone_and_confirm",
                    "action": action, "confirmed": p.confirmed, "attempts": p.attempts,
                    "blocked": p.blocked}

    def snapshot(self):
        with self.lock:
            ids = list(self.patients)
        return [self.location(i) for i in ids]

    def _get(self, pid):
        p = self.patients.get(pid)
        if not p:
            raise KeyError(pid)
        return p
