"""Autonomous table-positioning guidance from body keypoints (camera-agnostic, unit-tested).

Loop: measure the pose in table coordinates -> compare with the protocol -> give ONE instruction
-> wait for the patient to move -> measure again. When everything is within tolerance for
`stable_s`, the state becomes "positioned" and waits for the technologist to accept it.
The system guides; it never moves the table or starts the scan. No progress, an unreadable
pose, or a patient who needs physical help -> "needs_staff" and coaching stops.
"""
import json
import math
import threading
import time
from collections import deque

import numpy as np

# COCO-17 keypoint indices (YOLO pose)
KP = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4, "l_shoulder": 5, "r_shoulder": 6,
      "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
      "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16}
HEAD = ("nose", "l_eye", "r_eye", "l_ear", "r_ear")


def load_protocols(path="protocols.json"):
    with open(path) as f:
        return json.load(f)


def _mean(pts):
    pts = [p for p in pts if p is not None]
    return np.mean(pts, axis=0) if pts else None


def evaluate(kp, proto, timing, right_handed=True, scale=1.0):
    """kp: {name: (x, y, conf)} in table metres. Returns (readable, checks, measures).
    checks: ordered list of (name, ok, instruction_key or None); the first failing one is said."""
    c = timing["min_kp_conf"]
    g = {k: np.array(v[:2], float) if v is not None and v[2] >= c else None for k, v in kp.items()}
    sh = _mean([g.get("l_shoulder"), g.get("r_shoulder")])
    hp = _mean([g.get("l_hip"), g.get("r_hip")])
    if sh is None or hp is None:
        return False, [], {}
    head = _mean([g.get(k) for k in HEAD])
    feet = _mean([g.get("l_ankle"), g.get("r_ankle")])
    axis = hp - sh                                  # points from head end to feet end of the body
    length = np.linalg.norm(axis)
    if length < 0.15:                               # shoulders on top of hips: sitting up / not lying
        return True, [("lying", False, "lie_down")], {"torso_m": round(float(length), 2)}
    a = axis / length
    h = -a                                          # unit vector towards the head
    tol = {k: v * scale for k, v in proto["tol"].items() if isinstance(v, (int, float))}
    checks, m = [], {}

    # 1) head or feet towards the gantry (gantry end is x = 0)
    head_first = sh[0] < hp[0]
    want = proto.get("head_toward_gantry", True)
    checks.append(("orientation", bool(head_first == want), "turn_around"))

    # 2) body straight along the table
    ang = math.degrees(math.atan2(abs(a[1]), abs(a[0])))
    m["angle_deg"] = round(ang, 1)
    checks.append(("straight", ang <= tol["angle_deg"], "straighten"))

    # 3) facing up (overhead camera sees the face) — heuristic, only if the protocol wants it
    if proto.get("facing") == "up":
        face = any(kp.get(k) is not None and kp[k][2] >= c for k in ("nose", "l_eye", "r_eye"))
        checks.append(("facing", face, "lie_on_back"))

    # 4) centred across the table, instruction in the patient's own left/right
    mid_y = float((sh[1] + hp[1]) / 2)
    m["lateral_m"] = round(mid_y, 3)
    left = np.array([h[1], -h[0]])                  # patient's left for a supine patient, right-handed table frame
    if proto.get("facing") == "down":
        left = -left
    if not right_handed:
        left[1] = -left[1]
    move = np.array([0.0, -np.sign(mid_y)])
    side = "left" if float(np.dot(move, left)) > 0 else "right"
    size = "_small" if abs(mid_y) < 2.5 * proto["tol"]["lateral_m"] else ""
    checks.append(("centred", abs(mid_y) <= tol["lateral_m"], f"move_{side}{size}"))

    # 5) along the table: head (or feet) inside the protocol's range
    ref = head if proto.get("ref_point", "head") == "head" else feet
    if ref is None and proto.get("ref_point", "head") == "head":
        ref = sh + h * 0.25                          # head ~25 cm beyond the shoulders
    if ref is not None:
        lo, hi = proto["ref_x_m"]
        slack = (scale - 1) * 0.05
        m["ref_x_m"] = round(float(ref[0]), 3)
        if ref[0] > hi + slack:
            checks.append(("along", False, "slide_toward_scanner"))
        elif ref[0] < lo - slack:
            checks.append(("along", False, "slide_away"))
        else:
            checks.append(("along", True, None))

    # 6) arms
    wrists = [g.get("l_wrist"), g.get("r_wrist")]
    seen = [w for w in wrists if w is not None]
    if proto.get("arms") and seen:
        reach = [float(np.dot(w - sh, h)) for w in seen]   # + = towards the head
        head_reach = float(np.dot((head if head is not None else sh + h * 0.25) - sh, h))
        m["wrist_reach_m"] = [round(r, 2) for r in reach]
        if proto["arms"] == "up":
            ok = all(r >= head_reach - tol.get("arms_margin_m", 0.0) for r in reach) and len(seen) == 2
            checks.append(("arms", ok, "raise_arms"))
        else:
            ok = all(r <= -tol.get("arms_margin_m", 0.15) for r in reach)
            checks.append(("arms", ok, "arms_down"))
    elif proto.get("arms"):
        m["arms"] = "not visible"
    return True, checks, m


class PositioningSession:
    def __init__(self, pid, protocol_name, cfg, right_handed=True, lang="en", needs=None, now=None):
        now = time.time() if now is None else now
        self.pid, self.name, self.cfg = pid, protocol_name, cfg
        self.proto = cfg["protocols"][protocol_name]
        self.t = cfg["timing"]
        self.right_handed = right_handed
        self.lang = lang if lang in cfg["phrases"] else "en"
        self.state = "waiting_for_patient"
        self.reason = None
        self.instr = None                 # {"id", "key", "text", "t"}
        self.instr_id = 0
        self.repeats = 0
        self.hold_since = None
        self.bad_since = None
        self.unreadable_since = now
        self.started = now
        self.positioned_at = None
        self.checks, self.measures = [], {}
        self.buf = deque(maxlen=6)        # recent keypoint frames, median-smoothed
        self.lock = threading.Lock()
        needs = needs or {}
        if needs.get("mobility") in ("wheelchair", "assist", "stretcher"):
            self._to_staff("assistance_needed", now)   # coaching is not enough: call a person

    # ---- helpers -------------------------------------------------------------
    def _text(self, key):
        end = "feet" if self.proto.get("head_toward_gantry", True) is False else "head"
        words = self.cfg.get("end_words", {}).get(self.lang, {})
        return self.cfg["phrases"][self.lang][key].replace("{end}", words.get(end, end))

    def _say(self, key, now):
        self.instr_id += 1
        self.instr = {"id": self.instr_id, "key": key, "text": self._text(key), "t": now}

    def _to_staff(self, reason, now):
        self.state, self.reason = "needs_staff", reason
        self._say("staff", now)

    def _smoothed(self):
        out = {}
        for name in KP:
            obs = [f[name] for f in self.buf if f.get(name) is not None and f[name][2] >= self.t["min_kp_conf"]]
            if len(obs) >= max(1, len(self.buf) // 2):
                arr = np.array(obs)
                out[name] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), float(np.median(arr[:, 2])))
            else:
                out[name] = None
        return out

    # ---- main step -----------------------------------------------------------
    def update(self, kp, now=None):
        """kp: {name: (x, y, conf)} in table metres, or None if nobody is on the table."""
        now = time.time() if now is None else now
        with self.lock:
            if self.state in ("accepted", "needs_staff", "stopped"):
                return self.view(now, locked=True)
            if kp is not None:
                self.buf.append(kp)
            sm = self._smoothed() if self.buf and kp is not None else None
            calm = self.state in ("holding", "positioned")
            readable, checks, m = (evaluate(sm, self.proto, self.t, self.right_handed,
                                            self.t["hysteresis"] if calm else 1.0)
                                   if sm else (False, [], {}))
            self.checks, self.measures = checks, m
            if not readable:
                if self.state == "positioned":
                    self.state = "guiding"
                if now - self.unreadable_since > self.t["unreadable_s"]:
                    self._to_staff("pose_unreadable", now)
                elif self.instr is None and now - self.started > 2:
                    self._say("lie_down", now)
                return self.view(now, locked=True)
            self.unreadable_since = now
            failing = [c for c in checks if not c[1]]

            if self.state == "positioned":
                if failing:   # patient moved after being positioned (before the technologist accepted)
                    self.bad_since = self.bad_since or now
                    if now - self.bad_since >= self.t["moved_s"]:
                        self.state, self.hold_since, self.bad_since, self.positioned_at = "guiding", None, None, None
                else:
                    self.bad_since = None
                return self.view(now, locked=True)

            if not failing:
                if self.hold_since is None:
                    self.hold_since = now
                    self.state = "holding"
                    self._say("hold_still", now)
                elif now - self.hold_since >= self.t["stable_s"]:
                    self.state, self.positioned_at, self.repeats = "positioned", now, 0
                    self._say("positioned", now)
                return self.view(now, locked=True)

            self.hold_since = None
            self.state = "guiding"
            key = failing[0][2]
            since = now - self.instr["t"] if self.instr else 1e9
            if self.instr is None or (key != self.instr["key"] and since >= self.t["min_gap_s"]):
                self.repeats = 0 if (self.instr is None or key != self.instr["key"]) else self.repeats
                self._say(key, now)
            elif key == self.instr["key"] and since >= self.t["settle_s"]:
                self.repeats += 1
                if self.repeats >= self.t["max_repeats"]:
                    self._to_staff(f"no_progress:{key}", now)
                else:
                    self._say(key, now)
            return self.view(now, locked=True)

    # ---- staff ----------------------------------------------------------------
    def accept(self, now=None):
        with self.lock:
            if self.state != "positioned":
                return False
            self.state = "accepted"
            return True

    def reject(self, now=None):
        """Technologist says 'not right yet': coaching continues from scratch."""
        now = time.time() if now is None else now
        with self.lock:
            self.state, self.repeats, self.hold_since, self.positioned_at = "guiding", 0, None, None
            self.unreadable_since = now
            return True

    def stop(self):
        with self.lock:
            self.state = "stopped"

    def view(self, now=None, locked=False):
        now = time.time() if now is None else now
        return {"patient": self.pid, "protocol": self.name, "label": self.proto.get("label"),
                "state": self.state, "reason": self.reason,
                "instruction": {k: v for k, v in self.instr.items() if k != "t"} if self.instr else None,
                "checks": [{"check": n, "ok": bool(ok)} for n, ok, _ in self.checks],
                "measures": self.measures, "elapsed_s": round(now - self.started, 1),
                "positioned_for_s": round(now - self.positioned_at, 1) if self.positioned_at else None}


class PositioningManager:
    """All sessions, shared by the CT camera (updates) and the API (robot + technologist)."""

    def __init__(self, path="protocols.json"):
        self.cfg = load_protocols(path)
        self.sessions = {}
        self.lock = threading.Lock()

    def start(self, pid, protocol=None, right_handed=True, lang="en", needs=None, now=None):
        name = protocol or self.cfg["default"]
        if name not in self.cfg["protocols"]:
            raise KeyError(f"Unknown protocol '{name}'")
        s = PositioningSession(pid, name, self.cfg, right_handed, lang, needs, now)
        with self.lock:
            self.sessions[pid] = s
        return s

    def get(self, pid):
        with self.lock:
            return self.sessions.get(pid)

    def active(self):
        with self.lock:
            return [s for s in self.sessions.values() if s.state not in ("accepted", "stopped")]

    def remove(self, pid):
        with self.lock:
            self.sessions.pop(pid, None)
