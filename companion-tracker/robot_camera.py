"""Robot rear camera: keep the escorted patient in view and walk at their pace.

Mount a camera on the robot looking backwards (USB camera to a laptop/mini-PC on the tray, or a
phone streaming over Wi-Fi). Set its height and tilt in cameras.json; distance comes from them.
The robot app polls GET /api/robot/pace (~5x per second) and applies speed_mps (or just go/pause).

Which person to follow: the one whose appearance matches the escorted patient (same re-ID gallery
as the other cameras). If nobody matches but exactly one person is close behind, follow them as
"unverified". Someone else walking past never takes over a confirmed lock.
"""
import time

import cv2

import config
import matching
from embedding import apply_gains
from pace import DistanceEstimator, PaceController
from worker import CameraWorker

PHRASES = {
    "en": {"wait_for_you": "Take your time, I'll wait for you.", "lost": "I can't see you. I'll wait here.",
           "resume": "Let's continue.", "alert_staff": "I'll ask a colleague to help."},
    "de": {"wait_for_you": "Lassen Sie sich Zeit, ich warte auf Sie.", "lost": "Ich sehe Sie nicht. Ich warte hier.",
           "resume": "Gehen wir weiter.", "alert_staff": "Ich hole eine Kollegin zur Hilfe."},
}


class RobotCameraWorker(CameraWorker):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        c = self.cfg
        self.est = DistanceEstimator(c.get("hfov_deg", 90), c.get("camera_height_m", 1.0), c.get("tilt_deg", 0),
                                     c.get("person_height_m", 1.65))
        self.ctrl_args = dict(target_gap=c.get("target_gap_m", 1.2), wait_gap=c.get("wait_gap_m", 3.0),
                              resume_gap=c.get("resume_gap_m", 1.8), v_max=c.get("max_speed_mps", 0.8))
        self.ctrl = PaceController(**self.ctrl_args)
        self.pid, self.lang = None, "en"
        self.lock_tid, self.verified, self.bad, self.lock_seen = None, False, 0, 0.0
        self.last_score = {}           # tid -> (time, score)
        self.rejected = set()          # tracks that failed the appearance check
        self.robot_speed = None        # from POST /api/robot/telemetry, if the robot reports it
        self.pace = {"state": "idle"}
        self._last_state = None

    def health(self):
        h = super().health()
        h["pace"] = self.pace
        return h

    def _reset(self, pid, now):
        self.pid = pid
        self.lock_tid, self.verified, self.bad, self.last_score, self.lock_seen = None, False, 0, {}, now
        self.rejected = set()
        self.ctrl = PaceController(**self.ctrl_args)
        self.ctrl.reset(now)
        self.est.reset()
        if pid and self.journey:
            needs = self.journey.needs(pid)
            self.ctrl.apply_needs(needs)
            self.lang = needs.get("language", "en") if needs.get("language") in PHRASES else "en"

    def process(self, frame, people, gains, now):
        pid = self.journey.escorted() if self.journey else None
        if pid != self.pid:
            self._reset(pid, now)
        if not pid:
            self.pace = {"state": "idle", "patient": None, "speed_mps": 0.0, "command": "pause"}
            return
        tracked = {p[4]: p for p in people if p[4] >= 0}

        # appearance check of visible people (every 0.6 s per person, clean crops only)
        crops, tids = [], []
        for tid, (x1, y1, x2, y2, _, dc) in tracked.items():
            if now - self.last_score.get(tid, (0, 0))[0] < 0.6:
                continue
            others = [p[:4] for t, p in tracked.items() if t != tid]
            # close behind the robot, people are often cut off at the top/bottom: that's fine here,
            # only small, unsure or overlapped crops are skipped
            area = matching.area((x1, y1, x2, y2)) + 1e-9
            clean = (y2 - y1 >= config.MIN_CROP_HEIGHT and dc >= config.MIN_DET_CONF
                     and all(matching.inter((x1, y1, x2, y2), o) / area <= config.MAX_OVERLAP for o in others))
            if clean:
                crops.append(apply_gains(frame[y1:y2, x1:x2], gains))
                tids.append(tid)
        for tid, e in zip(tids, self.embedder.batch(crops) if crops else []):
            if e is not None:
                self.last_score[tid] = (now, self.store.scores(e).get(pid, 0.0))

        # who are we following?
        if self.lock_tid in tracked:
            self.lock_seen = now
            t_s, s = self.last_score.get(self.lock_tid, (0, None))
            if s is not None and t_s == now:              # scored this round
                if s >= self.keep_t:
                    self.verified, self.bad = True, 0
                else:
                    self.bad += 1
                    if self.bad >= 3:                     # it's not them after all
                        self.rejected.add(self.lock_tid)
                        self.lock_tid, self.verified, self.bad = None, False, 0
        elif self.lock_tid is not None and now - self.lock_seen > 3:
            self.lock_tid, self.verified, self.bad = None, False, 0
        fresh = {t: s for t, (ts, s) in self.last_score.items() if t in tracked and now - ts < 2}
        best = max(fresh.items(), key=lambda kv: kv[1], default=(None, 0))
        if not self.verified and best[0] is not None and best[1] >= self.bind_t:
            self.lock_tid, self.verified, self.bad, self.lock_seen = best[0], True, 0, now
        if self.lock_tid is None and len(tracked) == 1 and next(iter(tracked)) not in self.rejected:
            only = next(iter(tracked))
            d, _ = self.est.estimate(tracked[only][:4], frame.shape)
            if d is not None and d < 3.5:
                self.lock_tid, self.verified, self.lock_seen = only, False, now   # follow, flagged unconfirmed

        distance, method = None, None
        if self.lock_tid in tracked:
            distance, method = self.est.estimate(tracked[self.lock_tid][:4], frame.shape)
        out = self.ctrl.update(distance, now, self.robot_speed)
        out.update(patient=pid, track=self.lock_tid, verified=self.verified, distance_m=
                   round(distance, 2) if distance is not None else None, method=method,
                   say_text=PHRASES[self.lang].get(out["say"]) if out["say"] else None)
        if out["state"] == "alert_staff" and self._last_state != "alert_staff":
            out["say_text"] = PHRASES[self.lang]["alert_staff"]
        self.pace = out
        if out["state"] != self._last_state:
            self.journey.event(pid, f"pace_{out['state']}", self.name, {"gap_m": out["gap_m"]}, now)
            self._last_state = out["state"]

    def draw_extra(self, img):
        p = self.pace
        if self.lock_tid is not None:
            for x1, y1, x2, y2, tid, _ in self.people:
                if tid == self.lock_tid:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0) if self.verified else (0, 200, 255), 8)
        text = f"{p.get('state')}  gap {p.get('gap_m')} m  v {p.get('speed_mps')} m/s"
        h = img.shape[0]
        cv2.putText(img, text, (16, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 6)
        cv2.putText(img, text, (16, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 120, 0), 3)
