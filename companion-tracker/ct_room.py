"""CT room camera: who is on the table, and autonomous positioning guidance from their pose.

Use a pose model for this camera ("model": "yolo26n-pose.pt"): it detects, tracks and gives 17
body keypoints in one pass. Best mounted on the ceiling above the table, looking down.

Identity in the CT room comes from the journey: the patient the robot brought in. A lying person
can't be matched by appearance reliably, so if exactly one person is on the table, it's them;
if several are, guidance waits (and the dashboard shows why).
"""
import json
import os
import time

import cv2
import numpy as np

from positioning import KP
from worker import CameraWorker, FloorMap

CT_STAGES = ["to_ct_room", "in_ct_room", "on_table", "positioning", "positioned"]
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 5), (0, 6)]


class TableMap(FloorMap):
    """Image -> table coordinates (x from the gantry end along the table, y across, 0 = centre)."""

    def __init__(self, cal):
        super().__init__(cal)
        self.length = float(cal.get("table_length_m", 2.0))
        self.width = float(cal.get("table_width_m", 0.5))
        self.right_handed = bool(cal.get("right_handed", True))
        self.Hinv = np.linalg.inv(self.H)

    def kps_to_table(self, kp_px):
        """(17, 3) pixel keypoints -> {name: (x, y, conf)} in table metres."""
        out = {}
        for name, i in KP.items():
            u, v, c = kp_px[i]
            out[name] = (*self.to_floor(float(u), float(v)), float(c)) if c > 0.05 else None
        return out

    def on_table(self, kp_px, conf=0.4, margin=0.15):
        hips = [kp_px[KP[k]] for k in ("l_hip", "r_hip") if kp_px[KP[k]][2] >= conf]
        if not hips:
            return False
        x, y = self.to_floor(float(np.mean([h[0] for h in hips])), float(np.mean([h[1] for h in hips])))
        return -margin <= x <= self.length + margin and abs(y) <= self.width / 2 + margin

    def outline_px(self):
        """Table corners in the image (ignores lens distortion: for drawing only)."""
        L, W = self.length, self.width / 2
        pts = np.array([[[0, -W]], [[0, W]], [[L, W]], [[L, -W]]], np.float64)
        return cv2.perspectiveTransform(pts, self.Hinv).reshape(-1, 2).astype(np.int32)


def load_table(cfg):
    path = cfg.get("table_calibration")
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return TableMap(json.load(f))


class CtRoomWorker(CameraWorker):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.table = load_table(self.cfg)
        self.ct_status = {"patient": None, "on_table": 0, "note": "no table calibration" if not self.table else ""}
        self.target_tid = None
        self._last_state = {}

    def health(self):
        h = super().health()
        h["table_calibrated"] = self.table is not None
        h["ct"] = self.ct_status
        return h

    def after_process(self, frame, people, now):
        if not self.table or not self.journey or not self.positioning:
            return
        on_table = [tid for tid, k in self.kps.items() if self.table.on_table(k)]
        pid = self.journey.in_stage(CT_STAGES)
        self.ct_status = {"patient": pid, "on_table": len(on_table), "note": ""}
        target = None
        if pid:
            bound = [t for t in on_table if self.tracks.get(t) and self.tracks[t].pid == pid]
            if bound:
                target = bound[0]
            elif len(on_table) == 1:
                target = on_table[0]                    # the patient the robot brought in
            elif len(on_table) > 1:
                self.ct_status["note"] = "several people on the table area: guidance paused"
        self.target_tid = target
        if pid and target is not None:
            self.journey.observe_zone(pid, "ct_table", self.name, now)

        session = self.positioning.get(pid) if pid else None
        stage = self.journey.stage(pid) if pid else None
        if pid and stage == "on_table" and (session is None or session.state in ("stopped",)):
            v = self.journey.view(pid)
            needs = v["needs"]
            session = self.positioning.start(pid, v["protocol"], self.table.right_handed,
                                             needs.get("language", "en"), needs, now)
            self.journey.event(pid, "positioning_started", self.name, {"protocol": session.name}, now)
        if session is None or session.state in ("accepted", "stopped"):
            return
        kp = self.table.kps_to_table(self.kps[target]) if target is not None else None
        view = session.update(kp, now)
        prev = self._last_state.get(pid)
        if view["state"] != prev:
            self._last_state[pid] = view["state"]
            if view["state"] == "positioned":
                self.journey.event(pid, "positioned", self.name, {"measures": view["measures"]}, now)
            elif view["state"] == "needs_staff":
                self.journey.event(pid, "positioning_needs_staff", self.name, {"reason": view["reason"]}, now)
            elif view["state"] == "guiding" and prev == "positioned":
                self.journey.event(pid, "positioning_started", self.name, {"protocol": session.name,
                                                                            "note": "patient moved"}, now)

    def draw_extra(self, img):
        if not self.table:
            return
        cv2.polylines(img, [self.table.outline_px()], True, (255, 200, 0), 3)
        k = self.kps.get(self.target_tid) if self.target_tid is not None else None
        if k is not None:
            for a, b in SKELETON:
                if k[a][2] > 0.4 and k[b][2] > 0.4:
                    cv2.line(img, tuple(map(int, k[a][:2])), tuple(map(int, k[b][:2])), (0, 255, 255), 4)
        pid = self.ct_status.get("patient")
        s = self.positioning.get(pid) if pid and self.positioning else None
        if s and s.instr:
            text = f"{s.state}: {s.instr['key']}"
            h = img.shape[0]
            cv2.putText(img, text, (16, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 6)
            cv2.putText(img, text, (16, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 120, 255), 3)
