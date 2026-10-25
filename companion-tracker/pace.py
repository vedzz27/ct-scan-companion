"""Walk at the patient's pace: distance to the patient from the robot's rear camera -> speed command.

Distance: from where the feet touch the floor (camera height + tilt known), which needs no
assumption about the person. When the feet are out of view (patient close), from body height in
pixels, using the person's height learned while the feet were visible.
Speed: patient speed = robot speed - change of gap; command = patient speed corrected towards the
target gap, smoothed and limited. Too far behind -> stop and wait. Not seen -> stop, then alert.
"""
import math
import time
from collections import deque


class DistanceEstimator:
    def __init__(self, hfov_deg=90.0, cam_height_m=1.0, tilt_deg=10.0, person_height_m=1.65, edge_px=4):
        self.hfov, self.ch, self.tilt = hfov_deg, cam_height_m, tilt_deg
        self.default_h = person_height_m
        self.person_h = None            # learned per escort
        self.edge = edge_px

    def focal(self, width):
        return (width / 2) / math.tan(math.radians(self.hfov / 2))

    def reset(self):
        self.person_h = None

    def estimate(self, box, frame_shape):
        """Returns (distance_m or None, method)."""
        h_img, w_img = frame_shape[:2]
        f = self.focal(w_img)
        x1, y1, x2, y2 = box
        feet_ok = y2 < h_img - self.edge
        head_ok = y1 > self.edge
        d_feet = None
        if feet_ok:
            angle = math.radians(self.tilt) + math.atan((y2 - h_img / 2) / f)
            if angle > math.radians(1.5):
                d_feet = self.ch / math.tan(angle)
        if d_feet is not None and head_ok and 0.4 < d_feet < 8:
            # learn how tall this person is, for when the feet leave the picture
            h = d_feet * (y2 - y1) / f
            if 1.0 < h < 2.1:
                self.person_h = h if self.person_h is None else 0.85 * self.person_h + 0.15 * h
        if d_feet is not None:
            return d_feet, "feet"
        H = self.person_h or self.default_h
        if head_ok:
            if not feet_ok and H - self.ch > 0.2:
                # feet cut off (close): the top of the head is H - camera height above the lens
                up = math.atan((h_img / 2 - y1) / f) - math.radians(self.tilt)
                if up > math.radians(3):
                    return (H - self.ch) / math.tan(up), "head"
            if y2 - y1 > 10 and feet_ok:
                return f * H / (y2 - y1), "height"     # whole body visible but far away
        # head and feet both outside the picture: the person is closer than the distance at which
        # either would still be visible. Report that bound (conservative for the speed controller).
        vhalf = math.atan((h_img / 2) / f)
        t = math.radians(self.tilt)
        bounds = [self.ch / math.tan(vhalf + t)]
        if vhalf - t > math.radians(1) and H > self.ch:
            bounds.append((H - self.ch) / math.tan(vhalf - t))
        return round(min(bounds), 2), "closer_than"


class PaceController:
    def __init__(self, target_gap=1.2, wait_gap=3.0, resume_gap=1.8, v_max=0.8, v_go=0.15,
                 k=0.6, accel=0.4, lost_s=2.0, alert_s=15.0):
        self.target_gap, self.wait_gap, self.resume_gap = target_gap, wait_gap, resume_gap
        self.v_max, self.v_go, self.k, self.accel = v_max, v_go, k, accel
        self.lost_s, self.alert_s = lost_s, alert_s
        self.reset()

    def reset(self, now=None):
        self.state = "idle"
        self.v_cmd = 0.0
        self.gap = None
        self.hist = deque(maxlen=40)       # (t, distance, robot speed)
        self.last_seen = time.time() if now is None else now
        self.last_t = None
        self.patient_speed = None

    def apply_needs(self, needs):
        mob = (needs or {}).get("mobility")
        if mob in ("slow", "walker", "cane"):
            self.v_max, self.target_gap = 0.5, 1.0
        elif mob == "wheelchair":
            self.v_max, self.target_gap = 0.7, 1.5

    def update(self, distance, now=None, robot_speed=None):
        now = time.time() if now is None else now
        dt = 0.0 if self.last_t is None else min(0.5, now - self.last_t)
        self.last_t = now
        v_robot = self.v_cmd if robot_speed is None else robot_speed
        say = None

        if distance is None:
            unseen = now - self.last_seen
            if unseen > self.alert_s:
                self.state, target = "alert_staff", 0.0
            elif unseen > self.lost_s:
                if self.state != "lost":
                    say = "lost"
                self.state, target = "lost", 0.0
            else:
                target = self.v_cmd * 0.7            # brief dropout: ease off, don't stop dead
        else:
            if now - self.last_seen > self.lost_s:      # seen again after a gap: old readings are stale
                self.gap = None
                self.hist.clear()
            self.last_seen = now
            self.gap = distance if self.gap is None else 0.6 * self.gap + 0.4 * distance
            self.hist.append((now, distance, v_robot))
            win = [g for g in self.hist if now - g[0] <= 2.0]
            if len(win) >= 4 and win[-1][0] - win[0][0] > 0.5:
                ts = [g[0] for g in win]
                tm, dm = sum(ts) / len(ts), sum(g[1] for g in win) / len(win)
                rate = (sum((g[0] - tm) * (g[1] - dm) for g in win)
                        / max(1e-6, sum((x - tm) ** 2 for x in ts)))       # least-squares slope, + = gap growing
                ps = max(0.0, sum(g[2] for g in win) / len(win) - rate)   # robot's speed over the same window
                self.patient_speed = ps if self.patient_speed is None else 0.8 * self.patient_speed + 0.2 * ps
            ps = self.patient_speed if self.patient_speed is not None else 0.5
            if self.state == "following":
                if self.gap > self.wait_gap:
                    self.state, say = "waiting", "wait_for_you"
            elif self.gap <= (self.wait_gap if self.state == "idle" else self.resume_gap):
                say = "resume" if self.state in ("waiting", "lost", "alert_staff") else None
                self.state = "following"
            elif self.state != "waiting":
                self.state, say = "waiting", "wait_for_you"
            if self.state == "following":
                target = ps - self.k * (self.gap - self.target_gap)
                target = max(0.0, min(self.v_max, target))
            else:
                target = 0.0
        # smooth acceleration, but always allow an immediate stop
        if target < self.v_cmd and (target == 0.0 and self.state in ("waiting", "lost", "alert_staff")):
            self.v_cmd = 0.0
        else:
            step = self.accel * dt
            self.v_cmd += max(-step * 2, min(step, target - self.v_cmd))
        self.v_cmd = round(max(0.0, self.v_cmd), 3)
        return {"state": self.state, "speed_mps": self.v_cmd,
                "command": "go" if self.v_cmd >= self.v_go else "pause",
                "gap_m": round(self.gap, 2) if self.gap is not None else None,
                "patient_speed_mps": round(self.patient_speed, 2) if self.patient_speed is not None else None,
                "say": say}
