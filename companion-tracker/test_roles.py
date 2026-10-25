"""Integration tests for the CT room and robot camera workers with synthetic people:  python test_roles.py"""
import json
import math
import os
import tempfile

import cv2
import numpy as np

from calibrate_floor import right_handed, table_world
from ct_room import CtRoomWorker
from journey import Journey
from positioning import KP, PositioningManager
from robot_camera import RobotCameraWorker
from store import Store
from test_journey import body, project
from test_matching import ColourEmbedder, frame_with


def table_cal(tmp):
    """Overhead camera: table x -> image u, table y -> image v (as seen from above)."""
    W = table_world(2.0, 0.5)
    img = [(300 + 600 * x, 600 + 600 * y) for x, y in W]
    H, _ = cv2.findHomography(np.array(img, float), np.array(W, float))
    path = os.path.join(tmp, "table.json")
    json.dump({"homography": H.tolist(), "table_length_m": 2.0, "table_width_m": 0.5,
               "right_handed": right_handed(H)}, open(path, "w"))
    return path


def to_px(kp_table):
    arr = np.zeros((17, 3))
    for name, i in KP.items():
        x, y, c = kp_table[name]
        arr[i] = (300 + 600 * x, 600 + 600 * y, c)
    return arr


def test_ct_room_flow():
    tmp = tempfile.mkdtemp()
    s, j, pm = Store(), Journey(), PositioningManager()
    p = s.check_in("Ana", [np.ones(4, np.float32) / 2])
    j.register(p.id, {"language": "en"})
    j.event(p.id, "guide_start", "robot", {"target": "ct_room"})
    w = CtRoomWorker("ct", {"source": 0, "role": "ct_room", "table_calibration": table_cal(tmp)},
                     s, ColourEmbedder(), j, pm)
    assert w.table is not None and w.table.right_handed is False
    frame = np.zeros((1200, 1920, 3), np.uint8)
    # From above: head on screen-left, feet right, face up -> patient's LEFT is screen-up (-y).
    # Patient lies at +y (screen-down = their right side), so they must move to their LEFT.
    pose = dict(head_x=0.3, y=0.12, arms="up")
    t, said, last = 1000.0, [], 0
    for _ in range(200):
        t += 0.25
        w.kps = {7: to_px(body(**pose))}
        w.after_process(frame, [], t)
        sess = pm.get(p.id)
        if sess and sess.instr and sess.instr["id"] != last:
            last = sess.instr["id"]
            said.append(sess.instr["key"])
            if sess.instr["key"].startswith("move_left"):
                pose["y"] -= 0.06            # their left = -y in this camera view
            elif sess.instr["key"].startswith("move_right"):
                pose["y"] += 0.06
        if j.stage(p.id) == "positioned":
            break
    assert said[0].startswith("move_left"), said
    assert j.stage(p.id) == "positioned", (j.stage(p.id), said)
    events = [e["event"] for e in j.timeline(p.id)]
    assert events.count("positioning_started") == 1 and "positioned" in events
    # a second person climbs onto the table area: guidance pauses instead of guessing
    w.kps = {7: to_px(body(**pose)), 8: to_px(body(head_x=0.4, y=0.0))}
    w.after_process(frame, [], t + 1)
    assert w.target_tid is None and "several" in w.ct_status["note"]


def test_robot_pace_flow():
    s, j = Store(), Journey()
    emb = ColourEmbedder()
    f, _ = frame_with([(100, 100, 300, 600, 0, "red")])
    p = s.check_in("Ana", [emb(f[100:600, 100:300])])
    j.register(p.id, {"mobility": "slow"})
    cfg = {"source": 0, "role": "robot", "hfov_deg": 90, "camera_height_m": 1.0, "tilt_deg": 10}
    w = RobotCameraWorker("robot", cfg, s, emb, j, None)
    t = 500.0
    w.process(np.zeros((720, 1280, 3), np.uint8), [], None, t)
    assert w.pace["state"] == "idle"
    j.event(p.id, "guide_start", "robot", {"target": "ct_room"}, now=t)

    def scene(d_patient, d_other=None):
        (x1, y1, x2, y2), shape = project(d_patient, w.est)
        people = [(x1, y1, x2, y2, 1, "red")]
        if d_other:
            (a1, b1, a2, b2), _ = project(d_other, w.est)
            people.append((a1 + 300, b1, a2 + 300, b2, 2, "blue"))
        fr = np.full((720, 1280, 3), 128, np.uint8)
        lst = []
        for a, b, c, d, tid, col in people:
            fr[max(b, 0):min(d, 720), a:c] = {"red": (30, 30, 220), "blue": (220, 60, 30)}[col]
            lst.append((a, max(b, 0), c, min(d, 720), tid, 0.9))
        return fr, lst

    # patient walks behind at 1.3 m, a stranger closer at 1.0 m: follow the patient, not the stranger
    for k in range(40):
        t += 0.2
        fr, lst = scene(1.3, 1.0 if k > 10 else None)
        w.process(fr, lst, None, t)
    assert w.lock_tid == 1 and w.verified, (w.lock_tid, w.pace)
    assert w.pace["state"] == "following" and abs(w.pace["distance_m"] - 1.3) < 0.1, w.pace
    assert w.pace["speed_mps"] <= 0.5                     # "slow" mobility need caps the speed
    # patient falls far behind
    for k in range(10):
        t += 0.2
        fr, lst = scene(3.6)
        w.process(fr, lst, None, t)
    assert w.pace["state"] == "waiting" and w.pace["speed_mps"] == 0.0
    # patient out of view
    for k in range(20):
        t += 0.2
        w.process(np.full((720, 1280, 3), 128, np.uint8), [], None, t)
    assert w.pace["state"] == "lost" and "pace_lost" in [e["event"] for e in j.timeline(p.id)]
    # arrived: escort ends, controller idles
    j.event(p.id, "guide_arrived", "robot", now=t)
    w.process(np.zeros((720, 1280, 3), np.uint8), [], None, t + 0.2)
    assert w.pace["state"] == "idle"


if __name__ == "__main__":
    for fn in (test_ct_room_flow, test_robot_pace_flow):
        fn()
        print("ok ", fn.__name__)
    print("all role tests passed")
