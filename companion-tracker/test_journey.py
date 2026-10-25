"""Tests for journey, positioning and pace logic, no cameras:  python test_journey.py"""
import math
import os
import tempfile

import numpy as np

from journey import Journey
from pace import DistanceEstimator, PaceController
from positioning import PositioningManager, evaluate, load_protocols

CFG = load_protocols()


# ---- journey -------------------------------------------------------------------
def test_journey_flow():
    log = os.path.join(tempfile.mkdtemp(), "log.jsonl")
    j = Journey(log_path=log)
    t = 1000.0
    j.register("P-1", {"language": "de"}, now=t)
    assert j.stage("P-1") == "checked_in"
    for k in range(4):                                              # needs 2 s of sightings
        j.observe_zone("P-1", "waiting_A", "waiting", t + 5 + 0.5 * k)
    assert j.stage("P-1") == "checked_in"
    j.observe_zone("P-1", "ct_table", "ct", t + 6.6)                # one stray sighting elsewhere...
    j.observe_zone("P-1", "waiting_A", "waiting", t + 7.0)          # ...doesn't reset the waiting timer
    assert j.stage("P-1") == "waiting"
    j.event("P-1", "qr_verified", "robot", now=t + 60)               # logged, stage unchanged
    v = j.event("P-1", "call", "robot", now=t + 61)
    assert v["robot_task"]["task"] == "guide_to" and v["robot_task"]["target"] == "changing_room"
    j.event("P-1", "guide_start", "robot", {"target": "changing_room"}, now=t + 62)
    assert j.escorted() == "P-1" and j.stage("P-1") == "to_changing_room"
    j.event("P-1", "guide_arrived", "robot", now=t + 120)
    assert j.stage("P-1") == "changing" and j.escorted() is None
    j.event("P-1", "changing_ready", "tablet", now=t + 300)
    j.event("P-1", "guide_start", "robot", {"target": "ct_room"}, now=t + 301)
    # walking past the waiting-room camera does NOT move the journey backwards
    for k in range(8):
        j.observe_zone("P-1", "waiting_B", "waiting", t + 310 + 0.5 * k)
    assert j.stage("P-1") == "to_ct_room"
    # CT camera sees them at the door: arrival recognised by camera, escort ends
    for k in range(6):
        j.observe_zone("P-1", "ct_door", "ct", t + 340 + 0.5 * k)
    assert j.stage("P-1") == "in_ct_room" and j.escorted() is None
    assert any(e["event"] == "guide_arrived" and e["source"] == "ct" for e in j.timeline("P-1"))
    for k in range(6):
        j.observe_zone("P-1", "ct_table", "ct", t + 360 + 0.5 * k)
    assert j.view("P-1")["robot_task"]["task"] == "position_check"
    j.event("P-1", "positioning_started", "ct", {"protocol": "head_first_supine_arms_up"}, now=t + 365)
    j.event("P-1", "positioned", "ct", now=t + 400)
    v = j.event("P-1", "positioning_accepted", "technologist", now=t + 410)
    assert v["robot_task"]["task"] == "leave_room"
    j.event("P-1", "robot_left", "robot", now=t + 420)
    j.event("P-1", "scan_started", "workflow", now=t + 430)
    assert j.view("P-1")["robot_task"] is None                       # robot is outside
    v = j.event("P-1", "scan_finished", "workflow", now=t + 600)
    assert v["robot_task"]["task"] == "aftercare"
    d = j.durations("P-1", now=t + 700)
    assert d["waiting"] > 50 and d["positioning"] == 35.0, d
    assert j.metrics(now=t + 700)["mean_s"]["changing"] == 180.0
    assert "Ana" not in open(log).read()                              # no names in the log
    j.remove("P-1")
    assert open(log).read() == "" and not j.timeline("P-1")


# ---- positioning -----------------------------------------------------------------
def body(head_x=0.3, y=0.0, angle=0.0, arms="up", head_first=True, conf=0.9, face=True):
    """Keypoints in table metres for a supine patient; x from the gantry end."""
    s = 1 if head_first else -1
    a = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))]) * s   # head -> feet
    perp = np.array([-a[1], a[0]])
    head = np.array([head_x if head_first else 2.0 - head_x, y])
    sh = head + a * 0.25
    hp = sh + a * 0.5
    an = hp + a * 0.85
    kp = {}
    put = lambda n, p, c=conf: kp.__setitem__(n, (float(p[0]), float(p[1]), c))
    put("nose", head, conf if face else 0.1)
    put("l_eye", head + perp * 0.03, conf if face else 0.1)
    put("r_eye", head - perp * 0.03, conf if face else 0.1)
    put("l_ear", head + perp * 0.07, 0.5)
    put("r_ear", head - perp * 0.07, 0.5)
    for side, sg in (("l", 1), ("r", -1)):
        put(f"{side}_shoulder", sh + perp * 0.2 * sg)
        put(f"{side}_hip", hp + perp * 0.15 * sg)
        put(f"{side}_knee", hp + a * 0.45 + perp * 0.1 * sg)
        put(f"{side}_ankle", an + perp * 0.1 * sg)
        if arms == "up":
            put(f"{side}_elbow", sh - a * 0.2 + perp * 0.15 * sg)
            put(f"{side}_wrist", head - a * 0.15 + perp * 0.08 * sg)
        else:
            put(f"{side}_elbow", sh + a * 0.3 + perp * 0.25 * sg)
            put(f"{side}_wrist", hp + perp * 0.27 * sg)
    return kp


def first_fail(kp, proto="head_first_supine_arms_up"):
    ok, checks, _ = evaluate(kp, CFG["protocols"][proto], CFG["timing"])
    bad = [c for c in checks if not c[1]]
    return bad[0][2] if bad else None


def test_evaluate():
    assert first_fail(body()) is None
    assert first_fail(body(head_first=False)) == "turn_around"
    assert first_fail(body(angle=15)) == "straighten"
    # patient lying towards their own left (+y, head-first supine): must move to their right
    assert first_fail(body(y=0.08)) == "move_right_small"
    assert first_fail(body(y=-0.2)) == "move_left"
    assert first_fail(body(head_x=0.7)) == "slide_toward_scanner"
    assert first_fail(body(arms="down")) == "raise_arms"
    assert first_fail(body(arms="up"), "head_first_supine_arms_down") == "arms_down"
    assert first_fail(body(face=False)) == "lie_on_back"
    assert first_fail(body(head_x=0.1, head_first=False), "feet_first_supine_arms_up") is None


def test_session_guides_to_positioned():
    pm = PositioningManager()
    s = pm.start("P-1", now=0.0)
    pose = dict(head_x=0.75, y=-0.15, angle=0.0, arms="down")
    said, t, last_id = [], 0.0, 0
    for _ in range(400):
        t += 0.25
        v = s.update(body(**pose), t)
        ins = v["instruction"]
        if ins and ins["id"] != last_id:
            last_id = ins["id"]
            said.append(ins["key"])
            # the patient does what they're told, a bit at a time
            k = ins["key"]
            if k.startswith("move_left"):
                pose["y"] += 0.08
            elif k.startswith("move_right"):
                pose["y"] -= 0.08
            elif k == "slide_toward_scanner":
                pose["head_x"] -= 0.25
            elif k == "slide_away":
                pose["head_x"] += 0.1
            elif k == "raise_arms":
                pose["arms"] = "up"
        if v["state"] == "positioned":
            break
    assert v["state"] == "positioned", (v, said)
    assert said[0] == "move_left" and "raise_arms" in said and said[-1] == "positioned", said
    # patient moves before the technologist accepts -> back to guiding
    for _ in range(10):
        t += 0.25
        v = s.update(body(**dict(pose, y=0.2)), t)
    assert v["state"] == "guiding"


def test_session_staff_cases():
    pm = PositioningManager()
    s = pm.start("P-2", now=0.0)          # patient ignores the instruction -> staff
    t = 0.0
    for _ in range(200):
        t += 0.25
        v = s.update(body(y=0.2), t)
    assert v["state"] == "needs_staff" and v["reason"] == "no_progress:move_right", v
    s = pm.start("P-3", now=0.0)          # nobody readable on the table -> staff
    for k in range(50):
        v = s.update(None, k * 0.25)
    assert v["state"] == "needs_staff" and v["reason"] == "pose_unreadable"
    s = pm.start("P-4", needs={"mobility": "wheelchair"}, now=0.0)
    assert s.view()["state"] == "needs_staff" and s.view()["reason"] == "assistance_needed"
    s = pm.start("P-5", lang="de", now=0.0)
    for k in range(20):
        v = s.update(body(y=0.2), k * 0.25)
    assert "rechts" in v["instruction"]["text"]


# ---- pace --------------------------------------------------------------------------
def project(d, est, H=1.7, w=1280, h=720):
    """Box of a person standing d metres behind the robot camera."""
    f = est.focal(w)
    def row(height):   # image row of a point at this height above the floor, d metres away
        ang = math.atan2(est.ch - height, d) - math.radians(est.tilt)   # below the optical axis
        return h / 2 + f * math.tan(ang)
    return (600, int(row(H)), 680, int(row(0.0))), (h, w)


def test_distance():
    est = DistanceEstimator(hfov_deg=90, cam_height_m=1.0, tilt_deg=10)
    for d in (1.5, 2.5, 4.0):
        box, shape = project(d, est)
        got, how = est.estimate(box, shape)
        assert how == "feet" and abs(got - d) < 0.05 * d, (d, got)
    assert abs(est.person_h - 1.7) < 0.05
    est = DistanceEstimator(hfov_deg=90, cam_height_m=1.0, tilt_deg=0)
    est.person_h = 1.7
    box, shape = project(1.4, est)                    # close: feet below the picture, head visible
    assert box[3] > shape[0] and box[1] > 0
    got, how = est.estimate(box, shape)
    assert how == "head" and abs(got - 1.4) < 0.1, (got, how)
    box, shape = project(0.9, est)                    # very close: neither head nor feet visible
    got, how = est.estimate((box[0], -20, box[2], shape[0] + 20), shape)
    assert how == "closer_than" and 0.9 < got < 1.3, (got, how)


def test_pace_follows_patient():
    pc = PaceController(target_gap=1.2)
    gap, t, v_patient = 1.2, 0.0, 0.6
    for _ in range(300):                               # 30 s walk, patient at 0.6 m/s
        t += 0.1
        out = pc.update(gap, t)
        gap += (out["speed_mps"] - v_patient) * 0.1
    assert out["state"] == "following" and abs(out["speed_mps"] - 0.6) < 0.08 and abs(gap - 1.2) < 0.25, (out, gap)
    v_patient = 0.0                                    # patient stops to rest
    for _ in range(100):
        t += 0.1
        out = pc.update(gap, t)
        gap += (out["speed_mps"] - v_patient) * 0.1
    assert out["speed_mps"] < 0.1 and gap < 3.0, (out, gap)
    out = pc.update(3.5, t + 0.1)                      # one odd reading is smoothed away...
    assert out["state"] == "following"
    for k in range(5):                                 # ...but the robot really is far ahead
        out = pc.update(3.5, t + 0.2 + 0.1 * k)
    assert out["state"] == "waiting" and out["speed_mps"] == 0.0 and out["say"] is None
    t += 0.8
    for k in range(25):                                # not seen at all
        out = pc.update(None, t + 0.1 * k)
    assert out["state"] == "lost" and out["speed_mps"] == 0.0
    out = pc.update(None, t + 20)
    assert out["state"] == "alert_staff"
    out = pc.update(1.5, t + 21)
    assert out["state"] == "following" and out["say"] == "resume"


if __name__ == "__main__":
    for fn in (test_journey_flow, test_evaluate, test_session_guides_to_positioned, test_session_staff_cases,
               test_distance, test_pace_follows_patient):
        fn()
        print("ok ", fn.__name__)
    print("all journey tests passed")
