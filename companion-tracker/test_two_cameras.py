"""Front (check-in, face) + back (room, whole body) camera hand-off.  Run: python test_two_cameras.py"""
import time

import numpy as np

import config
import matching
from store import Store
from worker import CameraWorker


def vec(seed, n=64):
    r = np.random.default_rng(seed).normal(size=n).astype(np.float32)
    return r / np.linalg.norm(r)


class FakeEmbedder:
    name = "osnet"


def room(store, region=None):
    cfg = {"source": 1, "role": "tracking"}
    if region:
        cfg["checkin_region"] = region
    w = CameraWorker("waiting", cfg, store, FakeEmbedder())
    w.grabber.size = (1280, 720)
    return w


def person(w, tid, body, box, face=None, now=None):
    now = now or time.time()
    ts = w.tracks.setdefault(tid, matching.TrackState(now))
    ts.last_seen, ts.box = now, box
    for _ in range(4):
        ts.embs.append(body)
    if face is not None:
        for _ in range(config.FACE_MIN_FOR_MATCH):
            ts.faces.append(face)
    return ts


def test_only_person():
    w = room(Store())
    person(w, 1, vec(1), (100, 100, 300, 700))
    assert w.checkin_partner(None)[0] == 1
    print("ok  only unknown person in the room view is linked")


def test_face_decides():
    w = room(Store())
    me, other = vec(10, 512), vec(11, 512)
    person(w, 1, vec(1), (100, 100, 300, 700), face=other)
    person(w, 2, vec(2), (600, 100, 800, 700), face=me)
    tid, why = w.checkin_partner(me)
    assert tid == 2 and "face" in why, (tid, why)
    print("ok  room person with the same face is linked:", why)


def test_different_face_ruled_out():
    w = room(Store())
    me = vec(10, 512)
    person(w, 1, vec(1), (100, 100, 300, 700), face=vec(12, 512))   # clearly someone else
    person(w, 2, vec(2), (600, 100, 800, 700))                      # face not seen
    assert w.checkin_partner(me)[0] == 2
    print("ok  people with a different face are ruled out")


def test_ambiguous_no_guess():
    w = room(Store())
    person(w, 1, vec(1), (100, 100, 300, 700))
    person(w, 2, vec(2), (600, 100, 800, 700))
    tid, why = w.checkin_partner(None)
    assert tid is None and "2 possible" in why
    # ...unless a check-in area is configured (left half of the room picture)
    w2 = room(Store(), region=[[0, 0], [0.5, 0], [0.5, 1], [0, 1]])
    person(w2, 1, vec(1), (100, 100, 300, 700))
    person(w2, 2, vec(2), (900, 100, 1100, 700))
    assert w2.checkin_partner(None)[0] == 1
    print("ok  never guesses between several people; check-in area decides")


def test_known_people_ignored_and_learned_later():
    s = Store()
    w = room(s)
    anna = s.check_in("Anna", [vec(1)])
    ts = person(w, 1, vec(1), (100, 100, 300, 700))
    ts.pid = anna.id                                   # already known: never the newcomer
    assert w.checkin_partner(None)[0] is None
    # not linked at check-in: later, the room camera sees the face -> found despite other body view
    me = vec(30, 512)
    p = s.check_in("Vedika", [vec(40)], faces=[me])   # front camera: close-up body + face
    bind, keep, _ = config.thresholds("osnet")
    assert s.scores(vec(41))[p.id] < keep               # room body view alone: no match
    assert s.scores(vec(41), face=me)[p.id] >= bind     # with the face: match -> track binds
    print("ok  known people ignored; unlinked person found later by face")


for t in (test_only_person, test_face_decides, test_different_face_ruled_out, test_ambiguous_no_guess,
          test_known_people_ignored_and_learned_later):
    t()
print("all two-camera tests passed")


def test_handoff_after_checkin():
    s = Store()
    w = room(s)
    me = vec(50, 512)
    p = s.check_in("Vedika", [vec(60)], faces=[me])     # front camera: face only
    s.add_pending(p.id)
    t = time.time()
    # someone else walks in first (different face): announced as new patient, not linked
    other = person(w, 1, vec(1), (100, 100, 300, 700), face=vec(51, 512), now=t)
    w.detect_newcomers({1: other.box}, (720, 1280, 3), t)
    assert other.pid is None
    # Vedika appears in the room camera: face matches -> linked, body views kept
    mine = person(w, 2, vec(2), (600, 100, 800, 700), face=me, now=t)
    w.detect_newcomers({1: other.box, 2: mine.box}, (720, 1280, 3), t)
    assert mine.pid == p.id and not s.pending_ids()
    assert s.scores(vec(2))[p.id] > 0.99                 # body alone now recognises her
    assert s.list_new() == [] or all(n["track"] != 2 for n in s.list_new())
    print("ok  front-camera check-in -> back camera takes the whole body")


def test_handoff_only_unknown_person_without_face():
    s = Store()
    w = room(s)
    p = s.check_in("Ravi", [vec(70)], faces=[vec(71, 512)])
    s.add_pending(p.id)
    ts = person(w, 3, vec(3), (100, 100, 300, 700))      # back view, no face
    w.detect_newcomers({3: ts.box}, (720, 1280, 3), time.time())
    assert ts.pid == p.id
    print("ok  only unknown person in view is linked even from behind")


test_handoff_after_checkin()
test_handoff_only_unknown_person_without_face()


def test_handoff_biggest_person_with_others_in_view():
    s = Store()
    w = room(s)
    p = s.check_in("Vedika", [vec(80)], faces=[vec(81, 512)])
    s.add_pending(p.id)
    t = time.time()
    far = person(w, 1, vec(1), (500, 300, 560, 450), now=t)       # someone small in the back
    near = person(w, 2, vec(2), (700, 100, 1270, 715), now=t)     # the person at the desk
    w.detect_newcomers({1: far.box, 2: near.box}, (720, 1280, 3), t)
    assert near.pid == p.id and far.pid is None
    print("ok  several people: the clearly closest one is linked")


test_handoff_biggest_person_with_others_in_view()
