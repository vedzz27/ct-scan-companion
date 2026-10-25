"""New person detection + present/absent status, without cameras or models.  Run: python test_newcomer.py"""
import time

import numpy as np

import config
import matching
from store import Store
from worker import CameraWorker


def vec(seed):
    r = np.random.default_rng(seed).normal(size=64).astype(np.float32)
    return r / np.linalg.norm(r)


class FakeEmbedder:
    name = "osnet"


def make_worker(store):
    return CameraWorker("waiting", {"source": 0, "role": "tracking"}, store, FakeEmbedder())


def seen(w, tid, emb, now, n=None):
    ts = w.tracks.setdefault(tid, matching.TrackState(now))
    ts.last_seen = now
    for _ in range(n or config.NEW_PERSON_MIN_VIEWS):
        ts.embs.append(emb)
    return ts


def test_newcomer_flow():
    s = Store()
    w = make_worker(s)
    anna = s.check_in("Anna", vec(1))
    box = (100, 100, 200, 400)
    t0 = time.time()

    # Anna's track (looks like Anna) and an unknown person's track
    seen(w, 1, vec(1), t0)
    seen(w, 2, vec(2), t0)
    w.detect_newcomers({1: box, 2: box}, (720, 1280, 3), t0)
    assert s.list_new(t0) == [], "nobody is reported before NEW_PERSON_AFTER_S"

    t1 = t0 + config.NEW_PERSON_AFTER_S + 0.1
    w.detect_newcomers({1: box, 2: box}, (720, 1280, 3), t1)
    new = s.list_new(t1)
    assert [n["track"] for n in new] == [2], f"only the unknown person is new, got {new}"

    # robot asks the name -> enroll from the track's views, track is bound, newcomer cleared
    nid = new[0]["id"]
    p = s.check_in("Ravi", w.track_views(2), zone=new[0]["zone"])
    w.enroll_track(2, p.id)
    assert s.get_new(nid) is None and w.tracks[2].pid == p.id
    w.detect_newcomers({1: box, 2: box}, (720, 1280, 3), t1 + 10)
    assert s.list_new(t1 + 10) == [], "an enrolled person is never a newcomer again"

    # the same person coming back on a new track resembles Ravi -> not a newcomer
    seen(w, 3, vec(2), t1)
    w.detect_newcomers({3: box}, (720, 1280, 3), t1)
    w.detect_newcomers({3: box}, (720, 1280, 3), t1 + config.NEW_PERSON_AFTER_S + 1)
    assert s.list_new(t1 + config.NEW_PERSON_AFTER_S + 1) == []

    # a newcomer who walks away disappears from the list
    seen(w, 4, vec(5), t1)
    w.detect_newcomers({4: box}, (720, 1280, 3), t1)
    w.detect_newcomers({4: box}, (720, 1280, 3), t1 + config.NEW_PERSON_AFTER_S + 1)
    assert len(s.list_new(t1 + config.NEW_PERSON_AFTER_S + 1)) == 1
    assert s.list_new(t1 + config.NEW_PERSON_AFTER_S + 1 + config.NEW_PERSON_STALE_S + 1) == []

    # present / absent
    s.observe(anna.id, "waiting", "waiting", 0.9, now=t0 + 2)
    assert s.location(anna.id, t0 + 10)["status"] == "present"
    loc = s.location(anna.id, t0 + 2 + config.ABSENT_AFTER_S + 1)
    assert loc["status"] == "absent" and loc["absent_since"] is not None
    s.observe(anna.id, "waiting", "waiting", 0.9, now=t0 + 300)   # came back from the toilet
    assert s.location(anna.id, t0 + 301)["status"] == "present"
    print("all newcomer tests passed")


test_newcomer_flow()
