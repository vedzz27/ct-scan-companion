"""Tests for the detection logic without cameras or models:  python test_matching.py"""
import numpy as np

import config
import matching
from store import Store
from worker import CameraWorker


class ColourEmbedder:
    """Stand-in for OSNet: the crop's mean colour decides the vector (red person, blue person)."""
    name = "osnet"
    BASIS = {"red": np.array([1, 0, 0, 0.1]), "blue": np.array([0, 1, 0, 0.1]), "green": np.array([0, 0, 1, 0.1])}

    def batch(self, crops):
        out = []
        for c in crops:
            b, g, r = c.reshape(-1, 3).mean(axis=0)
            key = "red" if r > max(g, b) else "blue" if b > g else "green"
            v = self.BASIS[key] + np.random.default_rng().normal(0, 0.03, 4)
            out.append((v / np.linalg.norm(v)).astype(np.float32))
        return out

    def __call__(self, c):
        return self.batch([c])[0]


COL = {"red": (30, 30, 220), "blue": (220, 60, 30), "green": (40, 200, 40)}


def frame_with(people):
    """people: [(x1, y1, x2, y2, tid, colour)] -> frame and the worker's people list."""
    f = np.full((1200, 1920, 3), 128, np.uint8)
    lst = []
    for x1, y1, x2, y2, tid, col in people:
        f[y1:y2, x1:x2] = COL[col]
        lst.append((x1, y1, x2, y2, tid, 0.9))
    return f, lst


def make_worker(store):
    w = CameraWorker("waiting", {"source": 0, "zone": "waiting_area",
                                 "regions": [{"zone": "A", "box": [0, 0, 0.5, 1]},
                                             {"zone": "B", "poly": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]],
                                              "robot_point": "stop_B"}]},
                     store, ColourEmbedder())
    return w


def run(w, people, t, steps=6, dt=0.8):
    for _ in range(steps):
        f, lst = frame_with(people)
        w.process(f, lst, None, t)
        t += dt
    return t


def test_quality():
    shape = (1200, 1920)
    assert matching.crop_quality((100, 100, 200, 400), 0.9, [], shape) is None
    assert matching.crop_quality((100, 100, 200, 150), 0.9, [], shape) == "small"
    assert matching.crop_quality((0, 100, 100, 400), 0.9, [], shape) == "edge"
    assert matching.crop_quality((100, 100, 200, 400), 0.3, [], shape) == "low_conf"
    assert matching.crop_quality((100, 100, 200, 400), 0.9, [(150, 100, 260, 400)], shape) == "occluded"
    assert matching.crop_quality((100, 100, 500, 250), 0.9, [], shape) == "shape"   # lying / merged box


def test_assign_unique():
    rows = {1: {"A": 0.9, "B": 0.7}, 2: {"A": 0.85, "B": 0.6}}
    a = matching.assign(rows, keep=0.5)
    assert {a[1][0], a[2][0]} == {"A", "B"}             # both want A, only one gets it
    assert matching.assign({1: {"A": 0.3}}, keep=0.5) == {}


def test_hysteresis():
    ts = matching.TrackState(0)
    row = {"A": 0.8, "B": 0.3}
    assert matching.decide(ts, ("A", 0.8), row, 0.62, 0.5, 0.08, 0) == "none"   # 1st good round
    assert matching.decide(ts, ("A", 0.8), row, 0.62, 0.5, 0.08, 1) == "bind"   # 2nd: bound
    for i in range(config.UNBIND_STREAK - 1):                                  # brief dips: kept
        assert matching.decide(ts, None, {"A": 0.4}, 0.62, 0.5, 0.08, 2 + i) == "keep"
    assert matching.decide(ts, None, {"A": 0.4}, 0.62, 0.5, 0.08, 9) == "unbind"
    ts = matching.TrackState(0)
    matching.decide(ts, ("A", 0.8), {"A": 0.8, "B": 0.78}, 0.62, 0.5, 0.08, 0)   # margin too small
    assert matching.decide(ts, ("A", 0.8), {"A": 0.8, "B": 0.78}, 0.62, 0.5, 0.08, 1) == "none"


def test_follow_and_swap():
    s = Store()
    emb = ColourEmbedder()
    f, _ = frame_with([(100, 100, 300, 600, 0, "red")])
    red = s.check_in("Red", [emb(f[100:600, 100:300])])
    f, _ = frame_with([(100, 100, 300, 600, 0, "blue")])
    blue = s.check_in("Blue", [emb(f[100:600, 100:300])])
    w = make_worker(s)

    # both sit apart: red on the left (zone A), blue on the right (zone B)
    t = run(w, [(200, 300, 400, 800, 1, "red"), (1300, 300, 1500, 800, 2, "blue")], 1000.0)
    assert w.tracks[1].pid == red.id and w.tracks[2].pid == blue.id
    assert s.location(red.id, t)["zone"] == "A" and s.location(blue.id, t)["robot_target"] == "stop_B"
    assert s.location(red.id, t)["action"] == "go_to_zone_and_confirm"

    # they walk past each other: boxes overlap -> both flagged, robot must not use names
    f, lst = frame_with([(700, 300, 900, 800, 1, "red"), (800, 300, 1000, 800, 2, "blue")])
    w.process(f, lst, None, t)
    assert s.location(red.id, t)["ambiguous"] and not s.location(red.id, t)["may_use_name"]

    # the tracker swapped the IDs: track 1 now carries the blue person on the right
    t = run(w, [(1300, 300, 1500, 800, 1, "blue"), (200, 300, 400, 800, 2, "red")], t + 0.8)
    assert w.tracks[1].pid == blue.id and w.tracks[2].pid == red.id, w.debug()
    loc = s.location(red.id, t)
    assert loc["zone"] == "A" and not loc["ambiguous"] and loc["may_use_name"], loc

    # a stranger never gets a patient's identity
    t = run(w, [(200, 300, 400, 800, 1, "red"), (1300, 300, 1500, 800, 3, "green")], t + 1)
    assert w.tracks[3].pid is None


def test_still_neighbours_not_flagged():
    """Two people sitting side by side with slightly overlapping boxes: no swap alarm."""
    s = Store()
    emb = ColourEmbedder()
    f, _ = frame_with([(100, 100, 300, 600, 0, "red"), (900, 100, 1100, 600, 0, "blue")])
    red = s.check_in("Red", [emb(f[100:600, 100:300])])
    blue = s.check_in("Blue", [emb(f[100:600, 900:1100])])
    w = make_worker(s)
    t = run(w, [(500, 300, 760, 800, 1, "red"), (740, 300, 1000, 800, 2, "blue")], 2000.0, steps=10)
    assert w.tracks[1].pid == red.id and w.tracks[2].pid == blue.id, w.debug()
    assert not s.location(red.id, t)["ambiguous"]


def test_anchors_survive_learning():
    s = Store()
    v = np.array([1, 0, 0, 0], np.float32)
    p = s.check_in("A", [v])
    for _ in range(30):   # lots of learned views that drift away
        s.add_embedding(p.id, np.array([0, 1, 0, 0], np.float32))
    assert s.scores(v)[p.id] > 0.99


if __name__ == "__main__":
    for fn in (test_quality, test_assign_unique, test_hysteresis, test_follow_and_swap,
               test_still_neighbours_not_flagged, test_anchors_survive_learning):
        fn()
        print("ok ", fn.__name__)
    print("all matching tests passed")
