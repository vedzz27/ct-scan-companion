import time
import numpy as np
import config
from store import Store

def vec(seed):
    r = np.random.default_rng(seed).normal(size=64).astype(np.float32)
    return r / np.linalg.norm(r)

def test_all():
    s = Store()
    a = s.check_in("Anna", vec(1)); b = s.check_in("Ben", vec(2))
    pid, score, margin, _ = s.match(vec(1))
    assert pid == a.id and score > 0.99 and margin > 0.1
    t0 = time.time() + 2   # after the 1.5 s check-in debounce
    s.observe(a.id, "waiting_B", "waiting", 0.9, now=t0)
    loc = s.location(a.id, t0)
    assert loc["zone"] == "waiting_B" and loc["action"] == "go_to_zone_and_confirm"
    # confidence decays and robot is told to search the last zone
    later = t0 + 100
    assert s.location(a.id, later)["action"] == "go_to_last_zone_and_scan"
    assert s.location(a.id, t0 + 400)["action"] == "alert_staff"
    # 3 failed confirmations block the workflow
    for _ in range(3): r = s.confirm(a.id, False)
    assert r["blocked"] and s.confirm(a.id, True)["blocked"]
    # gallery limit
    for i in range(config.MAX_ACTIVE_PATIENTS - 2): s.check_in(f"x{i}", vec(20 + i))
    try: s.check_in("overflow", vec(99)); assert False
    except ValueError: pass
    # 48 h purge
    assert s.purge(time.time() + 49 * 3600) == config.MAX_ACTIVE_PATIENTS and not s.patients
    # early delete
    c = s.check_in("C", vec(3)); assert s.remove(c.id) and not s.patients
    print("all store tests passed")

test_all()
