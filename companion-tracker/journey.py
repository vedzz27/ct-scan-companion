"""Patient journey layer: one stage per patient, one timestamped event log for everything.

Inputs:  camera zone observations (with a dwell time, forward-only), and explicit events from the
         robot (QR verified, guide started/arrived, feedback), the technologist (positioning accepted)
         and the CT workflow (scan started/finished).
Outputs: current stage, the robot's next task (from journey.json), timelines and stage durations.
No names or images are logged: only the anonymous patient id.
"""
import json
import os
import threading
import time


class Journey:
    def __init__(self, path="journey.json", log_path=None, retention_s=48 * 3600):
        with open(path) as f:
            self.cfg = json.load(f)
        self.order = {s: i for i, s in enumerate(self.cfg["stages"])}
        self.log_path = log_path
        self.lock = threading.Lock()
        self.state = {}      # pid -> dict
        self.events = []     # [{pid, t, event, stage, source, detail}]
        if log_path and os.path.exists(log_path):   # drop events older than the retention limit
            cutoff = time.time() - retention_s
            with open(log_path) as f:
                keep = [ln for ln in f if self._t(ln) >= cutoff]
            with open(log_path, "w") as f:
                f.writelines(keep)

    @staticmethod
    def _t(line):
        try:
            return json.loads(line).get("t", 0)
        except ValueError:
            return 0

    # ---- writing ------------------------------------------------------------
    def _log(self, pid, event, source, detail=None, now=None):
        st = self.state.get(pid, {})
        e = {"pid": pid, "t": round(time.time() if now is None else now, 2), "event": event,
             "stage": st.get("stage"), "source": source}
        if detail:
            e["detail"] = detail
        self.events.append(e)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(e) + "\n")
        return e

    def _set_stage(self, pid, stage, source, now, detail=None):
        st = self.state[pid]
        if stage == st["stage"]:
            return False
        st["stage"], st["since"] = stage, now
        self._log(pid, "stage", source, dict(detail or {}, to=stage), now)
        return True

    def register(self, pid, needs=None, zone="check_in", now=None):
        now = time.time() if now is None else now
        with self.lock:
            self.state[pid] = {"stage": None, "since": now, "zone": zone, "seen": {},
                               "escort": None, "needs": needs or {}, "protocol": None}
            self._log(pid, "checkin", "kiosk", {"zone": zone}, now)
            self._set_stage(pid, "checked_in", "kiosk", now)

    def observe_zone(self, pid, zone, camera, now=None):
        """Called for every confident camera sighting. A zone counts after zone_dwell_s,
        and a zone can only move the journey forward (a glimpse of the waiting room while
        walking to the CT room doesn't send the patient back)."""
        now = time.time() if now is None else now
        with self.lock:
            st = self.state.get(pid)
            if not st:
                return
            # each zone has its own "seen continuously since" timer (gaps under zone_gap_s are bridged: cameras may run at only 1-2 fps),
            # so a stray sighting somewhere else doesn't cancel it
            s = st["seen"].get(zone)
            if s is None or now - s[1] > self.cfg.get("zone_gap_s", 2.5):
                st["seen"][zone] = [now, now]
            else:
                s[1] = now
            if zone == st["zone"] or now - st["seen"][zone][0] < self.cfg.get("zone_dwell_s", 2.0):
                return
            st["zone"] = zone
            self._log(pid, "zone_enter", camera, {"zone": zone}, now)
            stage = self.cfg["zone_stage"].get(zone)
            if stage and self.order.get(stage, -1) > self.order.get(st["stage"], -1):
                self._set_stage(pid, stage, camera, now, {"zone": zone})
                for target, g in self.cfg["guide"].items():   # camera saw them arrive
                    if st["escort"] == target and self.order[stage] >= self.order[g["arrived"]]:
                        st["escort"] = None
                        self._log(pid, "guide_arrived", camera, {"target": target}, now)

    def event(self, pid, kind, source="api", detail=None, now=None):
        """Explicit events can set any stage (people and the workflow are the authority)."""
        now = time.time() if now is None else now
        detail = detail or {}
        with self.lock:
            st = self.state.get(pid)
            if not st:
                raise KeyError(pid)
            self._log(pid, kind, source, detail, now)
            if kind == "guide_start":
                target = detail.get("target")
                g = self.cfg["guide"].get(target)
                st["escort"] = target
                if g:
                    self._set_stage(pid, g["during"], source, now)
            elif kind == "guide_arrived":
                target = detail.get("target") or st["escort"]
                g = self.cfg["guide"].get(target)
                st["escort"] = None
                if g:
                    self._set_stage(pid, g["arrived"], source, now)
            elif kind == "guide_stopped":
                st["escort"] = None
            else:
                stage = self.cfg["event_stage"].get(kind)
                if stage:
                    self._set_stage(pid, stage, source, now)
            if kind == "positioning_started" and detail.get("protocol"):
                st["protocol"] = detail["protocol"]
            return self._view(pid, now)

    def set_protocol(self, pid, protocol):
        with self.lock:
            if pid in self.state:
                self.state[pid]["protocol"] = protocol

    def remove(self, pid):
        with self.lock:
            self.state.pop(pid, None)
            self.events = [e for e in self.events if e["pid"] != pid]
            if self.log_path and os.path.exists(self.log_path):   # retention also applies to the file
                keep = []
                with open(self.log_path) as f:
                    for line in f:
                        try:
                            if json.loads(line).get("pid") != pid:
                                keep.append(line)
                        except ValueError:
                            pass
                with open(self.log_path, "w") as f:
                    f.writelines(keep)

    # ---- reading ------------------------------------------------------------
    def stage(self, pid):
        with self.lock:
            st = self.state.get(pid)
            return st["stage"] if st else None

    def needs(self, pid):
        with self.lock:
            st = self.state.get(pid)
            return dict(st["needs"]) if st else {}

    def _task(self, pid):
        st = self.state[pid]
        t = self.cfg["robot_task"].get(st["stage"])
        if not t:
            return None
        t = dict(t)
        if "target" in t:
            t["robot_point"] = self.cfg.get("robot_targets", {}).get(t["target"], t["target"])
        return t

    def _view(self, pid, now):
        st = self.state[pid]
        return {"id": pid, "stage": st["stage"], "since_s": round(now - st["since"], 1),
                "zone": st["zone"], "escort": st["escort"], "needs": st["needs"],
                "protocol": st["protocol"], "robot_task": self._task(pid)}

    def view(self, pid, now=None):
        with self.lock:
            if pid not in self.state:
                raise KeyError(pid)
            return self._view(pid, now or time.time())

    def all(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            return [self._view(p, now) for p in self.state]

    def robot_queue(self, now=None):
        """Patients with something for the robot to do, oldest stage first."""
        now = time.time() if now is None else now
        with self.lock:
            items = [self._view(p, now) for p in self.state if self._task(p)]
        return sorted(items, key=lambda v: -v["since_s"])

    def escorted(self):
        """The patient the robot is walking with right now (or None)."""
        with self.lock:
            cands = [(st["since"], p) for p, st in self.state.items() if st["escort"]]
            return max(cands)[1] if cands else None

    def in_stage(self, stages):
        """Most recent patient in one of these stages (e.g. who is in the CT room)."""
        with self.lock:
            cands = [(st["since"], p) for p, st in self.state.items() if st["stage"] in stages]
            return max(cands)[1] if cands else None

    def timeline(self, pid):
        with self.lock:
            return [e for e in self.events if e["pid"] == pid]

    def durations(self, pid, now=None):
        """Seconds spent in each stage, from the stage-change events."""
        now = time.time() if now is None else now
        with self.lock:
            ch = [e for e in self.events if e["pid"] == pid and e["event"] == "stage"]
        out = {}
        for a, b in zip(ch, ch[1:] + [None]):
            end = b["t"] if b else now
            out[a["detail"]["to"]] = round(out.get(a["detail"]["to"], 0) + end - a["t"], 1)
        return out

    def metrics(self, now=None):
        """Average time per stage across current patients: where the department loses time."""
        with self.lock:
            pids = list(self.state)
        per = [self.durations(p, now) for p in pids]
        stages = [s for s in self.cfg["stages"] if any(s in d for d in per)]
        return {"patients": len(per),
                "mean_s": {s: round(sum(d[s] for d in per if s in d) / sum(1 for d in per if s in d), 1)
                           for s in stages}}
