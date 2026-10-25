"""One camera = one grabber thread (always holds the newest frame) + one processing thread
(detect, track, embed clean crops, assign tracks to patients, report positions)."""
import csv
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

import config
import matching
from embedding import apply_gains, colour_gains
from face import get_face_embedder, mean_face


_MODEL_LOAD_LOCK = threading.Lock()
_CAMERA_OPEN_LOCK = threading.Lock()


# ---- capture -----------------------------------------------------------------
class FrameGrabber(threading.Thread):
    """Reads as fast as the camera delivers and keeps only the newest frame, so a slow detector
    never works on seconds-old pictures (OpenCV otherwise queues frames)."""

    def __init__(self, name, cfg):
        super().__init__(daemon=True, name=f"grab-{name}")
        self.cfg = cfg
        self.cond = threading.Condition()
        self.frame, self.t, self.seq = None, 0.0, 0
        self.status, self.fps, self.size = "starting", 0.0, None
        self.stop_flag = False
        self.reopen = False        # set when the camera source was changed while running

    def open(self):
        """Cameras are opened one at a time (a Mac can fail to start two at the same moment),
        and each gets a moment to deliver its first picture before the next one opens."""
        with _CAMERA_OPEN_LOCK:
            cap = self._open()
            for _ in range(20):                     # warm-up: wait for a first real picture
                ok, f = cap.read()
                if ok and f is not None and f.size:
                    break
                time.sleep(0.1)
            time.sleep(0.5)
            return cap

    def _open(self):
        from cams import resolve_source
        try:
            src = resolve_source(self.cfg["source"])    # "name:FaceTime" -> today's camera number
        except Exception as e:
            print(f"[{self.name}] {e}", flush=True)
            self.status = "no signal"
            return cv2.VideoCapture()                  # empty: the loop retries every second
        if src != self.cfg["source"]:
            print(f"[{self.name}] camera '{self.cfg['source'][5:]}' is number {src} right now", flush=True)
        if isinstance(src, int) and sys.platform == "win32":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
        if self.cfg.get("fourcc"):   # MJPG keeps several USB cameras within USB bandwidth
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg["fourcc"]))
        for key, prop in (("width", cv2.CAP_PROP_FRAME_WIDTH), ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                          ("fps", cv2.CAP_PROP_FPS)):
            if key in self.cfg:
                cap.set(prop, self.cfg[key])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cfg.get("lock_colour"):   # stop auto white balance shifting clothing colours
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            if "wb_temperature" in self.cfg:
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.cfg["wb_temperature"])
        if "exposure" in self.cfg:        # DirectShow: log2 seconds, e.g. -6 = 1/64 s
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, self.cfg["exposure"])
        return cap

    def run(self):
        cap, n, t0 = self.open(), 0, time.time()
        src = self.cfg["source"]
        is_file = isinstance(src, str) and os.path.isfile(src)   # a recorded video: replay it in a loop
        file_dt = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15) if is_file else 0
        while not self.stop_flag:
            if self.reopen:                                          # source changed (camera swap)
                self.reopen = False
                cap.release()
                cap = self.open()
                print(f"[{self.name}] now using camera {self.cfg['source']}", flush=True)
                src = self.cfg["source"]
                is_file = isinstance(src, str) and os.path.isfile(src)
                continue
            if is_file:
                time.sleep(file_dt)                                  # play at recorded speed
            ok, frame = cap.read()
            if is_file and not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                if self.status != "no signal":
                    print(f"[{self.name}] camera {self.cfg['source']} gives no picture, retrying...", flush=True)
                self.status = "no signal"
                time.sleep(1)
                cap.release()
                cap = self.open()
                continue
            if self.status == "no signal":
                print(f"[{self.name}] camera {self.cfg['source']} is back", flush=True)
            now = time.time()
            with self.cond:
                self.frame, self.t, self.seq = frame, now, self.seq + 1
                self.cond.notify_all()
            self.status, self.size = "ok", (frame.shape[1], frame.shape[0])
            n += 1
            if now - t0 >= 2:
                self.fps, n, t0 = n / (now - t0), 0, now
        cap.release()

    def wait_newer(self, seq, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq or self.stop_flag, timeout=timeout)
            return self.frame, self.t, self.seq


# ---- floor geometry ----------------------------------------------------------
class FloorMap:
    """Foot point in the image -> (x, y) metres on the floor, using the calibration written by
    calibrate_floor.py. Lens distortion of the wide-angle lens is removed first if available."""

    def __init__(self, cal):
        self.H = np.array(cal["homography"], np.float64)
        self.K = np.array(cal["K"], np.float64) if cal.get("K") else None
        self.D = np.array(cal["D"], np.float64) if cal.get("D") else None
        self.fisheye = cal.get("model") == "fisheye"

    def to_floor(self, u, v):
        pt = np.array([[[u, v]]], np.float64)
        if self.K is not None:
            pt = (cv2.fisheye.undistortPoints(pt, self.K, self.D, P=self.K) if self.fisheye
                  else cv2.undistortPoints(pt, self.K, self.D, P=self.K))
        x, y = cv2.perspectiveTransform(pt, self.H)[0, 0]
        return float(x), float(y)


def load_floor(cfg):
    path = cfg.get("floor_calibration")
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return FloorMap(json.load(f))


# ---- the per-camera worker ---------------------------------------------------
class CameraWorker(threading.Thread):
    def __init__(self, name, cfg, store, embedder, journey=None, positioning=None):
        super().__init__(daemon=True, name=f"proc-{name}")
        self.name, self.cfg, self.store, self.embedder = name, cfg, store, embedder
        self.journey, self.positioning = journey, positioning
        self.kps = {}             # track_id -> (17, 3) keypoints in pixels, when a pose model is used
        self.role = cfg.get("role", "tracking")
        self.default_zone = cfg.get("zone", name)
        self.regions = cfg.get("regions", [])
        self.floor = load_floor(cfg)
        self.bind_t, self.keep_t, self.margin_t = config.thresholds(embedder.name)
        self.grabber = FrameGrabber(name, cfg)
        self.stop_flag = False
        self.lock = threading.Lock()
        self.jpeg = None
        self.frame = None
        self.people = []          # [(x1, y1, x2, y2, track_id, det_conf)]
        self.people_seq = 0
        self.processed_at = 0.0
        self.capture_lock = threading.Lock()
        self.gains = None
        self.tracks: dict[int, matching.TrackState] = {}
        self.model_name = ""
        self.proc_fps = 0.0
        self.infer_ms = 0.0
        self.error = ""
        self.boost_until = 0.0
        self._log = None
        # faces run in their own thread so tracking never waits for the face model
        self.face_async = False
        self.face_cond = threading.Condition()
        self.face_jobs = {}        # tid -> (head crop, picture time); newest only
        self.face_results = []     # (tid, picture time, Face or None)

    @property
    def status(self):
        if self.error:
            return f"error: {self.error}"
        return self.grabber.status

    def health(self):
        age = time.time() - self.grabber.t if self.grabber.t else None
        return {"name": self.name, "role": self.role, "status": self.status,
                "frame_size": self.grabber.size, "camera_fps": round(self.grabber.fps, 1),
                "processed_fps": round(self.proc_fps, 1), "detector_ms": round(self.infer_ms),
                "detection_age_s": round(time.time() - self.processed_at, 2) if self.processed_at else None,
                "last_frame_age_s": round(age, 2) if age is not None else None,
                "detector": self.model_name, "embedder": self.embedder.name,
                "thresholds": {"bind": self.bind_t, "keep": self.keep_t, "margin": self.margin_t},
                "floor_calibrated": self.floor is not None, "tracks": len(self.tracks),
                "checkin": self.checkin_readiness() if self.role == "kiosk" else None}

    # ---- check-in ----------------------------------------------------------
    def checkin_candidate(self, frame, people):
        """Use the same selection rules for live guidance and enrollment."""
        if frame is None or time.time() - self.processed_at > config.CHECKIN_MAX_FRAME_AGE_S:
            return None, "Waiting for a fresh camera detection."
        people = sorted(people, key=lambda p: matching.area(p[:4]), reverse=True)
        if not people:
            return None, "No person visible. Stand in front of the camera."
        main = people[0]
        if len(people) > 1 and matching.area(main[:4]) < config.CHECKIN_DOMINANCE * matching.area(people[1][:4]):
            return None, "More than one person at the kiosk. Only the patient should step forward."
        if main[4] < 0:
            return None, "Hold still while the camera starts tracking you."
        q = matching.crop_quality(main[:4], main[5], [p[:4] for p in people[1:]], frame.shape)
        if q in self.cfg.get("accept_crops", []):   # close-up camera: people are always cut off
            q = None
        reason = {"small": "Please come closer to the camera.",
                  "edge": "Please step into the middle of the picture.",
                  "occluded": "Someone is standing in front. Please step forward."}
        if q:
            return None, reason.get(q, "Please stand still with your whole body visible.")
        return main, "Person detected. Ready to check in."

    def checkin_readiness(self):
        with self.lock:
            main, reason = self.checkin_candidate(self.frame, self.people)
        return {"ready": main is not None, "message": reason}

    def capture_burst(self, n=None, timeout=None):
        """Enroll one stable track from fresh, distinct detections only."""
        if not self.capture_lock.acquire(blocking=False):
            return None, "Another check-in is in progress. Please wait."
        try:
            return self._capture_burst(n, timeout)
        finally:
            self.capture_lock.release()

    def _capture_burst(self, n=None, timeout=None):
        n = max(1, config.CHECKIN_SAMPLES if n is None else n)
        timeout = config.CHECKIN_TIMEOUT_S if timeout is None else timeout
        crops, tid_used, last_box = [], None, None
        last_reason = "Waiting for a fresh camera detection."
        with self.lock:
            seen_seq = self.people_seq
        end = time.monotonic() + timeout
        self.boost_until = time.time() + timeout
        while time.monotonic() < end and len(crops) < n:
            with self.lock:
                frame, people, seq, gains = self.frame, list(self.people), self.people_seq, self.gains
                main, reason = self.checkin_candidate(frame, people)
            if seq == seen_seq:
                time.sleep(0.03)
                continue
            seen_seq = seq
            if main is None:
                last_reason = reason
                crops, tid_used, last_box = [], None, None
                continue
            # Reject sudden jumps even if ByteTrack reused an ID for a different person.
            if tid_used is not None and (main[4] != tid_used or matching.iou(main[:4], last_box) < 0.3):
                crops = []
            x1, y1, x2, y2 = main[:4]
            crops.append(apply_gains(frame[y1:y2, x1:x2].copy(), gains))
            tid_used, last_box = main[4], main[:4]
            last_reason = "Not enough stable views yet. Please hold still and try again."
        return (crops, None) if len(crops) == n else (None, last_reason)

    def capture_person(self):
        """Backward-compatible single crop."""
        crops, _ = self.capture_burst(n=1, timeout=1.5)
        return crops[0] if crops else None

    # ---- geometry ----------------------------------------------------------
    def zone_for(self, box, shape):
        h, w = shape[:2]
        fx, fy = (box[0] + box[2]) / 2 / w, box[3] / h   # foot point, normalised
        for r in self.regions:
            if "poly" in r:
                poly = np.array(r["poly"], np.float32)
                if cv2.pointPolygonTest(poly, (float(fx), float(fy)), False) >= 0:
                    return r["zone"], r.get("robot_point")
            else:
                a, b, c, d = r["box"]
                if a <= fx <= c and b <= fy <= d:
                    return r["zone"], r.get("robot_point")
        return self.default_zone, self.cfg.get("robot_point")

    def floor_xy(self, box):
        if not self.floor:
            return None
        return self.floor.to_floor((box[0] + box[2]) / 2, box[3])

    # ---- main loop ---------------------------------------------------------
    def load_model(self):
        from ultralytics import YOLO
        for name in (self.cfg.get("model", config.YOLO_MODEL), config.YOLO_FALLBACK):
            try:
                with _MODEL_LOAD_LOCK:     # cameras start together: download the file only once
                    m = YOLO(name)
                self.model_name = name
                return m
            except Exception as e:   # e.g. no internet for the first download, or an old ultralytics
                print(f"[{self.name}] could not load {name}: {e}")
        raise RuntimeError("No YOLO model could be loaded")

    def run(self):
        self.grabber.start()                       # camera on first: its picture shows while the model loads
        print(f"[{self.name}] camera starting, loading detector...", flush=True)
        try:
            model = self.load_model()
        except Exception as e:
            self.error = str(e)
            print(f"[{self.name}] detector failed: {e}", flush=True)
            return
        if self.role == "tracking" and self.cfg.get("faces", True) and get_face_embedder() is not None:
            self.face_async = True
            threading.Thread(target=self.face_loop, daemon=True, name=f"face-{self.name}").start()
        imgsz = int(self.cfg.get("imgsz", config.YOLO_IMGSZ))
        period = 1.0 / float(self.cfg.get("target_fps", config.TARGET_FPS))
        idle_period = 1.0 / float(self.cfg.get("idle_fps", self.cfg.get("target_fps", config.TARGET_FPS)))
        seq, n, t0 = 0, 0, time.time()
        while not self.stop_flag:
            frame, t_frame, seq = self.grabber.wait_newer(seq)
            if frame is None or time.time() - t_frame > config.CHECKIN_MAX_FRAME_AGE_S:
                continue
            if seq == getattr(self, "_processed_seq", -1):
                continue
            self._processed_seq = seq
            start = time.time()
            res = model.track(frame, persist=True, classes=[0], conf=float(self.cfg.get("conf", config.YOLO_CONF)), imgsz=imgsz,
                              tracker=config.TRACKER_CFG, verbose=False)[0]
            if self.infer_ms == 0:
                print(f"[{self.name}] detection running (first frame took {time.time() - start:.1f} s)", flush=True)
            self.infer_ms = 0.8 * self.infer_ms + 0.2 * (time.time() - start) * 1000
            people = []
            if res.boxes is not None and len(res.boxes):
                h, w = frame.shape[:2]
                xyxy = res.boxes.xyxy.cpu().numpy()
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
                xyxy = xyxy.astype(int)
                confs = res.boxes.conf.cpu().numpy()
                ids = (res.boxes.id.cpu().numpy().astype(int)
                       if res.boxes.id is not None else [-1] * len(xyxy))
                people = [(*b, int(i), float(c)) for b, i, c in zip(xyxy.tolist(), ids, confs)]
                kp = getattr(res, "keypoints", None)
                self.kps = ({int(i): k for i, k in zip(ids, kp.data.cpu().numpy())}
                            if kp is not None and kp.data is not None and len(kp.data) == len(xyxy) else {})
            else:
                self.kps = {}
            gains = colour_gains(frame) if self.cfg.get("normalise_colour", True) else None
            with self.lock:
                self.frame, self.people, self.gains = frame, people, gains
                self.people_seq += 1
                self.processed_at = t_frame
            if self.role != "kiosk":
                try:
                    self.process(frame, people, gains, t_frame)
                except Exception as e:   # keep the camera alive; show the problem on /api/health
                    print(f"[{self.name}] processing error: {e!r}")
            self.render(frame, people)
            n += 1
            if time.time() - t0 >= 2:
                self.proc_fps, n, t0 = n / (time.time() - t0), 0, time.time()
            busy = self.role != "kiosk" or time.time() < self.boost_until
            rest = (period if busy else idle_period) - (time.time() - start)
            if rest > 0:
                time.sleep(rest)
        self.grabber.stop_flag = True

    def process(self, frame, people, gains, now):
        tracked = [p for p in people if p[4] >= 0]
        boxes = {p[4]: tuple(p[:4]) for p in tracked}
        for p in tracked:
            ts = self.tracks.setdefault(p[4], matching.TrackState(now))
            ts.last_seen = now
            ts.box = tuple(p[:4])
            ts.note_position(p[:4], now)

        # 1) people touching each other while moving: the tracker may swap their IDs
        moving = {tid for tid in boxes if self.tracks[tid].moving(now)}
        for a, b in matching.close_pairs(boxes, moving):
            self.tracks[a].mark_close(now)
            self.tracks[b].mark_close(now)

        # 2) embed only clean crops, all in one batch
        todo, crops = [], []
        fresh = set()
        for x1, y1, x2, y2, tid, dc in tracked:
            ts = self.tracks[tid]
            others = [b for t, b in boxes.items() if t != tid]
            ts.quality = matching.crop_quality((x1, y1, x2, y2), dc, others, frame.shape) or "ok"
            if ts.quality in self.cfg.get("accept_crops", []):   # e.g. a desk webcam: everyone is cut off
                ts.quality = "ok"
            if ts.quality == "ok" and now - ts.last_embed >= config.EMBED_EVERY_S:
                todo.append(tid)
                crops.append(apply_gains(frame[y1:y2, x1:x2], gains))
        for tid, e in zip(todo, self.embedder.batch(crops) if crops else []):
            if e is not None:
                self.tracks[tid].embs.append(e)
                self.tracks[tid].last_embed = now
                fresh.add(tid)

        # 2b) faces: also from moving people and boxes cut off at the edge, just not hidden ones
        fresh |= self.collect_faces(frame, tracked, now)

        # 3) re-match tracks that have fresh appearance, jointly (one patient per track)
        due = [tid for tid in fresh
               if len(self.tracks[tid].embs) >= 2 and (self.tracks[tid].recheck
                                                       or now - self.tracks[tid].last_eval >= config.EVAL_EVERY_S)]
        if due:
            means = {tid: self.tracks[tid].mean_emb() for tid in due}
            rows = {tid: self.store.scores(m, self.track_face(tid)) for tid, m in means.items()}
            # tracks bound earlier but not due now still hold their patient: keep them in the assignment
            for tid, ts in self.tracks.items():
                if tid in boxes and tid not in rows and ts.pid:
                    rows[tid] = {ts.pid: ts.score}
            assigned = matching.assign(rows, self.keep_t)
            for tid in due:
                ts = self.tracks[tid]
                ts.last_eval = now
                was_recheck = ts.recheck
                event = matching.decide(ts, assigned.get(tid), rows[tid], self.bind_t, self.keep_t,
                                        self.margin_t, now)
                if event in ("bind", "switch"):   # an old, lost track can't keep holding this patient
                    for other_tid, other in self.tracks.items():
                        if other_tid != tid and other.pid == ts.pid:
                            other.pid, other.score = None, 0.0
                if was_recheck and ts.pid:
                    # step 1 empties the views while people touch, so this decision used only
                    # clean crops taken after they separated: the identity is settled again
                    ts.ambiguous_until = 0.0
                elif (ts.pid and not was_recheck and not ts.ambiguous(now)
                      and ts.score >= self.bind_t + config.GALLERY_ADD_BONUS):
                    self.store.add_embedding(ts.pid, ts.embs[-1])   # learn this camera's view of them
                if ts.pid and ts.faces and not ts.ambiguous(now) and not was_recheck:
                    self.learn_face(ts)
                self.log(now, tid, ts, rows[tid], event)

        # 4) report every bound track's position
        for tid, box in boxes.items():
            ts = self.tracks[tid]
            if not ts.pid:
                continue
            amb = ts.ambiguous(now) or ts.recheck
            ts.conf = matching.confidence(ts.score, self.keep_t, amb)
            zone, target = self.zone_for(box, frame.shape)
            self.store.observe(ts.pid, zone, self.name, ts.conf, now=now, xy=self.floor_xy(box),
                               ambiguous=amb, robot_target=target)
            if self.journey and not amb and ts.conf >= 0.6:
                self.journey.observe_zone(ts.pid, zone, self.name, now)

        # 5) unknown people: visible, unbound for a while, clean views, and nobody in the gallery fits
        if self.role == "tracking" and config.NEW_PERSON_ENABLED:
            self.detect_newcomers(boxes, frame.shape, now)
        self.after_process(frame, people, now)

        for tid in [t for t, s in self.tracks.items() if now - s.last_seen > 15]:
            self.store.clear_new(self.name, tid)
            del self.tracks[tid]

    def detect_newcomers(self, boxes, shape, now):
        for tid, box in boxes.items():
            ts = self.tracks[tid]
            if ts.pid or ts.cand:          # bound, or about to bind: a known person
                ts.unbound_since = None
                self.store.clear_new(self.name, tid)
                continue
            if ts.unbound_since is None:
                ts.unbound_since = now
            if self.try_handoff(tid, ts, boxes, now):   # the person just checked in at the front camera
                continue
            if (ts.ambiguous(now) or len(ts.embs) < config.NEW_PERSON_MIN_VIEWS
                    or now - ts.unbound_since < config.NEW_PERSON_AFTER_S):
                continue
            scores = self.store.scores(ts.mean_emb(), self.track_face(tid))
            best = max(scores.values()) if scores else 0.0
            if best >= self.keep_t:        # resembles someone enrolled: let matching decide, don't greet
                self.store.clear_new(self.name, tid)
                continue
            zone, target = self.zone_for(box, shape)
            self.store.report_new(self.name, tid, zone, self.floor_xy(box), target, now=now, best_score=best)

    def try_handoff(self, tid, ts, boxes, now):
        """Someone checked in at the front (face) camera and this back camera hasn't seen their body
        yet. Link this track to them if its face matches, or if it is the only unknown person in view.
        Returns True if linked or still undecided (then don't announce a 'new patient' yet)."""
        pend = self.store.pending_ids(now)
        if not pend or ts.ambiguous(now) or len(ts.embs) < 1:
            return bool(pend)
        face = self.track_face(tid)
        target = None
        if face is not None:
            fs = {p: s for p, s in self.store.face_scores(face).items() if p in pend}
            if fs and max(fs.values()) >= config.FACE_SAME_T:
                target = max(fs, key=fs.get)
            elif fs and all(s <= config.FACE_DIFF_T for s in fs.values()):
                return False                       # clearly not the person checking in: a newcomer
        if target is None and len(pend) == 1:
            unknown = [t for t in boxes if not self.tracks[t].pid and not self.tracks[t].ambiguous(now)]
            if unknown == [tid]:
                target = pend[0]                   # the only unknown person in view
            elif tid in unknown:
                # several people: the one checking in stands at the desk, right in front of this camera,
                # so take the clearly biggest person (at least HANDOFF_DOMINANCE x the next one)
                big = sorted(unknown, key=lambda t: matching.area(boxes[t]), reverse=True)
                if big[0] == tid and matching.area(boxes[tid]) >= config.HANDOFF_DOMINANCE * matching.area(boxes[big[1]]):
                    target = pend[0]
        if target is None:
            return True                            # wait: maybe this is them, decide when the face is seen
        self.store.link_body(target, list(ts.embs))
        self.enroll_track(tid, target)
        print(f"[{self.name}] linked track {tid} to {target} (whole body from the room camera)", flush=True)
        return True

    def collect_faces(self, frame, tracked, now):
        """Look for a face in the head region of each visible person. Unknown people are checked
        most often, so a newcomer's face is ready by the time the robot asks their name.
        Returns the track ids that got a new face."""
        fe = get_face_embedder()
        if fe is None or not self.cfg.get("faces", True):
            return set()
        due = []
        for x1, y1, x2, y2, tid, _ in tracked:
            ts = self.tracks[tid]
            every = config.FACE_EVERY_S if ts.pid else config.FACE_EVERY_S_NEW
            if ts.quality != "occluded" and now - ts.last_face >= every:
                due.append((0 if not ts.pid else 1, ts.last_face, tid, (x1, y1, x2, y2)))
        if not self.face_async:                       # tests / no thread: do it right here
            for _, _, tid, box in sorted(due)[:config.FACE_MAX_PER_FRAME]:   # unknown people first
                self.tracks[tid].last_face = now
                self.face_results.append((tid, now, fe.from_person(frame, box)))
        else:
            from face import head_region
            with self.face_cond:
                for _, _, tid, box in sorted(due)[:config.FACE_MAX_PER_FRAME]:
                    self.tracks[tid].last_face = now
                    x1, y1, x2, y2 = head_region(box, frame.shape)
                    self.face_jobs[tid] = (frame[y1:y2, x1:x2].copy(), now)
                self.face_cond.notify()
        # apply finished results here, in the tracking thread (no shared state touched elsewhere)
        with self.face_cond:
            results, self.face_results = self.face_results, []
        got = set()
        for tid, t_pic, f in results:
            ts = self.tracks.get(tid)
            if f is None or ts is None or t_pic < ts.faces_reset_at:   # gone, or picture from before a possible swap
                continue
            ts.faces.append(f.emb)
            ts.face_px = f.width
            got.add(tid)
        return got

    def face_loop(self):
        fe = get_face_embedder()
        while not self.stop_flag:
            with self.face_cond:
                self.face_cond.wait_for(lambda: self.face_jobs or self.stop_flag, timeout=1.0)
                if not self.face_jobs:
                    continue
                # unknown people (unbound tracks) first, then the oldest request
                tid = min(self.face_jobs, key=lambda t: (bool(self.tracks.get(t) and self.tracks[t].pid),
                                                        self.face_jobs[t][1]))
                crop, t_pic = self.face_jobs.pop(tid)
            try:
                f = fe.from_region(crop)
            except Exception as e:
                print(f"[{self.name}] face error: {e!r}", flush=True)
                f = None
            with self.face_cond:
                self.face_results.append((tid, t_pic, f))

    def learn_face(self, ts):
        """Add this track's face to its patient: when it clearly matches a known face, or when the
        patient has no face yet and the body match is very confident."""
        face = ts.faces[-1]
        known = self.store.face_scores(face).get(ts.pid)
        if known is None:
            if ts.score >= self.bind_t + config.GALLERY_ADD_BONUS:
                self.store.add_face(ts.pid, face)
        elif known >= config.FACE_SAME_T + 0.1:
            self.store.add_face(ts.pid, face)

    def checkin_partner(self, kiosk_face=None):
        """Hand-off from the check-in (front) camera: which person in THIS (back) camera is the one
        checking in right now? Returns (track id or None, reason).
          1. a visible unknown person whose face matches the check-in face -> that one
          2. people whose face clearly differs are ruled out
          3. of the rest, the one inside "checkin_region" (cameras.json), or the only one left
        Several possible people and no way to tell -> None: the body is learned later, as soon as
        this camera sees the face (the face match then binds the track)."""
        now = time.time()
        size = self.grabber.size
        cands = []
        for tid, ts in list(self.tracks.items()):
            if (ts.pid or now - ts.last_seen > 1.5 or ts.ambiguous(now)
                    or len(ts.embs) < 2 or getattr(ts, "box", None) is None):
                continue
            cands.append((tid, ts))
        if not cands:
            return None, "nobody unknown visible in the room camera"
        if kiosk_face is not None:
            scored = []
            for tid, ts in cands:
                f = self.track_face(tid)
                scored.append((None if f is None else float(np.dot(f, kiosk_face)), tid, ts))
            same = [s for s in scored if s[0] is not None and s[0] >= config.FACE_SAME_T]
            if same:
                best = max(same, key=lambda s: s[0])
                return best[1], f"face match {best[0]:.2f}"
            cands = [(tid, ts) for sim, tid, ts in scored if sim is None or sim > config.FACE_DIFF_T]
            if not cands:
                return None, "everyone visible has a different face"
        region = self.cfg.get("checkin_region")
        if region and size:
            poly = np.array(region, np.float32)
            inside = []
            for tid, ts in cands:
                x1, y1, x2, y2 = ts.box
                fx, fy = (x1 + x2) / 2 / size[0], y2 / size[1]
                if cv2.pointPolygonTest(poly, (float(fx), float(fy)), False) >= 0:
                    inside.append((tid, ts))
            cands = inside
            if len(cands) == 1:
                return cands[0][0], "standing in the check-in area"
        if len(cands) == 1:
            return cands[0][0], "only unknown person in view"
        return None, f"{len(cands)} possible people in the room camera"

    def enroll_track(self, tid, pid):
        """Bind a newcomer's track to the patient just created from its views (robot check-in)."""
        ts = self.tracks.get(tid)
        if ts:
            ts.pid, ts.score, ts.cand, ts.cand_streak, ts.unbound_since = pid, self.bind_t, None, 0, None
        self.store.clear_new(self.name, tid)

    def track_views(self, tid):
        ts = self.tracks.get(tid)
        return list(ts.embs) if ts else []

    def track_face(self, tid):
        """The track's face for matching: the mean of its recent faces, once there are enough."""
        ts = self.tracks.get(tid)
        if not ts or len(ts.faces) < config.FACE_MIN_FOR_MATCH:
            return None
        return mean_face(ts.faces)

    def track_faces(self, tid):
        ts = self.tracks.get(tid)
        return list(ts.faces) if ts else []

    def track_visible(self, tid, within=1.5):
        ts = self.tracks.get(tid)
        return bool(ts) and time.time() - ts.last_seen < within

    def after_process(self, frame, people, now):
        """Hook for role-specific work (CT room positioning, robot pace)."""

    def draw_extra(self, img):
        """Hook for role-specific overlay."""

    # ---- debugging / evaluation -------------------------------------------
    def debug(self):
        now = time.time()
        return [{"track": tid, "visible": now - ts.last_seen < 1.0, "patient": ts.pid, "candidate": ts.cand, "score": round(ts.score, 3),
                 "conf": ts.conf, "ambiguous": ts.ambiguous(now), "crop": ts.quality,
                 "views": len(ts.embs), "faces": len(ts.faces), "face_px": ts.face_px,
                 "last_round": ts.debug} for tid, ts in self.tracks.items()]

    def log(self, now, tid, ts, row, event):
        if not config.SCORE_LOG:
            return
        if self._log is None:
            new = not os.path.exists(config.SCORE_LOG)
            self._log = open(config.SCORE_LOG, "a", newline="")
            self._w = csv.writer(self._log)
            if new:
                self._w.writerow(["time", "camera", "track", "event", "bound", "best", "best_score",
                                  "second_score", "ambiguous", "all_scores"])
        ranked = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
        best = ranked[0] if ranked else (None, 0)
        second = ranked[1][1] if len(ranked) > 1 else ""
        self._w.writerow([round(now, 2), self.name, tid, event, ts.pid, best[0], round(best[1], 3),
                          round(second, 3) if second != "" else "", ts.ambiguous(now),
                          " ".join(f"{p}:{s:.3f}" for p, s in ranked)])
        self._log.flush()

    def render(self, frame, people):
        img = frame.copy()
        h, w = img.shape[:2]
        th = max(1, w // 900)
        for r in self.regions:
            if "poly" in r:
                pts = (np.array(r["poly"]) * [w, h]).astype(np.int32)
                cv2.polylines(img, [pts], True, (200, 160, 40), th)
                x0, y0 = pts[0]
            else:
                a, b, c, d = r["box"]
                x0, y0 = int(a * w), int(b * h)
                cv2.rectangle(img, (x0, y0), (int(c * w), int(d * h)), (200, 160, 40), th)
            cv2.putText(img, r["zone"], (x0 + 8, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 160, 40), 2)
        now = time.time()
        for x1, y1, x2, y2, tid, _ in people:
            ts = self.tracks.get(tid)
            if ts and ts.pid:
                amb = ts.ambiguous(now)
                color = (0, 140, 255) if amb else (80, 160, 20)   # orange = may have swapped
                label = f"{ts.pid} {ts.score:.2f}" + (" F" if ts.faces else "") + (" ?" if amb else "")
            else:
                color = (160, 160, 160)
                label = f"#{tid}" if tid >= 0 else "?"
                if ts and ts.cand:
                    label += f" ~{ts.cand}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            cv2.circle(img, ((x1 + x2) // 2, y2), 9, color, -1)          # the point used for zones
            cv2.putText(img, label, (x1, max(y1 - 12, 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6)
            cv2.putText(img, label, (x1, max(y1 - 12, 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        self.draw_extra(img)
        info = f"{self.name}  {self.proc_fps:.1f} fps  det {self.infer_ms:.0f} ms"
        cv2.putText(img, info, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6)
        cv2.putText(img, info, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (40, 40, 40), 2)
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (800, int(800 * h / w))), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
