"""API for the dashboard, the robot and the CT workflow.  Run:  uvicorn main:app --host 0.0.0.0 --port 8000

Camera roles (cameras.json): kiosk (check-in), tracking (waiting room...), ct_room (pose +
positioning guidance), robot (rear camera on the robot: patient distance and pace).
"""
import json
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from embedding import get_embedder
from face import get_face_embedder
from journey import Journey
from positioning import PositioningManager
from store import Store

store = Store()
journey = Journey(config.JOURNEY_FILE, config.JOURNEY_LOG or None, config.RETENTION_HOURS * 3600)
positioning = PositioningManager(config.PROTOCOLS_FILE)
workers: dict = {}


def _forget(pid):
    journey.remove(pid)
    positioning.remove(pid)


def purge_loop():
    while True:
        time.sleep(config.PURGE_INTERVAL_S)
        store.purge()
        for v in journey.all():                    # retention applies to the journey log too
            if v["id"] not in store.patients:
                _forget(v["id"])


def make_worker(name, cfg, embedder):
    role = cfg.get("role", "tracking")
    if role == "ct_room":
        from ct_room import CtRoomWorker as W
    elif role == "robot":
        from robot_camera import RobotCameraWorker as W
    else:
        from worker import CameraWorker as W
    return W(name, cfg, store, embedder, journey, positioning)


@asynccontextmanager
async def lifespan(app):
    if config.CAMERAS_ENABLED:
        embedder = get_embedder()
        store.embedder_name = embedder.name     # thresholds follow the embedder actually loaded
        get_face_embedder()                     # load the face model once, before cameras start
        with open(config.CAMERAS_FILE) as f:
            for name, cfg in json.load(f).items():
                if name.startswith("_") or cfg.get("enabled", True) is False:
                    continue
                w = make_worker(name, cfg, embedder)
                workers[name] = w
                w.start()
    threading.Thread(target=purge_loop, daemon=True).start()
    yield
    for w in workers.values():
        w.stop_flag = True


app = FastAPI(title="Companion tracker", lifespan=lifespan)


class CheckIn(BaseModel):
    name: str = ""
    camera: str = "kiosk"
    needs: dict = {}          # e.g. {"language": "de", "mobility": "slow|walker|wheelchair", "quiet": true}
    protocol: str | None = None


class Confirm(BaseModel):
    accepted: bool


class Event(BaseModel):
    type: str
    source: str = "api"
    detail: dict = {}


class Telemetry(BaseModel):
    speed_mps: float | None = None


class Start(BaseModel):
    protocol: str | None = None


def _worker(camera):
    w = workers.get(camera)
    if not w:
        raise HTTPException(503, f"Camera '{camera}' is not running")
    return w


def _robot_worker():
    for w in workers.values():
        if w.role == "robot":
            return w
    return None


# ---- check-in and identity -------------------------------------------------------
@app.post("/api/checkin")
def checkin(body: CheckIn):
    w = _worker(body.camera)
    if w.role != "kiosk":
        raise HTTPException(400, "Check-in requires a kiosk camera")
    crops, reason = w.capture_burst()          # ~1 s of clean crops of one person
    if crops is None:
        raise HTTPException(409, reason)
    embs = [e for e in get_embedder().batch(crops) if e is not None]   # images discarded here
    fe = get_face_embedder()
    faces = []
    if fe is not None:
        for c in crops:                  # front camera is close: the face can be anywhere in the crop
            f = fe.from_region(c)
            if f is not None:
                faces.append(f.emb)
    del crops
    # hand-off: find the same person in the room (back) camera and take their whole-body views
    link, link_reason, room_embs = None, "no room camera", []
    kface = None
    if faces:
        kface = np.mean(faces, axis=0)
        kface /= np.linalg.norm(kface) + 1e-9
    for rw in workers.values():
        if rw.role != "tracking":
            continue
        tid, link_reason = rw.checkin_partner(kface)
        if tid is not None:
            link, room_embs = (rw, tid), rw.track_views(tid)
            break
    if len(embs) < config.CHECKIN_SAMPLES or not all(np.isfinite(e).all() and np.linalg.norm(e) > 0 for e in embs):
        raise HTTPException(409, "Could not capture enough usable appearance views. Please try again.")
    mean = np.mean(embs, axis=0)
    mean /= np.linalg.norm(mean) + 1e-9
    pid_dup, score_dup, _, _ = store.match(mean)
    bind_t, _, _ = config.thresholds(get_embedder().name)
    try:
        p = store.check_in(body.name, embs + room_embs, zone=w.default_zone, faces=faces)
    except ValueError as e:
        raise HTTPException(409, str(e))
    if link:
        link[0].enroll_track(link[1], p.id)
    else:
        store.add_pending(p.id)          # the back camera links them as soon as it sees them
    journey.register(p.id, body.needs, zone=w.default_zone)
    if body.protocol:
        journey.set_protocol(p.id, body.protocol)
    out = store.location(p.id)
    out["views"] = len(embs)
    out["faces"] = len(faces)
    out["room_views"] = len(room_embs)
    out["room_link"] = (f"linked to {link[0].name} track {link[1]} ({link_reason})" if link
                        else f"room camera: waiting up to {config.HANDOFF_WAIT_S:.0f} s to see the whole body ({link_reason})")
    out["journey"] = journey.view(p.id)
    if pid_dup and score_dup >= bind_t:
        out["warning"] = (f"Looks very similar to {pid_dup} (score {score_dup:.2f}). "
                          "Same person checked in twice, or very similar clothing?")
    return out


# ---- newcomers: unknown people seen by the tracking cameras ---------------------------
class NewcomerCheckIn(BaseModel):
    name: str = ""
    needs: dict = {}
    protocol: str | None = None


@app.get("/api/newcomers")
def newcomers():
    """People in the room who match nobody enrolled, oldest first, with where they are."""
    return store.list_new()


@app.post("/api/newcomers/{nid}/checkin")
def newcomer_checkin(nid: str, body: NewcomerCheckIn):
    """The robot went to the newcomer and got their name: enroll them from the appearance views
    the tracking camera already collected (no extra photo needed) and bind their track."""
    n = store.get_new(nid)
    if not n:
        raise HTTPException(404, "Unknown or no longer visible newcomer")
    w = _worker(n["camera"])
    embs = [e for e in w.track_views(n["track"]) if e is not None and np.isfinite(e).all()]
    if len(embs) < config.NEW_PERSON_MIN_VIEWS:
        raise HTTPException(409, "Not enough clean views of this person yet. Try again in a moment.")
    # The person is still standing there talking to the robot: keep collecting face views for a
    # moment if there aren't enough yet (the camera checks unknown people several times a second).
    if get_face_embedder() is not None:
        end = time.time() + config.FACE_ENROLL_WAIT_S
        while (len(w.track_faces(n["track"])) < config.FACE_ENROLL_MIN and time.time() < end
               and w.track_visible(n["track"])):
            time.sleep(0.1)
    faces = w.track_faces(n["track"])
    try:
        p = store.check_in(body.name, embs, zone=n["zone"], faces=faces)
    except ValueError as e:
        raise HTTPException(409, str(e))
    w.enroll_track(n["track"], p.id)
    store.observe(p.id, n["zone"], n["camera"], 1.0, xy=n["xy"], robot_target=n["robot_target"])
    journey.register(p.id, body.needs, zone=n["zone"])
    if body.protocol:
        journey.set_protocol(p.id, body.protocol)
    out = store.location(p.id)
    out["views"] = len(embs)
    out["faces"] = len(faces)
    if not faces:
        out["warning"] = ("No usable face seen (too small, turned away or blurred): this person is "
                          "recognised by body/clothing only for now. Their face is learned later "
                          "when the camera sees it clearly.")
    out["journey"] = journey.view(p.id)
    return out


@app.get("/api/room")
def room():
    """Waiting-room summary: who is present, who is absent, and unknown people to greet."""
    ps = store.snapshot()
    return {"present": [p for p in ps if p["status"] == "present"],
            "absent": [p for p in ps if p["status"] == "absent"],
            "newcomers": store.list_new()}


@app.get("/api/patients")
def patients():
    out = store.snapshot()
    for p in out:
        try:
            v = journey.view(p["id"])
            p["stage"], p["robot_task"] = v["stage"], v["robot_task"]
        except KeyError:
            pass
    return out


@app.get("/api/patients/{pid}/location")
def location(pid: str):
    try:
        return store.location(pid)
    except KeyError:
        raise HTTPException(404, "Unknown patient")


@app.post("/api/patients/{pid}/confirm")
def confirm(pid: str, body: Confirm):
    """The robot's QR + voice/tap confirmation result (QR scanning itself lives in the robot code)."""
    try:
        r = store.confirm(pid, body.accepted)
        journey.event(pid, "qr_verified" if body.accepted else "qr_rejected", "robot", {"attempts": r["attempts"]})
        return r
    except KeyError:
        raise HTTPException(404, "Unknown patient")


@app.delete("/api/patients/{pid}")
def finish(pid: str):
    if not store.remove(pid):
        raise HTTPException(404, "Unknown patient")
    _forget(pid)
    return {"deleted": pid}


@app.post("/api/purge")
def purge():
    n = store.purge()
    for v in journey.all():
        if v["id"] not in store.patients:
            _forget(v["id"])
    return {"deleted": n}


# ---- journey ------------------------------------------------------------------------
@app.get("/api/journey")
def journey_all():
    return journey.all()


@app.get("/api/journey/metrics")
def journey_metrics():
    return journey.metrics()


@app.get("/api/journey/{pid}")
def journey_one(pid: str):
    try:
        return dict(journey.view(pid), timeline=journey.timeline(pid), durations_s=journey.durations(pid))
    except KeyError:
        raise HTTPException(404, "Unknown patient")


@app.post("/api/journey/{pid}/event")
def journey_event(pid: str, ev: Event):
    """Robot / tablet / workflow events, e.g. call, guide_start {"target": "ct_room"}, guide_arrived,
    changing_ready, robot_left, scan_started, scan_finished, aftercare_started, aftercare_done, finish."""
    try:
        return journey.event(pid, ev.type, ev.source, ev.detail)
    except KeyError:
        raise HTTPException(404, "Unknown patient")


# ---- robot ----------------------------------------------------------------------------
@app.get("/api/robot/next")
def robot_next():
    """The next thing the robot should do (oldest first), with the patient's location."""
    q = journey.robot_queue()
    if not q:
        new = store.list_new()          # nothing in the journey queue: greet the oldest newcomer
        if new:
            return {"task": "greet_new_person", "newcomer": new[0],
                    "then": f"POST /api/newcomers/{new[0]['id']}/checkin with the person's name"}
        return {"task": None}
    v = q[0]
    try:
        v["location"] = store.location(v["id"])
    except KeyError:
        pass
    return v


@app.get("/api/robot/queue")
def robot_queue():
    return journey.robot_queue()


@app.get("/api/robot/pace")
def robot_pace():
    """Poll ~5 Hz while escorting: speed_mps / command (go|pause), say_text when it changes."""
    w = _robot_worker()
    if not w:
        raise HTTPException(503, "No robot camera configured")
    return w.pace


@app.post("/api/robot/telemetry")
def robot_telemetry(t: Telemetry):
    w = _robot_worker()
    if w:
        w.robot_speed = t.speed_mps
    return {"ok": True}


# ---- positioning (CT room) ---------------------------------------------------------------
def _session(pid):
    s = positioning.get(pid)
    if not s:
        raise HTTPException(404, "No positioning session for this patient")
    return s


@app.get("/api/positioning/{pid}")
def positioning_view(pid: str):
    """The robot speaks instruction.text whenever instruction.id changes."""
    return _session(pid).view()


@app.post("/api/positioning/{pid}/start")
def positioning_start(pid: str, body: Start):
    try:
        v = journey.view(pid)
    except KeyError:
        raise HTTPException(404, "Unknown patient")
    ct = next((w for w in workers.values() if w.role == "ct_room"), None)
    rh = ct.table.right_handed if ct and ct.table else True
    try:
        s = positioning.start(pid, body.protocol or v["protocol"], rh, v["needs"].get("language", "en"), v["needs"])
    except KeyError as e:
        raise HTTPException(400, str(e))
    journey.event(pid, "positioning_started", "technologist", {"protocol": s.name})
    return s.view()


@app.post("/api/positioning/{pid}/accept")
def positioning_accept(pid: str):
    """Technologist accepts the position. Only then may the robot leave and the scan start."""
    s = _session(pid)
    if not s.accept():
        raise HTTPException(409, f"Not positioned yet (state: {s.state})")
    journey.event(pid, "positioning_accepted", "technologist", {"measures": s.measures})
    return s.view()


@app.post("/api/positioning/{pid}/reject")
def positioning_reject(pid: str):
    s = _session(pid)
    s.reject()
    journey.event(pid, "positioning_rejected", "technologist")
    return s.view()


@app.post("/api/positioning/{pid}/stop")
def positioning_stop(pid: str):
    """Technologist takes over: coaching stops immediately."""
    s = _session(pid)
    s.stop()
    journey.event(pid, "positioning_stopped", "technologist")
    return s.view()


# ---- cameras / health -------------------------------------------------------------------
@app.post("/api/cameras/swap")
def swap_cameras(a: str = "kiosk", b: str = "waiting"):
    """Swap which physical camera two roles use (macOS may renumber cameras between starts).
    Takes effect immediately and is saved in cameras.json."""
    wa, wb = workers.get(a), workers.get(b)
    if not wa or not wb:
        raise HTTPException(404, "Both cameras must be running to swap them")
    wa.cfg["source"], wb.cfg["source"] = wb.cfg["source"], wa.cfg["source"]
    for w in (wa, wb):
        w.tracks.clear()                      # old tracks belong to the other picture
        w.grabber.reopen = True
    with open(config.CAMERAS_FILE) as f:
        cams = json.load(f)
    cams[a]["source"], cams[b]["source"] = wa.cfg["source"], wb.cfg["source"]
    with open(config.CAMERAS_FILE, "w") as f:
        json.dump(cams, f, indent=2)
    return {a: wa.cfg["source"], b: wb.cfg["source"]}


@app.get("/api/cameras")
def cameras():
    return [{"name": n, "role": w.role, "status": w.status} for n, w in workers.items()]


@app.get("/api/health")
def health():
    """Per camera: is it delivering frames, how fast, which models. Not ok if any camera's last
    frame is older than 3 s (the challenge's feedback limit)."""
    cams = [w.health() for w in workers.values()]
    ok = all(c["status"] == "ok" and c["last_frame_age_s"] is not None and c["last_frame_age_s"] < 3
             for c in cams)
    fe = get_face_embedder() if config.CAMERAS_ENABLED else None
    return {"ok": ok, "embedder": get_embedder().name if config.CAMERAS_ENABLED else None,
            "faces": (fe.name + " " + config.FACE_MODEL) if fe else "off",
            "patients": len(store.patients), "cameras": cams}


@app.get("/api/debug/tracks")
def debug_tracks():
    """Live match scores per track: use this to set MATCH_THRESHOLD / KEEP_THRESHOLD."""
    return {n: w.debug() for n, w in workers.items() if w.role != "kiosk"}


@app.get("/api/cameras/{name}.jpg")
def camera_image(name: str, live: bool = False):
    w = workers.get(name)
    if not w:
        raise HTTPException(404, "No such camera")
    if live:
        # Keep live video independent of inference; old boxes belong to their original frame.
        with w.grabber.cond:
            frame, captured_at = w.grabber.frame, w.grabber.t
        if frame is None or time.time() - captured_at > 3:
            raise HTTPException(503, "No fresh camera image")
        import cv2
        h, width = frame.shape[:2]
        preview = cv2.resize(frame, (800, max(1, round(800 * h / width))))
        ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise HTTPException(503, "Could not encode camera image")
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    if w.jpeg is None:
        # detection still loading: show the plain camera picture so the camera can be checked already
        frame = w.grabber.frame
        if frame is None:
            raise HTTPException(404, "No image yet")
        import cv2
        h, wd = frame.shape[:2]
        img = cv2.resize(frame, (800, int(800 * h / wd)))
        cv2.putText(img, "detection loading...", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
        cv2.putText(img, "detection loading...", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        ok, buf = cv2.imencode(".jpg", img)
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    with w.lock:
        return Response(w.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/patients/{pid}/find.jpg")
def find_image(pid: str):
    """Testing aid: the tracking cameras' current pictures with ONLY this patient highlighted
    (everyone else greyed), so a tester can judge 'is that really them?'."""
    import cv2
    try:
        name = store.location(pid)["name"]
    except KeyError:
        raise HTTPException(404, "Unknown patient")
    tiles, now = [], time.time()
    for n, w in workers.items():
        if w.role not in ("tracking", "ct_room"):
            continue
        with w.lock:
            frame, people = (None if w.frame is None else w.frame.copy()), list(w.people)
        if frame is None:
            continue
        img = (frame * 0.55).astype(np.uint8)          # dim everyone else
        found = False
        for x1, y1, x2, y2, tid, _ in people:
            ts = w.tracks.get(tid)
            if ts and ts.pid == pid:
                found = True
                img[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                amb = ts.ambiguous(now)
                col = (0, 140, 255) if amb else (0, 220, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 8)
                label = f"{name}  {ts.score:.2f}" + ("  UNSURE" if amb else "")
                cv2.putText(img, label, (x1, max(50, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 8)
                cv2.putText(img, label, (x1, max(50, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 4)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 2)
        head = f"{n}: {'FOUND ' + name if found else name + ' NOT FOUND in this camera'}"
        cv2.putText(img, head, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 8)
        cv2.putText(img, head, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0) if found else (0, 0, 255), 4)
        h, wd = img.shape[:2]
        tiles.append(cv2.resize(img, (960, int(960 * h / wd))))
    if not tiles:
        raise HTTPException(503, "No tracking camera image yet")
    width = max(t.shape[1] for t in tiles)
    out = np.vstack([cv2.copyMakeBorder(t, 0, 0, 0, width - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles])
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/")
def dashboard():
    return FileResponse("static/index.html")
