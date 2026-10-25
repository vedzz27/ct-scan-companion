# Companion tracker (prototype)

Follows each checked-in patient through the CT visit (check-in, waiting room, escort, CT room, table
positioning, aftercare) and tells the robot where they are, how sure it is, and what to do next.
QR scanning stays in the robot code; it reports the result to `POST /api/patients/{id}/confirm`.

## Cameras (cameras.json, one entry per camera, `role` decides what it does)
| Camera | role | Does |
|---|---|---|
| Check-in | `kiosk` | ~1 s of clean pictures -> appearance vectors (images discarded) |
| Waiting room (SVPRO) | `tracking` | follows everyone, matches them to patients, zones -> journey |
| CT room | `ct_room` | pose model: who is on the table, positioning guidance |
| Robot rear camera | `robot` | distance to the escorted patient -> walking speed |
More rooms later = more `tracking` entries with their own zones (e.g. `changing_room`, `exit`).

## Run
    pip install -r requirements.txt
    python download_reid.py            # once: person re-ID weights (~9 MB) into weights/
    python test_store.py               # logic tests, no camera needed
    python test_matching.py
    python test_journey.py
    python test_roles.py
    python test_checkin.py
    uvicorn main:app --host 0.0.0.0 --port 8000
    # open http://localhost:8000

Settings are environment variables. In PowerShell set them before starting, e.g.
`$env:SCORE_LOG="scores.csv"; uvicorn main:app --host 0.0.0.0 --port 8000`.

`source` in cameras.json is a webcam number (0, 1...), a stream URL, or a video file (loops; for replaying tests).
Demo the 48 h purge quickly: `RETENTION_HOURS=0.01 PURGE_INTERVAL_S=10 uvicorn main:app`

## Reliable first-camera check-in
The first camera is selected by `role: "kiosk"`, not by its USB number. The current
configuration uses USB source **1** for check-in and source **0** for the waiting room.
Check the live previews and swap the source numbers in cameras.json if your devices differ.

The dashboard shows whether a fresh, tracked person is ready, with guidance for framing,
distance and obstructions. Check-in collects all six views from distinct new detections of
the same stable track. Missing/unclean detections reset the burst; a track change or sudden
box jump restarts it. Concurrent check-ins on the same camera are rejected. The default
capture timeout is 8 s; `CHECKIN_MAX_FRAME_AGE_S` (2 s) limits detection age.

The detector cutoff is 0.15 so ByteTrack can use weak detections to extend existing tracks;
identity crops still require confidence 0.5. Override `conf` per camera if necessary.
Matching and gallery updates now require a new clean appearance sample. Swap rechecks
remain uncertain until fresh evidence resolves them.

Run `python test_checkin.py` for camera-free enrollment regression tests, then use the
live multi-person test below to measure accuracy with your actual cameras and lighting.

## How detection works now
1. **Capture**: a grabber thread per camera always holds the newest frame, so the detector never
   works on old, queued pictures. `/api/health` shows camera fps, processed fps and frame age.
2. **Detect + track**: YOLO26n at 1280 px (the SVPRO's 126° lens makes people small) + ByteTrack
   tuned for a fixed camera (`tracker.yaml`: lost tracks survive ~8 s behind a pillar).
3. **Clean crops only**: a crop is embedded only if it's tall enough, not cut off at the edge, not
   largely covered by someone else, and a confident detection. Bad crops were what made matching unstable.
4. **Re-ID model**: OSNet (trained on 126k images of 4,101 people, MSMT17) turns each crop into a vector
   that captures clothing texture, shape and colour, much more than the old colour histogram.
   Colours are white-balanced per camera first, so the kiosk and the waiting camera agree better.
5. **Assignment**: all tracks in a camera are matched to the patients together (Hungarian), so two
   people can never both be "Ana". A track needs 2 good rounds in a row to bind, and 3 bad ones to let go.
6. **Swap guard**: when two people touch while moving (walking past each other), both are flagged
   `ambiguous`, their collected views are discarded, and the identity is re-decided from clean crops
   once they separate. While ambiguous, `may_use_name` is false and the action is
   `go_to_last_zone_and_scan`. Neighbours sitting still next to each other don't trigger this.
7. **Gallery**: the check-in views (6 per patient, taken over ~1 s) are kept permanently; new views
   from the waiting camera are learned only from very confident matches, so identities can't drift.
8. **Position**: zone from the foot point (box or polygon regions), the robot's named stop point for
   that zone (`robot_point`), and floor `xy` in metres if the camera is calibrated.

## The journey
`journey.json` holds the stages, which zone means which stage, and which robot task belongs to each
stage. Everything is written to one event log (`journey_log.jsonl`: anonymous id, time, event, source,
no names), which also gives time per stage (`/api/journey/metrics`).

    checked_in -> waiting -> called -> (to_changing_room -> changing -> ready) -> to_ct_room -> in_ct_room
    -> on_table -> positioning -> positioned -> position_accepted -> robot_left -> scanning
    -> scan_finished -> aftercare -> exiting -> done

- Cameras move the stage forward when a patient is seen in a zone for 2 s; they never move it back
  (walking past the waiting room on the way to the CT room doesn't undo anything).
- The robot, tablet, technologist and CT workflow send events: `POST /api/journey/{id}/event`
  `{"type": "call" | "guide_start" (detail {"target": "ct_room"}) | "guide_arrived" | "changing_ready" |
  "robot_left" | "scan_started" | "scan_finished" | "aftercare_started" | "aftercare_done" | "finish"}`.
  Whether contrast was given etc. comes from the workflow in `detail`; the system never guesses it.
- The robot asks `GET /api/robot/next`: the oldest patient with a task, the task
  (`guide_to` + robot_point, `position_check`, `leave_room`, `aftercare`) and the patient's location.
- Check-in takes `needs`: `{"language": "de", "mobility": "slow|walker|wheelchair|assist"}` and an optional
  `protocol`. Language picks the phrases; mobility slows the robot and, at the table, calls staff
  instead of coaching.

## Walking at the patient's pace (robot camera)
Mount a camera on the robot looking backwards at about 1 m height, roughly level (`tilt_deg` 0), and
enter height, tilt and horizontal field of view in cameras.json. A phone streaming video over Wi-Fi works
as `source`. Distance comes from where the feet meet the floor; when the patient is close and the feet
leave the picture, from the head position and the height learned earlier.
While escorting (`guide_start` ... `guide_arrived`), poll `GET /api/robot/pace` ~5 times a second:
- `speed_mps`: walking speed matching the patient, keeping `target_gap_m` (1.2 m). If the robot can't take
  a speed, use `command`: `go` / `pause`.
- `state`: `following`, `waiting` (patient > 3 m behind: stop until they're within 1.8 m), `lost`
  (not seen 2 s: stop), `alert_staff` (not seen 15 s).
- `say_text` when something should be said ("Take your time, I'll wait for you."), `verified` = the
  followed person matches the patient's appearance (a stranger walking closer is ignored).
- Optional: `POST /api/robot/telemetry {"speed_mps": ...}` with the robot's real speed makes the
  patient-speed estimate more accurate.

## Table positioning (CT room camera)
Ceiling camera above the table, pose model (`yolo26n-pose.pt`). Calibrate the table once:
`python calibrate_floor.py ct --table` (click the 4 table-top corners, gantry end first). The patient's
left/right is then worked out automatically.
When the escorted patient lies on the table, guidance starts by itself with the patient's protocol
(`protocols.json`: head/feet first, arms up/down, how centred, tolerances, phrases in en/de):
- One instruction at a time, in the patient's own left/right ("Please move a little to your left"),
  then it waits for them to move and measures again.
- Within tolerance for 2 s -> `positioned` ("Perfect, please stay still"). If they move before the
  technologist accepts, guidance resumes.
- No progress after 3 tries, nobody readable on the table for 8 s, two people on the table, or a mobility
  need -> `needs_staff` and coaching stops.
- The robot speaks `instruction.text` from `GET /api/positioning/{id}` whenever `instruction.id` changes.
- The technologist decides: `POST /api/positioning/{id}/accept` (only when positioned) -> robot task
  `leave_room`; `/reject` -> coaching restarts; `/stop` -> coaching ends now. The system never moves the
  table or starts the scan.
Tolerances in protocols.json are demo values; real ones must come from the clinical team. Keypoints sit
a few cm above the table surface, so a camera straight above the table gives the most accurate centring.

## Test: does it find the right person among people? (cameras only)
Check in 3-4 people, put them all in the waiting-room view, run `python find_test.py`. For each patient
it shows the waiting-room picture with only the person it thinks is them highlighted; press y (correct),
n (wrong person), m (missed), a (absent). Name each round by situation (seated, swapped, crossing...).
`python find_test.py --summary` gives correct/wrong/missed per situation. In a browser:
`http://localhost:8000/api/patients/<id>/find.jpg`.

## Test with 2 cameras, no robot yet
Start the app, then in a second terminal `python fake_robot.py`. It shows every patient's zone,
where the robot would drive and whether it may use the name. `f 1` fetches patient 1: you answer
whether they were really there and the QR result (sent to the app like the robot will). `s` prints
the accuracy summary; every fetch is saved in fetch_log.csv.

## Demo script (3 cameras + robot)
1. Check in at the kiosk (`needs` with language). 2. Sit in the waiting room: stage `waiting`.
3. Robot: `/api/robot/next` -> `call` -> QR (your code) -> `confirm` -> `guide_start` ct_room; walk slowly
   and stop once: the robot waits. 4. CT camera sees you on the table: guidance starts; follow it.
5. Technologist accepts -> robot leaves -> `scan_started` / `scan_finished` -> robot returns for aftercare.
6. Show `/api/journey/{id}` (timeline) and `/api/journey/metrics` (time per stage).

## Robot / dashboard API
- `GET /api/patients/{id}/location` → zone, xy (m), robot_target, confidence, age_s, ambiguous,
  may_use_name, action (`go_to_zone_and_confirm` | `go_to_last_zone_and_scan` | `alert_staff`)
- `POST /api/patients/{id}/confirm {"accepted": true}` → also resets confidence and clears ambiguous
- `POST /api/checkin {"name": "...", "camera": "kiosk"}` → 409 with a message for the patient
  ("Please come closer", "More than one person at the kiosk", ...); `warning` if they look like an existing patient
- `GET /api/patients` (now with `stage` and `robot_task`), `GET /api/journey`, `GET /api/journey/{id}`
- `GET /api/health` (per camera incl. pace / CT status), `GET /api/debug/tracks` (live scores per track)

## The waiting-room camera (SVPRO SV-U3GS0234C-CH120)
126° horizontal view, 1920×1200, global shutter (no smearing when people walk).
Approximate person height in the picture: standing 1.7 m → 277 px at 3 m, 166 px at 5 m, 119 px at 7 m;
seated → 212 / 127 / 91 px. Crops under 110 px are tracked but not used for identity, so keep seats
within ~6 m. Mount it 2.2-2.6 m high, tilted 20-30° down, facing away from windows. Lock the colour
(`lock_colour`, optionally `wb_temperature`); if the picture flickers or is too dark, set `exposure`
(DirectShow, e.g. -6).

## Tune (in this order)
1. Start with `SCORE_LOG=scores.csv`, let 3-4 teammates check in and walk around, sit, swap seats.
   In the CSV, `best_score` for the right person vs `second_score` shows where to put the line.
   Set `MATCH_THRESHOLD` (bind) between them, `KEEP_THRESHOLD` ~0.1 lower, `MATCH_MARGIN` for the gap.
   Defaults for OSNet: 0.62 / 0.50 / 0.08.
2. Record a test once (`python record.py waiting`) and replay it as the camera source after each change.
3. Four cameras on one laptop CPU is heavy (check `processed_fps` in `/api/health`; 2+ is enough for
   the journey, 5+ for smooth pace). The kiosk idles at 2 fps between check-ins. Too slow? `YOLO_IMGSZ=960`, or `python download_reid.py osnet_x0_25`
   and `OSNET_WEIGHTS=weights/osnet_x0_25_msmt17.pt`. With an NVIDIA GPU use `YOLO_MODEL=yolo26s.pt`.

## Floor in metres (optional, ~20 min)
    python calibrate_floor.py waiting                  # click 4-6 taped floor marks, type their X Y
    python calibrate_floor.py waiting --zone waiting_A # click a zone outline, paste the printed line
    python calibrate_lens.py waiting                   # only if positions are off near the edges
Then each location has `xy` in metres, the same coordinate frame for every camera.

## Privacy
No images are stored by the app: crops become vectors in memory and are discarded. Data is deleted on
"Finish" or after 48 h. (`record.py` recordings do contain images; delete them after testing.)

## Not built yet
A changing-room camera (the robot reports `guide_arrived` / the tablet `changing_ready` instead), the robot
app side (polling and speaking), face-up/down detection beyond the face-visible heuristic, measured
accuracy numbers.

## Camera responsiveness
The dashboard defaults to live video, refreshed up to 10 times per second with one
outstanding image request per camera. Uncheck **Smooth live video** to inspect detection
boxes on the original processed frames. Live video continues while the detector loads or
runs slowly; it does not imply that tracking runs at the same rate.

The kiosk uses 640 px inference and 4 fps while idle (8 fps during check-in).
The waiting camera keeps 1280 px inference for distant people. Re-ID crops are sampled
at most every 0.25 s per track (`EMBED_EVERY_S`); positions still update every detection.
`/api/health` reports camera fps, processed fps, detector time and `detection_age_s` so
capture problems can be distinguished from slow inference. Run `python test_camera_speed.py`
for preview and scheduling regression tests. Restart the server after changing settings.
