# CT Scan Companion

Camera-based person tracking for a hospital waiting room, built to guide a service robot
(Pudu KettyBot Pro) through a CT-scan visit. The system notices when a **new patient** walks in,
enrolls them by **face and appearance**, keeps track of who is **present or absent**, and tells
the robot where each patient is.

> Research prototype. Not a medical device. Use only with the consent of everyone in view.

## What it does

- **Detects people** live from one or two cameras (YOLO + ByteTrack).
- **Announces new patients**: anyone the system doesn't know gets a "New patient" banner on the
  dashboard and a spoken announcement.
- **Recognises people** by face (InsightFace: SCRFD detector + ArcFace) *and* by clothing/body
  appearance (OSNet re-ID). Face and body are fused:
  - same face → recognised even after changing clothes
  - different face → never confused with someone in the same clothes
  - blurred or side-on faces are skipped automatically, so movement never causes a wrong match
- **Present / absent**: a patient not seen for a while is marked *absent* (e.g. gone to the
  toilet) and *present* again when they come back, without checking in twice.
- **Two-camera hand-off** (optional): a front camera takes the face at check-in, a back/room
  camera takes the whole body of the same person and tracks them through the room.
- **Robot API**: the robot asks `GET /api/robot/next` for its next task (e.g. greet a new
  patient, fetch a patient) together with the patient's location.
- **Privacy by design**: no images are stored. Crops are turned into anonymous vectors in memory
  and discarded; all data is deleted on "Finish" or after 48 h.

## Quick start (macOS, MacBook camera only)

Requires Python 3.10+ (3.12 recommended).

```bash
# 1. one-time setup (keep the environment outside iCloud-synced folders)
python3.12 -m venv ~/venvs/companion
source ~/venvs/companion/bin/activate
cd companion-tracker
pip install -r requirements.txt
python download_reid.py            # body re-ID weights (~9 MB)

# 2. run
ABSENT_AFTER_S=20 uvicorn main:app --host 0.0.0.0 --port 8001 --no-access-log
```

Open <http://localhost:8001>. On first start macOS asks for camera access for Terminal: allow it.
The face model (~280 MB) and the YOLO detector download automatically on the first run.

**Try it:** sit in front of the camera for ~5 s. The **New patient** banner appears. Type your
name in the row under *New patients in the room* and press **Check in** while facing the camera.
Walk out of view for 20 s (you become *absent*) and come back (*present* again).

## Cameras

Cameras are configured in `companion-tracker/cameras.json` and picked **by name**, so macOS
renumbering cameras (e.g. when an iPhone connects) doesn't matter:

```json
"waiting": { "source": "name:FaceTime", "role": "tracking", ... }
```

Run `python camtest.py` to list the cameras with their names and save one test photo each.

| Setup | `kiosk` (front, face) | `waiting` (room, whole body) |
|---|---|---|
| MacBook only (default) | `"enabled": false` | `"name:FaceTime"` |
| Two cameras | `"name:FaceTime"` | `"name:Global Shutter"` (external camera) |

With two cameras, check in with **Take photos and check in**: the face comes from the front
camera and the room camera links the same person's whole body (by face, or the only/closest
unknown person in view, waiting up to 60 s).

## Main settings

Environment variables (see `companion-tracker/config.py` for all of them):

| Variable | Default | Meaning |
|---|---|---|
| `ABSENT_AFTER_S` | 90 | seconds unseen before a patient counts as absent |
| `NEW_PERSON_AFTER_S` | 4 | seconds an unknown person must be visible before "New patient" |
| `FACE_ENABLED` | 1 | face recognition on/off (body re-ID always runs) |
| `FACE_MIN_PX` | 40 | smallest usable face width in pixels |
| `HANDOFF_WAIT_S` | 60 | two-camera mode: how long the room camera may take to find the person |
| `MAX_ACTIVE_PATIENTS` | 20 | gallery size |

## API (selection)

| Endpoint | Purpose |
|---|---|
| `GET /api/room` | who is present, absent, and new |
| `GET /api/newcomers` | unknown people in view, with zone and position |
| `POST /api/newcomers/{id}/checkin` `{"name": "..."}` | enroll a newcomer (body + face from the room camera) |
| `POST /api/checkin` `{"name": "...", "camera": "kiosk"}` | check-in at the front camera |
| `GET /api/patients` | all patients with status, zone, confidence, robot action |
| `GET /api/robot/next` | the robot's next task |
| `GET /api/health`, `GET /api/debug/tracks` | camera status and live match scores |

## Tests

No cameras needed:

```bash
cd companion-tracker
python test_store.py && python test_matching.py && python test_newcomer.py \
  && python test_face.py && python test_two_cameras.py && python test_journey.py \
  && python test_roles.py && python test_checkin.py
```

`test_face.py` also runs the real face model on a sample photo (same face ≈ 0.97 similarity,
different people ≤ 0.12, motion-blurred faces rejected).

## Project layout

```
companion-tracker/
  main.py          FastAPI app: dashboard, check-in, robot and room API
  worker.py        one thread per camera: detect, track, embed, match, newcomers, hand-off
  face.py          face detection + ArcFace embeddings (quality-filtered)
  embedding.py     OSNet body re-ID embeddings
  matching.py      track-to-patient assignment, face/body fusion, swap guard
  store.py         in-memory patients, newcomers, presence
  cams.py          pick cameras by name (macOS)
  journey.py       CT visit stages and robot tasks (journey.json)
  positioning.py   CT table positioning guidance (protocols.json)
  static/index.html  dashboard
```

The full CT-room workflow (escorting at the patient's pace, table positioning, calibration,
tuning) is documented in [`companion-tracker/README.md`](companion-tracker/README.md).

## Licences and notes

- InsightFace's pretrained models (`buffalo_l`) are licensed for **non-commercial research only**.
  For commercial use, swap in e.g. OpenCV YuNet + SFace.
- Faces are biometric data under GDPR (Art. 9): get explicit consent and run a data protection
  impact assessment before using this with real patients.

## Author

Vedika Darokar · [GitHub](https://github.com/vedzz27) · [LinkedIn](https://www.linkedin.com/in/vedikadarokar)
