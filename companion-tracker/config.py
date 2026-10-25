"""All tunable settings. Override any of them with environment variables."""
import os


def _f(name, default):
    return float(os.getenv(name, str(default)))


# ---- data handling -----------------------------------------------------------
RETENTION_HOURS = _f("RETENTION_HOURS", 48)          # hard ceiling for stored data
PURGE_INTERVAL_S = _f("PURGE_INTERVAL_S", 60)
MAX_ACTIVE_PATIENTS = int(os.getenv("MAX_ACTIVE_PATIENTS", "20"))  # was 5 (slide); a waiting room needs more

# ---- appearance matching -----------------------------------------------------
EMBEDDER = os.getenv("EMBEDDER", "osnet")              # "osnet" | "fused" | "hsv"
OSNET_WEIGHTS = os.getenv("OSNET_WEIGHTS", "weights/osnet_x1_0_msmt17.pt")
FUSED_OSNET_WEIGHT = _f("FUSED_OSNET_WEIGHT", 0.8)

# Starting points per embedder: (bind threshold, keep threshold, margin to 2nd best).
# Different embedders give very different score ranges, so tune with /api/debug/tracks
# or SCORE_LOG=1 (see README) and then set MATCH_THRESHOLD / KEEP_THRESHOLD / MATCH_MARGIN.
_DEFAULTS = {"osnet": (0.62, 0.50, 0.08), "fused": (0.66, 0.55, 0.07), "hsv": (0.88, 0.82, 0.03)}


def thresholds(embedder_name):
    bind, keep, margin = _DEFAULTS.get(embedder_name, _DEFAULTS["hsv"])
    return (_f("MATCH_THRESHOLD", bind), _f("KEEP_THRESHOLD", keep), _f("MATCH_MARGIN", margin))


BIND_STREAK = int(os.getenv("BIND_STREAK", "2"))     # consecutive good matches before binding a track
UNBIND_STREAK = int(os.getenv("UNBIND_STREAK", "3"))  # consecutive bad matches before letting go
EMBED_EVERY_S = _f("EMBED_EVERY_S", 0.25)  # cap expensive re-ID at 4 Hz per person
EVAL_EVERY_S = _f("EVAL_EVERY_S", 0.7)              # how often each track is re-matched
GALLERY_ADD_BONUS = _f("GALLERY_ADD_BONUS", 0.06)   # only learn new views when this far above bind

# ---- face recognition (next to body re-ID; see face.py) ---------------------
FACE_ENABLED = os.getenv("FACE_ENABLED", "1") == "1"
FACE_MODEL = os.getenv("FACE_MODEL", "buffalo_l")       # buffalo_l (accurate) | buffalo_s (fast)
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "320"))   # detector input for a head region
FACE_DET_SCORE = _f("FACE_DET_SCORE", 0.6)
FACE_MIN_PX = int(os.getenv("FACE_MIN_PX", "40"))        # smaller faces are too unreliable to use
FACE_MAX_YAW = _f("FACE_MAX_YAW", 0.35)                  # 0 = frontal; skip strong side views
FACE_MIN_SHARPNESS = _f("FACE_MIN_SHARPNESS", 35)        # Laplacian var. of the aligned face; below
                                                         # ~35 motion blur makes ArcFace unreliable (measured)
FACE_MIN_FOR_MATCH = int(os.getenv("FACE_MIN_FOR_MATCH", "2"))  # faces a track needs before its face counts
FACE_EVERY_S = _f("FACE_EVERY_S", 0.5)                   # face check per known person
FACE_EVERY_S_NEW = _f("FACE_EVERY_S_NEW", 0.15)          # unknown people / newcomers: as often as possible
# ArcFace cosine: same person usually > 0.45, different people usually < 0.2.
FACE_SAME_T = _f("FACE_SAME_T", 0.45)    # face this similar: same person, even with other clothes
FACE_DIFF_T = _f("FACE_DIFF_T", 0.20)    # face this different: not that person, even with same clothes
FACE_ENROLL_MIN = int(os.getenv("FACE_ENROLL_MIN", "2"))     # faces wanted at check-in
FACE_ENROLL_WAIT_S = _f("FACE_ENROLL_WAIT_S", 3.0)          # keep looking this long during check-in

# ---- crop quality (only clean crops are embedded) ---------------------------
MIN_CROP_HEIGHT = int(os.getenv("MIN_CROP_HEIGHT", "110"))   # px in the full-resolution frame
MIN_DET_CONF = _f("MIN_DET_CONF", 0.5)
MAX_OVERLAP = _f("MAX_OVERLAP", 0.25)          # fraction of the box covered by another person
EDGE_MARGIN_PX = int(os.getenv("EDGE_MARGIN_PX", "6"))    # boxes touching the border are cut off

# ---- swap detection ----------------------------------------------------------
CLOSE_IOU = _f("CLOSE_IOU", 0.05)              # two people this close: tracker IDs may swap...
MOVE_FRACTION = _f("MOVE_FRACTION", 0.15)      # ...if one moved > this x body height in 1.5 s
AMBIGUOUS_HOLD_S = _f("AMBIGUOUS_HOLD_S", 4.0)  # stay "ambiguous" this long after being close

# ---- confidence / robot actions ---------------------------------------------
CONFIDENCE_HALF_LIFE_S = _f("CONFIDENCE_HALF_LIFE_S", 60)
FRESH_S = _f("FRESH_S", 30)        # "seen recently" window
GIVE_UP_S = _f("GIVE_UP_S", 180)   # after this, alert staff
MAX_CONFIRM_ATTEMPTS = 3           # slide: max 3 attempts

# ---- presence ----------------------------------------------------------------
ABSENT_AFTER_S = _f("ABSENT_AFTER_S", 90)   # not seen by any camera this long -> status "absent"

# ---- new person detection (tracking cameras) -----------------------------------
# A visible track that matches nobody in the gallery is reported as a newcomer, so the robot
# can go to them, ask their name and enroll them from the views the camera already collected.
NEW_PERSON_ENABLED = os.getenv("NEW_PERSON_ENABLED", "1") == "1"
NEW_PERSON_AFTER_S = _f("NEW_PERSON_AFTER_S", 4.0)   # unmatched this long (clean views) -> newcomer
NEW_PERSON_MIN_VIEWS = int(os.getenv("NEW_PERSON_MIN_VIEWS", "4"))  # clean crops needed first
NEW_PERSON_STALE_S = _f("NEW_PERSON_STALE_S", 5.0)   # drop a newcomer not seen this long

# ---- detector ----------------------------------------------------------------
# yolo26n: fast on a laptop CPU. yolo26s: better at small/far people, needs ~2x time (GPU ideal).
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt")
YOLO_FALLBACK = os.getenv("YOLO_FALLBACK", "yolov8n.pt")   # used if YOLO_MODEL can't be loaded
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))   # wide 126° lens makes people small: keep >= 1280
YOLO_CONF = _f("YOLO_CONF", 0.15)  # retain weak detections for ByteTrack's second pass
TRACKER_CFG = os.getenv("TRACKER_CFG", "tracker.yaml")
TARGET_FPS = _f("TARGET_FPS", 8)    # processing rate per camera; the grabber always keeps the newest frame

# ---- check-in capture --------------------------------------------------------
CHECKIN_SAMPLES = int(os.getenv("CHECKIN_SAMPLES", "6"))   # crops collected per check-in
CHECKIN_TIMEOUT_S = _f("CHECKIN_TIMEOUT_S", 8.0)
CHECKIN_MAX_FRAME_AGE_S = _f("CHECKIN_MAX_FRAME_AGE_S", 2.0)
CHECKIN_DOMINANCE = _f("CHECKIN_DOMINANCE", 1.4)  # nearest person must be this much bigger than the next

# ---- misc --------------------------------------------------------------------
CAMERAS_FILE = os.getenv("CAMERAS_FILE", "cameras.json")
CAMERAS_ENABLED = os.getenv("CAMERAS_ENABLED", "1") == "1"
SCORE_LOG = os.getenv("SCORE_LOG", "")   # path to a CSV: every match decision is appended (for tuning/eval)

# ---- journey ------------------------------------------------------------------
JOURNEY_FILE = os.getenv("JOURNEY_FILE", "journey.json")        # stages, zones, robot tasks (data)
JOURNEY_LOG = os.getenv("JOURNEY_LOG", "journey_log.jsonl")     # event log on disk ("" = memory only)
PROTOCOLS_FILE = os.getenv("PROTOCOLS_FILE", "protocols.json")  # positioning targets + phrases
FACE_MAX_PER_FRAME = int(os.getenv("FACE_MAX_PER_FRAME", "3"))   # face checks per camera frame (CPU budget)
HANDOFF_WAIT_S = _f("HANDOFF_WAIT_S", 60)   # after a front-camera check-in, wait this long for the back camera
HANDOFF_DOMINANCE = _f("HANDOFF_DOMINANCE", 2.0)   # "clearly the closest person" at hand-off
