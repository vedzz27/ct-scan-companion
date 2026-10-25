"""Record a camera to a video file, so test walks can be replayed after changing settings.

    python record.py waiting                  # writes recordings/waiting_<time>.avi, Ctrl+C or q to stop
Replay: copy cameras.json, set the camera's "source" to the .avi path, and start the app with
CAMERAS_FILE=that_copy.json. The file loops at its recorded speed.
Recordings contain people's images: delete them after the test (they are not part of the app's data).
"""
import json
import os
import sys
import time

import cv2

import config
from worker import FrameGrabber


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "waiting"
    with open(config.CAMERAS_FILE) as f:
        cfg = json.load(f)[name]
    cap = FrameGrabber(name, cfg).open()
    ok, frame = cap.read()
    if not ok:
        sys.exit("No picture from the camera")
    os.makedirs("recordings", exist_ok=True)
    path = os.path.join("recordings", f"{name}_{time.strftime('%H%M%S')}.avi")
    fps = 15
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, frame.shape[1::-1])
    print(f"Recording to {path} ... press q in the window (or Ctrl+C) to stop")
    next_t = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if time.time() >= next_t:              # keep a steady 15 fps in the file
                out.write(frame)
                next_t += 1 / fps
            cv2.imshow("recording - q to stop", cv2.resize(frame, None, fx=0.4, fy=0.4))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    out.release()
    cap.release()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
