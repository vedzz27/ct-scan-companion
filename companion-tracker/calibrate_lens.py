"""Optional: measure the wide-angle lens distortion with a printed checkerboard.

    python calibrate_lens.py waiting                 # 9x6 inner corners (a 10x7 squares board)
    python calibrate_lens.py waiting --cols 7 --rows 5 --fisheye

Hold the board at different places and angles, especially near the picture edges and corners.
Press Space to keep a view (green = corners found), Enter after ~15 views. Writes calib/<camera>_lens.json,
which calibrate_floor.py and the app pick up. Only worth it if floor positions are off at the image edges.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import config
from worker import FrameGrabber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--fisheye", action="store_true")
    a = ap.parse_args()
    with open(config.CAMERAS_FILE) as f:
        cap = FrameGrabber("lens", json.load(f)[a.camera]).open()
    pattern = (a.cols, a.rows)
    objp = np.zeros((a.cols * a.rows, 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2)
    views, size = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        size = frame.shape[1::-1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        show = frame.copy()
        if found:
            cv2.drawChessboardCorners(show, pattern, corners, found)
        cv2.putText(show, f"views: {len(views)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("lens - Space keep, Enter finish, Esc quit", cv2.resize(show, None, fx=0.6, fy=0.6))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return
        if k == 32 and found:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            views.append(corners)
        if k == 13 and len(views) >= 8:
            break
    cap.release()
    cv2.destroyAllWindows()
    obj = [objp] * len(views)
    if a.fisheye:
        K, D = np.zeros((3, 3)), np.zeros((4, 1))
        rms, K, D, _, _ = cv2.fisheye.calibrate(obj, views, size, K, D,
                                                flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
    else:
        rms, K, D, _, _ = cv2.calibrateCamera([o.reshape(-1, 3) for o in obj], views, size, None, None,
                                              flags=cv2.CALIB_RATIONAL_MODEL)
    print(f"RMS reprojection error: {rms:.2f} px (below ~1 px is good)")
    if rms > 2:
        print("High error: take more views closer to the edges, keep the board flat and still.")
    os.makedirs("calib", exist_ok=True)
    out = os.path.join("calib", f"{a.camera}_lens.json")
    with open(out, "w") as f:
        json.dump({"model": "fisheye" if a.fisheye else "pinhole", "K": K.tolist(),
                   "D": np.asarray(D).ravel().tolist(), "image_size": list(size), "rms": rms}, f, indent=1)
    print(f"Saved {out}. Now run: python calibrate_floor.py {a.camera}")


if __name__ == "__main__":
    sys.exit(main())
