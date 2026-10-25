"""Map the waiting-room floor to metres, or draw a zone polygon. Optional but recommended.

Floor:  python calibrate_floor.py waiting
  Put 4-6 markers (tape crosses) on the floor, measure their positions in metres from one
  corner of the room, click each marker in the picture, and type its X Y in the terminal.
  Writes calib/<camera>_floor.json; cameras.json "floor_calibration" points to it.
  Afterwards every patient location also has "xy" in metres, which the robot can use.

Zone:   python calibrate_floor.py waiting --zone waiting_A
  Click the corners of the zone on the floor, press Enter: prints a "poly" line for cameras.json.

Table:  python calibrate_floor.py ct --table
  Click the 4 table-top corners in this order: 1 and 2 at the GANTRY end, then 3 at the foot end on
  the same side as 2, then 4 at the foot end on the same side as 1. Type the table length and width.
  Writes calib/<camera>_table.json (cameras.json "table_calibration"). Left/right for the patient is
  worked out automatically from how the camera sees the table.

Use --image picture.jpg instead of the live camera if you like.
If calib/<camera>_lens.json exists (calibrate_lens.py), lens distortion is removed first.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import config


def grab(cam_cfg, image):
    if image:
        return cv2.imread(image)
    from worker import FrameGrabber
    cap = FrameGrabber("calib", cam_cfg).open()
    frame = None
    for _ in range(30):                       # let exposure settle
        ok, f = cap.read()
        frame = f if ok else frame
    cap.release()
    return frame


def table_world(length, width):
    """World coordinates of the clicked corners: x from the gantry end, y across (0 = centre)."""
    return [(0.0, -width / 2), (0.0, width / 2), (length, width / 2), (length, -width / 2)]


def right_handed(H):
    """Is (table x, table y, up) right-handed as seen by a camera above the table?
    Seen from above, right-handed means x turns counter-clockwise into y. Image v points down,
    so that shows up as a negative cross product in image coordinates."""
    Hinv = np.linalg.inv(np.asarray(H, np.float64))
    p = cv2.perspectiveTransform(np.array([[[0.5, 0.0]], [[0.7, 0.0]], [[0.5, 0.1]]]), Hinv).reshape(-1, 2)
    ex, ey = p[1] - p[0], p[2] - p[0]
    return bool(ex[0] * ey[1] - ex[1] * ey[0] < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--zone")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--image")
    a = ap.parse_args()
    with open(config.CAMERAS_FILE) as f:
        cam = json.load(f)[a.camera]
    frame = grab(cam, a.image)
    if frame is None:
        sys.exit("No picture from the camera")
    h, w = frame.shape[:2]
    scale = min(1.0, 1400 / w)
    clicks = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x / scale, y / scale))

    win = "click points - Enter when done, Backspace to undo, Esc to quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = cv2.resize(frame, None, fx=scale, fy=scale)
        for i, (x, y) in enumerate(clicks):
            p = (int(x * scale), int(y * scale))
            cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if a.zone and len(clicks) > 1:
            pts = (np.array(clicks) * scale).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 200, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            return
        if k == 8 and clicks:
            clicks.pop()
        if k == 13:
            break
    cv2.destroyAllWindows()

    if a.zone:
        poly = [[round(x / w, 3), round(y / h, 3)] for x, y in clicks]
        print(json.dumps({"zone": a.zone, "poly": poly, "robot_point": a.zone}))
        print("Paste this into the camera's \"regions\" list in cameras.json.")
        return

    if len(clicks) < 4:
        sys.exit("Need at least 4 points")
    if a.table:
        clicks = clicks[:4]
        length = float(input("Table top length in metres [2.0]: ") or 2.0)
        width = float(input("Table top width in metres [0.5]: ") or 0.5)
        world = table_world(length, width)
    else:
        world = []
        for i, _ in enumerate(clicks):
            x, y = map(float, input(f"Point {i + 1}: X Y in metres: ").replace(",", " ").split())
            world.append((x, y))

    pts = np.array(clicks, np.float64).reshape(-1, 1, 2)
    cal = {}
    lens_file = os.path.join("calib", f"{a.camera}_lens.json")
    if os.path.exists(lens_file):
        with open(lens_file) as f:
            cal = json.load(f)
        K, D = np.array(cal["K"]), np.array(cal["D"])
        pts = (cv2.fisheye.undistortPoints(pts, K, D, P=K) if cal.get("model") == "fisheye"
               else cv2.undistortPoints(pts, K, D, P=K))
        print(f"Lens correction from {lens_file} applied.")
    H, _ = cv2.findHomography(pts, np.array(world, np.float64), 0)
    back = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    err = np.linalg.norm(back - np.array(world), axis=1)
    for i, e in enumerate(err):
        print(f"  point {i + 1}: error {e * 100:.0f} cm")
    if len(clicks) > 4:
        print(f"Mean error {err.mean() * 100:.0f} cm (above ~30 cm: re-measure or calibrate the lens)")
    cal.update({"homography": H.tolist(), "image_points": clicks, "world_points": world,
                "image_size": [w, h]})
    os.makedirs("calib", exist_ok=True)
    if a.table:
        cal.update(table_length_m=length, table_width_m=width, right_handed=right_handed(H))
        out = cam.get("table_calibration") or os.path.join("calib", f"{a.camera}_table.json")
    else:
        out = cam.get("floor_calibration") or os.path.join("calib", f"{a.camera}_floor.json")
    with open(out, "w") as f:
        json.dump(cal, f, indent=1)
    print(f"Saved {out}. Restart the app.")


if __name__ == "__main__":
    main()
