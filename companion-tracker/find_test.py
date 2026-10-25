"""Test: does the camera system find the RIGHT person among several people?  (no robot, no QR)

1. Start the app. Check in 3-4 people at the kiosk camera (dashboard button), one at a time.
2. Everyone goes into the waiting-room camera's view.
3. Run:  python find_test.py
   Type a short name for the situation (e.g. "seated", "swapped seats", "crossing", "standing group").
   For each patient a window shows the waiting-room picture with ONLY the person the system thinks
   is that patient highlighted. Press:
       y  correct person
       n  WRONG person highlighted
       m  missed: the patient is visible, but the system says NOT FOUND (or highlights nobody)
       a  absent: the patient really isn't in view (not counted as an error)
       r  refresh the picture (e.g. wait a few seconds after people moved)
       Esc  stop
   Then change the situation and run another round. Results go to find_log.csv.
   python find_test.py --summary   prints the totals per situation (your slide numbers).
"""
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

API = os.getenv("API", "http://localhost:8000/api")
LOG = "find_log.csv"


def get(path, raw=False):
    try:
        with urllib.request.urlopen(API + path, timeout=5) as r:
            data = r.read()
            return data if raw else json.loads(data)
    except urllib.error.HTTPError as e:
        return None
    except urllib.error.URLError:
        sys.exit(f"Can't reach {API}. Is the app running?")


def summary():
    if not os.path.exists(LOG):
        sys.exit("No results yet.")
    rows = list(csv.DictReader(open(LOG)))
    groups = {}
    for r in rows:
        groups.setdefault(r["situation"], []).append(r["verdict"])
    groups["ALL"] = [r["verdict"] for r in rows]
    print(f"\n{'situation':<22}{'tests':>6}{'correct':>9}{'wrong':>7}{'missed':>8}   correct %")
    for s, v in groups.items():
        counted = [x for x in v if x != "absent"]
        n = len(counted) or 1
        print(f"{s:<22}{len(counted):>6}{v.count('correct'):>9}{v.count('wrong'):>7}{v.count('missed'):>8}"
              f"   {100 * v.count('correct') / n:.0f}%")
    unsure_wrong = sum(1 for r in rows if r["verdict"] == "wrong" and r["unsure"] == "True")
    wrong = sum(1 for r in rows if r["verdict"] == "wrong")
    if wrong:
        print(f"\nOf the {wrong} wrong ones, {unsure_wrong} were flagged UNSURE (robot would not use the name).")


def main():
    if "--summary" in sys.argv:
        return summary()
    situation = input("Situation name for this round (e.g. seated / swapped / crossing): ").strip() or "unnamed"
    patients = get("/patients") or []
    if not patients:
        sys.exit("No patients checked in yet.")
    new = not os.path.exists(LOG)
    f = open(LOG, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["time", "situation", "patient", "name", "found", "score", "unsure", "verdict"])
    keys = {ord("y"): "correct", ord("n"): "wrong", ord("m"): "missed", ord("a"): "absent"}
    for p in patients:
        while True:
            jpg = get(f"/patients/{p['id']}/find.jpg", raw=True)
            if jpg is None:
                print("No picture yet")
                time.sleep(1)
                continue
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            title = f"Is this {p['name']}?  y correct / n wrong / m missed / a absent / r refresh / Esc stop"
            cv2.imshow("find test", img)
            cv2.setWindowTitle("find test", title)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                f.close()
                cv2.destroyAllWindows()
                return summary()
            if k == ord("r"):
                continue
            if k in keys:
                dbg = get("/debug/tracks") or {}
                bound = [t for cam in dbg.values() for t in cam if t["patient"] == p["id"] and t["visible"]]
                w.writerow([time.strftime("%H:%M:%S"), situation, p["id"], p["name"], bool(bound),
                            bound[0]["score"] if bound else "", bound[0]["ambiguous"] if bound else "",
                            keys[k]])
                f.flush()
                print(f"  {p['name']}: {keys[k]}")
                break
    f.close()
    cv2.destroyAllWindows()
    summary()


if __name__ == "__main__":
    main()
