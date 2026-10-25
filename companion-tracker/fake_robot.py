"""Pretend to be the robot, so the 2-camera setup can be demoed and measured before robot code exists.

Start the app first, then in a second terminal:   python fake_robot.py
Commands:
  Enter        refresh the patient list
  f <n>        fetch patient number n: shows where the robot would drive and what it may say,
               then you tell it where the person really was and the QR result
  s            summary of all fetches so far (your accuracy numbers)
  q            quit
Every fetch is appended to fetch_log.csv. No pictures, only ids, zones and results.
"""
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = os.getenv("API", "http://localhost:8000/api")
LOG = "fetch_log.csv"


def call(path, body=None):
    req = urllib.request.Request(API + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}
    except urllib.error.URLError:
        sys.exit(f"Can't reach {API}. Is the app running (uvicorn main:app ...)?")


def show(patients):
    if not patients:
        print("\n  No patients checked in. Use the dashboard: stand at the kiosk camera and check in.")
        return
    print(f"\n  {'#':<3}{'id':<8}{'name':<12}{'zone':<14}{'robot goes to':<16}{'conf':<6}{'seen':<8}action")
    for i, p in enumerate(patients, 1):
        flag = "  (people crossed: unsure)" if p.get("ambiguous") else ""
        print(f"  {i:<3}{p['id']:<8}{p['name'][:11]:<12}{p['zone']:<14}{str(p.get('robot_target')):<16}"
              f"{p['confidence']:<6}{str(p['age_s']) + 's':<8}{p['action']}{flag}")


def fetch(p):
    loc = call(f"/patients/{p['id']}/location")
    print(f"\n  ROBOT: driving to '{loc.get('robot_target') or loc['zone']}' (zone {loc['zone']}, "
          f"confidence {loc['confidence']}, last seen {loc['age_s']} s ago)")
    if loc["action"] == "alert_staff":
        print("  ROBOT: patient not seen for too long -> would ask staff for help.")
    elif loc["may_use_name"]:
        print(f"  ROBOT says: \"Hello {loc['name']}, could you please show me your QR code?\"")
    else:
        print("  ROBOT says: \"Hello, could you please show me your QR code?\"  (unsure: no name used)")
    where = input("  Was the patient really in that zone? [y/n, or type the real zone]: ").strip()
    correct = where.lower() in ("y", "yes", "")
    qr = input("  QR result: right patient? [y/n/skip]: ").strip().lower()
    result = None
    if qr in ("y", "n"):
        result = call(f"/patients/{p['id']}/confirm", {"accepted": qr == "y"})
        print(f"  -> confirm sent: {result}")
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "patient", "zone_predicted", "robot_target", "action", "confidence",
                        "ambiguous", "zone_correct", "real_zone", "qr"])
        w.writerow([time.strftime("%H:%M:%S"), p["id"], loc["zone"], loc.get("robot_target"), loc["action"],
                    loc["confidence"], loc.get("ambiguous"), correct, "" if correct else where, qr])


def summary():
    if not os.path.exists(LOG):
        print("  No fetches yet.")
        return
    rows = list(csv.DictReader(open(LOG)))
    n = len(rows)
    right = sum(r["zone_correct"] == "True" for r in rows)
    sure = [r for r in rows if r["action"] == "go_to_zone_and_confirm"]
    sure_right = sum(r["zone_correct"] == "True" for r in sure)
    unsure = sum(r["action"] == "go_to_last_zone_and_scan" for r in rows)
    staff = sum(r["action"] == "alert_staff" for r in rows)
    wrong_qr = sum(r["qr"] == "n" for r in rows)
    print(f"\n  Fetches: {n}")
    print(f"  Robot went to the right zone:          {right}/{n} ({100 * right / n:.0f}%)")
    print(f"  ...when the system said it was sure:   {sure_right}/{len(sure)}")
    print(f"  System said 'unsure', re-checked by QR: {unsure}")
    print(f"  Asked staff:                            {staff}")
    print(f"  QR showed it was the wrong person:      {wrong_qr}")


def main():
    print(__doc__)
    patients = []
    while True:
        patients = call("/patients")
        if isinstance(patients, dict):
            print(patients)
            patients = []
        show(patients)
        cmd = input("\n> ").strip().split()
        if not cmd:
            continue
        if cmd[0] == "q":
            break
        if cmd[0] == "s":
            summary()
        elif cmd[0] == "f" and len(cmd) == 2 and cmd[1].isdigit() and 1 <= int(cmd[1]) <= len(patients):
            fetch(patients[int(cmd[1]) - 1])
        else:
            print("  Commands: Enter, f <n>, s, q")


if __name__ == "__main__":
    main()
