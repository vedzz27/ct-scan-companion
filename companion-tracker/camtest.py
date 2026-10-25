"""Which camera has which number and name? Saves one photo per camera: camtest_0.jpg, camtest_1.jpg, ...
Run with the server stopped:  python camtest.py"""
import time

import cv2

from cams import list_cameras

names = {i: n for i, n, _ in list_cameras()}
if not names:
    print("(camera names unavailable: pip install pyobjc-framework-AVFoundation)")
for i in range(max(4, len(names))):
    c = cv2.VideoCapture(i)
    time.sleep(1)
    ok = False
    for _ in range(10):
        ok, f = c.read()
    print(i, "picture" if ok else "nothing", "|", names.get(i, "?"))
    if ok:
        cv2.imwrite(f"camtest_{i}.jpg", f)
    c.release()
