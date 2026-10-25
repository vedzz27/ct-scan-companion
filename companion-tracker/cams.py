"""Pick cameras by NAME instead of by number (macOS renumbers cameras when e.g. an iPhone connects).

In cameras.json, "source" may be a name part instead of a number, e.g.
    "source": "name:FaceTime"      (the MacBook camera)
    "source": "name:Siemens"       (the external camera)
The name is looked up case-insensitively in the list printed by:  python camtest.py

How: OpenCV on macOS numbers the cameras as AVFoundation lists them, sorted by their uniqueID.
We list them the same way (needs: pip install pyobjc-framework-AVFoundation) and return the number.
"""
import sys


def list_cameras():
    """[(opencv_index, name, unique_id)] on macOS, [] elsewhere or if pyobjc is missing."""
    if sys.platform != "darwin":
        return []
    try:
        import AVFoundation as AVF
    except ImportError:
        return []
    devs = list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo) or [])
    devs += list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeMuxed) or [])
    seen, out = set(), []
    for d in devs:
        if d.uniqueID() not in seen:
            seen.add(d.uniqueID())
            out.append(d)
    out.sort(key=lambda d: str(d.uniqueID()))        # same order OpenCV uses
    return [(i, str(d.localizedName()), str(d.uniqueID())) for i, d in enumerate(out)]


def resolve_source(src):
    """Number / URL / file stay as they are; "name:xyz" becomes the matching camera number."""
    if not (isinstance(src, str) and src.lower().startswith("name:")):
        return src
    want = src[5:].strip().lower()
    cams = list_cameras()
    if not cams:
        raise RuntimeError(f"Can't look up camera '{want}' by name: run  pip install pyobjc-framework-AVFoundation  "
                           "(macOS only), or use a camera number in cameras.json")
    hits = [c for c in cams if want in c[1].lower()]
    if not hits:
        raise RuntimeError(f"No camera named like '{want}'. Cameras found: " + ", ".join(f"{i}: {n}" for i, n, _ in cams))
    return hits[0][0]
