"""Download person re-ID weights (OSNet, trained on MSMT17) into ./weights and check they load.

    python download_reid.py                 # osnet_x1_0_msmt17 (default, ~9 MB, best accuracy)
    python download_reid.py osnet_x0_25     # tiny and fast, for a slow laptop
    python download_reid.py osnet_ibn_x1_0  # often transfers better to new places and cameras

If the download is blocked, open the printed Google Drive link in a browser, save the file into
the weights folder under the printed name, and run this script again to check it.
Weights: Kaiyang Zhou's torchreid model zoo (MIT).
"""
import os
import sys

FILES = {
    "osnet_x1_0": ("osnet_x1_0_msmt17.pt", "112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M"),
    "osnet_x0_25": ("osnet_x0_25_msmt17.pt", "1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"),
    "osnet_ibn_x1_0": ("osnet_ibn_x1_0_msmt17.pt", "1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ"),
}


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "osnet_x1_0"
    if key not in FILES:
        sys.exit(f"Choose one of: {', '.join(FILES)}")
    name, gid = FILES[key]
    os.makedirs("weights", exist_ok=True)
    path = os.path.join("weights", name)
    url = f"https://drive.google.com/uc?id={gid}"
    if not os.path.exists(path):
        try:
            import gdown
        except ImportError:
            sys.exit("pip install gdown   (then run this again)")
        print(f"Downloading {name} ...")
        if gdown.download(url, path, quiet=False) is None or not os.path.exists(path):
            sys.exit(f"Download failed. Get it manually:\n  {url}\nSave it as {path}")
    from osnet import load_osnet
    load_osnet(path)
    print(f"OK: {path} loads correctly.")
    if path.replace("\\", "/") != "weights/osnet_x1_0_msmt17.pt":
        print(f"Tell the app to use it:  set OSNET_WEIGHTS={path}  (PowerShell: $env:OSNET_WEIGHTS=\"{path}\")")


if __name__ == "__main__":
    main()
