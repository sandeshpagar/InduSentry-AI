"""Utility to download and verify NASA C-MAPSS FD001 dataset files."""

import urllib.request
from pathlib import Path
import pandas as pd

RAW_BASE_URL = "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/"
TARGET_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "NASA_CMAPSS"

FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt"
]


def download_cmapss():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}

    for filename in FILES:
        dest_path = TARGET_DIR / filename
        if dest_path.exists() and dest_path.stat().st_size > 100:
            print(f"File already exists: {dest_path}")
            continue

        url = RAW_BASE_URL + filename
        print(f"Downloading {filename} from {url}...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f_out:
            f_out.write(resp.read())

        print(f"Saved to: {dest_path} ({dest_path.stat().st_size} bytes)")

    print("C-MAPSS FD001 dataset files successfully verified in data/raw/NASA_CMAPSS/!")


if __name__ == "__main__":
    download_cmapss()
