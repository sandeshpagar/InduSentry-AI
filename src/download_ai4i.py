"""Utility to download the official AI4I 2020 Predictive Maintenance dataset from UCI."""

import urllib.request
import zipfile
import io
from pathlib import Path
import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
TARGET_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TARGET_FILE = TARGET_DIR / "ai4i2020.csv"


def download_ai4i():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 1000:
        print(f"Dataset already exists at: {TARGET_FILE}")
        return TARGET_FILE

    print(f"Downloading AI4I 2020 dataset from UCI: {UCI_ZIP_URL}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(UCI_ZIP_URL, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Find the csv inside
        csv_names = [name for name in z.namelist() if name.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV file found in downloaded UCI zip.")
        
        csv_file_name = csv_names[0]
        print(f"Extracting {csv_file_name} to {TARGET_FILE}...")
        with z.open(csv_file_name) as f_in, open(TARGET_FILE, "wb") as f_out:
            f_out.write(f_in.read())

    df = pd.read_csv(TARGET_FILE)
    print(f"Successfully downloaded AI4I 2020! Shape: {df.shape}")
    return TARGET_FILE


if __name__ == "__main__":
    download_ai4i()
