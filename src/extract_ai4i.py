"""Placement of the official UCI AI4I 2020 dataset."""
import os
import shutil
from pathlib import Path

SOURCE_FETCHED = Path(r"C:\Users\sande\.gemini\antigravity\brain\e1c0a020-a838-4ecc-b45b-ee1ba9e1d6b1\.system_generated\steps\104\content.md")
DEST_CSV = Path(r"d:\AMLISPro\data\raw\ai4i2020.csv")

def extract_clean_csv():
    DEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCE_FETCHED, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()
    
    # Header starts at line index 8 (line 9 in 1-indexed tool output)
    csv_lines = lines[8:10009]
    print(f"Extracting {len(csv_lines)} lines to {DEST_CSV}...")
    with open(DEST_CSV, "w", encoding="utf-8") as f_out:
        f_out.writelines(csv_lines)
    
    print("Done!")

if __name__ == "__main__":
    extract_clean_csv()
