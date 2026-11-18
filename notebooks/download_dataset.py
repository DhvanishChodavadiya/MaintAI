
import io
import zipfile
from pathlib import Path

import requests

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset directory
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Official UCI dataset download
URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

print("Downloading AI4I dataset...")

response = requests.get(URL, timeout=60)
response.raise_for_status()

# Open downloaded ZIP file
with zipfile.ZipFile(io.BytesIO(response.content)) as archive:

    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("No CSV file found in dataset archive.")

    # Extract original CSV
    csv_name = csv_files[0]
    csv_content = archive.read(csv_name)

    output_path = RAW_DATA_DIR / "ai4i2020.csv"
    output_path.write_bytes(csv_content)

print("Dataset downloaded successfully!")
print(f"Saved to: {output_path}")
