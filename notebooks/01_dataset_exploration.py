
import pandas as pd
from pathlib import Path

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

df = pd.read_csv(DATA_PATH)


import pandas as pd
from pathlib import Path

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

df = pd.read_csv(DATA_PATH)

# Display basic information
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- DATASET INFORMATION ---")
df.info()

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())
