
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
)


# --------------------------------------------------
# 2. REQUIRED DATASET COLUMNS
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


# --------------------------------------------------
# 3. SCHEMA VALIDATION
# --------------------------------------------------

def validate_schema(df: pd.DataFrame) -> bool:
    """
    Check whether the dataset contains
    all required columns.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{missing_columns}"
        )

    return True


# --------------------------------------------------
# 4. DATASET LOADING
# --------------------------------------------------

def load_dataset(
    file_path: str | Path | None = None
) -> pd.DataFrame:

    """
    Load the AI4I dataset and validate its schema.
    """

    if file_path is None:
        file_path = DEFAULT_DATA_PATH

    file_path = Path(file_path)

    # Check whether file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Load CSV
    df = pd.read_csv(file_path)

    # Validate dataset structure
    validate_schema(df)

    print("Dataset loaded successfully!")
    print("Schema validation passed!")

    return df
