
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


# --------------------------------------------------
# 5. DATA QUALITY VALIDATION
# --------------------------------------------------

def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Identify missing values in each column.
    """

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    return missing.to_dict()


def check_duplicates(df: pd.DataFrame) -> dict:
    """
    Check duplicate rows and identifiers.
    """

    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_udi": int(df["UDI"].duplicated().sum()),
        "duplicate_product_ids": int(
            df["Product ID"].duplicated().sum()
        )
    }


# --------------------------------------------------
# 6. DATA TYPE VALIDATION
# --------------------------------------------------

def validate_data_types(df: pd.DataFrame) -> dict:
    """
    Validate expected numeric and categorical columns.
    """

    numeric_columns = [
        "UDI",
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

    categorical_columns = [
        "Product ID",
        "Type"
    ]

    invalid_columns = []

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            invalid_columns.append(column)

    for column in categorical_columns:
        if not (
            pd.api.types.is_string_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
        ):
            invalid_columns.append(column)

    return {
        "valid": len(invalid_columns) == 0,
        "invalid_columns": invalid_columns
    }


# --------------------------------------------------
# 7. SENSOR VALUE VALIDATION
# --------------------------------------------------

def validate_sensor_values(df: pd.DataFrame) -> dict:
    """
    Detect impossible or suspicious sensor values.

    These are basic domain checks, not certified
    operating limits for a real industrial machine.
    """

    rules = {
        "Air temperature [K]": lambda x: x <= 0,
        "Process temperature [K]": lambda x: x <= 0,
        "Rotational speed [rpm]": lambda x: x <= 0,
        "Torque [Nm]": lambda x: x < 0,
        "Tool wear [min]": lambda x: x < 0
    }

    results = {}

    for column, rule in rules.items():

        invalid_count = int(rule(df[column]).sum())

        results[column] = invalid_count

    return results


# --------------------------------------------------
# 8. TARGET LABEL VALIDATION
# --------------------------------------------------

def validate_target_labels(df: pd.DataFrame) -> dict:
    """
    Ensure target and failure labels contain
    only binary values.
    """

    label_columns = [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    results = {}

    for column in label_columns:

        invalid_values = df.loc[
            ~df[column].isin([0, 1]),
            column
        ].unique().tolist()

        results[column] = invalid_values

    return results


# --------------------------------------------------
# 9. FAILURE LABEL CONSISTENCY
# --------------------------------------------------

def check_failure_consistency(df: pd.DataFrame) -> dict:
    """
    Check whether individual failure labels
    agree with the overall failure label.
    """

    failure_columns = [
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    any_failure_mode = df[failure_columns].eq(1).any(axis=1)

    overall_failure = df["Machine failure"].eq(1)

    mismatch = any_failure_mode != overall_failure

    return {
        "mismatch_count": int(mismatch.sum()),
        "failure_mode_without_overall_failure": int(
            (any_failure_mode & ~overall_failure).sum()
        ),
        "overall_failure_without_failure_mode": int(
            (~any_failure_mode & overall_failure).sum()
        ),
        "mismatch_percentage": round(
            (mismatch.sum() / len(df)) * 100, 2
        )
    }


# --------------------------------------------------
# 10. COMPLETE VALIDATION REPORT
# --------------------------------------------------

def run_quality_checks(df: pd.DataFrame) -> dict:
    """
    Execute all data quality checks.
    """

    validate_schema(df)

    return {
        "missing_values": check_missing_values(df),
        "duplicates": check_duplicates(df),
        "data_types": validate_data_types(df),
        "sensor_values": validate_sensor_values(df),
        "target_labels": validate_target_labels(df),
        "failure_consistency": check_failure_consistency(df)
    }
