
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Machine failure"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = [
    "Type"
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features_and_target(df: pd.DataFrame):
    """
    Separate input features (X) and prediction target (y).
    """

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    Stratification preserves the failure/non-failure ratio.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
