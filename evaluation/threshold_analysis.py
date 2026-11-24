
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_thresholds(model, X_test, y_test, thresholds=None):
    """
    Evaluate a trained classification model
    at different probability thresholds.
    """

    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist()
        })

    return pd.DataFrame(results)

def select_threshold_for_recall(results, min_recall=0.85):
    """
    Select the highest classification threshold that satisfies
    the minimum required recall.

    Parameters:
        results: DataFrame returned by evaluate_thresholds()
        min_recall: Minimum acceptable recall, between 0 and 1

    Returns:
        Dictionary containing the selected threshold and metrics.
    """

    if not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be between 0 and 1.")

    required_columns = {"threshold", "recall"}

    if not required_columns.issubset(results.columns):
        raise ValueError(
            "Results must contain 'threshold' and 'recall' columns."
        )

    valid_results = results[
        results["recall"] >= min_recall
    ]

    if valid_results.empty:
        raise ValueError(
            f"No tested threshold satisfies minimum recall "
            f"of {min_recall:.0%}."
        )

    best_result = valid_results.loc[
        valid_results["threshold"].idxmax()
    ]

    return best_result.to_dict()
