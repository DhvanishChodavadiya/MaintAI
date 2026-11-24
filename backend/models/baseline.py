
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from backend.data.preprocessing import create_preprocessor


def create_baseline_model():
    """
    Create a complete preprocessing and
    Logistic Regression pipeline.
    """

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),

            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    return model_pipeline
