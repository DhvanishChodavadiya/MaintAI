
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from backend.data.validation import load_dataset

from backend.data.preprocessing import (
    prepare_features_and_target,
    split_dataset
)

from backend.models.baseline import create_baseline_model


def main():

    # 1. Load dataset
    df = load_dataset()

    # 2. Prepare features and target
    X, y = prepare_features_and_target(df)

    # 3. Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # 4. Create complete ML pipeline
    model = create_baseline_model()

    # 5. Train model using training data only
    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    # 6. Make predictions on unseen testing data
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # 7. Calculate evaluation metrics
    print("\n========== MODEL EVALUATION ==========")

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    print(
        "Precision:",
        round(precision_score(y_test, y_pred, zero_division=0), 4)
    )

    print(
        "Recall:",
        round(recall_score(y_test, y_pred, zero_division=0), 4)
    )

    print(
        "F1 Score:",
        round(f1_score(y_test, y_pred, zero_division=0), 4)
    )

    print("\n========== CONFUSION MATRIX ==========")
    print(confusion_matrix(y_test, y_pred))

    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
