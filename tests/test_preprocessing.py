
from backend.data.validation import load_dataset

from backend.data.preprocessing import (
    prepare_features_and_target,
    split_dataset
)


def main():
    df = load_dataset()

    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\n========== FEATURE INFORMATION ==========")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    print("Features:", X.columns.tolist())

    print("\n========== DATA SPLIT ==========")
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training target:", y_train.shape)
    print("Testing target:", y_test.shape)

    print("\n========== FAILURE DISTRIBUTION ==========")
    print("Original:")
    print(y.value_counts(normalize=True))

    print("\nTraining:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting:")
    print(y_test.value_counts(normalize=True))


if __name__ == "__main__":
    main()
