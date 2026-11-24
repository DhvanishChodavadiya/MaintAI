
from backend.data.validation import load_dataset

from backend.data.preprocessing import (
    prepare_features_and_target,
    split_dataset,
    create_preprocessor
)


def main():

    # 1. Load dataset
    df = load_dataset()

    # 2. Separate features and target
    X, y = prepare_features_and_target(df)

    # 3. Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # 4. Create preprocessor
    preprocessor = create_preprocessor()

    # 5. Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # 6. Transform testing data
    X_test_processed = preprocessor.transform(X_test)

    # 7. Display results
    print("\n========== PREPROCESSING RESULTS ==========")

    print("Original training shape:", X_train.shape)
    print("Original testing shape:", X_test.shape)

    print("\nProcessed training shape:", X_train_processed.shape)
    print("Processed testing shape:", X_test_processed.shape)

    print("\nTraining target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
