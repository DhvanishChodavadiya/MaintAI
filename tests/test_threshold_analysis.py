
from backend.data.validation import load_dataset
from backend.data.preprocessing import (
    prepare_features_and_target,
    split_dataset
)
from backend.models.baseline import create_baseline_model
from evaluation.threshold_analysis import (
    evaluate_thresholds,
    select_threshold_for_recall
)

# 1. Load dataset
df = load_dataset()

# 2. Prepare features and target
X, y = prepare_features_and_target(df)

# 3. Split dataset (same settings as baseline)
X_train, X_test, y_train, y_test = split_dataset(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Create and train the baseline model once
model = create_baseline_model()
model.fit(X_train, y_train)

# 5. Evaluate different classification thresholds
results = evaluate_thresholds(
    model,
    X_test,
    y_test
)

# 6. Display results
print("\nTHRESHOLD ANALYSIS")
print("=" * 70)

for _, row in results.iterrows():
    print(f"\nThreshold: {row['threshold']}")
    print(f"Precision: {row['precision']:.4f}")
    print(f"Recall:    {row['recall']:.4f}")
    print(f"F1 Score:  {row['f1_score']:.4f}")
    print(f"Confusion Matrix: {row['confusion_matrix']}")


# 7. Select the best threshold for minimum recall of 85%
minimum_recall = 0.85

selected = select_threshold_for_recall(
    results,
    min_recall=minimum_recall
)

print("\nTHRESHOLD SELECTION")
print("=" * 70)
print(f"Minimum required recall: {minimum_recall:.0%}")
print(f"Selected threshold:     {selected['threshold']}")
print(f"Precision:              {selected['precision']:.4f}")
print(f"Recall:                 {selected['recall']:.4f}")
print(f"F1 Score:               {selected['f1_score']:.4f}")
print(f"Confusion Matrix:       {selected['confusion_matrix']}")



