
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATA QUALITY ANALYSIS ==========")

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check duplicate UDI values
print("\nDuplicate UDI values:")
print(df["UDI"].duplicated().sum())

# Check duplicate Product IDs
print("\nDuplicate Product IDs:")
print(df["Product ID"].duplicated().sum())

# Check unique product types
print("\nProduct types:")
print(df["Type"].value_counts())

# Check failure distribution
print("\nMachine failure distribution:")
print(df["Machine failure"].value_counts())

# Check failure percentages
print("\nMachine failure percentage:")
print(
    df["Machine failure"]
    .value_counts(normalize=True)
    .mul(100)
)

# Check individual failure types
failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print("\nIndividual failure counts:")
print(df[failure_columns].sum())

print("\nIndividual failure percentages:")
print(df[failure_columns].mean().mul(100))


import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Machine failure distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Machine failure"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Observations")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "failure_distribution.png")
plt.show()


# 2. Individual failure type distribution
failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

failure_counts = df[failure_columns].sum()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=failure_counts.index,
    y=failure_counts.values
)

plt.title("Individual Failure Type Distribution")
plt.xlabel("Failure Type")
plt.ylabel("Number of Failures")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "failure_types.png")
plt.show()


# 3. Sensor distributions
sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for column, ax in zip(sensor_columns, axes.flatten()):

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax
    )

    ax.set_title(column)

# Hide unused subplot
axes.flatten()[-1].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "sensor_distributions.png")
plt.show()


# 4. Correlation analysis
correlation_columns = sensor_columns + ["Machine failure"]

correlation = df[correlation_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Sensor Features and Machine Failure Correlation")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "correlation_heatmap.png")
plt.show()

print("\nEDA analysis completed.")
print(f"Charts saved in: {FIGURE_DIR}")

