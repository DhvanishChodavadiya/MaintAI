
from backend.data.validation import load_dataset


def main():

    df = load_dataset()

    print("\n========== DATASET INFORMATION ==========")

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)


if __name__ == "__main__":
    main()
