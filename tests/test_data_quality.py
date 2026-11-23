
from backend.data.validation import (
    load_dataset,
    run_quality_checks
)


def main():

    df = load_dataset()

    report = run_quality_checks(df)

    print("\n========== DATA QUALITY REPORT ==========")

    for category, results in report.items():

        print(f"\n{category.upper()}")

        if isinstance(results, dict):

            for key, value in results.items():
                print(f"{key}: {value}")

        else:
            print(results)


if __name__ == "__main__":
    main()
