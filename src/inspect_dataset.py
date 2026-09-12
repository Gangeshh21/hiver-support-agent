from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/twcs.csv")


def main():
    print("=" * 60)
    print("HIVER SUPPORT AGENT - DATASET INSPECTION")
    print("=" * 60)

    # Read only a small sample first
    df = pd.read_csv(DATA_PATH, nrows=10)

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nDataset file size:")
    size_mb = DATA_PATH.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.2f} MB")


if __name__ == "__main__":
    main()