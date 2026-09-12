import pandas as pd

DATA_PATH = "data/raw/twcs.csv"


def main():
    print("Reading dataset header...", flush=True)

    # Read only the header, so we don't load the huge dataset.
    columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()

    print("\n========== COLUMNS ==========")
    for column in columns:
        print(column)

    print("\n========== READING SAMPLE ==========", flush=True)

    # Initially inspect only 10,000 rows.
    df = pd.read_csv(DATA_PATH, nrows=10_000)

    print(f"Sample rows: {len(df):,}")

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== FIRST 10 ROWS ==========")
    print(df.head(10).to_string())

    print("\n========== UNIQUE VALUES ==========")

    for column in df.columns:
        if df[column].dtype == "object":
            print(f"{column}: {df[column].nunique():,} unique values")

    print("\nInspection complete!", flush=True)


if __name__ == "__main__":
    main()