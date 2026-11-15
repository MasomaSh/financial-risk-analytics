from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

fraud_path = DATA_DIR / "creditcard.csv"
credit_path = DATA_DIR / "default of credit card clients.xls"

fraud = pd.read_csv(fraud_path)
credit = pd.read_excel(credit_path, header=1)

def inspect_dataset(name, df):
    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    print("\nShape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nSummary statistics:")
    print(df.describe().T)

inspect_dataset("CREDIT CARD FRAUD", fraud)
inspect_dataset("CREDIT CARD DEFAULT", credit)

print("\nFRAUD TARGET DISTRIBUTION:")
print(fraud["Class"].value_counts())

print("\nDEFAULT TARGET DISTRIBUTION:")
print(credit["default payment next month"].value_counts())
