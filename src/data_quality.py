from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

fraud = pd.read_csv(DATA_DIR / "creditcard.csv")
credit = pd.read_excel(
    DATA_DIR / "default of credit card clients.xls",
    header=1
)

for name, df in [
    ("FRAUD DATASET", fraud),
    ("CREDIT DEFAULT DATASET", credit)
]:
    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:", df.isnull().sum().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nData types:")
    print(df.dtypes)
