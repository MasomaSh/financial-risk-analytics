from pathlib import Path
import pandas as pd

# Locate the project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"


def load_fraud_data():
    """Load the credit card fraud dataset."""
    file_path = RAW_DIR / "creditcard.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Fraud dataset loaded: {df.shape}")
    return df


def load_credit_default_data():
    """Load the credit card default dataset."""
    file_path = RAW_DIR / "default of credit card clients.xls"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset: {file_path}")

    # The first row contains descriptive information.
    df = pd.read_excel(file_path, header=1)

    print(f"Credit default dataset loaded: {df.shape}")
    return df


if __name__ == "__main__":
    fraud_df = load_fraud_data()
    credit_df = load_credit_default_data()

    print("\nFraud columns:")
    print(fraud_df.columns.tolist())

    print("\nCredit default columns:")
    print(credit_df.columns.tolist())
