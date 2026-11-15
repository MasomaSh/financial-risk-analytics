from pathlib import Path
import pandas as pd

from ingestion import (
    load_fraud_data,
    load_credit_default_data
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def clean_fraud_data(df):
    """Clean the credit card fraud dataset."""

    df = df.copy()

    original_rows = len(df)

    # Remove exact duplicate transactions
    df = df.drop_duplicates().copy()

    removed = original_rows - len(df)

    # Validate missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Fraud dataset contains missing values.")

    # Validate target
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected values in fraud target.")

    print(f"Fraud duplicates removed: {removed}")
    print(f"Clean fraud dataset: {df.shape}")

    return df


def clean_credit_default_data(df):
    """Clean the credit card default dataset."""

    df = df.copy()

    # Remove exact duplicate customer records
    df = df.drop_duplicates().copy()

    # Validate missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Credit default dataset contains missing values.")

    target = "default payment next month"

    # Validate target
    if not set(df[target].unique()).issubset({0, 1}):
        raise ValueError("Unexpected values in default target.")

    # ID identifies a customer; it is not a predictive feature.
    # Keep it in the cleaned table for traceability.
    print(f"Clean credit default dataset: {df.shape}")

    return df


def save_cleaned_data(fraud_df, credit_df):
    """Save both cleaned datasets."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fraud_path = PROCESSED_DIR / "fraud_cleaned.csv"
    credit_path = PROCESSED_DIR / "credit_default_cleaned.csv"

    fraud_df.to_csv(fraud_path, index=False)
    credit_df.to_csv(credit_path, index=False)

    print(f"\nSaved: {fraud_path}")
    print(f"Saved: {credit_path}")


if __name__ == "__main__":
    fraud = load_fraud_data()
    credit = load_credit_default_data()

    fraud_clean = clean_fraud_data(fraud)
    credit_clean = clean_credit_default_data(credit)

    save_cleaned_data(fraud_clean, credit_clean)

    print("\nBoth datasets cleaned successfully.")
