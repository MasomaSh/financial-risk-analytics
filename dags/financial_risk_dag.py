
from pathlib import Path
import os

import pandas as pd
import pendulum

from airflow.sdk import dag, task


# ==================================================
# PROJECT PATHS
# ==================================================

DATA_DIR = Path("/opt/airflow/data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FRAUD_RAW_FILE = RAW_DIR / "creditcard.csv"
DEFAULT_RAW_FILE = RAW_DIR / "default of credit card clients.xls"


# ==================================================
# DATABASE CONNECTION
# ==================================================

def get_database_engine():
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("RISK_DB_USER", "risk_user"),
        password=os.getenv("RISK_DB_PASSWORD", "risk_password"),
        host=os.getenv("RISK_DB_HOST", "risk-postgres"),
        port=int(os.getenv("RISK_DB_PORT", "5432")),
        database=os.getenv("RISK_DB_NAME", "financial_risk"),
    )

    return create_engine(connection_url)


# ==================================================
# MODEL TRAINING AND EVALUATION
# ==================================================

def evaluate_model(df, target_column, dataset_name):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    # Separate predictors and target.
    columns_to_drop = [target_column]

    # ID is an identifier, not a useful predictive feature.
    if "ID" in df.columns:
        columns_to_drop.append("ID")

    X = df.drop(columns=columns_to_drop)
    y = df[target_column]

    X = X.apply(pd.to_numeric, errors="raise")

    # Preserve the class proportions in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Scale features and train logistic regression.
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "dataset": dataset_name,
        "model": "Logistic Regression",
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "test_rows": int(len(y_test)),
    }

    results["confusion_matrix"] = confusion_matrix(
        y_test,
        predictions,
    ).tolist()

    print(f"\nResults for {dataset_name}:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

    return results


# ==================================================
# AIRFLOW DAG
# ==================================================

@dag(
    dag_id="financial_risk_analytics",
    description="Ingest, clean, store, and model two financial risk datasets.",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 11, 14, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["financial-risk", "machine-learning"],
)
def financial_risk_pipeline():

    # ----------------------------------------------
    # 1. INGESTION
    # ----------------------------------------------



    @task
    def ingest_fraud():
        if not FRAUD_RAW_FILE.exists():
            raise FileNotFoundError(
                f"Fraud dataset not found: {FRAUD_RAW_FILE}"
            )

        print(f"Fraud dataset found: {FRAUD_RAW_FILE}")
        return str(FRAUD_RAW_FILE)

    @task
    def ingest_default():
        if not DEFAULT_RAW_FILE.exists():
            raise FileNotFoundError(
                f"Credit default dataset not found: {DEFAULT_RAW_FILE}"
            )

        print(f"Credit default dataset found: {DEFAULT_RAW_FILE}")
        return str(DEFAULT_RAW_FILE)

    # ----------------------------------------------
    # 2. CLEAN AND TRANSFORM
    # ----------------------------------------------

    @task
    def clean_fraud(raw_file):
        df = pd.read_csv(raw_file)

        original_rows = len(df)
        df = df.drop_duplicates().copy()

        if df.isnull().sum().sum() > 0:
            raise ValueError("Fraud dataset contains missing values.")

        if not set(df["Class"].unique()).issubset({0, 1}):
            raise ValueError("Unexpected fraud target values.")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        output_file = PROCESSED_DIR / "fraud_cleaned.csv"
        df.to_csv(output_file, index=False)

        print(f"Fraud duplicates removed: {original_rows - len(df)}")
        print(f"Clean fraud dataset shape: {df.shape}")

        return str(output_file)

    @task
    def clean_default(raw_file):
        df = pd.read_excel(raw_file, header=1)

        original_rows = len(df)
        df = df.drop_duplicates().copy()

        target = "default payment next month"

        if df.isnull().sum().sum() > 0:
            raise ValueError(
                "Credit default dataset contains missing values."
            )

        if not set(df[target].unique()).issubset({0, 1}):
            raise ValueError("Unexpected credit default target values.")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        output_file = PROCESSED_DIR / "credit_default_cleaned.csv"
        df.to_csv(output_file, index=False)

        print(f"Credit default duplicates removed: {original_rows - len(df)}")
        print(f"Clean credit default dataset shape: {df.shape}")

        return str(output_file)

    # ----------------------------------------------
    # 3. LOAD INTO POSTGRESQL
    # ----------------------------------------------

    @task
    def load_fraud(cleaned_file):
        df = pd.read_csv(cleaned_file)
        engine = get_database_engine()

        try:
            df.to_sql(
                "fraud_cleaned",
                engine,
                if_exists="replace",
                index=False,
                
                chunksize=10000,
            )
        finally:
            engine.dispose()

        print(f"Loaded {len(df)} fraud records.")
        return "fraud_cleaned"

    @task
    def load_default(cleaned_file):
        df = pd.read_csv(cleaned_file)
        engine = get_database_engine()

        try:
            df.to_sql(
                "credit_default_cleaned",
                engine,
                if_exists="replace",
                index=False,
                chunksize=10000,
            )
        finally:
            engine.dispose()

        print(f"Loaded {len(df)} credit default records.")
        return "credit_default_cleaned"

    # ----------------------------------------------
    # 4. TRAIN AND EVALUATE MODELS
    # ----------------------------------------------

    @task
    def train_fraud(table_name):
        engine = get_database_engine()

        try:
            df = pd.read_sql_query(
                f'SELECT * FROM "{table_name}"',
                engine,
            )
        finally:
            engine.dispose()

        return evaluate_model(
            df,
            target_column="Class",
            dataset_name="Credit Card Fraud",
        )

    @task
    def train_default(table_name):
        engine = get_database_engine()

        try:
            df = pd.read_sql_query(
                f'SELECT * FROM "{table_name}"',
                engine,
            )
        finally:
            engine.dispose()

        return evaluate_model(
            df,
            target_column="default payment next month",
            dataset_name="Credit Card Default",
        )

    # ----------------------------------------------
    # 5. COMBINE MODEL RESULTS
    # ----------------------------------------------

    @task
    def combine_results(fraud_results, default_results):
        engine = get_database_engine()

        results = [fraud_results, default_results]

        summary = pd.DataFrame([
            {
                key: value
                for key, value in result.items()
                if key != "confusion_matrix"
            }
            for result in results
        ])

        try:
            summary.to_sql(
                "financial_risk_summary",
                engine,
                if_exists="replace",
                index=False,
            )
        finally:
            engine.dispose()

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        output_file = PROCESSED_DIR / "financial_risk_summary.csv"
        summary.to_csv(output_file, index=False)

        print("\nCombined financial risk summary:")
        print(summary.to_string(index=False))

        return "financial_risk_summary"

    # ----------------------------------------------
    # 6. ANALYZE AND VISUALIZE RESULTS
    # ----------------------------------------------

    @task
    def analyze_results(table_name):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        engine = get_database_engine()

        try:
            summary = pd.read_sql_query(
                f'SELECT * FROM "{table_name}"',
                engine,
            )
        finally:
            engine.dispose()

        print("\nFinal model evaluation:")
        print(summary.to_string(index=False))

        metrics = ["precision", "recall", "f1_score"]

        ax = summary.set_index("dataset")[metrics].plot(
            kind="bar",
            figsize=(10, 6),
        )

        ax.set_title("Financial Risk Model Performance")
        ax.set_ylabel("Score")
        ax.set_xlabel("Dataset")
        ax.set_ylim(0, 1)
        ax.legend(title="Evaluation Metric")

        plt.xticks(rotation=0)
        plt.tight_layout()

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        chart_file = PROCESSED_DIR / "risk_model_metrics.png"
        plt.savefig(chart_file, dpi=150)
        plt.close()

        print(f"Performance chart saved to: {chart_file}")

    # ----------------------------------------------
    # 7. DEFINE TASK ORDER AND PARALLELISM
    # ----------------------------------------------

    fraud_raw = ingest_fraud()
    default_raw = ingest_default()

    fraud_clean = clean_fraud(fraud_raw)
    default_clean = clean_default(default_raw)

    fraud_table = load_fraud(fraud_clean)
    default_table = load_default(default_clean)

    fraud_metrics = train_fraud(fraud_table)
    default_metrics = train_default(default_table)

    combined_summary = combine_results(
        fraud_metrics,
        default_metrics,
    )

    analyze_results(combined_summary)


financial_risk_pipeline()

