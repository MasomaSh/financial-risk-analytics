from pathlib import Path
from typing import Any, TypedDict
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from langgraph.graph import StateGraph, END


# --------------------------------------------------
# Paths and experiment configurations
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "fraud_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

EXPERIMENTS = [
    {
        "name": "Balanced Logistic Regression",
        "class_weight": "balanced",
        "C": 1.0,
    },
    {
        "name": "Higher Fraud Penalty",
        "class_weight": {0: 1, 1: 10},
        "C": 1.0,
    },
    {
        "name": "Stronger Fraud Penalty",
        "class_weight": {0: 1, 1: 25},
        "C": 0.5,
    },
]


# --------------------------------------------------
# Shared state passed between agents
# --------------------------------------------------

class AgentState(TypedDict, total=False):
    X_train: Any
    y_train: Any
    X_val: Any
    y_val: Any
    X_test: Any
    y_test: Any

    cycle: int
    max_cycles: int
    current_config: dict
    current_metrics: dict
    best_config: dict
    best_validation_metrics: dict
    history: list

    final_metrics: dict


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(
            y_true, probabilities
        )),
        "average_precision": float(average_precision_score(
            y_true, probabilities
        )),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


def train_model(config, X_train, y_train):
    model = LogisticRegression(
        class_weight=config["class_weight"],
        C=config["C"],
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


# --------------------------------------------------
# Agent 1: Data Scientist
# --------------------------------------------------

def data_scientist_agent(state: AgentState):
    print("\n[Data Scientist Agent] Loading and profiling fraud data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' was not found.")

    if df.isnull().any().any():
        raise ValueError("The cleaned dataset still contains missing values.")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("The fraud target must contain both 0 and 1.")

    print(f"Rows: {len(df):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Fraud rate: {y.mean():.4%}")

    # First reserve 20% for final testing.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Reserve 20% of the full dataset for validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    print(
        f"Training rows: {len(X_train):,} | "
        f"Validation rows: {len(X_val):,} | "
        f"Test rows: {len(X_test):,}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "cycle": 0,
        "max_cycles": len(EXPERIMENTS),
        "history": [],
        "best_validation_metrics": {"f1_score": -1.0},
    }


# --------------------------------------------------
# Agent 2: Hypothesis Generator
# --------------------------------------------------

def hypothesis_agent(state: AgentState):
    cycle = state["cycle"]
    config = EXPERIMENTS[cycle]

    print(f"\n[Hypothesis Agent] Experiment {cycle + 1}")
    print(f"Proposed experiment: {config['name']}")
    print(f"Class weights: {config['class_weight']}")

    return {"current_config": config}


# --------------------------------------------------
# Agent 3: ML Engineer
# --------------------------------------------------

def ml_engineer_agent(state: AgentState):
    print("\n[ML Engineer Agent] Training model...")

    model = train_model(
        state["current_config"],
        state["X_train"],
        state["y_train"],
    )

    probabilities = model.predict_proba(state["X_val"])[:, 1]
    metrics = calculate_metrics(state["y_val"], probabilities)

    print(
        f"Validation F1: {metrics['f1_score']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f}"
    )

    return {
        "current_metrics": metrics,
        "cycle": state["cycle"] + 1,
    }


# --------------------------------------------------
# Agent 4: Judge
# --------------------------------------------------

def judge_agent(state: AgentState):
    print("\n[Judge Agent] Comparing experiment results...")

    metrics = state["current_metrics"]
    config = state["current_config"]

    history = state["history"] + [{
        "experiment": config["name"],
        "class_weight": config["class_weight"],
        "C": config["C"],
        "validation_metrics": metrics,
    }]

    best_metrics = state["best_validation_metrics"]
    best_config = state.get("best_config")

    if metrics["f1_score"] > best_metrics["f1_score"]:
        best_metrics = metrics
        best_config = config
        print("New best experiment selected.")
    else:
        print("Previous best experiment remains selected.")

    return {
        "history": history,
        "best_config": best_config,
        "best_validation_metrics": best_metrics,
    }


def route_after_judge(state: AgentState):
    if state["cycle"] >= state["max_cycles"]:
        return "final_evaluation"

    return "hypothesis"


# --------------------------------------------------
# Agent 5: Final Evaluation
# --------------------------------------------------

def final_evaluation_agent(state: AgentState):
    print("\n[Final Evaluation Agent] Evaluating best model on test data...")

    best_config = state["best_config"]

    # Refit using training + validation data.
    X_train_val = pd.concat(
        [state["X_train"], state["X_val"]]
    )
    y_train_val = pd.concat(
        [state["y_train"], state["y_val"]]
    )

    final_model = train_model(
        best_config,
        X_train_val,
        y_train_val,
    )

    test_probabilities = final_model.predict_proba(
        state["X_test"]
    )[:, 1]

    final_metrics = calculate_metrics(
        state["y_test"],
        test_probabilities,
    )

    print(f"Selected model: {best_config['name']}")
    print(f"Test precision: {final_metrics['precision']:.4f}")
    print(f"Test recall: {final_metrics['recall']:.4f}")
    print(f"Test F1: {final_metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(
        f"Test average precision: "
        f"{final_metrics['average_precision']:.4f}"
    )

    results = {
        "selected_experiment": best_config["name"],
        "selected_configuration": best_config,
        "validation_metrics": state["best_validation_metrics"],
        "test_metrics": final_metrics,
        "experiment_history": state["history"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / "fraud_agent_results.json"

    with open(results_path, "w") as file:
        json.dump(results, file, indent=4)

    print(f"\nResults saved to: {results_path}")

    return {"final_metrics": final_metrics}


# --------------------------------------------------
# Build the LangGraph workflow
# --------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("data_scientist", data_scientist_agent)
    graph.add_node("hypothesis", hypothesis_agent)
    graph.add_node("ml_engineer", ml_engineer_agent)
    graph.add_node("judge", judge_agent)
    graph.add_node("final_evaluation", final_evaluation_agent)

    graph.set_entry_point("data_scientist")

    graph.add_edge("data_scientist", "hypothesis")
    graph.add_edge("hypothesis", "ml_engineer")
    graph.add_edge("ml_engineer", "judge")

    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "hypothesis": "hypothesis",
            "final_evaluation": "final_evaluation",
        },
    )

    graph.add_edge("final_evaluation", END)

    return graph.compile()


# --------------------------------------------------
# Run workflow and save graph
# --------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    # Save a Mermaid representation of the live agent graph.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graph_text = app.get_graph().draw_mermaid()
    graph_path = OUTPUT_DIR / "fraud_agent_graph.mmd"
    graph_path.write_text(graph_text)

    print("Agent graph saved to:", graph_path)
    print("\nStarting financial risk agent workflow...")

    result = app.invoke({})

    print("\nWorkflow completed successfully.")
    print("Final test metrics:")
    print(json.dumps(result["final_metrics"], indent=4))

