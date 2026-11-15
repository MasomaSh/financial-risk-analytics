# Financial Risk Analytics: Airflow Data Pipeline and Agent-Based Machine Learning

## Project Overview

This project builds an automated financial risk analytics pipeline using Apache Airflow, Docker, PostgreSQL, Python, and LangGraph.

It processes two credit-related datasets, cleans and transforms the data, stores the cleaned datasets in PostgreSQL, trains machine learning models, and generates a comparison of model performance.

The project has two parts:

1. **Part 1: Airflow Data Pipeline** — Automates data ingestion, cleaning, database loading, model training, and analysis.
2. **Part 2: Agent-Based Machine Learning Workflow** — Uses a LangGraph agent workflow to run multiple machine learning experiments, compare their performance, and select the best configuration.

The project focuses on two financial risk prediction tasks:

- Credit card fraud detection.
- Credit card payment default prediction.

---

## Datasets

### 1. Credit Card Fraud Detection

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`
- **Original records:** 284,807
- **Features:** 30 input features and one target variable (`Class`)
- **Target:** `Class` (0 = legitimate transaction, 1 = fraudulent transaction)

### 2. Default of Credit Card Clients

- **Source:** [UCI Machine Learning Repository – Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **File:** `default of credit card clients.xls`
- **Original records:** 30,000
- **Features:** 23 input features and one target variable (`default payment next month`)
- **Target:** 0 = no default, 1 = default

These datasets are related through the broader domain of credit-card financial risk. However, they contain different populations, observations, and feature definitions. They do not share a customer or transaction identifier, so joining individual records would not be statistically meaningful.

Instead, the pipeline combines their standardized model evaluation results into a single `financial_risk_summary` table. This provides a meaningful comparison while preserving the original datasets separately.

---

## Technologies

- Python
- Apache Airflow
- Docker and Docker Compose
- PostgreSQL
- pandas and NumPy
- scikit-learn
- SQLAlchemy
- Matplotlib
- LangGraph

---

## Project Structure

```text
financial-risk-analytics/
├── dags/
│   └── financial_risk_dag.py
├── data/
│   ├── raw/
│   │   ├── creditcard.csv
│   │   └── default of credit card clients.xls
│   └── processed/
│       ├── fraud_cleaned.csv
│       ├── credit_default_cleaned.csv
│       ├── financial_risk_summary.csv
│       ├── risk_model_metrics.png
│       ├── fraud_agent_results.json
│       ├── fraud_agent_graph.mmd
│       └── fraud_agent_graph.png
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── fraud_agent_workflow.py
│   │   └── render_agent_graph.py
│   ├── ingestion.py
│   ├── transformations.py
│   ├── inspect_data.py
│   └── data_quality.py
├── config/
├── logs/
├── plugins/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

Raw datasets, generated CSV files, logs, and environment files are excluded from GitHub through `.gitignore` where appropriate. The raw data files must be downloaded separately and placed in `data/raw/`.

---

# Part 1: Build an Airflow Data Pipeline

## Pipeline Overview

The Airflow DAG is named `financial_risk_analytics`.

It automates the workflow from data ingestion through cleaning, database loading, model training, and visualization.

The pipeline uses Docker Compose to run Airflow and its supporting services. PostgreSQL stores the cleaned datasets and model evaluation results.

## Pipeline Workflow

The DAG contains the following tasks:

1. **Ingestion:** Verify that both raw dataset files exist and pass their file paths to the cleaning tasks.
2. **Cleaning and transformation:** Remove duplicate records, validate missing values, and verify binary target variables.
3. **Database loading:** Load the cleaned datasets into separate PostgreSQL tables.
4. **Model training:** Read the cleaned tables from PostgreSQL and train a logistic regression model for each dataset.
5. **Result combination:** Combine the evaluation metrics into a unified financial risk summary table.
6. **Analysis:** Read the summary table from PostgreSQL and generate a visualization comparing model performance.

## Parallel Execution

The fraud and credit default branches are independent. Each branch performs ingestion, cleaning, database loading, and model training separately.

The two branches can execute in parallel:

```text
Fraud Dataset:
Ingest → Clean → Load to PostgreSQL → Train Model
                                              |
                                              v
                                      Combine Results
                                              |
                                              v
                                         Analyze
                                              ^
                                              |
Credit Default Dataset:                       |
Ingest → Clean → Load to PostgreSQL → Train Model
```

The `combine_results` task waits for both model evaluation tasks to finish. It combines their evaluation metrics, excluding the confusion matrices, into the `financial_risk_summary` PostgreSQL table and saves a CSV copy.

The `analyze_results` task then reads the summary table from PostgreSQL and saves `risk_model_metrics.png` in `data/processed/`.

## Scheduling

The DAG is configured with:

- **Schedule:** Daily (`@daily`)
- **Catchup:** Disabled
- **Maximum active DAG runs:** 1

The daily schedule allows the pipeline to run automatically. Setting `max_active_runs=1` prevents multiple DAG runs from executing simultaneously.

## Data Cleaning and Transformation

The pipeline performs the following operations:

- Removes duplicate records.
- Checks for missing values.
- Validates binary target variables.
- Saves cleaned datasets as CSV files.
- Loads the cleaned data into PostgreSQL.

After cleaning, the datasets contain:

| Dataset | Original records | Cleaned records |
|---|---:|---:|
| Credit Card Fraud | 284,807 | 283,726 |
| Credit Card Default | 30,000 | 30,000 |

The fraud dataset had 1,081 duplicate records removed. No records were removed from the credit default dataset.

## PostgreSQL Tables

The pipeline creates three tables:

| Table | Description |
|---|---|
| `fraud_cleaned` | Cleaned credit card transaction data |
| `credit_default_cleaned` | Cleaned credit card default data |
| `financial_risk_summary` | Combined model evaluation metrics |

The two cleaned datasets remain separate because they do not have a valid record-level join key. Their model evaluation results are combined at the summary level.

## Machine Learning and Evaluation

A logistic regression model is trained independently for each dataset.

The workflow uses:

- An 80/20 training and testing split.
- Stratified sampling to preserve target proportions.
- `StandardScaler` for feature scaling.
- Class weighting to help address class imbalance.
- A fixed random state of 42 for reproducibility.

The models are evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision

### Part 1 Model Results

| Metric | Credit Card Fraud | Credit Card Default |
|---|---:|---:|
| Precision | 0.0564 | 0.3672 |
| Recall | 0.8737 | 0.6202 |
| F1-score | 0.1059 | 0.4613 |
| ROC-AUC | 0.9657 | 0.7081 |
| Average Precision | 0.6719 | 0.4904 |

The fraud model achieves high recall, identifying a large proportion of fraudulent transactions. However, its precision is low, meaning it also flags many legitimate transactions as potentially fraudulent.

The credit default model achieves more balanced precision and recall, although its overall predictive performance is lower based on ROC-AUC.

Because the fraud dataset is highly imbalanced, accuracy alone would not provide a reliable measure of model performance. Recall, precision, F1-score, and average precision provide more useful information.

## Visualization

The pipeline generates a chart comparing precision, recall, and F1-score for both models.

**Generated file:** `data/processed/risk_model_metrics.png`

---

# Part 2: Agent-Based Machine Learning Workflow for Loops

## Overview

A LangGraph workflow coordinates specialized agents to run and evaluate machine learning experiments on the cleaned credit card fraud dataset produced in Part 1.

The workflow uses rule-based agent decisions and scikit-learn. It does not require an external LLM or API key.

The purpose of this workflow is to test different model configurations, compare their validation performance, select the best configuration, and evaluate the selected model on a separate test set.

## Agent Workflow

The workflow contains five agents:

### 1. Data Scientist Agent

- Loads the cleaned fraud dataset from `data/processed/fraud_cleaned.csv`.
- Checks that the target variable exists.
- Checks for missing values and valid target classes.
- Creates stratified training, validation, and test datasets.

### 2. Hypothesis Agent

- Selects the next experiment configuration.
- Changes model class weights and regularization settings to test different approaches to the imbalanced fraud classification problem.

### 3. ML Engineer Agent

- Trains a logistic regression model using the selected configuration.
- Evaluates the model on the validation dataset.
- Calculates precision, recall, F1-score, ROC-AUC, average precision, and a confusion matrix.

### 4. Judge Agent

- Compares the current experiment's validation F1-score with the best result so far.
- Keeps the best-performing model configuration.
- Decides whether another experiment should be run or whether the workflow should proceed to final evaluation.

### 5. Final Evaluation Agent

- Retrains the selected configuration using the combined training and validation data.
- Evaluates the selected model on the held-out test dataset.
- Saves the selected configuration and evaluation results to a JSON file.

## Agent Loop and Experiment Design

The workflow tests three logistic regression configurations with different class weights and regularization settings:

1. **Balanced Logistic Regression:** Uses `class_weight="balanced"`.
2. **Higher Fraud Penalty:** Uses class weights `{0: 1, 1: 10}`.
3. **Stronger Fraud Penalty:** Uses class weights `{0: 1, 1: 25}` and a different regularization setting.

The purpose of changing class weights is to give more importance to fraudulent transactions, which represent a very small proportion of the dataset.

The workflow divides the cleaned dataset into:

- 60% training data
- 20% validation data
- 20% final testing data

The validation F1-score is used to select the best experiment. The final test dataset is held out during experiment selection to provide an independent evaluation of the selected model.

The agent workflow follows this structure:

```text
Data Scientist Agent
        |
        v
Hypothesis Agent
        |
        v
ML Engineer Agent
        |
        v
Judge Agent
        |
        |------ If more experiments remain
        |                   |
        |                   v
        |           Hypothesis Agent
        |
        |------ If experiments are complete
                            |
                            v
                  Final Evaluation Agent
                            |
                            v
                           End
```

This loop allows the workflow to evaluate multiple model configurations before selecting the final model.

## Part 2 Experiment Results

The three experiments were evaluated using the validation dataset.

### Validation Results

| Experiment | Precision | Recall | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|
| Balanced Logistic Regression | 0.0456 | 0.8936 | 0.0868 | 0.9772 | 0.7879 |
| Higher Fraud Penalty | 0.6897 | 0.8511 | **0.7619** | 0.9588 | **0.7939** |
| Stronger Fraud Penalty | 0.5093 | 0.8723 | 0.6431 | 0.9647 | 0.7883 |

**Selected experiment: Higher Fraud Penalty**

The Judge agent selected Higher Fraud Penalty because it achieved the highest validation F1-score of 0.7619.

Compared with the balanced configuration, the higher fraud penalty substantially improved precision while maintaining strong recall. This indicates that the model reduced false positive fraud predictions without losing too much ability to identify fraudulent transactions.

### Final Test Results

After selecting the best configuration, the Final Evaluation Agent retrained the model using the combined training and validation data and evaluated it on the held-out test dataset.

| Metric | Final Test Result |
|---|---:|
| Precision | 0.7849 |
| Recall | 0.7684 |
| F1-score | **0.7766** |
| ROC-AUC | 0.9429 |
| Average Precision | 0.7006 |

### Final Test Confusion Matrix

Rows represent actual classes, and columns represent predicted classes.

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,631 | 20 |
| Actual Fraud | 22 | 73 |

The final model correctly identified 73 fraudulent transactions and missed 22. It also incorrectly flagged 20 legitimate transactions as fraudulent.

### Part 2 Findings

The experiments show that class weighting has a substantial effect on fraud detection performance.

The balanced logistic regression configuration achieved high recall but very low precision. It identified most fraudulent transactions but also generated many false positives.

Increasing the penalty for fraudulent transactions improved the balance between precision and recall. The Higher Fraud Penalty configuration achieved the best validation F1-score of 0.7619 and was selected by the Judge agent.

On the held-out test dataset, the selected model achieved an F1-score of 0.7766, precision of 0.7849, and recall of 0.7684.

These results demonstrate how the agent workflow can iteratively evaluate model configurations and select a model based on an appropriate metric for an imbalanced classification problem.

## Running the Agent Workflow

From the project root, run:

```bash
python3 src/agents/fraud_agent_workflow.py
```

The workflow compiles the LangGraph, generates a Mermaid representation of the agent graph, runs the agents, and saves the evaluation results.

To generate the agent graph visualization:

```bash
python3 src/agents/render_agent_graph.py
```

## Part 2 Outputs

| File | Description |
|---|---|
| `data/processed/fraud_agent_results.json` | Selected experiment, validation history, and final test metrics |
| `data/processed/fraud_agent_graph.mmd` | Mermaid representation of the compiled LangGraph |
| `data/processed/fraud_agent_graph.png` | Agent workflow graph visualization |

---

# Running the Project

## 1. Download the Datasets

Download both datasets from their respective sources and place them in:

```text
data/raw/
```

The expected files are:

```text
data/raw/creditcard.csv
data/raw/default of credit card clients.xls
```

## 2. Configure Environment Variables

Create a `.env` file containing the required Airflow and PostgreSQL settings.

Do not commit this file to GitHub because it contains credentials and configuration secrets.

## 3. Build and Start the Containers

```bash
docker compose build
docker compose up -d
```

## 4. Access Airflow

Open:

[http://localhost:8080](http://localhost:8080)

Find the `financial_risk_analytics` DAG and trigger a manual run, or allow the daily schedule to execute.

## 5. Verify PostgreSQL Tables

```bash
docker compose exec risk-postgres psql -U risk_user -d financial_risk -c "\dt"
```

Check the cleaned record counts:

```bash
docker compose exec risk-postgres psql -U risk_user -d financial_risk -c "SELECT 'fraud_cleaned' AS dataset, COUNT(*) AS rows FROM fraud_cleaned UNION ALL SELECT 'credit_default_cleaned', COUNT(*) FROM credit_default_cleaned;"
```

## 6. Run the Agent Workflow

The agent workflow uses the cleaned fraud dataset generated by Part 1.

```bash
python3 src/agents/fraud_agent_workflow.py
```

Generate the agent graph visualization:

```bash
python3 src/agents/render_agent_graph.py
```

---

# Project Limitations

- The datasets do not share a common record-level identifier, so the pipeline combines model evaluation results rather than joining individual observations.
- The datasets represent different financial risk prediction tasks and should not be interpreted as directly interchangeable.
- Logistic regression provides a baseline model. Further work could explore alternative algorithms, feature engineering, and threshold optimization.
- The fraud dataset is highly imbalanced, which affects precision and other classification metrics.
- The agent workflow evaluates a limited set of logistic regression configurations rather than searching across a large model space.

---

# Key Findings

The Airflow pipeline automates the processing and analysis of two credit-related datasets. The two independent data branches execute in parallel, and their model evaluation results are combined in PostgreSQL.

The Part 1 fraud model achieves high recall but low precision, illustrating the challenge of detecting rare fraudulent transactions without generating many false positives. The credit default model achieves more balanced precision and recall.

The Part 2 agent workflow extends the fraud analysis by testing three logistic regression configurations. The Hypothesis and ML Engineer agents run experiments, while the Judge agent compares validation results and selects the Higher Fraud Penalty configuration.

The selected model achieves a final test F1-score of 0.7766, with precision of 0.7849 and recall of 0.7684.

Together, the two workflows demonstrate automated data engineering, database integration, model evaluation, and iterative agent-based experimentation.

---

# Submission Checklist

The repository should include:

- [ ] Airflow DAG (`dags/financial_risk_dag.py`)
- [ ] Data ingestion and transformation code
- [ ] Modified LangGraph agent workflow (`src/agents/fraud_agent_workflow.py`)
- [ ] Agent graph rendering code (`src/agents/render_agent_graph.py`)
- [ ] Dockerfile and Docker Compose configuration
- [ ] Requirements file
- [ ] README with setup and execution instructions
- [ ] Screenshot of successful Airflow DAG execution
- [ ] Screenshot of the Airflow DAG graph view
- [ ] Screenshot of the modified agent graph
- [ ] Screenshot of the agent model evaluation results

Raw datasets, generated data files, logs, and credentials should not be committed unless specifically required by the instructor.