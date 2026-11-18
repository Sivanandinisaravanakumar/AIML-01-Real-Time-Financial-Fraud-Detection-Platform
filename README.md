# AI-Based Credit Card Fraud Detection System

An end-to-end machine learning capstone project for credit card fraud detection using traditional ML, real-time streaming, feature stores, model monitoring, and containerized deployment.

> **Educational prototype only.** This project is not intended for real financial decisions or production banking use.

## 1. Project Overview

This project explores two fraud detection datasets and two complementary machine learning approaches:

- **ULB Credit Card Fraud Dataset:** An anonymized dataset with `Time`, `Amount`, and PCA-transformed transaction features (`V1`–`V28`).
- **IEEE-CIS Fraud Detection Dataset:** A larger transaction and identity dataset used for feature engineering, XGBoost training, and additional graph-based experiments.

The system supports transaction scoring through FastAPI and publishes prediction events to Apache Kafka.

## 2. Architecture

```text
                 Transaction Request
                         |
                         v
                    FastAPI API
                         |
             +-----------+-----------+
             |                       |
             v                       v
        Feature Store            ML Model
          (Feast)               (XGBoost)
             |                       |
             +-----------+-----------+
                         |
                         v
                 Fraud Prediction
                         |
                         v
                  Kafka Producer
                         |
                         v
               fraud-transactions
                         |
                         v
                  Kafka Consumer
                         |
                         v
                Transaction Alerts
```

Additional project components:

- Streamlit dashboard for analyst case review and monitoring.
- Evidently AI for data drift analysis.
- Graph Neural Network experiments for transaction relationship analysis.
- Docker and Kubernetes deployment for the IEEE-CIS API.

## 3. Tech Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Graph learning | PyTorch, PyTorch Geometric |
| API | FastAPI, Uvicorn |
| Feature store | Feast |
| Streaming | Apache Kafka |
| Dashboard | Streamlit |
| Monitoring | Evidently AI |
| Experiment tracking | MLflow (training runs logged by `src/train_ieee_model.py`; UI: `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5002`) |
| Containerization | Docker |
| Orchestration | Kubernetes (Docker Desktop) |

## 4. Model Performance

### Original ULB model

The following metrics are reported by the original ULB project README:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9805 |
| PR-AUC | 0.8382 |
| Precision | 0.4398 |
| Recall | 0.8571 |

### IEEE-CIS XGBoost model

Evaluation on the project's validation split:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.8810 |
| PR-AUC | 0.3507 |
| Precision | 0.3161 |
| Recall | 0.4549 |
| F1-score | 0.3730 |

The selected validation threshold is approximately **0.7657**. Threshold selection affects the precision-recall trade-off.

### Graph Neural Network experiment

The second GNN experiment achieved:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.8503 |
| Average Precision | 0.5749 |
| Precision | 0.3514 |
| Recall | 0.6575 |
| F1-score | 0.4580 |

### Hybrid experiment

The XGBoost and GNN scores were combined experimentally. The hybrid did not outperform the XGBoost validation result in this experiment. The GNN is therefore treated as an additional research signal, not as a proven improvement.

> Metrics are dataset- and split-specific. They should not be compared as though they came from the same evaluation dataset or protocol.

## 5. Project Structure

```text
AIML-01-Fraud-Detection/
│
├── src/
│   ├── api.py
│   ├── ieee_api.py
│   ├── ieee_feature_engineering.py
│   ├── train_ieee_model.py
│   ├── ieee_kafka_consumer.py
│   ├── feast_api.py
│   ├── evaluate_hybrid.py
│   ├── drift_monitor.py
│   └── dashboard.py
│
├── models/
│   └── ieee_cis/
│
├── feature_repo/
│   └── feature_repo/
│       ├── feature_store.yaml
│       ├── feature_definitions.py
│       └── data/
│
├── data/
│   └── ieee_cis/
│
├── processed/
├── reports/
├── k8s/
│   └── ieee-deployment.yaml
│
├── Dockerfile
├── Dockerfile.ieee
├── Dockerfile.ieee.dockerignore
├── requirements-ieee.txt
└── README.md
```

Large source datasets and local runtime files are excluded from the repository where appropriate. Feast runtime files are included in the deployment build because the API uses the local feature store.

## 6. Dataset

### ULB Credit Card Fraud Dataset

Contains anonymized credit card transactions with:
- `Time`
- `Amount`
- `V1` to `V28`
- Fraud label

### IEEE-CIS Fraud Detection Dataset

The IEEE-CIS dataset contains transaction and identity information. Feature engineering includes:
- Transaction hour and day
- Transaction velocity features
- Previous transaction counts over time windows
- Average transaction amount features
- Device and address change indicators
- Device missing-value indicators

Dataset source: [IEEE-CIS Fraud Detection – Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

Download the dataset through Kaggle and place the extracted files in the expected local data directory before running the feature engineering pipeline.

## 7. API Usage

### Original ULB API

The original API uses:

```text
POST /predict
```

Run the original API using Docker:

```powershell
docker build -t fraud-detection-api .
docker compose up -d kafka
docker run -d --name fraud-api -p 8000:8000 fraud-detection-api
```

Swagger documentation:

```text
http://localhost:8000/docs
```

### IEEE-CIS API

The IEEE-CIS API uses FastAPI, the XGBoost model, and Feast online features.

The API returns fields including:
- Transaction ID
- Fraud probability
- Decision threshold
- Prediction and label
- Model name
- Feature source
- Kafka message status

The API is exposed on port `8000` inside the container.

## 8. Kafka Streaming

The application publishes transaction predictions to the Kafka topic:

```text
fraud-transactions
```

Kafka listener configuration:

| Client | Bootstrap server |
|---|---|
| Windows host consumer | `localhost:9092` |
| Docker/Kubernetes API | `host.docker.internal:9093` |

Start Kafka:

```powershell
docker compose up -d kafka
```

Run the original ULB consumer:

```powershell
python -m src.kafka_consumer
```

Run the IEEE-CIS consumer:

```powershell
python -m src.ieee_kafka_consumer
```

The consumer receives transaction messages and displays prediction details.

## 9. Feast Feature Store

Feast is used to retrieve online transaction features for the IEEE-CIS prediction API.

The local feature store configuration is located at:

```text
feature_repo/feature_repo/feature_store.yaml
```

The feature repository contains feature definitions and local registry and online-store data.

The online feature view excludes the target label to avoid label leakage during inference.

## 10. Streamlit Dashboard

The dashboard provides:
- Model and prediction overview
- Historical analyst case queue
- Case status and feedback updates
- Data drift monitoring
- Kafka live transaction alerts

Run the dashboard from the project environment:

```powershell
streamlit run src/dashboard.py
```

## 11. Data Drift Monitoring

Evidently AI is used to compare training and test feature distributions.

The monitoring report and associated JSON outputs are stored under:

```text
reports/
```

The current drift analysis identified drift in a subset of monitored features. The configured aggregate drift threshold did not trigger an automatic retraining action.

The retraining trigger is an indicator for review, not a fully automated retraining pipeline.

## 12. Docker Deployment

Build the IEEE-CIS API image:

```powershell
docker build -f Dockerfile.ieee -t fraud-detection-ieee-api:latest .
```

The image contains:
- IEEE-CIS API code
- XGBoost model artifacts
- Feast feature definitions and local store data
- Python runtime dependencies

## 13. Kubernetes Deployment (Docker Desktop)

Verified on a local Docker Desktop Kubernetes cluster (context: `docker-desktop`).

    docker compose up -d kafka
    docker build -f Dockerfile.ieee -t fraud-detection-ieee-api:latest .
    kubectl apply -f k8s/ieee-deployment.yaml
    kubectl rollout status deployment/ieee-fraud-api
    curl.exe http://localhost:30082/

The manifest uses `imagePullPolicy: Never`, so the image must exist in the local
Docker daemon. NodePort `30082` maps to container port `8000`. The pod reaches Kafka
at `host.docker.internal:9093`. This is a local deployment, not a public cloud URL.
## 14. Important Limitations

- Metrics in `models/ieee_cis/metrics.json` are at threshold 0.5 (PR-AUC 0.351, ROC-AUC 0.881, recall 74%, precision 16%). The API uses a tuned threshold of 0.7657 (precision 31.6%, recall 45.5%, F1 0.373).
- The Live Kafka Alerts page shows transactions consumed while the dashboard is open. The demo feed replays IEEE-CIS test rows (the test set has no labels).
- The 85% drift figure from the live monitor is computed on streamed data that includes manual test and replay records. It demonstrates the trigger, not real production drift.
- Retraining is gated on labelled data. The production model is never replaced automatically; candidate evaluation and promotion are not verified end to end.
- GNN weights (`*.pt`) are not packaged in the API image. The API scores with XGBoost + Feast. The XGBoost + GNN blend was evaluated offline and did not outperform XGBoost alone; GNN and hybrid results are experimental.
- GNN metrics (e.g. v2: AP 0.575, ROC-AUC 0.850) are card-node level on 2,979 validation cards and are not directly comparable to the transaction-level XGBoost PR-AUC (0.351). The first GNN attempt (v1) did not learn (ROC-AUC 0.5); v2 improved on it.
- The XGBoost training run (params, metrics, artifacts) is tracked in MLflow by `src/train_ieee_model.py`. GNN, hybrid and threshold summary metrics were logged afterwards from saved JSON files (`scripts/log_mlflow_runs.py`).
- Kubernetes was verified on Docker Desktop. Minikube was not used for the final deployment.
- This is an educational prototype, not a production banking system.
- Dataset anonymization limits the interpretation of some features.
- Validation metrics do not guarantee performance on future transactions.
- Fraud thresholds must be selected according to operational requirements.
- The GNN and hybrid results are experimental.
- Drift detection does not automatically retrain or redeploy the model.
- The local Feast SQLite configuration is intended for this project setup, not as a production multi-instance feature-store architecture.
- Kafka events are currently published to a shared topic by the two project APIs; consumers should identify the dataset/model fields in each message.

## 15. Future Improvements

- Evaluate models on a strictly independent test set.
- Improve temporal validation and leakage prevention.
- Explore model calibration and cost-sensitive threshold selection.
- Improve graph construction and GNN evaluation.
- Add automated model retraining with approval controls.
- Add authentication, monitoring, and production-grade feature-store infrastructure.
- Separate Kafka topics or consumer groups by dataset and application.

## 16. Disclaimer

This project was developed for educational and capstone demonstration purposes. Predictions must not be used as the sole basis for approving, declining, or investigating real financial transactions.


## 17. Experiment Comparison (MLflow, validation split)

| Run | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| XGBoost baseline | 0.50 | 0.158 | 0.744 | 0.261 | 0.351 | 0.881 |
| XGBoost tuned threshold | 0.766 | 0.316 | 0.455 | 0.373 | 0.351 | 0.881 |
| Hybrid XGBoost + GNN | 0.50 | 0.111 | 0.836 | 0.195 | 0.333 | 0.875 |
| Hybrid XGBoost + GNN | 0.797 | 0.304 | 0.455 | 0.365 | 0.333 | 0.875 |
| GCN v2 (card nodes) | - | 0.351 | 0.658 | 0.458 | 0.575 | 0.850 |
| GCN v1 (card nodes) | - | 0.000 | 0.000 | 0.000 | 0.123 | 0.500 |

- Moving the XGBoost threshold from 0.5 to 0.766 roughly doubled precision (0.158 to 0.316), lowered recall (0.744 to 0.455) and raised F1 (0.261 to 0.373).
- The hybrid blend did not beat tuned XGBoost (F1 0.365 vs 0.373), so only XGBoost is served by the API.
- GCN rows use 2,979 card nodes. XGBoost and hybrid use 118,108 validation transactions, so GCN numbers are not directly comparable.
- GCN v1 collapsed to predicting no fraud.
- Thresholds were selected on the validation set, so the numbers are optimistic. The Kaggle test file has no labels.

Log the saved outputs into MLflow and open the UI:

```powershell
.\.venv\Scripts\python.exe scripts\log_mlflow_runs.py
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow_ieee.db --host 127.0.0.1 --port 5001
```

Open http://127.0.0.1:5001, switch to Model training, open the `IEEE-CIS-Fraud-Detection` experiment and compare runs. Running the logging script twice adds duplicate runs.

## 18. Running the Demo

Run every command from the project root.

```powershell
docker compose up -d kafka
kubectl get pods
# API docs: http://localhost:30082/docs

# Window A: consumer
.\.venv\Scripts\python.exe -m src.ieee_kafka_consumer

# Window B: dashboard (http://localhost:8501)
.\.venv\Scripts\python.exe -m streamlit run src\dashboard.py

# Window C: replay labelled transactions
.\.venv\Scripts\python.exe scripts\replay_fraud_demo.py --fraud 10 --legit 40 --delay 1
```

`replay_fraud_demo.py` samples the latest 118,108 labelled rows of `train_features.csv` and mixes known fraud and legitimate transactions so alerts appear. The demo stream is fraud-enriched and is not a realistic fraud rate.

## 19. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Seven tests pass.

## 20. Monitoring and Retraining Gate

| Check | Compares | Output |
|---|---|---|
| `src/drift_monitor.py` | train vs test split | `reports/ieee_data_drift_report.html` |
| `src/live_drift_monitor.py` | train vs last 1,000 Kafka-streamed transactions | `reports/ieee_live_drift_report.html`, `reports/live_retraining_trigger.json` |

The live monitor excludes the raw timestamp and card/address identifier columns, which drift by construction. It needs Evidently, which is installed in `.venv-feast`:

```powershell
.\.venv-feast\Scripts\python.exe src\live_drift_monitor.py
.\.venv\Scripts\python.exe src\retraining_orchestrator.py
```

The drift share depends on the sample, so a result is a prompt to review, not proof that the model degraded. The orchestrator waits for at least 1,000 confirmed labelled rows (`WAITING_FOR_LABELS`), only trains a candidate model, and never changes the production model.

The dashboard Data Drift page shows the offline train-vs-test report. Live stream drift is written to `reports/live_retraining_trigger.json`.

## 21. Architecture Diagram

See [docs/architecture.md](docs/architecture.md).

## 22. Additional Limitations

- The GNN and hybrid are offline experiments and are not served by the API.
- The Kaggle test file has no labels, so final metrics need an independent labelled split.
