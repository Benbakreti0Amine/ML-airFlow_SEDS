# Airflow MLflow Iris Classification Pipeline

An end-to-end machine learning pipeline for classifying the Iris dataset, integrating Apache Airflow for workflow orchestration and MLflow for experiment tracking and model management.

## Features

- **Data Loading and Preprocessing**
  - Loads the Iris dataset
  - Splits into training and testing sets
  - Applies feature standardization

- **Model Training**
  - Trains multiple models:
    - K-Nearest Neighbors
    - Gradient Boosting
    - Support Vector Machine (SVM)

- **Experiment Tracking**
  - Logs metrics, parameters, and trained models in MLflow experiment

- **Monitoring and Retraining**
  - Monitors model performance
  - Automatically retrains models if accuracy falls below threshold

## Prerequisites

- Python 3.8 or higher
- Apache Airflow
- MLflow
- Required Python libraries (see `requirements.txt`)

## Setup and Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up MLflow tracking:
   ```bash
   # Start the MLflow server
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```

4. Start Airflow services:
   ```bash
   # Start the webserver
   airflow webserver

   # Start the scheduler
   airflow scheduler
   ```

5. Trigger the DAG:
   ```bash
   # Via command line
   airflow dags trigger mlflow_iris_pipeline
   ```
   Alternatively, trigger via the Airflow UI

6. Monitor the pipeline:
   - Access MLflow dashboard at http://localhost:5000

## Project Workflow

### Data Preparation
- Dataset splitting (train/test)
- Feature standardization

### Model Training
- Trains three ML models:
  - KNN
  - Gradient Boosting
  - SVM
- Logs performance metrics to MLflow:
  - Accuracy
  - F1-score

### Monitoring
- Performance threshold monitoring
- Automated accuracy checks (threshold: 0.9)

### Retraining
- Automated model retraining when needed
- Performance logging and tracking

## Directory Structure

```
.
├── dags/               # Airflow DAG scripts
├── artifacts/          # MLflow artifacts (models, metrics)
├── mlruns/            # MLflow experiment logs
└── requirements.txt    # Project dependencies
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
