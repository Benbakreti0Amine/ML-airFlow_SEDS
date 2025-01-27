import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_split_data():

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    np.save(os.path.join(BASE_DIR, "X_train_raw.npy"), X_train)
    np.save(os.path.join(BASE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(BASE_DIR, "X_test_raw.npy"), X_test)
    np.save(os.path.join(BASE_DIR, "y_test.npy"), y_test)


def preprocess_data():

    X_train = np.load(os.path.join(BASE_DIR, "X_train_raw.npy"))
    X_test = np.load(os.path.join(BASE_DIR, "X_test_raw.npy"))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save(os.path.join(BASE_DIR, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(BASE_DIR, "X_test.npy"), X_test_scaled)

    print("Data preprocessing complete.")

def train_and_log_model():

    X_train = np.load(os.path.join(BASE_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(BASE_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(BASE_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(BASE_DIR, "y_test.npy"))

    models = [
    ("KNN-3", KNeighborsClassifier(n_neighbors=3)),
    ("KNN-5", KNeighborsClassifier(n_neighbors=5)),
    ("KNN-7", KNeighborsClassifier(n_neighbors=7)),
    ("Gradient Boosting (50 Estimators)", GradientBoostingClassifier(n_estimators=50, max_depth=3)),
    ("Gradient Boosting (100 Estimators)", GradientBoostingClassifier(n_estimators=100, max_depth=3)),
    ("SVM (RBF Kernel)", SVC(kernel='rbf', C=1, probability=True)),
    ("SVM (Linear Kernel)", SVC(kernel='linear', C=1, probability=True))
    ]


    mlflow.set_tracking_uri("http://localhost:5000")
    if not mlflow.get_experiment_by_name("Iris-Classification"):
        mlflow.create_experiment(
            "Iris-Classification",
            artifact_location="./artifacts/Iris-Classification"
        )
    mlflow.set_experiment("Iris-Classification")

    for model_name, model in models:
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True)


            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("f1-score_macro", report["macro avg"]["f1-score"])

            mlflow.sklearn.log_model(model, model_name) # bech n9dro nkhdmo b model for predictions in the future.

def monitor_model():

    bad_runs = mlflow.search_runs(
        experiment_names=["Iris-Classification"],
        filter_string="metrics.accuracy < 0.9",
        max_results=1
    )

    if not bad_runs.empty:
        print("Bad runs found with accuracy < 0.9, retraining required.")
        return 'retrain_model_task'
    else:
        print("No bad runs found, accuracy is acceptable.")
        return 'done'

def retrain_model():

    train_and_log_model()

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    dag_id='mlflow_iris_pipeline',
    default_args=default_args,
    description='An Airflow pipeline with MLflow integration for Iris dataset',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    load_and_split_data_task = PythonOperator(
        task_id='load_and_split_data',
        python_callable=load_and_split_data,
    )

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    train_and_log_model_task = PythonOperator(
        task_id='train_and_log_model',
        python_callable=train_and_log_model,
    )

    monitor_model_task = PythonOperator(
        task_id='monitor_model',
        python_callable=monitor_model,
        provide_context=True,
    )

    retrain_model_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model,
    )

    load_and_split_data_task >> preprocess_data_task >> train_and_log_model_task >> monitor_model_task
    monitor_model_task >> retrain_model_task
