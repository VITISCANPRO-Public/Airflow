"""
dag_monitoring.py — Weekly monitoring DAG.

Checks two retraining triggers:
- Volume trigger  : >= MIN_NEW_IMAGES new labeled images on S3
- Delay trigger   : >= MAX_DAYS_WITHOUT_RETRAINING since last training
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
import boto3
import json
import logging

logger = logging.getLogger(__name__)

S3_BUCKET                   = "vitiscanpro-bucket"
S3_NEW_IMAGES_DIR           = "new-images/"
S3_METADATA_KEY             = "datasets/combined/last_training_metadata.json"
MIN_NEW_IMAGES              = 200
MAX_DAYS_WITHOUT_RETRAINING = 60
MLFLOW_TRACKING_URI         = "https://mouniat-vitiscanpro-hf.hf.space"
F1_THRESHOLD                = 0.90
RECALL_THRESHOLD            = 0.90

default_args = {
    "owner":            "vitiscan",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}


def check_retraining_triggers(**context) -> str:
    """
    Checks volume and delay triggers.
    Returns 'trigger_ingestion' if either condition is met, 'check_model_performance' otherwise.
    """
    s3 = boto3.client("s3")

    # ── Criteria N°1 : Number of new photos ───
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_NEW_IMAGES_DIR)
    image_files = [
        obj for obj in response.get("Contents", [])
        if obj["Key"].endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    new_image_count = len(image_files)
    logger.info(f"New labeled images available: {new_image_count}/{MIN_NEW_IMAGES}")

    if new_image_count >= MIN_NEW_IMAGES:
        logger.info(f"Volume trigger reached ({new_image_count} images) → triggering ingestion")
        context["ti"].xcom_push(key="trigger_reason", value="volume")
        return "trigger_ingestion"

    # ── Criteria N°2 : delay since last training ───
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=S3_METADATA_KEY)
        metadata = json.loads(response["Body"].read().decode("utf-8"))
        last_training_date = datetime.fromisoformat(metadata["created_at"])
        days_since_training = (datetime.now() - last_training_date).days
        logger.info(f"Days since last training: {days_since_training}/{MAX_DAYS_WITHOUT_RETRAINING}")

        if days_since_training >= MAX_DAYS_WITHOUT_RETRAINING:
            logger.info(f"Delay trigger reached ({days_since_training} days) → triggering ingestion")
            context["ti"].xcom_push(key="trigger_reason", value="delay")
            return "trigger_ingestion"

    except Exception:
        logger.warning("No previous training metadata found → triggering ingestion for first run")
        context["ti"].xcom_push(key="trigger_reason", value="first_run")
        return "trigger_ingestion"

    logger.info("No retraining trigger reached → checking model performance only")
    return "check_model_performance"


def check_model_performance(**context) -> str:
    """
    Fetches latest model metrics from MLflow.
    Returns 'send_alert' if below thresholds, 'no_action' otherwise.
    """
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(
            "Vitiscan_CNN_resnet_inrae_resnet18_Fine-tuning"
        )
        if not experiment:
            logger.warning("MLflow experiment not found")
            return "no_action"

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            logger.warning("No runs found in MLflow")
            return "no_action"

        latest_run = runs[0]
        f1_score   = latest_run.data.metrics.get("test_f1_macro", 0)
        recall     = latest_run.data.metrics.get("test_recall_macro", 0)

        logger.info(f"Latest model — F1: {f1_score:.3f} | Recall: {recall:.3f}")
        logger.info(f"Thresholds   — F1: {F1_THRESHOLD} | Recall: {RECALL_THRESHOLD}")

        context["ti"].xcom_push(key="f1_score", value=f1_score)
        context["ti"].xcom_push(key="recall",   value=recall)

        if f1_score < F1_THRESHOLD or recall < RECALL_THRESHOLD:
            logger.warning("Model performance below thresholds → sending alert")
            return "send_alert"

        logger.info("Model performance within thresholds → no action needed")
        return "no_action"

    except Exception as e:
        logger.error(f"Error fetching MLflow metrics: {e}")
        return "no_action"


def send_alert(**context) -> None:
    """
    Sends an alert when model performance drops below thresholds.
    In production: would send email or Slack notification.
    """
    ti       = context["ti"]
    f1_score = ti.xcom_pull(key="f1_score", task_ids="check_model_performance")
    recall   = ti.xcom_pull(key="recall",   task_ids="check_model_performance")

    message = (
        f"VITISCAN MODEL ALERT\n"
        f"Model performance below acceptable thresholds:\n"
        f"  F1 Macro  : {f1_score:.3f} (threshold: {F1_THRESHOLD})\n"
        f"  Recall    : {recall:.3f} (threshold: {RECALL_THRESHOLD})\n"
        f"Action required: manual review or forced retraining.\n"
        f"MLflow dashboard: {MLFLOW_TRACKING_URI}"
    )

    logger.warning(message)
    # In production: send email or Slack notification here


# ── DAG definition ───

with DAG(
    dag_id="dag_monitoring",
    description="Weekly monitoring: checks retraining triggers and model performance",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["vitiscan", "monitoring"],
) as dag:

    check_triggers = BranchPythonOperator(
        task_id="check_retraining_triggers",
        python_callable=check_retraining_triggers,
    )

    trigger_ingestion = TriggerDagRunOperator(
        task_id="trigger_ingestion",
        trigger_dag_id="dag_data_ingestion",
        wait_for_completion=False,
        conf={"triggered_by": "dag_monitoring"},
    )

    check_performance = BranchPythonOperator(
        task_id="check_model_performance",
        python_callable=check_model_performance,
    )

    alert = PythonOperator(
        task_id="send_alert",
        python_callable=send_alert,
    )

    no_action = EmptyOperator(task_id="no_action")

    # ── Dependencies ───

    check_triggers >> trigger_ingestion
    check_triggers >> check_performance >> [alert, no_action]