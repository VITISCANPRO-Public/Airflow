"""
dag_monitoring.py — Weekly monitoring DAG.

This DAG is the entry point of the Vitiscan MLOps pipeline.
It runs automatically every week and checks two conditions:

1. RETRAINING TRIGGERS:
   - Volume: >= MIN_NEW_IMAGES new labeled images on S3
   - Delay: >= MAX_DAYS_WITHOUT_RETRAINING days since last training
   → If either condition is met, triggers dag_data_ingestion

2. PERFORMANCE CHECK:
   - If no trigger is activated, checks production model metrics
   - If F1 < F1_THRESHOLD or Recall < RECALL_THRESHOLD → sends alert

Flow architecture:
    check_retraining_triggers
            │
    ┌───────┴───────┐
    ▼               ▼
trigger_ingestion   check_model_performance
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                send_alert    no_action
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

# Standard library
from datetime import datetime, timedelta
import json
import logging

# Third-party libraries
# FIX: mlflow is now imported globally (not inside the function)
# Benefit: if mlflow is not installed, the error appears at DAG parsing
# instead of at task execution (several hours later)
import boto3
import mlflow

# Airflow imports
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

# Local imports
# FIX: using centralized config.py file
# Before: constants duplicated in each DAG
# After: single source of truth
from config import (
    S3_BUCKET,
    S3_NEW_IMAGES_DIR,
    S3_METADATA_KEY,
    MIN_NEW_IMAGES,
    MAX_DAYS_WITHOUT_RETRAINING,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    F1_THRESHOLD,
    RECALL_THRESHOLD,
    VALID_EXTENSIONS,
)

# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT DAG ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

default_args = {
    "owner": "vitiscan",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def list_s3_images(s3_client, bucket: str, prefix: str) -> list[str]:
    """
    List ALL images in an S3 prefix, with pagination.
    
    FIX: This function uses a paginator to handle the case where there are
    more than 1000 objects. The S3 list_objects_v2 API returns max 1000
    objects per call. Without pagination, images would be silently lost.
    
    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: Prefix (path) to list
        
    Returns:
        List of S3 keys (paths) for images found
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    images = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(VALID_EXTENSIONS):
                images.append(obj["Key"])
    
    return images


# ══════════════════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def check_retraining_triggers(**context) -> str:
    """
    Check conditions for triggering a retraining run.
    
    Two independent criteria are checked:
    1. VOLUME: number of new images >= MIN_NEW_IMAGES
    2. DELAY: days since last training >= MAX_DAYS_WITHOUT_RETRAINING
    
    Returns:
        'trigger_ingestion' if either criterion is met
        'check_model_performance' otherwise
    """
    s3 = boto3.client("s3")

    # ── Criterion #1: Number of new images ────────────────────────────────────
    # FIX: using helper function with pagination
    new_images = list_s3_images(s3, S3_BUCKET, S3_NEW_IMAGES_DIR)
    new_image_count = len(new_images)
    
    logger.info(f"New labeled images available: {new_image_count}/{MIN_NEW_IMAGES}")

    if new_image_count >= MIN_NEW_IMAGES:
        logger.info(
            f"Volume trigger reached ({new_image_count} images) → triggering ingestion"
        )
        context["ti"].xcom_push(key="trigger_reason", value="volume")
        return "trigger_ingestion"

    # ── Criterion #2: Days since last training ────────────────────────────────
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=S3_METADATA_KEY)
        metadata = json.loads(response["Body"].read().decode("utf-8"))
        last_training_date = datetime.fromisoformat(metadata["created_at"])
        days_since_training = (datetime.now() - last_training_date).days
        
        logger.info(
            f"Days since last training: {days_since_training}/{MAX_DAYS_WITHOUT_RETRAINING}"
        )

        if days_since_training >= MAX_DAYS_WITHOUT_RETRAINING:
            logger.info(
                f"Delay trigger reached ({days_since_training} days) → triggering ingestion"
            )
            context["ti"].xcom_push(key="trigger_reason", value="delay")
            return "trigger_ingestion"

    except s3.exceptions.NoSuchKey:
        # Metadata file doesn't exist → first run
        logger.warning(
            "No previous training metadata found → triggering ingestion for first run"
        )
        context["ti"].xcom_push(key="trigger_reason", value="first_run")
        return "trigger_ingestion"
    except Exception as e:
        logger.error(f"Error reading metadata: {e}")
        # On error, continue to performance check
        pass

    logger.info("No retraining trigger reached → checking model performance only")
    return "check_model_performance"


def check_model_performance(**context) -> str:
    """
    Check metrics of the model currently in production.
    
    FIX: Uses MLflow Model Registry to identify the production model
    instead of assuming it's the most recent run.
    
    Returns:
        'send_alert' if metrics are below thresholds
        'no_action' if everything is fine
    """
    # FIX: mlflow is now imported globally (line 42)
    # Before: import mlflow ← was here, inside the function
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        # ── Get Production model via Model Registry ───────────────────────────
        # FIX: We use the "Production" stage from the Registry
        # instead of assuming it's the most recent run
        prod_versions = client.get_latest_versions(
            MLFLOW_MODEL_NAME, 
            stages=["Production"]
        )
        
        if not prod_versions:
            logger.warning(
                f"No model in Production stage for '{MLFLOW_MODEL_NAME}'. "
                "Falling back to latest run."
            )
            # Fallback: use latest run if no model in Production
            experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
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
            f1_score = latest_run.data.metrics.get("test_f1_macro", 0)
            recall = latest_run.data.metrics.get("test_recall_macro", 0)
        else:
            # Use the Production model
            prod_version = prod_versions[0]
            prod_run = client.get_run(prod_version.run_id)
            f1_score = prod_run.data.metrics.get("test_f1_macro", 0)
            recall = prod_run.data.metrics.get("test_recall_macro", 0)
            logger.info(f"Checking Production model v{prod_version.version}")

        logger.info(f"Latest model — F1: {f1_score:.3f} | Recall: {recall:.3f}")
        logger.info(f"Thresholds   — F1: {F1_THRESHOLD} | Recall: {RECALL_THRESHOLD}")

        context["ti"].xcom_push(key="f1_score", value=f1_score)
        context["ti"].xcom_push(key="recall", value=recall)

        if f1_score < F1_THRESHOLD or recall < RECALL_THRESHOLD:
            logger.warning("Model performance below thresholds → sending alert")
            return "send_alert"

        logger.info("Model performance within thresholds → no action needed")
        return "no_action"

    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow error: {e}")
        return "no_action"
    except Exception as e:
        logger.error(f"Error fetching MLflow metrics: {e}")
        return "no_action"


def send_alert(**context) -> None:
    """
    Send an alert when model performance is below thresholds.
    
    In production, this function should:
    - Send an email via SMTP
    - Post a Slack message
    - Create a Jira/PagerDuty ticket
    
    For now, it simply logs the alert.
    """
    ti = context["ti"]
    f1_score = ti.xcom_pull(key="f1_score", task_ids="check_model_performance")
    recall = ti.xcom_pull(key="recall", task_ids="check_model_performance")

    message = (
        f"🚨 VITISCAN MODEL ALERT 🚨\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Model performance below acceptable thresholds:\n"
        f"  F1 Macro  : {f1_score:.3f} (threshold: {F1_THRESHOLD})\n"
        f"  Recall    : {recall:.3f} (threshold: {RECALL_THRESHOLD})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Action required: manual review or forced retraining.\n"
        f"MLflow dashboard: {MLFLOW_TRACKING_URI}"
    )

    logger.warning(message)
    
    # TODO: In production, add here:
    # - send_email(to="mlops-team@vitiscan.com", subject="Model Alert", body=message)
    # - slack_webhook.post(message)
    # - pagerduty.create_incident(message)


# ══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="dag_monitoring",
    description="Weekly monitoring: checks retraining triggers and model performance",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",  # Runs every Monday at midnight
    catchup=False,       # Don't backfill missed runs
    tags=["vitiscan", "monitoring"],
) as dag:

    # ── Tasks ─────────────────────────────────────────────────────────────────

    check_triggers = BranchPythonOperator(
        task_id="check_retraining_triggers",
        python_callable=check_retraining_triggers,
        doc_md="""
        Check if retraining should be triggered.
        Criteria: new image volume OR time since last training.
        """,
    )

    trigger_ingestion = TriggerDagRunOperator(
        task_id="trigger_ingestion",
        trigger_dag_id="dag_data_ingestion",
        wait_for_completion=False,  # Don't wait for ingestion to complete
        conf={"triggered_by": "dag_monitoring"},
        doc_md="Triggers the data ingestion DAG.",
    )

    check_performance = BranchPythonOperator(
        task_id="check_model_performance",
        python_callable=check_model_performance,
        doc_md="""
        Check production model metrics.
        Sends alert if F1 or Recall are below thresholds.
        """,
    )

    alert = PythonOperator(
        task_id="send_alert",
        python_callable=send_alert,
        doc_md="Sends an alert (logs for now, email/Slack in production).",
    )

    no_action = EmptyOperator(
        task_id="no_action",
        doc_md="Terminal state: no action required, model is performing well.",
    )

    # ── Dependencies ──────────────────────────────────────────────────────────

    check_triggers >> trigger_ingestion
    check_triggers >> check_performance >> [alert, no_action]