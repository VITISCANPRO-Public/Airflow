"""
dag_monitoring.py — Weekly monitoring DAG with Evidently drift detection.

This DAG is the entry point of the Vitiscan MLOps pipeline.
It runs automatically every week and performs three types of checks:

1. RETRAINING TRIGGERS:
   - Volume: >= MIN_NEW_IMAGES new labeled images on S3
   - Delay: >= MAX_DAYS_WITHOUT_RETRAINING days since last training
   → If either condition is met, triggers dag_data_ingestion

2. DATA DRIFT DETECTION (NEW - Evidently):
   - Extracts features from new images
   - Compares with reference features (from training set)
   - Generates drift report and uploads to S3
   - Alerts if drift exceeds threshold

3. PERFORMANCE CHECK:
   - If no retraining trigger, checks production model metrics
   - If F1 < F1_THRESHOLD or Recall < RECALL_THRESHOLD → sends alert

Flow architecture:
    check_retraining_triggers
            │
    ┌───────┴───────────────────────────┐
    │                                   │
    ▼                                   ▼
trigger_ingestion              check_data_drift (Evidently)
                                        │
                                ┌───────┴───────┐
                                ▼               ▼
                        send_drift_alert   check_model_performance
                                                   │
                                           ┌──────┴──────┐
                                           ▼             ▼
                                       send_alert    no_action
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
# Standard library
import json
import logging
from datetime import datetime, timedelta

# Third-party libraries
import boto3
import mlflow
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Local imports
from config import (
    DRIFT_DETECTION_ENABLED,
    DRIFT_THRESHOLD,
    F1_THRESHOLD,
    MAX_DAYS_WITHOUT_RETRAINING,
    MIN_IMAGES_FOR_DRIFT,
    MIN_NEW_IMAGES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MONITORED_FEATURES,
    RECALL_THRESHOLD,
    S3_BUCKET,
    S3_DRIFT_REPORTS_DIR,
    S3_METADATA_KEY,
    S3_NEW_IMAGES_DIR,
    S3_REFERENCE_FEATURES_KEY,
    VALID_EXTENSIONS,
)

from utils.drift_detection import (
    check_drift_detected,
    extract_image_features,
    generate_drift_report,
    load_reference_features,
    upload_report_to_s3,
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

    Uses a paginator to handle >1000 objects (S3 API limit per call).

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
        'check_data_drift' otherwise (proceed to drift analysis)
    """
    s3 = boto3.client("s3")

    # ── Criterion #1: Number of new images ────────────────────────────────────
    new_images = list_s3_images(s3, S3_BUCKET, S3_NEW_IMAGES_DIR)
    new_image_count = len(new_images)

    logger.info(
        f"New labeled images available: {new_image_count}/{MIN_NEW_IMAGES}"
    )

    if new_image_count >= MIN_NEW_IMAGES:
        logger.info(
            f"Volume trigger reached ({new_image_count} images) "
            "→ triggering ingestion"
        )
        context["ti"].xcom_push(key="trigger_reason", value="volume")
        context["ti"].xcom_push(key="new_images", value=new_images)
        return "trigger_ingestion"

    # ── Criterion #2: Days since last training ────────────────────────────────
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=S3_METADATA_KEY)
        metadata = json.loads(response["Body"].read().decode("utf-8"))
        last_training_date = datetime.fromisoformat(metadata["created_at"])
        days_since_training = (datetime.now() - last_training_date).days

        logger.info(
            f"Days since last training: "
            f"{days_since_training}/{MAX_DAYS_WITHOUT_RETRAINING}"
        )

        if days_since_training >= MAX_DAYS_WITHOUT_RETRAINING:
            logger.info(
                f"Delay trigger reached ({days_since_training} days) "
                "→ triggering ingestion"
            )
            context["ti"].xcom_push(key="trigger_reason", value="delay")
            return "trigger_ingestion"

    except s3.exceptions.NoSuchKey:
        logger.warning(
            "No previous training metadata found "
            "→ triggering ingestion for first run"
        )
        context["ti"].xcom_push(key="trigger_reason", value="first_run")
        return "trigger_ingestion"
    except Exception as e:
        logger.error(f"Error reading metadata: {e}")

    # Store new images for drift analysis
    context["ti"].xcom_push(key="new_images", value=new_images)

    logger.info("No retraining trigger reached → proceeding to drift analysis")
    return "check_data_drift"


def check_data_drift(**context) -> str:
    """
    Analyze data drift using Evidently.

    Compares features from new images against the reference dataset
    (features extracted from the training set).

    Returns:
        'send_drift_alert' if significant drift detected
        'check_model_performance' otherwise
    """
    # Check if drift detection is enabled
    if not DRIFT_DETECTION_ENABLED:
        logger.info("Drift detection is disabled → skipping")
        return "check_model_performance"

    s3 = boto3.client("s3")
    ti = context["ti"]

    # Get list of new images from previous task
    new_images = ti.xcom_pull(key="new_images", task_ids="check_retraining_triggers")

    if not new_images:
        new_images = list_s3_images(s3, S3_BUCKET, S3_NEW_IMAGES_DIR)

    # Check minimum sample size
    if len(new_images) < MIN_IMAGES_FOR_DRIFT:
        logger.info(
            f"Not enough images for drift analysis: "
            f"{len(new_images)} < {MIN_IMAGES_FOR_DRIFT}"
        )
        return "check_model_performance"

    # ── Load reference features ───────────────────────────────────────────────
    reference_df = load_reference_features(s3, S3_BUCKET, S3_REFERENCE_FEATURES_KEY)

    if reference_df.empty:
        logger.warning(
            "No reference features found. "
            "Run scripts/generate_reference_features.py first. "
            "Skipping drift analysis."
        )
        return "check_model_performance"

    # ── Extract features from new images ──────────────────────────────────────
    logger.info(f"Extracting features from {len(new_images)} new images...")
    current_df = extract_image_features(new_images, s3_client=s3)

    if current_df.empty:
        logger.warning("Failed to extract features from new images")
        return "check_model_performance"

    # ── Generate drift report ─────────────────────────────────────────────────
    logger.info("Generating Evidently drift report...")
    report, results = generate_drift_report(
        reference_data=reference_df,
        current_data=current_df,
        feature_columns=MONITORED_FEATURES,
    )

    # ── Upload report to S3 ───────────────────────────────────────────────────
    report_key = upload_report_to_s3(
        report=report,
        results=results,
        s3_client=s3,
        bucket=S3_BUCKET,
        prefix=S3_DRIFT_REPORTS_DIR,
    )

    # ── Check if drift exceeds threshold ──────────────────────────────────────
    drift_exceeded, message = check_drift_detected(results, DRIFT_THRESHOLD)

    logger.info(message)

    # Push results for downstream tasks
    ti.xcom_push(key="drift_results", value=results)
    ti.xcom_push(key="drift_report_key", value=report_key)
    ti.xcom_push(key="drift_exceeded", value=drift_exceeded)

    if drift_exceeded:
        return "send_drift_alert"

    return "check_model_performance"


def send_drift_alert(**context) -> None:
    """
    Send an alert when significant data drift is detected.

    This indicates that new images are statistically different from
    the training data, which could lead to degraded model performance.
    """
    ti = context["ti"]
    results = ti.xcom_pull(key="drift_results", task_ids="check_data_drift")
    report_key = ti.xcom_pull(key="drift_report_key", task_ids="check_data_drift")

    drift_share = results.get("drift_share", 0)
    drifted_features = results.get("drifted_features", [])

    message = (
        "VITISCAN DATA DRIFT ALERT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Significant data drift detected in new images:\n"
        f"  Drift share    : {drift_share:.1%} (threshold: {DRIFT_THRESHOLD:.1%})\n"
        f"  Drifted features: {', '.join(drifted_features)}\n"
        f"  Reference size : {results.get('n_reference', 'N/A')} images\n"
        f"  Current size   : {results.get('n_current', 'N/A')} images\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Recommended actions:\n"
        "  1. Review the drift report on S3\n"
        "  2. Investigate the cause (lighting, camera, season?)\n"
        "  3. Consider updating the reference dataset\n"
        "  4. May need to retrain the model\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Full report: s3://{S3_BUCKET}/{report_key}"
    )

    logger.warning(message)

    # TODO: In production, add:
    # - send_email(to="mlops-team@vitiscan.com", subject="Drift Alert", body=message)
    # - slack_webhook.post(message)


def check_model_performance(**context) -> str:
    """
    Check metrics of the model currently in production.

    Uses MLflow Model Registry to identify the production model
    and fetch its metrics.

    Returns:
        'send_alert' if metrics are below thresholds
        'no_action' if everything is fine
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        # ── Get Production model via Model Registry ───────────────────────────
        prod_versions = client.get_latest_versions(
            MLFLOW_MODEL_NAME,
            stages=["Production"]
        )

        if not prod_versions:
            logger.warning(
                f"No model in Production stage for '{MLFLOW_MODEL_NAME}'. "
                "Falling back to latest run."
            )
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
            prod_version = prod_versions[0]
            prod_run = client.get_run(prod_version.run_id)
            f1_score = prod_run.data.metrics.get("test_f1_macro", 0)
            recall = prod_run.data.metrics.get("test_recall_macro", 0)
            logger.info(f"Checking Production model v{prod_version.version}")

        logger.info(
            f"Latest model — F1: {f1_score:.3f} | Recall: {recall:.3f}"
        )
        logger.info(
            f"Thresholds   — F1: {F1_THRESHOLD} | Recall: {RECALL_THRESHOLD}"
        )

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
    """
    ti = context["ti"]
    f1_score = ti.xcom_pull(key="f1_score", task_ids="check_model_performance")
    recall = ti.xcom_pull(key="recall", task_ids="check_model_performance")

    message = (
        "VITISCAN MODEL PERFORMANCE ALERT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Model performance below acceptable thresholds:\n"
        f"  F1 Macro  : {f1_score:.3f} (threshold: {F1_THRESHOLD})\n"
        f"  Recall    : {recall:.3f} (threshold: {RECALL_THRESHOLD})\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Action required: manual review or forced retraining.\n"
        f"MLflow dashboard: {MLFLOW_TRACKING_URI}"
    )

    logger.warning(message)


# ══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="dag_monitoring",
    description="Weekly monitoring: triggers, drift detection, performance check",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["vitiscan", "monitoring", "evidently"],
) as dag:

    # ── Task Definitions ──────────────────────────────────────────────────────

    check_triggers = BranchPythonOperator(
        task_id="check_retraining_triggers",
        python_callable=check_retraining_triggers,
        doc_md="Check if retraining should be triggered (volume or delay).",
    )

    trigger_ingestion = TriggerDagRunOperator(
        task_id="trigger_ingestion",
        trigger_dag_id="dag_data_ingestion",
        wait_for_completion=False,
        conf={"triggered_by": "dag_monitoring"},
        doc_md="Triggers the data ingestion DAG.",
    )

    drift_check = BranchPythonOperator(
        task_id="check_data_drift",
        python_callable=check_data_drift,
        doc_md="Analyze data drift using Evidently.",
    )

    drift_alert = PythonOperator(
        task_id="send_drift_alert",
        python_callable=send_drift_alert,
        doc_md="Send alert when significant drift is detected.",
    )

    check_performance = BranchPythonOperator(
        task_id="check_model_performance",
        python_callable=check_model_performance,
        doc_md="Check production model metrics via MLflow.",
    )

    perf_alert = PythonOperator(
        task_id="send_alert",
        python_callable=send_alert,
        doc_md="Send alert when model performance is below thresholds.",
    )

    no_action = EmptyOperator(
        task_id="no_action",
        doc_md="Terminal state: no action required.",
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    # Flow:
    #   check_triggers
    #       ├── trigger_ingestion (if volume/delay trigger)
    #       └── check_data_drift (if no trigger)
    #               ├── send_drift_alert (if drift detected)
    #               └── check_model_performance (if no drift)
    #                       ├── send_alert (if perf issue)
    #                       └── no_action (if all good)

    check_triggers >> trigger_ingestion
    check_triggers >> drift_check

    drift_check >> drift_alert
    drift_check >> check_performance

    check_performance >> [perf_alert, no_action]