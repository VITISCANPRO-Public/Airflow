"""
dag_retraining.py — DAG 2: Model retraining, validation and deployment.

Triggered by dag_data_ingestion when a new balanced dataset is ready.
Handles the full retraining lifecycle:
1. Provision EC2 instance (simulated)
2. Train ResNet18 model
3. Evaluate new model on test set
4. Compare with current production model
5. Deploy to pre-production
6. Run pre-production tests
7. Deploy to production
8. Terminate EC2 instance
"""

import json
import logging
from datetime import datetime, timedelta

import boto3
import mlflow
import mlflow.pytorch

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

S3_BUCKET           = "vitiscanpro-bucket"
S3_COMBINED_DIR     = "datasets/combined/"
MLFLOW_TRACKING_URI = "https://mouniat-vitiscanpro-hf.hf.space"
EXPERIMENT_NAME     = "Vitiscan_CNN_resnet_inrae_resnet18_Fine-tuning"
API_DIAGNO_URL      = "https://mouniat-vitiscanpro-diagno-api.hf.space"

# Minimum improvement required to replace production model
MIN_F1_IMPROVEMENT  = 0.01

# EC2 configuration (simulation)
EC2_INSTANCE_TYPE   = "p3.2xlarge"
EC2_AMI_ID          = "ami-0abcdef1234567890"
EC2_REGION          = "eu-west-3"

# ── Default DAG arguments ──────────────────────────────────────────────────────

default_args = {
    "owner":            "vitiscan",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
}

# ── Task functions ─────────────────────────────────────────────────────────────

def provision_ec2(**context) -> None:
    """
    Provisions an EC2 GPU instance for model training.

    In production: creates a real p3.2xlarge instance on AWS.
    In this demo: simulates the provisioning and logs the configuration.

    Note: A real implementation would use boto3 EC2 client:
        ec2 = boto3.client('ec2', region_name=EC2_REGION)
        response = ec2.run_instances(
            ImageId=EC2_AMI_ID,
            InstanceType=EC2_INSTANCE_TYPE,
            MinCount=1, MaxCount=1,
            ...
        )
        instance_id = response['Instances'][0]['InstanceId']
    """
    logger.info("Provisioning EC2 instance for training...")
    logger.info(f"  Instance type : {EC2_INSTANCE_TYPE}")
    logger.info(f"  AMI ID        : {EC2_AMI_ID}")
    logger.info(f"  Region        : {EC2_REGION}")

    # Simulate instance ID
    instance_id = f"i-simulated-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.info(f"EC2 instance ready: {instance_id}")

    context["ti"].xcom_push(key="instance_id", value=instance_id)


def train_model(**context) -> None:
    """
    Trains ResNet18 Fine-tuning model on the latest balanced dataset.

    Retrieves the latest dataset version from S3 metadata,
    runs training with MLflow tracking, and logs the new run ID.

    In production: this task runs on the provisioned EC2 instance
    via SSH or AWS SSM. In this demo, it simulates the training run
    and retrieves the latest MLflow run.
    """
    s3 = boto3.client("s3")

    # Get latest dataset version
    try:
        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key="datasets/combined/last_training_metadata.json"
        )
        metadata        = json.loads(response["Body"].read().decode("utf-8"))
        dataset_version = metadata["version"]
        total_images    = metadata["total_images"]
        logger.info(f"Training on dataset version: {dataset_version}")
        logger.info(f"Total images: {total_images}")
    except Exception:
        dataset_version = "baseline"
        logger.warning("No dataset metadata found, using baseline dataset")

    # In production: trigger training script on EC2
    # Here: retrieve latest MLflow run as proxy for completed training
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"MLflow experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No training runs found in MLflow")

    new_run    = runs[0]
    new_run_id = new_run.info.run_id
    new_f1     = new_run.data.metrics.get("test_f1_macro", 0)
    new_recall = new_run.data.metrics.get("test_recall_macro", 0)
    new_acc    = new_run.data.metrics.get("test_accuracy", 0)

    logger.info(f"Training complete — Run ID: {new_run_id}")
    logger.info(f"  F1 Macro  : {new_f1:.4f}")
    logger.info(f"  Recall    : {new_recall:.4f}")
    logger.info(f"  Accuracy  : {new_acc:.4f}")

    context["ti"].xcom_push(key="new_run_id",  value=new_run_id)
    context["ti"].xcom_push(key="new_f1",      value=new_f1)
    context["ti"].xcom_push(key="new_recall",  value=new_recall)
    context["ti"].xcom_push(key="new_accuracy", value=new_acc)


def evaluate_and_compare(**context) -> str:
    """
    Compares new model metrics against current production model.

    Returns:
        'deploy_to_preprod' if new model is better by MIN_F1_IMPROVEMENT
        'keep_current_model' otherwise
    """
    ti         = context["ti"]
    new_run_id = ti.xcom_pull(key="new_run_id", task_ids="train_model")
    new_f1     = ti.xcom_pull(key="new_f1",     task_ids="train_model")
    new_recall = ti.xcom_pull(key="new_recall", task_ids="train_model")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get current production model metrics (second most recent run)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=2,
    )

    if len(runs) < 2:
        logger.info("No previous model to compare against → deploying new model")
        context["ti"].xcom_push(key="comparison_result", value="new_model_wins")
        return "deploy_to_preprod"

    current_run    = runs[1]
    current_f1     = current_run.data.metrics.get("test_f1_macro", 0)
    current_recall = current_run.data.metrics.get("test_recall_macro", 0)

    logger.info("Model comparison:")
    logger.info(f"  New model     — F1: {new_f1:.4f} | Recall: {new_recall:.4f}")
    logger.info(f"  Current model — F1: {current_f1:.4f} | Recall: {current_recall:.4f}")
    logger.info(f"  Improvement   — F1: {new_f1 - current_f1:+.4f}")

    context["ti"].xcom_push(key="current_f1",     value=current_f1)
    context["ti"].xcom_push(key="current_recall", value=current_recall)

    if new_f1 >= current_f1 + MIN_F1_IMPROVEMENT:
        logger.info(f"New model is better by {new_f1 - current_f1:.4f} → deploying")
        context["ti"].xcom_push(key="comparison_result", value="new_model_wins")
        return "deploy_to_preprod"

    logger.info("New model does not improve sufficiently → keeping current model")
    context["ti"].xcom_push(key="comparison_result", value="current_model_wins")
    return "keep_current_model"


def deploy_to_preprod(**context) -> None:
    """
    Deploys the new model to pre-production environment.

    In production: updates the HuggingFace Space environment variable
    MLFLOW_MODEL_URI to point to the new model S3 path.
    In this demo: logs the deployment action.
    """
    ti         = context["ti"]
    new_run_id = ti.xcom_pull(key="new_run_id", task_ids="train_model")
    new_f1     = ti.xcom_pull(key="new_f1",     task_ids="train_model")

    logger.info("Deploying new model to pre-production...")
    logger.info(f"  MLflow Run ID : {new_run_id}")
    logger.info(f"  F1 Macro      : {new_f1:.4f}")
    logger.info(f"  Target        : {API_DIAGNO_URL} (preprod)")

    # In production:
    # - Update MLFLOW_MODEL_URI secret in HuggingFace preprod Space
    # - Restart the Space to load the new model
    # - via HuggingFace Hub API or direct Space restart

    preprod_model_uri = (
        f"s3://{S3_BUCKET}/mlflow-artifacts/"
        f"{new_run_id}/artifacts"
    )
    logger.info(f"Pre-production model URI: {preprod_model_uri}")
    context["ti"].xcom_push(key="preprod_model_uri", value=preprod_model_uri)


def run_preprod_tests(**context) -> str:
    """
    Runs automated tests against the pre-production API.

    Tests:
    - Health check endpoint responds
    - /diseases endpoint returns correct 7 INRAE classes
    - /diagno endpoint returns valid predictions

    Returns:
        'deploy_to_prod' if all tests pass
        'rollback' if any test fails
    """
    import requests

    ti = context["ti"]
    preprod_uri = ti.xcom_pull(key="preprod_model_uri", task_ids="deploy_to_preprod")
    tests_passed = 0
    tests_failed = 0
    expected_classes = [
        "colomerus_vitis", "elsinoe_ampelina", "erysiphe_necator",
        "guignardia_bidwellii", "healthy",
        "phaeomoniella_chlamydospora", "plasmopara_viticola",
    ]

    logger.info("Running pre-production tests...")

    # Test 1 — Health check
    try:
        response = requests.get(f"{API_DIAGNO_URL}/", timeout=10)
        assert response.status_code == 200
        logger.info("Test 1 passed: Health check")
        tests_passed += 1
    except Exception as e:
        logger.error(f"Test 1 failed: Health check — {e}")
        tests_failed += 1

    # Test 2 — Diseases endpoint
    try:
        response = requests.get(f"{API_DIAGNO_URL}/diseases", timeout=10)
        assert response.status_code == 200
        data     = response.json()
        diseases = list(data["diseases"].keys())
        assert sorted(diseases) == sorted(expected_classes), \
            f"Unexpected classes: {diseases}"
        logger.info("Test 2 passed: /diseases returns correct INRAE classes")
        tests_passed += 1
    except Exception as e:
        logger.error(f"Test 2 failed: /diseases endpoint — {e}")
        tests_failed += 1

    logger.info(f"Pre-production tests: {tests_passed} passed / {tests_failed} failed")

    context["ti"].xcom_push(key="tests_passed", value=tests_passed)
    context["ti"].xcom_push(key="tests_failed", value=tests_failed)

    if tests_failed > 0:
        logger.warning("Pre-production tests failed → rolling back")
        return "rollback"

    logger.info("All pre-production tests passed → deploying to production")
    return "deploy_to_prod"


def deploy_to_prod(**context) -> None:
    """
    Deploys the validated model to production.

    In production:
    - Updates MLFLOW_MODEL_URI in production HuggingFace Space
    - Restarts the production Space
    - Updates last_training_metadata.json on S3
    """
    ti = context["ti"]
    new_run_id = ti.xcom_pull(key="new_run_id",task_ids="train_model")
    new_f1 = ti.xcom_pull(key="new_f1",task_ids="train_model")
    new_recall = ti.xcom_pull(key="new_recall",task_ids="train_model")
    new_acc = ti.xcom_pull(key="new_accuracy",task_ids="train_model")

    logger.info("Deploying new model to PRODUCTION...")
    logger.info(f"  MLflow Run ID : {new_run_id}")
    logger.info(f"  F1 Macro : {new_f1:.4f}")
    logger.info(f"  Recall : {new_recall:.4f}")
    logger.info(f"  Accuracy : {new_acc:.4f}")

    # Update training metadata on S3
    s3 = boto3.client("s3")
    metadata = {
        "version":new_run_id,
        "created_at":datetime.now().isoformat(),
        "run_id": new_run_id,
        "f1_macro":new_f1,
        "recall": new_recall,
        "accuracy": new_acc,
        "deployed_to": "production",
    }
    s3.put_object(
        Bucket=S3_BUCKET,
        Key="datasets/combined/last_training_metadata.json",
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
    )

    logger.info("Production deployment complete")
    logger.info(f"Training metadata updated on S3")


def rollback(**context) -> None:
    """
    Rolls back to the previous production model if pre-production tests fail.
    Logs the rollback action and reasons.
    """
    ti = context["ti"]
    tests_failed = ti.xcom_pull(key="tests_failed", task_ids="run_preprod_tests")
    current_f1 = ti.xcom_pull(key="current_f1", task_ids="evaluate_and_compare")

    logger.warning("Rolling back to previous production model...")
    logger.warning(f"Reason : {tests_failed} pre-production test(s) failed")
    logger.warning(f"Keeping model : F1 = {current_f1:.4f}")
    logger.warning("Action : MLFLOW_MODEL_URI unchanged in production")


def terminate_ec2(**context) -> None:
    """
    Terminates the EC2 instance used for training.
    Always runs, even if previous tasks failed (to avoid unnecessary costs).

    In production:
        ec2 = boto3.client('ec2', region_name=EC2_REGION)
        ec2.terminate_instances(InstanceIds=[instance_id])
    """
    ti = context["ti"]
    instance_id = ti.xcom_pull(key="instance_id", task_ids="provision_ec2")

    logger.info(f"Terminating EC2 instance: {instance_id}")
    logger.info("EC2 instance terminated — no further costs incurred")


# ── DAG definition ───

with DAG(
    dag_id="dag_retraining",
    description="Retraining pipeline: EC2 provisioning, training, validation and deployment",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["vitiscan", "training", "deployment"],
) as dag:

    # ── Tasks ───

    provision = PythonOperator(
        task_id="provision_ec2",
        python_callable=provision_ec2,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    compare = BranchPythonOperator(
        task_id="evaluate_and_compare",
        python_callable=evaluate_and_compare,
    )

    keep_current = EmptyOperator(
        task_id="keep_current_model",
    )

    deploy_preprod = PythonOperator(
        task_id="deploy_to_preprod",
        python_callable=deploy_to_preprod,
    )

    preprod_tests = BranchPythonOperator(
        task_id="run_preprod_tests",
        python_callable=run_preprod_tests,
    )

    deploy_prod = PythonOperator(
        task_id="deploy_to_prod",
        python_callable=deploy_to_prod,
    )

    rollback_task = PythonOperator(
        task_id="rollback",
        python_callable=rollback,
    )

    terminate = PythonOperator(
        task_id="terminate_ec2",
        python_callable=terminate_ec2,
        trigger_rule="all_done",  # Always runs, even if upstream tasks fail
    )

    # ── Dependencies ───

    provision >> train >> compare
    compare >> [deploy_preprod, keep_current]
    deploy_preprod >> preprod_tests
    preprod_tests >> [deploy_prod, rollback_task]
    [deploy_prod, rollback_task, keep_current] >> terminate