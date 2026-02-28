"""
dag_retraining.py — DAG for model retraining, validation and deployment.

This DAG is triggered by dag_data_ingestion when a new balanced dataset is ready.
It handles the complete retraining lifecycle:

1. PROVISION : Provision an EC2 GPU instance (simulated in this version)
2. TRAIN     : Train ResNet18 model and register it in MLflow Registry
3. COMPARE   : Compare with Production model (via Model Registry)
4. PREPROD   : Deploy to pre-production if new model is better
5. TEST      : Run automated tests on pre-prod API
6. PROD      : Promote to Production if tests pass
7. TERMINATE : Terminate EC2 instance (always executed)

Flow architecture:
                    provision_ec2
                         ↓
                    train_model
                         ↓
                evaluate_and_compare
                    ↙         ↘
        deploy_to_preprod    keep_current_model
               ↓                     ↓
        run_preprod_tests            │
            ↙      ↘                 │
    deploy_to_prod  rollback         │
            ↘         ↓         ↙
              → terminate_ec2 ←

Important note about terminate_ec2:
    This task uses trigger_rule="all_done" to ALWAYS execute,
    even if upstream tasks fail. This prevents cloud costs
    from a forgotten EC2 instance.
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

# Standard library
from datetime import datetime, timedelta
import json
import logging

# Third-party libraries
# FIX: all imports are now global
# Before: requests was imported inside run_preprod_tests()
# Problem: if requests is not installed, the error appears late at execution
import boto3
import mlflow
import mlflow.pytorch
import requests

# Airflow imports
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Local imports
# FIX: using centralized config.py file
from config import (
    S3_BUCKET,
    S3_METADATA_KEY,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    API_DIAGNO_URL,
    MIN_F1_IMPROVEMENT,
    VALID_CLASSES,
    EC2_INSTANCE_TYPE,
    EC2_AMI_ID,
    EC2_REGION,
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
    "retry_delay": timedelta(minutes=10),
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def provision_ec2(**context) -> None:
    """
    Provision an EC2 GPU instance for model training.
    
    SIMULATED VERSION
    In this demo version, the instance is simulated.
    In production, this function should:
    
    ```python
    ec2 = boto3.client('ec2', region_name=EC2_REGION)
    response = ec2.run_instances(
        ImageId=EC2_AMI_ID,
        InstanceType=EC2_INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        KeyName='vitiscan-training-key',
        SecurityGroupIds=['sg-xxx'],
        SubnetId='subnet-xxx',
        IamInstanceProfile={'Name': 'vitiscan-ec2-role'},
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': 'vitiscan-training'}]
        }]
    )
    instance_id = response['Instances'][0]['InstanceId']
    
    # Wait for instance to be running
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    ```
    
    Pushes to XCom:
        instance_id: EC2 instance ID (simulated here)
    """
    logger.info("Provisioning EC2 instance for training...")
    logger.info(f"  Instance type : {EC2_INSTANCE_TYPE}")
    logger.info(f"  AMI ID        : {EC2_AMI_ID}")
    logger.info(f"  Region        : {EC2_REGION}")

    # Simulation: generate a fake instance ID
    instance_id = f"i-simulated-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.info(f"EC2 instance ready: {instance_id}")

    context["ti"].xcom_push(key="instance_id", value=instance_id)


def train_model(**context) -> None:
    """
    Train the ResNet18 model and register it in MLflow Model Registry.
    
    SIMULATED VERSION
    This function retrieves the latest MLflow run as a proxy for a
    completed training run. In production, it should:
    
    1. Connect to EC2 instance via SSH/SSM
    2. Launch training script
    3. Wait for training to complete
    4. Retrieve metrics from MLflow
    
    IMPORTANT FIX:
    This version registers the model in MLflow Model Registry.
    This allows:
    - Explicit version management (v1, v2, v3...)
    - Using stages (None → Staging → Production → Archived)
    - Comparing with the correct production model (not just the latest run)
    
    Pushes to XCom:
        new_run_id    : MLflow run ID
        new_f1        : F1 macro score of the new model
        new_recall    : Recall macro of the new model
        new_accuracy  : Accuracy of the new model
        model_name    : Model name in the Registry
        model_version : Model version created
    """
    s3 = boto3.client("s3")

    # ── Get dataset version ───────────────────────────────────────────────────
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=S3_METADATA_KEY)
        metadata = json.loads(response["Body"].read().decode("utf-8"))
        dataset_version = metadata["version"]
        total_images = metadata["total_images"]
        logger.info(f"Training on dataset version: {dataset_version}")
        logger.info(f"Total images: {total_images}")
    except Exception:
        dataset_version = "baseline"
        total_images = 0
        logger.warning("No dataset metadata found, using baseline dataset")

    # ── Get latest MLflow run (training simulation) ───────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No training runs found in MLflow")

    new_run = runs[0]
    new_run_id = new_run.info.run_id
    new_f1 = new_run.data.metrics.get("test_f1_macro", 0)
    new_recall = new_run.data.metrics.get("test_recall_macro", 0)
    new_acc = new_run.data.metrics.get("test_accuracy", 0)

    logger.info(f"Training complete — Run ID: {new_run_id}")
    logger.info(f"  F1 Macro  : {new_f1:.4f}")
    logger.info(f"  Recall    : {new_recall:.4f}")
    logger.info(f"  Accuracy  : {new_acc:.4f}")

    # ── FIX: Register model in Model Registry ─────────────────────────────────
    # Before: we simply compared with runs[1] (fragile and unreliable)
    # After: we use the Registry with versions and stages
    
    model_uri = f"runs:/{new_run_id}/model"
    
    try:
        # Create registered model if it doesn't exist yet
        client.create_registered_model(
            MLFLOW_MODEL_NAME,
            description="Vitiscan grape leaf disease classifier (ResNet18)"
        )
        logger.info(f"Created new registered model: {MLFLOW_MODEL_NAME}")
    except mlflow.exceptions.MlflowException:
        # Model already exists, that's OK
        logger.info(f"Registered model '{MLFLOW_MODEL_NAME}' already exists")

    # Create a new model version
    model_version = client.create_model_version(
        name=MLFLOW_MODEL_NAME,
        source=model_uri,
        run_id=new_run_id,
        description=f"Trained on dataset {dataset_version} with {total_images} images",
    )
    
    logger.info(f"Registered model version: {model_version.version}")

    # ── Push information for downstream tasks ─────────────────────────────────
    context["ti"].xcom_push(key="new_run_id", value=new_run_id)
    context["ti"].xcom_push(key="new_f1", value=new_f1)
    context["ti"].xcom_push(key="new_recall", value=new_recall)
    context["ti"].xcom_push(key="new_accuracy", value=new_acc)
    context["ti"].xcom_push(key="model_name", value=MLFLOW_MODEL_NAME)
    context["ti"].xcom_push(key="model_version", value=model_version.version)


def evaluate_and_compare(**context) -> str:
    """
    Compare the new model with the model currently in Production.
    
    IMPORTANT FIX:
    This version uses MLflow Model Registry to identify the production
    model via its "Production" stage, instead of assuming it's the
    2nd most recent run (fragile approach).
    
    Decision logic:
        - If no model in Production → deploy the new one
        - If new_f1 >= prod_f1 + MIN_F1_IMPROVEMENT → deploy the new one
        - Otherwise → keep current model
    
    Returns:
        'deploy_to_preprod' if the new model is better
        'keep_current_model' otherwise
    """
    ti = context["ti"]
    new_run_id = ti.xcom_pull(key="new_run_id", task_ids="train_model")
    new_f1 = ti.xcom_pull(key="new_f1", task_ids="train_model")
    new_recall = ti.xcom_pull(key="new_recall", task_ids="train_model")
    model_name = ti.xcom_pull(key="model_name", task_ids="train_model")
    new_version = ti.xcom_pull(key="model_version", task_ids="train_model")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # ── FIX: Get Production model via Registry ────────────────────────────────
    # Before (fragile):
    #   runs = client.search_runs(..., max_results=2)
    #   current_run = runs[1]  # Assumed 2nd run is prod
    #
    # After (robust):
    #   We use the "Production" stage from Model Registry
    
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not prod_versions:
            # No model in production → deploy directly
            logger.info("No model in Production stage → deploying new model")
            ti.xcom_push(key="comparison_result", value="no_production_model")
            return "deploy_to_preprod"

        # Get Production model metrics
        prod_version = prod_versions[0]
        prod_run = client.get_run(prod_version.run_id)
        prod_f1 = prod_run.data.metrics.get("test_f1_macro", 0)
        prod_recall = prod_run.data.metrics.get("test_recall_macro", 0)

        # ── Display comparison ────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        logger.info(
            f"  New model     (v{new_version}) — F1: {new_f1:.4f} | Recall: {new_recall:.4f}"
        )
        logger.info(
            f"  Prod model    (v{prod_version.version}) — F1: {prod_f1:.4f} | Recall: {prod_recall:.4f}"
        )
        logger.info(f"  F1 difference : {new_f1 - prod_f1:+.4f}")
        logger.info(f"  Required improvement : {MIN_F1_IMPROVEMENT}")
        logger.info("=" * 60)

        ti.xcom_push(key="prod_f1", value=prod_f1)
        ti.xcom_push(key="prod_recall", value=prod_recall)
        ti.xcom_push(key="prod_version", value=prod_version.version)

        # ── Deployment decision ───────────────────────────────────────────────
        if new_f1 >= prod_f1 + MIN_F1_IMPROVEMENT:
            logger.info(
                f"✓ New model improves F1 by {new_f1 - prod_f1:.4f} → deploying"
            )
            ti.xcom_push(key="comparison_result", value="new_model_wins")
            return "deploy_to_preprod"

        logger.info("✗ New model does not improve sufficiently → keeping current model")
        ti.xcom_push(key="comparison_result", value="production_model_wins")
        return "keep_current_model"

    except mlflow.exceptions.MlflowException as e:
        logger.warning(f"Error accessing Model Registry: {e}")
        logger.info("Proceeding with deployment as no production baseline exists")
        ti.xcom_push(key="comparison_result", value="registry_error")
        return "deploy_to_preprod"


def deploy_to_preprod(**context) -> None:
    """
    Deploy the new model to pre-production environment.
    
    FIX: This version sets the model to "Staging" stage in the
    Model Registry, allowing explicit tracking of which model
    is in pre-production.
    
    In production, this function should also:
    - Update MLFLOW_MODEL_URI secret on HuggingFace Spaces (preprod)
    - Restart the Space to load the new model
    """
    ti = context["ti"]
    new_run_id = ti.xcom_pull(key="new_run_id", task_ids="train_model")
    new_f1 = ti.xcom_pull(key="new_f1", task_ids="train_model")
    model_name = ti.xcom_pull(key="model_name", task_ids="train_model")
    new_version = ti.xcom_pull(key="model_version", task_ids="train_model")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # ── Transition model to "Staging" stage ───────────────────────────────────
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Staging",
    )

    logger.info("Deploying new model to pre-production...")
    logger.info(f"  Model         : {model_name}")
    logger.info(f"  Version       : {new_version}")
    logger.info(f"  Stage         : Staging")
    logger.info(f"  MLflow Run ID : {new_run_id}")
    logger.info(f"  F1 Macro      : {new_f1:.4f}")
    logger.info(f"  Target API    : {API_DIAGNO_URL} (preprod)")

    # In production, add here:
    # - huggingface_hub.update_space_variables(space_id, {"MLFLOW_MODEL_URI": model_uri})
    # - huggingface_hub.restart_space(space_id)

    preprod_model_uri = f"models:/{model_name}/{new_version}"
    logger.info(f"Pre-production model URI: {preprod_model_uri}")
    
    context["ti"].xcom_push(key="preprod_model_uri", value=preprod_model_uri)


def run_preprod_tests(**context) -> str:
    """
    Run automated tests against the pre-production API.
    
    Tests performed:
    1. Health check: / endpoint responds with status 200
    2. Diseases endpoint: /diseases returns the 7 INRAE classes
    
    In production, also add:
    - Prediction test with a reference image
    - Latency test (response time < 2s)
    - Load test (10 parallel requests)
    
    Returns:
        'deploy_to_prod' if all tests pass
        'rollback' if any test fails
    """
    # FIX: requests is now imported globally (line 33)
    # Before: import requests ← was here, inside the function
    
    ti = context["ti"]
    tests_passed = 0
    tests_failed = 0

    logger.info("=" * 60)
    logger.info("RUNNING PRE-PRODUCTION TESTS")
    logger.info("=" * 60)

    # ── Test 1: Health check ──────────────────────────────────────────────────
    try:
        response = requests.get(f"{API_DIAGNO_URL}/", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        logger.info("✓ Test 1 passed: Health check")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Test 1 failed: Health check — {e}")
        tests_failed += 1

    # ── Test 2: Diseases endpoint ─────────────────────────────────────────────
    try:
        response = requests.get(f"{API_DIAGNO_URL}/diseases", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        diseases = list(data["diseases"].keys())
        
        # FIX: using VALID_CLASSES from config.py
        # instead of a hardcoded list
        assert sorted(diseases) == sorted(VALID_CLASSES), \
            f"Unexpected classes: {diseases}"
            
        logger.info("✓ Test 2 passed: /diseases returns correct INRAE classes")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Test 2 failed: /diseases endpoint — {e}")
        tests_failed += 1

    # ── Summary and decision ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Pre-production tests: {tests_passed} passed / {tests_failed} failed")
    logger.info("=" * 60)

    ti.xcom_push(key="tests_passed", value=tests_passed)
    ti.xcom_push(key="tests_failed", value=tests_failed)

    if tests_failed > 0:
        logger.warning("Pre-production tests failed → rolling back")
        return "rollback"

    logger.info("All pre-production tests passed → deploying to production")
    return "deploy_to_prod"


def deploy_to_prod(**context) -> None:
    """
    Promote the validated model to Production in the Model Registry.
    
    Actions:
    1. Archive the old Production model (if exists)
    2. Promote the new model to "Production" stage
    3. Update metadata on S3
    
    In production, also add:
    - Update MLFLOW_MODEL_URI secret on HuggingFace Spaces (prod)
    - Restart production Space
    - Send Slack/email notification for successful deployment
    """
    ti = context["ti"]
    model_name = ti.xcom_pull(key="model_name", task_ids="train_model")
    new_version = ti.xcom_pull(key="model_version", task_ids="train_model")
    new_run_id = ti.xcom_pull(key="new_run_id", task_ids="train_model")
    new_f1 = ti.xcom_pull(key="new_f1", task_ids="train_model")
    new_recall = ti.xcom_pull(key="new_recall", task_ids="train_model")
    new_acc = ti.xcom_pull(key="new_accuracy", task_ids="train_model")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # ── Archive old Production model ──────────────────────────────────────────
    try:
        old_prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for old_version in old_prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=old_version.version,
                stage="Archived",
            )
            logger.info(f"Archived previous production model: v{old_version.version}")
    except mlflow.exceptions.MlflowException:
        logger.info("No previous production model to archive")

    # ── Promote new model to Production ───────────────────────────────────────
    client.transition_model_version_stage(
        name=model_name,
        version=new_version,
        stage="Production",
    )

    logger.info("=" * 60)
    logger.info("🚀 PRODUCTION DEPLOYMENT SUCCESSFUL")
    logger.info("=" * 60)
    logger.info(f"  Model    : {model_name}")
    logger.info(f"  Version  : {new_version}")
    logger.info(f"  F1 Macro : {new_f1:.4f}")
    logger.info(f"  Recall   : {new_recall:.4f}")
    logger.info(f"  Accuracy : {new_acc:.4f}")
    logger.info("=" * 60)

    # ── Update metadata on S3 ─────────────────────────────────────────────────
    s3 = boto3.client("s3")
    
    metadata = {
        "version": new_version,
        "created_at": datetime.now().isoformat(),
        "run_id": new_run_id,
        "f1_macro": new_f1,
        "recall": new_recall,
        "accuracy": new_acc,
        "deployed_to": "production",
        "model_registry": f"{model_name}/v{new_version}",
    }
    
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_METADATA_KEY,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    logger.info(f"Training metadata updated on S3: s3://{S3_BUCKET}/{S3_METADATA_KEY}")


def rollback(**context) -> None:
    """
    Rollback if pre-production tests fail.
    
    Actions:
    1. Remove model from "Staging" stage
    2. Log failure reasons
    3. Keep current Production model unchanged
    
    Note: This function does NOT delete the model version.
    It simply removes it from Staging stage so it won't be
    accidentally promoted to Production.
    """
    ti = context["ti"]
    tests_failed = ti.xcom_pull(key="tests_failed", task_ids="run_preprod_tests")
    model_name = ti.xcom_pull(key="model_name", task_ids="train_model")
    new_version = ti.xcom_pull(key="model_version", task_ids="train_model")
    prod_f1 = ti.xcom_pull(key="prod_f1", task_ids="evaluate_and_compare")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # ── Remove model from Staging stage ───────────────────────────────────────
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="None",  # Remove from Staging
        )
    except mlflow.exceptions.MlflowException as e:
        logger.warning(f"Could not transition model out of Staging: {e}")

    logger.warning("=" * 60)
    logger.warning("ROLLBACK — KEEPING CURRENT PRODUCTION MODEL")
    logger.warning("=" * 60)
    logger.warning(f"  Reason        : {tests_failed} pre-production test(s) failed")
    logger.warning(f"  Failed model  : v{new_version}")
    logger.warning(f"  Keeping model : Production (F1 = {prod_f1:.4f})")
    logger.warning("  Action        : MLFLOW_MODEL_URI unchanged in production")
    logger.warning("=" * 60)


def terminate_ec2(**context) -> None:
    """
    Terminate the EC2 instance used for training.
    
    CRITICAL 
    This task uses trigger_rule="all_done" to ALWAYS execute,
    even if upstream tasks fail. This is CRUCIAL to prevent:
    - Unnecessary cloud costs (p3.2xlarge = ~$3/hour)
    - Forgotten orphan instances
    
    In production:
    ```python
    ec2 = boto3.client('ec2', region_name=EC2_REGION)
    ec2.terminate_instances(InstanceIds=[instance_id])
    
    # Wait for termination
    waiter = ec2.get_waiter('instance_terminated')
    waiter.wait(InstanceIds=[instance_id])
    ```
    """
    ti = context["ti"]
    instance_id = ti.xcom_pull(key="instance_id", task_ids="provision_ec2")

    logger.info(f"Terminating EC2 instance: {instance_id}")
    
    # In production: ec2.terminate_instances(InstanceIds=[instance_id])
    
    logger.info("EC2 instance terminated — no further cloud costs incurred")


# ══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="dag_retraining",
    description="Retraining pipeline: EC2 provisioning, training, validation and deployment",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,  # Triggered by dag_data_ingestion only
    catchup=False,
    tags=["vitiscan", "training", "deployment"],
) as dag:

    # ── Tasks ─────────────────────────────────────────────────────────────────

    provision = PythonOperator(
        task_id="provision_ec2",
        python_callable=provision_ec2,
        doc_md="Provision an EC2 GPU instance for training.",
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        doc_md="Train ResNet18 and register model in MLflow Registry.",
    )

    compare = BranchPythonOperator(
        task_id="evaluate_and_compare",
        python_callable=evaluate_and_compare,
        doc_md="Compare new model with Production model.",
    )

    keep_current = EmptyOperator(
        task_id="keep_current_model",
        doc_md="Terminal state: new model doesn't improve sufficiently.",
    )

    deploy_preprod = PythonOperator(
        task_id="deploy_to_preprod",
        python_callable=deploy_to_preprod,
        doc_md="Deploy new model to pre-production (Staging stage).",
    )

    preprod_tests = BranchPythonOperator(
        task_id="run_preprod_tests",
        python_callable=run_preprod_tests,
        doc_md="Run automated tests on pre-prod API.",
    )

    deploy_prod = PythonOperator(
        task_id="deploy_to_prod",
        python_callable=deploy_to_prod,
        doc_md="Promote model to Production in the Registry.",
    )

    rollback_task = PythonOperator(
        task_id="rollback",
        python_callable=rollback,
        doc_md="Rollback if pre-production tests fail.",
    )

    terminate = PythonOperator(
        task_id="terminate_ec2",
        python_callable=terminate_ec2,
        # CRITICAL: always executed, even if upstream tasks fail
        trigger_rule="all_done",
        doc_md="Terminate EC2 instance (always executed to avoid costs).",
    )

    # ── Dependencies ──────────────────────────────────────────────────────────

    provision >> train >> compare
    compare >> [deploy_preprod, keep_current]
    deploy_preprod >> preprod_tests
    preprod_tests >> [deploy_prod, rollback_task]
    [deploy_prod, rollback_task, keep_current] >> terminate