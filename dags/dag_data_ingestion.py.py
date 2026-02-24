"""
dag_data_ingestion.py — DAG 1: Ingestion and preparation of new labeled images.

Triggered by dag_monitoring when a retraining condition is met.
Handles the full data preparation lifecycle:
1. List new labeled images available on S3
2. Validate image format and class membership
3. Integrate validated images into the combined dataset
4. Balance the dataset across all classes
5. Update dataset metadata on S3
6. Trigger dag_retraining
"""

import json
import logging
import random
from datetime import datetime, timedelta

import boto3

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

S3_BUCKET         = "vitiscanpro-bucket"
S3_NEW_IMAGES_DIR = "new-images/"
S3_COMBINED_DIR   = "datasets/combined/"
S3_METADATA_KEY   = "datasets/combined/last_training_metadata.json"

VALID_EXTENSIONS  = (".jpg", ".jpeg", ".png", ".webp")

# The 7 INRAE disease classes — must match dag_retraining and the Diagnostic API
VALID_CLASSES = [
    "colomerus_vitis",
    "elsinoe_ampelina",
    "erysiphe_necator",
    "guignardia_bidwellii",
    "healthy",
    "phaeomoniella_chlamydospora",
    "plasmopara_viticola",
]

# Target number of images per class after balancing
TARGET_IMAGES_PER_CLASS = 350

# ── Default DAG arguments ──────────────────────────────────────────────────────

default_args = {
    "owner":            "vitiscan",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

# ── Task functions ─────────────────────────────────────────────────────────────

def list_new_images(**context) -> None:
    """
    Lists all new labeled images available in the S3 new-images/ directory.

    Expected S3 structure:
        new-images/<class_name>/<image_file>

    For example:
        new-images/plasmopara_viticola/img_001.jpg
        new-images/healthy/img_042.png

    Pushes to XCom:
        new_images: list of S3 keys for all valid-extension files found
    """
    s3 = boto3.client("s3")

    response   = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_NEW_IMAGES_DIR)
    all_objects = response.get("Contents", [])

    new_images = [
        obj["Key"] for obj in all_objects
        if obj["Key"].lower().endswith(VALID_EXTENSIONS)
    ]

    logger.info(f"New images found in S3: {len(new_images)}")
    for img in new_images[:10]:    # log first 10 to avoid flooding logs
        logger.info(f"  {img}")
    if len(new_images) > 10:
        logger.info(f"  ... and {len(new_images) - 10} more")

    if not new_images:
        raise ValueError(
            f"No new images found in s3://{S3_BUCKET}/{S3_NEW_IMAGES_DIR}. "
            "Check that images are uploaded with the correct structure: "
            "new-images/<class_name>/<image_file>"
        )

    context["ti"].xcom_push(key="new_images", value=new_images)


def validate_images(**context) -> None:
    """
    Validates that each new image:
    - Belongs to a known INRAE disease class (parsed from its S3 path)
    - Has an accepted file extension

    Expected path format: new-images/<class_name>/<filename>

    Pushes to XCom:
        valid_images  : list of S3 keys that passed validation
        invalid_images: list of S3 keys that failed (with reason logged)
    """
    ti         = context["ti"]
    new_images = ti.xcom_pull(key="new_images", task_ids="list_new_images")

    valid_images   = []
    invalid_images = []

    for s3_key in new_images:
        # Parse class name from path: new-images/<class_name>/<filename>
        parts = s3_key.split("/")

        # parts[0] = "new-images", parts[1] = class_name, parts[2] = filename
        if len(parts) < 3:
            logger.warning(f"Invalid path structure (expected 3 parts): {s3_key}")
            invalid_images.append(s3_key)
            continue

        class_name = parts[1]

        if class_name not in VALID_CLASSES:
            logger.warning(
                f"Unknown class '{class_name}' for image {s3_key}. "
                f"Valid classes: {VALID_CLASSES}"
            )
            invalid_images.append(s3_key)
            continue

        valid_images.append(s3_key)

    logger.info("Validation complete:")
    logger.info(f"  Valid images   : {len(valid_images)}")
    logger.info(f"  Invalid images : {len(invalid_images)}")

    if not valid_images:
        raise ValueError(
            "No valid images passed validation. "
            "Make sure images are stored under a valid class subfolder: "
            f"{VALID_CLASSES}"
        )

    ti.xcom_push(key="valid_images",   value=valid_images)
    ti.xcom_push(key="invalid_images", value=invalid_images)


def integrate_images(**context) -> None:
    """
    Copies validated images from new-images/<class>/ to datasets/combined/<class>/.

    In production: uses S3 copy operations (no data transfer cost within same bucket).
    In this demo: simulates the copy and logs the operations.

    After copying, removes the images from new-images/ to avoid re-processing them
    in the next monitoring cycle.

    Pushes to XCom:
        integrated_count: total number of images successfully copied
        class_counts    : dict mapping each class to its new total image count
    """
    ti           = context["ti"]
    valid_images = ti.xcom_pull(key="valid_images", task_ids="validate_images")

    s3               = boto3.client("s3")
    integrated_count = 0
    class_counts     = {cls: 0 for cls in VALID_CLASSES}

    # Count existing images per class in datasets/combined/
    for cls in VALID_CLASSES:
        prefix   = f"{S3_COMBINED_DIR}{cls}/"
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        existing = [
            obj for obj in response.get("Contents", [])
            if obj["Key"].lower().endswith(VALID_EXTENSIONS)
        ]
        class_counts[cls] = len(existing)
        logger.info(f"Existing images for '{cls}': {len(existing)}")

    # Copy each valid image to datasets/combined/<class>/
    for s3_key in valid_images:
        parts      = s3_key.split("/")
        class_name = parts[1]
        filename   = parts[2]

        destination_key = f"{S3_COMBINED_DIR}{class_name}/{filename}"

        try:
            # In production: real S3 copy
            s3.copy_object(
                Bucket     = S3_BUCKET,
                CopySource = {"Bucket": S3_BUCKET, "Key": s3_key},
                Key        = destination_key,
            )

            # Remove from new-images/ after successful copy
            s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)

            class_counts[class_name] += 1
            integrated_count         += 1
            logger.info(f"Integrated: {s3_key} → {destination_key}")

        except Exception as e:
            logger.error(f"Failed to integrate {s3_key}: {e}")

    logger.info(f"Integration complete: {integrated_count} images copied")
    logger.info("Class distribution after integration:")
    for cls, count in class_counts.items():
        logger.info(f"  {cls}: {count} images")

    ti.xcom_push(key="integrated_count", value=integrated_count)
    ti.xcom_push(key="class_counts",     value=class_counts)


def balance_dataset(**context) -> None:
    """
    Balances the dataset so that each class has exactly TARGET_IMAGES_PER_CLASS images.

    Strategy:
    - If a class has MORE than TARGET images: randomly select images to archive
      (moved to datasets/archived/<class>/ to avoid permanent data loss)
    - If a class has FEWER than TARGET images: log a warning (undersampling
      will be handled during model training via weighted sampling)

    Pushes to XCom:
        final_class_counts : dict mapping each class to its final image count
        total_images       : total number of images in the balanced dataset
    """
    ti           = context["ti"]
    class_counts = ti.xcom_pull(key="class_counts", task_ids="integrate_images")

    s3                 = boto3.client("s3")
    final_class_counts = {}

    for cls in VALID_CLASSES:
        prefix   = f"{S3_COMBINED_DIR}{cls}/"
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        images   = [
            obj["Key"] for obj in response.get("Contents", [])
            if obj["Key"].lower().endswith(VALID_EXTENSIONS)
        ]
        current_count = len(images)

        if current_count > TARGET_IMAGES_PER_CLASS:
            # Randomly select images to archive — keep exactly TARGET_IMAGES_PER_CLASS
            random.shuffle(images)
            to_archive = images[TARGET_IMAGES_PER_CLASS:]

            archived_count = 0
            for img_key in to_archive:
                filename    = img_key.split("/")[-1]
                archive_key = f"datasets/archived/{cls}/{filename}"
                try:
                    s3.copy_object(
                        Bucket     = S3_BUCKET,
                        CopySource = {"Bucket": S3_BUCKET, "Key": img_key},
                        Key        = archive_key,
                    )
                    s3.delete_object(Bucket=S3_BUCKET, Key=img_key)
                    archived_count += 1
                except Exception as e:
                    logger.error(f"Failed to archive {img_key}: {e}")

            final_count = current_count - archived_count
            logger.info(
                f"'{cls}': {current_count} → {final_count} images "
                f"({archived_count} archived to datasets/archived/{cls}/)"
            )

        elif current_count < TARGET_IMAGES_PER_CLASS:
            final_count = current_count
            shortage    = TARGET_IMAGES_PER_CLASS - current_count
            logger.warning(
                f"'{cls}': only {current_count} images "
                f"({shortage} below target of {TARGET_IMAGES_PER_CLASS}). "
                "Weighted sampling will compensate during training."
            )

        else:
            final_count = current_count
            logger.info(f"'{cls}': {current_count} images — already balanced ✓")

        final_class_counts[cls] = final_count

    total_images = sum(final_class_counts.values())

    logger.info(f"Dataset balancing complete. Total images: {total_images}")
    logger.info("Final class distribution:")
    for cls, count in final_class_counts.items():
        logger.info(f"  {cls}: {count}")

    context["ti"].xcom_push(key="final_class_counts", value=final_class_counts)
    context["ti"].xcom_push(key="total_images",       value=total_images)


def update_metadata(**context) -> None:
    """
    Updates the dataset metadata file on S3 after successful ingestion.

    Writes to: datasets/combined/last_training_metadata.json

    This file is read by:
    - dag_retraining : to know which dataset version to train on
    - dag_monitoring : to calculate days since last training
    """
    ti                 = context["ti"]
    final_class_counts = ti.xcom_pull(key="final_class_counts", task_ids="balance_dataset")
    total_images       = ti.xcom_pull(key="total_images",       task_ids="balance_dataset")
    integrated_count   = ti.xcom_pull(key="integrated_count",   task_ids="integrate_images")

    version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    metadata = {
        "version":            version,
        "created_at":         datetime.now().isoformat(),
        "total_images":       total_images,
        "new_images_added":   integrated_count,
        "class_distribution": final_class_counts,
        "target_per_class":   TARGET_IMAGES_PER_CLASS,
        "classes":            VALID_CLASSES,
        "deployed_to":        "pending_retraining",
    }

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket      = S3_BUCKET,
        Key         = S3_METADATA_KEY,
        Body        = json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType = "application/json",
    )

    logger.info(f"Metadata updated on S3: s3://{S3_BUCKET}/{S3_METADATA_KEY}")
    logger.info(f"  Version      : {version}")
    logger.info(f"  Total images : {total_images}")
    logger.info(f"  New images   : {integrated_count}")


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="dag_data_ingestion",
    description="Data ingestion pipeline: validate, integrate and balance new labeled images",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,        # Triggered by dag_monitoring only — never runs on its own
    catchup=False,
    tags=["vitiscan", "ingestion", "data"],
) as dag:

    # ── Tasks ──────────────────────────────────────────────────────────────────

    list_images = PythonOperator(
        task_id="list_new_images",
        python_callable=list_new_images,
    )

    validate = PythonOperator(
        task_id="validate_images",
        python_callable=validate_images,
    )

    integrate = PythonOperator(
        task_id="integrate_images",
        python_callable=integrate_images,
    )

    balance = PythonOperator(
        task_id="balance_dataset",
        python_callable=balance_dataset,
    )

    update_meta = PythonOperator(
        task_id="update_metadata",
        python_callable=update_metadata,
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="dag_retraining",
        wait_for_completion=False,
        conf={"triggered_by": "dag_data_ingestion"},
    )

    # ── Dependencies ───────────────────────────────────────────────────────────

    list_images >> validate >> integrate >> balance >> update_meta >> trigger_retraining