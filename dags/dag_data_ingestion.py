"""
dag_data_ingestion.py — DAG for ingestion and preparation of new labeled images.

This DAG is triggered by dag_monitoring when a retraining condition is met.
It handles the complete data preparation lifecycle:

1. LIST    : List new images on S3 (new-images/)
2. VALIDATE: Check format and INRAE class membership
3. INTEGRATE: Copy validated images to the combined dataset
4. BALANCE : Balance dataset (max TARGET_IMAGES_PER_CLASS per class)
5. METADATA: Update metadata on S3
6. TRIGGER : Trigger dag_retraining

Flow architecture:
    list_new_images → validate_images → integrate_images → balance_dataset
                                                                  ↓
                              trigger_retraining ← update_metadata

Expected S3 structure:
    new-images/
    ├── colomerus_vitis/
    │   ├── img_001.jpg
    │   └── img_002.png
    ├── healthy/
    │   └── img_003.webp
    └── plasmopara_viticola/
        └── img_004.jpeg
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

# Standard library
import json
import logging
import random
from datetime import datetime, timedelta

# Third-party libraries
import boto3

# Airflow imports
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Local imports
from dags.config import (
    S3_ARCHIVED_DIR,
    S3_BUCKET,
    S3_COMBINED_DIR,
    S3_METADATA_KEY,
    S3_NEW_IMAGES_DIR,
    TARGET_IMAGES_PER_CLASS,
    VALID_CLASSES,
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

    IMPORTANT FIX:
    The S3 list_objects_v2 API returns MAXIMUM 1000 objects per call.
    Without pagination, if you have 1500 images, you silently lose 500!

    This function uses a paginator that automatically handles multiple
    calls and retrieves ALL objects.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: Prefix (path) to list

    Returns:
        List of S3 keys (full paths) for images found

    Example:
        >>> s3 = boto3.client("s3")
        >>> images = list_s3_images(s3, "my-bucket", "new-images/healthy/")
        >>> print(images)
        ['new-images/healthy/img_001.jpg', 'new-images/healthy/img_002.png']
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    images = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(VALID_EXTENSIONS):
                images.append(obj["Key"])

    return images


def count_s3_images(s3_client, bucket: str, prefix: str) -> int:
    """
    Count the number of images in an S3 prefix, with pagination.

    Optimization: if you only need the count and not the list,
    this function avoids storing all paths in memory.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: Prefix (path) to count

    Returns:
        Number of images found
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    count = 0
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(VALID_EXTENSIONS):
                count += 1

    return count


# ══════════════════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def list_new_images(**context) -> None:
    """
    List all new labeled images available on S3.

    Expected S3 structure:
        new-images/<class_name>/<image_file>

    Example:
        new-images/plasmopara_viticola/img_001.jpg
        new-images/healthy/img_042.png

    Pushes to XCom:
        new_images: list of S3 keys for all images found
    """
    s3 = boto3.client("s3")

    new_images = list_s3_images(s3, S3_BUCKET, S3_NEW_IMAGES_DIR)

    logger.info(f"New images found in S3: {len(new_images)}")

    for img in new_images[:10]:
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
    Validate that each new image:
    - Belongs to a known INRAE class (parsed from its S3 path)
    - Has an accepted file extension

    Expected path format: new-images/<class_name>/<filename>

    Pushes to XCom:
        valid_images  : list of S3 keys that passed validation
        invalid_images: list of S3 keys that failed (with reason logged)
    """
    ti = context["ti"]
    new_images = ti.xcom_pull(key="new_images", task_ids="list_new_images")

    valid_images = []
    invalid_images = []

    for s3_key in new_images:
        parts = s3_key.split("/")

        if len(parts) < 3:
            logger.warning(
                f"Invalid path structure (expected 3 parts): {s3_key}"
            )
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

    ti.xcom_push(key="valid_images", value=valid_images)
    ti.xcom_push(key="invalid_images", value=invalid_images)


def integrate_images(**context) -> None:
    """
    Copy validated images from new-images/<class>/ to datasets/combined/<class>/.

    S3 operations:
    - Copy: copies image to combined dataset (no data transfer cost,
            this is a server-side operation within the same bucket)
    - Delete: removes image from new-images/ after successful copy

    This prevents re-processing the same images in the next monitoring cycle.

    Pushes to XCom:
        integrated_count: total number of images successfully copied
        class_counts    : dict mapping each class to its new image total
    """
    ti = context["ti"]
    valid_images = ti.xcom_pull(key="valid_images", task_ids="validate_images")

    s3 = boto3.client("s3")
    integrated_count = 0
    class_counts = {cls: 0 for cls in VALID_CLASSES}

    # ── Count existing images per class ───────────────────────────────────────
    for cls in VALID_CLASSES:
        prefix = f"{S3_COMBINED_DIR}{cls}/"
        count = count_s3_images(s3, S3_BUCKET, prefix)
        class_counts[cls] = count
        logger.info(f"Existing images for '{cls}': {count}")

    # ── Copy each validated image to datasets/combined/<class>/ ───────────────
    for s3_key in valid_images:
        parts = s3_key.split("/")
        class_name = parts[1]
        filename = parts[2]

        destination_key = f"{S3_COMBINED_DIR}{class_name}/{filename}"

        try:
            s3.copy_object(
                Bucket=S3_BUCKET,
                CopySource={"Bucket": S3_BUCKET, "Key": s3_key},
                Key=destination_key,
            )

            s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)

            class_counts[class_name] += 1
            integrated_count += 1
            logger.info(f"Integrated: {s3_key} → {destination_key}")

        except Exception as e:
            logger.error(f"Failed to integrate {s3_key}: {e}")

    logger.info(f"Integration complete: {integrated_count} images copied")
    logger.info("Class distribution after integration:")
    for cls, count in class_counts.items():
        logger.info(f"  {cls}: {count} images")

    ti.xcom_push(key="integrated_count", value=integrated_count)
    ti.xcom_push(key="class_counts", value=class_counts)


def balance_dataset(**context) -> None:
    """
    Balance the dataset so each class has at most TARGET_IMAGES_PER_CLASS.

    Strategy:
    - If a class has MORE than TARGET images:
      → Randomly select excess images
      → Move to datasets/archived/<class>/ (never permanently deleted)

    - If a class has FEWER than TARGET images:
      → Log a warning (weighted sampling will compensate during training)

    Archiving instead of deleting allows:
    - Recovering data if needed
    - Keeping a complete history
    - Avoiding accidental loss of labeled data (expensive to produce)

    Pushes to XCom:
        final_class_counts: dict mapping each class to its final count
        total_images      : total number of images in the balanced dataset
    """
    s3 = boto3.client("s3")
    final_class_counts = {}

    for cls in VALID_CLASSES:
        prefix = f"{S3_COMBINED_DIR}{cls}/"

        images = list_s3_images(s3, S3_BUCKET, prefix)
        current_count = len(images)

        if current_count > TARGET_IMAGES_PER_CLASS:
            # ── Class has excess: archive extra images ────────────────────────
            random.shuffle(images)
            to_archive = images[TARGET_IMAGES_PER_CLASS:]

            archived_count = 0
            for img_key in to_archive:
                filename = img_key.split("/")[-1]
                archive_key = f"{S3_ARCHIVED_DIR}{cls}/{filename}"

                try:
                    s3.copy_object(
                        Bucket=S3_BUCKET,
                        CopySource={"Bucket": S3_BUCKET, "Key": img_key},
                        Key=archive_key,
                    )
                    s3.delete_object(Bucket=S3_BUCKET, Key=img_key)
                    archived_count += 1
                except Exception as e:
                    logger.error(f"Failed to archive {img_key}: {e}")

            final_count = current_count - archived_count
            logger.info(
                f"'{cls}': {current_count} → {final_count} images "
                f"({archived_count} archived)"
            )

        elif current_count < TARGET_IMAGES_PER_CLASS:
            # ── Class has deficit: warning ────────────────────────────────────
            final_count = current_count
            shortage = TARGET_IMAGES_PER_CLASS - current_count
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
        status = "✓" if count == TARGET_IMAGES_PER_CLASS else "⚠"
        logger.info(f"  {status} {cls}: {count}")

    context["ti"].xcom_push(key="final_class_counts", value=final_class_counts)
    context["ti"].xcom_push(key="total_images", value=total_images)


def update_metadata(**context) -> None:
    """
    Update the dataset metadata file on S3.

    Writes to: datasets/combined/last_training_metadata.json

    This file is read by:
    - dag_retraining: to know which dataset version to use
    - dag_monitoring: to calculate days since last training
    """
    ti = context["ti"]
    final_class_counts = ti.xcom_pull(
        key="final_class_counts", task_ids="balance_dataset"
    )
    total_images = ti.xcom_pull(key="total_images", task_ids="balance_dataset")
    integrated_count = ti.xcom_pull(
        key="integrated_count", task_ids="integrate_images"
    )

    version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "total_images": total_images,
        "new_images_added": integrated_count,
        "class_distribution": final_class_counts,
        "target_per_class": TARGET_IMAGES_PER_CLASS,
        "classes": VALID_CLASSES,
        "deployed_to": "pending_retraining",
    }

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_METADATA_KEY,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    logger.info(f"Metadata updated on S3: s3://{S3_BUCKET}/{S3_METADATA_KEY}")
    logger.info(f"  Version      : {version}")
    logger.info(f"  Total images : {total_images}")
    logger.info(f"  New images   : {integrated_count}")


# ══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="dag_data_ingestion",
    description="Data ingestion: validate, integrate and balance new images",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["vitiscan", "ingestion", "data"],
) as dag:

    list_images = PythonOperator(
        task_id="list_new_images",
        python_callable=list_new_images,
        doc_md="List all new images in s3://bucket/new-images/",
    )

    validate = PythonOperator(
        task_id="validate_images",
        python_callable=validate_images,
        doc_md="Validate format and INRAE class membership.",
    )

    integrate = PythonOperator(
        task_id="integrate_images",
        python_callable=integrate_images,
        doc_md="Copy validated images to the combined dataset.",
    )

    balance = PythonOperator(
        task_id="balance_dataset",
        python_callable=balance_dataset,
        doc_md=f"Balance dataset to {TARGET_IMAGES_PER_CLASS} images/class.",
    )

    update_meta = PythonOperator(
        task_id="update_metadata",
        python_callable=update_metadata,
        doc_md="Update dataset metadata on S3.",
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="dag_retraining",
        wait_for_completion=False,
        conf={"triggered_by": "dag_data_ingestion"},
        doc_md="Trigger the model retraining DAG.",
    )

    # ── Dependencies ──────────────────────────────────────────────────────────

    (
        list_images
        >> validate
        >> integrate
        >> balance
        >> update_meta
        >> trigger_retraining
    )
