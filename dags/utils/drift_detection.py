"""
drift_detection.py — Evidently-based data drift detection for Vitiscan.

This module provides functions to:
1. Extract statistical features from images
2. Generate drift reports comparing reference vs current data
3. Detect if significant drift has occurred
4. Upload reports to S3 for archival

How it works:
    Since Evidently works with tabular data (DataFrames), we can't pass
    raw images directly. Instead, we extract numerical features from each
    image (brightness, contrast, color channels, etc.) and compare the
    distributions of these features between:
    - Reference dataset: features from the training set (stable baseline)
    - Current dataset: features from new images (what we're monitoring)

    If the statistical distribution of these features changes significantly,
    Evidently flags it as "drift", which could indicate:
    - Different camera/lighting conditions
    - Seasonal changes in leaf appearance
    - Data quality issues
    - Need for model retraining

Usage in DAG:
    from utils.drift_detection import (
        extract_image_features,
        generate_drift_report,
        check_drift_detected
    )

    # Extract features from new images
    current_features = extract_image_features(new_image_paths)

    # Load reference features (from training set)
    reference_features = pd.read_csv("s3://bucket/reference_features.csv")

    # Generate and analyze drift report
    report, drift_detected = generate_drift_report(reference_features, current_features)
"""

import io
import json
import logging
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════


def extract_single_image_features(image_path: str, s3_client=None) -> dict:
    """
    Extract numerical features from a single image.

    These features capture the statistical properties of the image
    that we want to monitor for drift.

    Args:
        image_path: Local file path or S3 key
        s3_client: boto3 S3 client (required if image_path is S3 key)

    Returns:
        Dictionary of feature name → value

    Features extracted:
        - brightness: mean pixel value (0-1 scale)
        - contrast: standard deviation of pixels (0-1 scale)
        - aspect_ratio: width / height
        - file_size_kb: file size in kilobytes
        - red_mean, green_mean, blue_mean: mean of each color channel
        - width, height: image dimensions in pixels
    """
    try:
        # Load image (from S3 or local)
        if s3_client and image_path.startswith("s3://"):
            # Parse S3 URI
            parts = image_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response["Body"].read()
            img = Image.open(io.BytesIO(image_data))
            file_size_kb = len(image_data) / 1024
        elif s3_client and not image_path.startswith("/"):
            # Assume it's an S3 key (not full URI)
            from config import S3_BUCKET
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=image_path)
            image_data = response["Body"].read()
            img = Image.open(io.BytesIO(image_data))
            file_size_kb = len(image_data) / 1024
        else:
            # Local file
            img = Image.open(image_path)
            file_size_kb = os.path.getsize(image_path) / 1024

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to numpy array for calculations
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1

        # Extract features
        features = {
            # Overall brightness (mean of all pixels)
            "brightness": float(np.mean(img_array)),
            # Contrast (standard deviation)
            "contrast": float(np.std(img_array)),
            # Dimensions
            "width": img.width,
            "height": img.height,
            "aspect_ratio": round(img.width / img.height, 3),
            # File size
            "file_size_kb": round(file_size_kb, 2),
            # Color channel means
            "red_mean": float(np.mean(img_array[:, :, 0])),
            "green_mean": float(np.mean(img_array[:, :, 1])),
            "blue_mean": float(np.mean(img_array[:, :, 2])),
        }

        return features

    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return None


def extract_image_features(
    image_paths: list[str],
    s3_client=None,
    include_class: bool = True
) -> pd.DataFrame:
    """
    Extract features from multiple images and return as DataFrame.

    Args:
        image_paths: List of image paths (local or S3 keys)
        s3_client: boto3 S3 client (required for S3 paths)
        include_class: If True, extract class name from path

    Returns:
        DataFrame with one row per image and columns for each feature

    Example output:
        | image_path | brightness | contrast | class |
        |------------|------------|----------|-------|
        | img_001.jpg| 0.45       | 0.21     | healthy |
        | img_002.jpg| 0.52       | 0.19     | plasmopara |
    """
    all_features = []

    for path in image_paths:
        features = extract_single_image_features(path, s3_client)

        if features is not None:
            features["image_path"] = path

            # Extract class from path if requested
            # Expected format: .../class_name/image.jpg
            if include_class:
                try:
                    parts = path.replace("\\", "/").split("/")
                    # Find the class name (second to last part)
                    class_name = parts[-2] if len(parts) >= 2 else "unknown"
                    features["class"] = class_name
                except Exception:
                    features["class"] = "unknown"

            all_features.append(features)

    if not all_features:
        logger.warning("No features extracted from any images")
        return pd.DataFrame()

    df = pd.DataFrame(all_features)
    logger.info(f"Extracted features from {len(df)} images")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_columns: list[str] = None,
    include_data_quality: bool = True
) -> tuple[Report, dict]:
    """
    Generate an Evidently drift report comparing reference and current data.

    Args:
        reference_data: DataFrame with features from training/reference set
        current_data: DataFrame with features from new/current images
        feature_columns: List of columns to analyze (default: all numeric)
        include_data_quality: Also include data quality metrics

    Returns:
        Tuple of (Report object, results dictionary)

    The results dictionary contains:
        - dataset_drift: bool (True if overall drift detected)
        - drift_share: float (fraction of features with drift, 0.0-1.0)
        - drifted_features: list of feature names that drifted
        - feature_drift_scores: dict of feature → drift score
    """
    # Default to all numeric columns except metadata
    if feature_columns is None:
        exclude_cols = ["image_path", "class"]
        feature_columns = [
            col for col in reference_data.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(reference_data[col])
        ]

    logger.info(f"Analyzing drift for features: {feature_columns}")

    # Ensure both DataFrames have the same columns
    common_cols = list(set(feature_columns) & set(current_data.columns))
    if len(common_cols) < len(feature_columns):
        missing = set(feature_columns) - set(common_cols)
        logger.warning(f"Missing columns in current data: {missing}")
        feature_columns = common_cols

    # Create column mapping for Evidently
    column_mapping = ColumnMapping(
        numerical_features=feature_columns,
        categorical_features=[],
        target=None,
        prediction=None,
    )

    # Build metrics list
    metrics = [DataDriftPreset()]
    if include_data_quality:
        metrics.append(DataQualityPreset())

    # Create and run report
    report = Report(metrics=metrics)
    report.run(
        reference_data=reference_data[feature_columns],
        current_data=current_data[feature_columns],
        column_mapping=column_mapping,
    )

    # Extract results
    report_dict = report.as_dict()

    # Parse drift results
    results = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "drifted_features": [],
        "feature_drift_scores": {},
        "n_reference": len(reference_data),
        "n_current": len(current_data),
        "timestamp": datetime.now().isoformat(),
    }

    # Find the DataDriftPreset results
    for metric in report_dict.get("metrics", []):
        metric_result = metric.get("result", {})

        # Overall drift detection
        if "dataset_drift" in metric_result:
            results["dataset_drift"] = metric_result["dataset_drift"]

        if "share_of_drifted_columns" in metric_result:
            results["drift_share"] = metric_result["share_of_drifted_columns"]

        # Per-feature drift
        if "drift_by_columns" in metric_result:
            for col_name, col_data in metric_result["drift_by_columns"].items():
                drift_detected = col_data.get("drift_detected", False)
                drift_score = col_data.get("drift_score", 0.0)

                results["feature_drift_scores"][col_name] = {
                    "drift_detected": drift_detected,
                    "drift_score": drift_score,
                    "stattest_name": col_data.get("stattest_name", "unknown"),
                }

                if drift_detected:
                    results["drifted_features"].append(col_name)

    logger.info(
        f"Drift analysis complete: "
        f"drift_detected={results['dataset_drift']}, "
        f"drift_share={results['drift_share']:.1%}, "
        f"drifted_features={results['drifted_features']}"
    )

    return report, results


def check_drift_detected(
    results: dict,
    drift_threshold: float = 0.3
) -> tuple[bool, str]:
    """
    Determine if drift exceeds acceptable thresholds.

    Args:
        results: Results dictionary from generate_drift_report()
        drift_threshold: Maximum acceptable drift share (0.0-1.0)

    Returns:
        Tuple of (drift_exceeded: bool, message: str)
    """
    drift_share = results.get("drift_share", 0.0)
    drifted_features = results.get("drifted_features", [])

    if drift_share >= drift_threshold:
        message = (
            f"DRIFT ALERT: {drift_share:.1%} of features show drift "
            f"(threshold: {drift_threshold:.1%}). "
            f"Drifted features: {', '.join(drifted_features)}"
        )
        return True, message

    message = (
        f"✓ Drift within acceptable limits: {drift_share:.1%} "
        f"(threshold: {drift_threshold:.1%})"
    )
    return False, message


# ══════════════════════════════════════════════════════════════════════════════
# S3 OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════


def upload_report_to_s3(
    report: Report,
    results: dict,
    s3_client,
    bucket: str,
    prefix: str = "monitoring/evidently/reports/"
) -> str:
    """
    Upload drift report (HTML) and results (JSON) to S3.

    Args:
        report: Evidently Report object
        results: Results dictionary
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix for reports

    Returns:
        S3 key of the HTML report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save HTML report
    html_key = f"{prefix}drift_report_{timestamp}.html"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        report.save_html(f.name)
        f.flush()
        with open(f.name, "rb") as html_file:
            s3_client.put_object(
                Bucket=bucket,
                Key=html_key,
                Body=html_file.read(),
                ContentType="text/html",
            )
    os.unlink(f.name)

    # Save JSON results
    json_key = f"{prefix}drift_results_{timestamp}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=json_key,
        Body=json.dumps(results, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    logger.info(f"Uploaded drift report to s3://{bucket}/{html_key}")
    logger.info(f"Uploaded drift results to s3://{bucket}/{json_key}")

    return html_key


def load_reference_features(s3_client, bucket: str, key: str) -> pd.DataFrame:
    """
    Load reference features DataFrame from S3.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 key to the CSV file

    Returns:
        DataFrame with reference features
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response["Body"].read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_content))
        logger.info(f"Loaded reference features: {len(df)} rows from s3://{bucket}/{key}")
        return df
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Reference features not found at s3://{bucket}/{key}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading reference features: {e}")
        return pd.DataFrame()


def save_reference_features(
    df: pd.DataFrame,
    s3_client,
    bucket: str,
    key: str
) -> bool:
    """
    Save reference features DataFrame to S3.

    Args:
        df: DataFrame with reference features
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 key for the CSV file

    Returns:
        True if successful, False otherwise
    """
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )
        logger.info(f"Saved reference features: {len(df)} rows to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"Error saving reference features: {e}")
        return False