"""
config.py — Centralized configuration for all Vitiscan DAGs.

This file centralizes ALL constants used across the DAGs.
Benefits:
- Single source of truth (no duplication)
- Values can be overridden via environment variables
- Easier testing (this module can be mocked)
- Configuration changes without modifying DAG code

Values are read from environment variables.
Default values are used if environment variables are not defined.

Usage:
    from config import S3_BUCKET, MIN_NEW_IMAGES, DRIFT_THRESHOLD
"""
import os

# AWS S3 CONFIGURATION ═══════════════════════════════════════════════

# Main bucket for all Vitiscan data
S3_BUCKET = os.environ.get("VITISCAN_S3_BUCKET", "vitiscanpro-bucket")

# Directory where new labeled images are uploaded
S3_NEW_IMAGES_DIR = "new-images/"

# Directory for the combined dataset (validated and balanced images)
S3_COMBINED_DIR = "datasets/combined/"

# Metadata file for the last training run
S3_METADATA_KEY = "datasets/combined/last_training_metadata.json"

# Directory for archived images
S3_ARCHIVED_DIR = "datasets/archived/"

# Evidently reports and reference data
S3_EVIDENTLY_DIR = "monitoring/evidently/"
S3_REFERENCE_FEATURES_KEY = "monitoring/reference_features.csv"
S3_DRIFT_REPORTS_DIR = "monitoring/evidently/reports/"



# MLFLOW CONFIGURATION ═══════════════════════════════════════════════

# MLflow tracking server URL (hosted on HuggingFace Spaces)
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://mouniat-vitiscanpro-hf.hf.space"
)

# MLflow experiment name for run tracking
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "Vitiscan_CNN_resnet_inrae_resnet18_Fine-tuning"
)

# Model name in MLflow Model Registry
# Used for version management and stages (Staging, Production, Archived)
MLFLOW_MODEL_NAME = os.environ.get(
    "MLFLOW_MODEL_NAME",
    "vitiscan-resnet18"
)


# API CONFIGURATION ═══════════════════════════════════════════════════

# Diagnostic API URL (hosted on HuggingFace Spaces)
API_DIAGNO_URL = os.environ.get(
    "VITISCAN_API_DIAGNO_URL",
    "https://mouniat-vitiscanpro-diagno-api.hf.space"
)



# RETRAINING TRIGGERS (dag_monitoring) ════════════════════════════════

# Minimum number of new images to trigger retraining
# If >= MIN_NEW_IMAGES images are available in new-images/, ingestion is triggered
MIN_NEW_IMAGES = int(os.environ.get("VITISCAN_MIN_NEW_IMAGES", "200"))

# Maximum number of days without retraining
# If last training was more than MAX_DAYS_WITHOUT_RETRAINING days ago,
# ingestion is triggered even if there aren't enough new images
MAX_DAYS_WITHOUT_RETRAINING = int(os.environ.get("VITISCAN_MAX_DAYS", "60"))



# MODEL PERFORMANCE THRESHOLDS (dag_monitoring) ════════════════════════

# Minimum F1 macro score threshold. Below this threshold, an alert is sent
F1_THRESHOLD = float(os.environ.get("VITISCAN_F1_THRESHOLD", "0.90"))

# Minimum recall macro threshold. Below this threshold, an alert is sent
RECALL_THRESHOLD = float(os.environ.get("VITISCAN_RECALL_THRESHOLD", "0.90"))

# Minimum F1 improvement required to deploy a new model
MIN_F1_IMPROVEMENT = float(os.getenv("VITISCAN_MIN_F1_IMPROVEMENT", "0.01"))



# EVIDENTLY — DATA DRIFT DETECTION ═════════════════════════════════════

# Drift detection threshold (0.0 to 1.0)
# If more than this fraction of features show drift, trigger an alert
# Example: 0.3 means if 30% or more features have drifted → alert
DRIFT_THRESHOLD = float(os.getenv("VITISCAN_DRIFT_THRESHOLD", "0.3"))

# Minimum number of images required to run drift detection
# (need enough samples for statistical significance)
MIN_IMAGES_FOR_DRIFT = int(os.getenv("VITISCAN_MIN_IMAGES_FOR_DRIFT", "50"))

# Enable/disable drift detection (useful for debugging)
DRIFT_DETECTION_ENABLED = os.getenv(
    "VITISCAN_DRIFT_DETECTION_ENABLED", "true"
).lower() == "true"

# Features to monitor for drift
# These are extracted from images and compared between reference and current
MONITORED_FEATURES = [
    "brightness",      # Mean pixel value (0-255 normalized to 0-1)
    "contrast",        # Standard deviation of pixels
    "aspect_ratio",    # Width / Height
    "file_size_kb",    # File size in kilobytes
    "red_mean",        # Mean of red channel
    "green_mean",      # Mean of green channel
    "blue_mean",       # Mean of blue channel
]


# DATASET CONFIGURATION ═════════════════════════════════════════════════════════

# Target number of images per class after balancing
# Excess images are archived (never permanently deleted)
TARGET_IMAGES_PER_CLASS = int(os.environ.get("VITISCAN_TARGET_PER_CLASS", "350"))

# Valid classes. WARNING: This list must exactly match the subfolder names in S3
# (new-images/<class_name>/) and the CNN model classes
VALID_CLASSES = (
    "colomerus_vitis",
    "elsinoe_ampelina",
    "erysiphe_necator",
    "guignardia_bidwellii",
    "healthy",
    "phaeomoniella_chlamydospora",
    "plasmopara_viticola",
)

# Valid image extensions
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")



# EC2 CONFIGURATION (dag_retraining - simulation) ═══════════════════════════════

# EC2 instance type for training (GPU)
EC2_INSTANCE_TYPE = os.environ.get("VITISCAN_EC2_INSTANCE_TYPE", "p3.2xlarge")

# EC2 AMI ID (Deep Learning AMI)
EC2_AMI_ID = os.environ.get("VITISCAN_EC2_AMI_ID", "ami-0abcdef1234567890")

# AWS region for EC2 instance
EC2_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-3")


# API ENDPOINTS ═══════════════════════════════════════════════════════════════

API_DIAGNO_URL = os.getenv(
    "VITISCAN_API_DIAGNO_URL",
    "https://mouniat-vitiscanpro-diagno-api.hf.space"
)

# DISPLAY CONFIGURATION SUMMARY (for debugging) ════════════════════════════════

if __name__ == "__main__":
    print("Vitiscan Airflow Configuration")
    print("=" * 50)
    print(f"S3_BUCKET: {S3_BUCKET}")
    print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
    print(f"MIN_NEW_IMAGES: {MIN_NEW_IMAGES}")
    print(f"MAX_DAYS_WITHOUT_RETRAINING: {MAX_DAYS_WITHOUT_RETRAINING}")
    print(f"F1_THRESHOLD: {F1_THRESHOLD}")
    print(f"DRIFT_THRESHOLD: {DRIFT_THRESHOLD}")
    print(f"DRIFT_DETECTION_ENABLED: {DRIFT_DETECTION_ENABLED}")
    print(f"TARGET_IMAGES_PER_CLASS: {TARGET_IMAGES_PER_CLASS}")
