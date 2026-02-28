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
"""

import os

# S3 CONFIGURATION ═══════════════════════════════════════════════════

# Main bucket for all Vitiscan data
S3_BUCKET = os.environ.get("VITISCAN_S3_BUCKET", "vitiscanpro-bucket")

# Directory where new labeled images are uploaded
S3_NEW_IMAGES_DIR = "new-images/"

# Directory for the combined dataset (validated and balanced images)
S3_COMBINED_DIR = "datasets/combined/"

# Metadata file for the last training run
S3_METADATA_KEY = "datasets/combined/last_training_metadata.json"


# MLFLOW CONFIGURATION ══════════════════════════════════════════════

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

# API CONFIGURATION ════════════════════════════════════════════════════════════════

# Diagnostic API URL (hosted on HuggingFace Spaces)
API_DIAGNO_URL = os.environ.get(
    "VITISCAN_API_DIAGNO_URL",
    "https://mouniat-vitiscanpro-diagno-api.hf.space"
)

# MONITORING THRESHOLDS (dag_monitoring) ═════════════════════════════════════════

# Minimum number of new images to trigger retraining
# If >= MIN_NEW_IMAGES images are available in new-images/, ingestion is triggered
MIN_NEW_IMAGES = int(os.environ.get("VITISCAN_MIN_NEW_IMAGES", "200"))

# Maximum number of days without retraining
# If last training was more than MAX_DAYS_WITHOUT_RETRAINING days ago,
# ingestion is triggered even if there aren't enough new images
MAX_DAYS_WITHOUT_RETRAINING = int(os.environ.get("VITISCAN_MAX_DAYS", "60"))


# MODEL PERFORMANCE THRESHOLDS (dag_monitoring) ═════════════════════════════════

# Minimum F1 macro score threshold
# Below this threshold, an alert is sent
F1_THRESHOLD = float(os.environ.get("VITISCAN_F1_THRESHOLD", "0.90"))

# Minimum recall macro threshold
# Below this threshold, an alert is sent
RECALL_THRESHOLD = float(os.environ.get("VITISCAN_RECALL_THRESHOLD", "0.90"))


# DATASET BALANCING (dag_data_ingestion) ════════════════════════════════════════

# Target number of images per class after balancing
# Excess images are archived (never permanently deleted)
TARGET_IMAGES_PER_CLASS = int(os.environ.get("VITISCAN_TARGET_PER_CLASS", "350"))


# TRAINING CONFIGURATION (dag_retraining) ═══════════════════════════════════════

# Minimum F1 score improvement required to replace production model
# New model must have F1 >= F1_production + MIN_F1_IMPROVEMENT
MIN_F1_IMPROVEMENT = float(os.environ.get("VITISCAN_MIN_F1_IMPROVEMENT", "0.01"))


# EC2 CONFIGURATION (dag_retraining - simulation) ═══════════════════════════════

# EC2 instance type for training (GPU)
EC2_INSTANCE_TYPE = os.environ.get("VITISCAN_EC2_INSTANCE_TYPE", "p3.2xlarge")

# EC2 AMI ID (Deep Learning AMI)
EC2_AMI_ID = os.environ.get("VITISCAN_EC2_AMI_ID", "ami-0abcdef1234567890")

# AWS region for EC2 instance
EC2_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-3")


# INRAE DISEASE CLASSES ════════════════════════════════════════════════════════

# The 7 grape vine disease classes defined by INRAE
# WARNING: This list must exactly match the subfolder names in S3
# (new-images/<class_name>/) and the CNN model classes
VALID_CLASSES = [
    "colomerus_vitis",              # Grape erineum mite
    "elsinoe_ampelina",             # Grape anthracnose
    "erysiphe_necator",             # Powdery mildew
    "guignardia_bidwellii",         # Black rot
    "healthy",                      # Healthy leaf
    "phaeomoniella_chlamydospora",  # Esca disease
    "plasmopara_viticola",          # Downy mildew
]


# IMAGE VALIDATION ═════════════════════════════════════════════════════════════

# Accepted file extensions for images
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")