# Vitiscan — Airflow MLOps Pipeline

Automated MLOps pipeline for the Vitiscan grape leaf disease classification system.
This repository contains three Apache Airflow DAGs that handle the full machine learning lifecycle: data monitoring, dataset preparation, and model retraining with automated deployment.

[![Airflow](https://img.shields.io/badge/Airflow-3.1.3-red)](https://airflow.apache.org)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.33-teal)](https://evidentlyai.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [DAGs Description](#dags-description)
  - [dag_monitoring](#dag_monitoring)
  - [dag_data_ingestion](#dag_data_ingestion)
  - [dag_retraining](#dag_retraining)
- [Data Drift Detection](#data-drift-detection)
  - [Why Drift Detection Matters](#why-drift-detection-matters)
  - [How It Works](#how-it-works)
  - [Extracted Features](#extracted-features)
  - [Generated Reports](#generated-reports)
  - [Interpreting Results](#interpreting-results)
- [How the DAGs Work Together](#how-the-dags-work-together)
- [Project Structure](#project-structure)
- [CI/CD Pipeline](#cicd-pipeline)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Setup & Installation](#setup--installation)
- [Running the DAGs](#running-the-dags)
- [Infrastructure Overview](#infrastructure-overview)
- [Disease Classes](#disease-classes)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

Vitiscan is an AI-powered application that diagnoses grape vine leaf diseases from photographs. A farmer takes a photo of a suspicious leaf and receives an instant diagnosis among 7 disease categories, along with a treatment plan.

This Airflow repository automates the **machine learning lifecycle** — ensuring the model is always trained on the latest labeled data, deployed only when it genuinely improves on the current production model, and continuously monitored for performance degradation.

> **Note:** This pipeline manages data and model lifecycle only.
> Application code quality and deployment is handled separately by GitHub Actions (see the `Diagnostic-API` repository).

### Key Features

| Feature | Tool | Description |
|---------|------|-------------|
| **Retraining triggers** | Airflow | Automatic retraining when ≥200 new images or 60+ days elapsed |
| **Data drift detection** | Evidently | Statistical comparison of new images vs training dataset |
| **Performance monitoring** | MLflow | Alert when F1 or Recall drops below 0.90 |
| **Safe deployment** | MLflow Registry | Pre-production testing before promoting to Production |

---

## Pipeline Architecture

```
Every Monday (scheduled)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            dag_monitoring                                │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  RETRAINING TRIGGERS                                            │     │
│  │                                                                 │     │
│  │  Trigger 1: New images ≥ 200 in S3 new-images/?                 │     │
│  │  Trigger 2: Last training > 60 days ago?                        │     │
│  └──────────────────────────┬──────────────────────────────────────┘     │
│                             │                                            │
│              ┌──────────────┴──────────────┐                             │
│              │ YES                         │ NO                          │
│              ▼                             ▼                             │
│     trigger_ingestion              ┌───────────────────────────────┐     │
│              │                     │  DATA DRIFT DETECTION         │     │
│              │                     │  (Evidently)                  │     │
│              │                     │                               │     │
│              │                     │  1. Extract features from     │     │
│              │                     │     new images (brightness,   │     │
│              │                     │     contrast, colors, etc.)   │     │
│              │                     │  2. Compare vs reference      │     │
│              │                     │     (training dataset)        │     │
│              │                     │  3. Statistical tests         │     │
│              │                     └───────────────┬───────────────┘     │
│              │                                     │                     │
│              │                     ┌───────────────┴───────────────┐     │
│              │                     │ Drift > 30%?                  │     │
│              │                     ▼                               ▼     │
│              │              send_drift_alert          check_performance  │
│              │                                              │            │
│              │                                     ┌────────┴────────┐   │
│              │                                     ▼                 ▼   │
│              │                              send_perf_alert     no_action│
└──────────────┼───────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          dag_data_ingestion                              │
│                                                                          │
│  1. List new images         4. Balance dataset (≤350 per class)          │
│  2. Validate classes        5. Update metadata                           │
│  3. Integrate to S3         6. Trigger retraining                        │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            dag_retraining                                │
│                                                                          │
│  1. Provision EC2 GPU instance                                           │
│  2. Train ResNet18 on new dataset                                        │
│  3. Register model in MLflow Model Registry                              │
│  4. Compare new vs production model (F1 score)                           │
│     ├── New model better → deploy to pre-production                      │
│     │       ├── Tests pass → promote to Production                       │
│     │       └── Tests fail → rollback                                    │
│     └── No improvement → keep current model                              │
│  5. Terminate EC2 instance (always)                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## DAGs Description

### dag_monitoring

**Schedule:** Every week (`@weekly`)  
**Trigger:** Automatic — no external trigger required  
**Tags:** `monitoring`, `evidently`

This is the entry point of the pipeline. It acts as a watchdog that runs weekly and decides whether action is needed.

**Three independent checks are performed:**

| Check | Condition | Action |
|-------|-----------|--------|
| **Volume trigger** | ≥ 200 new images in `new-images/` | → Trigger `dag_data_ingestion` |
| **Delay trigger** | Last training > 60 days ago | → Trigger `dag_data_ingestion` |
| **Data drift** | > 30% of features drifted (Evidently) | → Send drift alert |
| **Performance** | F1 < 0.90 or Recall < 0.90 | → Send performance alert |

| Task | Type | Description |
|---|---|---|
| `check_retraining_triggers` | BranchPythonOperator | Checks volume and delay triggers |
| `trigger_ingestion` | TriggerDagRunOperator | Triggers dag_data_ingestion |
| `check_data_drift` | BranchPythonOperator | Runs Evidently drift analysis |
| `send_drift_alert` | PythonOperator | Sends drift alert with report link |
| `check_model_performance` | BranchPythonOperator | Checks F1 and recall thresholds |
| `send_perf_alert` | PythonOperator | Sends performance alert |
| `no_action` | EmptyOperator | Terminal state when all metrics are healthy |

**Task graph:**
```
check_retraining_triggers
        │
        ├─────────────────────┐
        ▼                     ▼
trigger_ingestion      check_data_drift
                              │
                       ┌──────┴──────┐
                       ▼             ▼
               send_drift_alert   check_model_performance
                                        │
                                 ┌──────┴──────┐
                                 ▼             ▼
                          send_perf_alert   no_action
```

---

### dag_data_ingestion

**Schedule:** None — triggered by `dag_monitoring` only  
**Trigger:** `TriggerDagRunOperator` from `dag_monitoring`

Responsible for preparing the training dataset from newly labeled images. It ensures that only valid, well-structured images enter the training pipeline and that the dataset remains balanced across all 7 disease classes.

| Task | Type | Description |
|---|---|---|
| `list_new_images` | PythonOperator | Lists all images in `new-images/` on S3 (with pagination) |
| `validate_images` | PythonOperator | Checks class membership and file extensions |
| `integrate_images` | PythonOperator | Copies valid images to `datasets/combined/` |
| `balance_dataset` | PythonOperator | Archives excess images to maintain ≤ 350 per class |
| `update_metadata` | PythonOperator | Writes updated metadata to `last_training_metadata.json` |
| `trigger_retraining` | TriggerDagRunOperator | Triggers dag_retraining |

**Validation rules:**
- Images must be in a subfolder matching one of the 7 class names
- Accepted formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- Images in unknown subfolders are rejected and logged

**Balancing strategy:**
- Target: **350 images per class**
- Excess images are moved to `datasets/archived/<class>/` — never permanently deleted
- Classes with fewer than 350 images trigger a warning; training compensates via weighted sampling

---

### dag_retraining

**Schedule:** None — triggered by `dag_data_ingestion` only  
**Trigger:** `TriggerDagRunOperator` from `dag_data_ingestion`

Manages the complete model retraining and deployment lifecycle using MLflow Model Registry for version management.

| Task | Type | Description |
|---|---|---|
| `provision_ec2` | PythonOperator | Provisions a GPU instance for training (simulated) |
| `train_model` | PythonOperator | Trains ResNet18 and registers model in MLflow Registry |
| `evaluate_and_compare` | BranchPythonOperator | Compares new model vs Production model in Registry |
| `deploy_to_preprod` | PythonOperator | Transitions model to "Staging" stage |
| `run_preprod_tests` | BranchPythonOperator | Runs automated tests against pre-prod API |
| `deploy_to_prod` | PythonOperator | Promotes model to "Production" stage |
| `rollback` | PythonOperator | Removes model from Staging if tests fail |
| `keep_current_model` | EmptyOperator | Terminal state when new model does not improve |
| `terminate_ec2` | PythonOperator | Terminates EC2 instance (always runs) |

**MLflow Model Registry stages:**
```
None → Staging → Production
              ↘ Archived (previous production models)
```

**Deployment decision logic:**
```
New model F1 ≥ Current production F1 + 0.01 ?
    YES → deploy to Staging → run tests
              Tests pass? YES → promote to Production ✓
              Tests pass? NO  → rollback (remove from Staging)
    NO  → keep current model (no deployment)
```

The `terminate_ec2` task uses `trigger_rule="all_done"` — it always runs regardless of upstream task failures, preventing unnecessary cloud costs.

---

## Data Drift Detection

### Why Drift Detection Matters

In agricultural applications, the characteristics of incoming images can change over time due to:

| Cause | Example | Impact on Model |
|-------|---------|-----------------|
| **Equipment changes** | New smartphone with different camera | Color calibration differs from training data |
| **Seasonal variations** | Summer vs autumn lighting | Brightness/contrast patterns shift |
| **Geographic expansion** | Images from new regions | Different soil/climate affects leaf appearance |
| **User behavior** | Different photo angles or distances | Aspect ratios and compositions change |

If these changes are significant, the model may perform poorly on the new data distribution — even if test metrics looked good at training time. **Drift detection acts as an early warning system** before model performance visibly degrades.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ONE-TIME SETUP                                                         │
│                                                                         │
│  scripts/generate_reference_features.py                                 │
│       │                                                                 │
│       ▼                                                                 │
│  Extract features from ALL training images (~2450 images)               │
│       │                                                                 │
│       ▼                                                                 │
│  Save to S3: monitoring/reference_features.csv                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  WEEKLY ANALYSIS (dag_monitoring)                                       │
│                                                                         │
│  1. Download reference_features.csv from S3                             │
│       │                                                                 │
│       ▼                                                                 │
│  2. Extract features from NEW images in new-images/                     │
│       │                                                                 │
│       ▼                                                                 │
│  3. Evidently compares distributions using statistical tests            │
│     (Kolmogorov-Smirnov for numerical, chi-squared for categorical)     │
│       │                                                                 │
│       ▼                                                                 │
│  4. Decision: drift_share > DRIFT_THRESHOLD ?                           │
│       │                                                                 │
│       ├── YES → send_drift_alert (with S3 report link)                  │
│       └── NO  → check_model_performance (continue pipeline)             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Extracted Features

Evidently analyzes **8 numerical features** extracted from each image:

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `brightness` | Average pixel intensity (0-1) | Detects lighting condition changes |
| `contrast` | Standard deviation of pixel values | Identifies image quality variations |
| `aspect_ratio` | Width / Height | Reveals camera or crop changes |
| `red_mean` | Average red channel value | Detects color calibration shifts |
| `green_mean` | Average green channel value | Important for leaf color analysis |
| `blue_mean` | Average blue channel value | Reveals white balance changes |
| `saturation` | Color intensity (HSV) | Identifies washed-out or oversaturated images |
| `file_size_kb` | Compressed file size | Proxy for image complexity/resolution |

**Feature extraction code** is located in `dags/utils/drift_detection.py`.

### Generated Reports

Each drift analysis produces two outputs, stored on S3:

| File | Location | Purpose |
|------|----------|---------|
| **HTML Report** | `s3://vitiscanpro-bucket/monitoring/evidently/reports/drift_report_YYYYMMDD_HHMMSS.html` | Interactive visualization for human review |
| **JSON Results** | `s3://vitiscanpro-bucket/monitoring/evidently/reports/drift_results_YYYYMMDD_HHMMSS.json` | Machine-readable output for pipeline decisions |

**Example JSON output:**
```json
{
  "dataset_drift": true,
  "drift_share": 0.43,
  "number_of_columns": 8,
  "number_of_drifted_columns": 3,
  "drifted_features": ["brightness", "contrast", "blue_mean"],
  "feature_drift_scores": {
    "brightness": {"drift_detected": true, "drift_score": 0.002},
    "contrast": {"drift_detected": true, "drift_score": 0.015},
    "aspect_ratio": {"drift_detected": false, "drift_score": 0.234},
    "red_mean": {"drift_detected": false, "drift_score": 0.456},
    "green_mean": {"drift_detected": false, "drift_score": 0.321},
    "blue_mean": {"drift_detected": true, "drift_score": 0.008},
    "saturation": {"drift_detected": false, "drift_score": 0.187},
    "file_size_kb": {"drift_detected": false, "drift_score": 0.543}
  },
  "reference_size": 2450,
  "current_size": 127,
  "timestamp": "2024-03-15T10:30:00Z"
}
```

### Interpreting Results

**In the HTML report:**

| Indicator | Meaning |
|-----------|---------|
| 🟢 Green | No significant drift detected |
| 🔴 Red | Drift detected (p-value < 0.05) |
| **Drift Share** | Percentage of features showing drift (e.g., 0.43 = 43%) |

**Recommended actions when drift is detected:**

| Drift Share | Severity | Recommended Action |
|-------------|----------|-------------------|
| < 20% | Low | Monitor, likely normal variation |
| 20-40% | Medium | Investigate source, check recent uploads |
| > 40% | High | Pause ingestion, investigate before retraining |

**Common causes and solutions:**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `brightness` + `contrast` drift | Lighting/weather change | Normal seasonal variation — monitor |
| `blue_mean` drift only | White balance issue | Check camera settings on upload app |
| `aspect_ratio` drift | New device type | Update preprocessing if needed |
| `file_size_kb` drift | Compression change | Check upload pipeline settings |
| All features drift | Major equipment change | Consider updating reference dataset |

---

## How the DAGs Work Together

The three DAGs form a **cascade pipeline** where each DAG is responsible for one distinct stage and triggers the next one:

```
dag_monitoring  ──triggers──▶  dag_data_ingestion  ──triggers──▶  dag_retraining
  (weekly)                        (on demand)                        (on demand)
```

**Key design principles:**
- Each DAG has `schedule=None` except `dag_monitoring`, meaning they never run autonomously
- DAGs communicate via XCom for intra-DAG task data sharing
- DAGs communicate via S3 metadata (`last_training_metadata.json`) for inter-DAG information
- Model versioning is managed via MLflow Model Registry stages
- `terminate_ec2` always runs (`trigger_rule="all_done"`) to prevent cloud cost leakage

**Separation from GitHub Actions:**
This Airflow pipeline manages the **data and model lifecycle**. It does not interact with GitHub Actions, which independently handles **code quality** (unit tests, integration tests, and API deployment on code push). Their only shared point is HuggingFace Spaces, where both systems deploy to the same Diagnostic API — GitHub Actions deploys the code, Airflow deploys the model.

---

## Project Structure

```
vitiscan-airflow/
├── .github/
│   └── workflows/
│       └── ci.yml                       # CI/CD pipeline
├── dags/
│   ├── config.py                         # Centralized configuration
│   ├── dag_monitoring.py                 # Weekly watchdog DAG (+ drift detection)
│   ├── dag_data_ingestion.py             # Dataset preparation DAG
│   ├── dag_retraining.py                 # Model training and deployment DAG
│   └── utils/                            # Utility modules
│       ├── __init__.py
│       └── drift_detection.py            # Evidently integration module
├── scripts/
│   └── generate_reference_features.py    # One-time reference dataset generator
├── config/
│   └── airflow.cfg                       # Airflow configuration overrides
├── logs/                                 # Airflow task logs (auto-generated)
├── plugins/                              # Custom Airflow plugins (empty)
├── .env.template                         # Environment variables template
├── docker-compose.yaml                   # Docker Compose configuration
├── Dockerfile                            # Custom Airflow image with dependencies
├── requirements.txt                      # Python dependencies
└── README.md
```

**New files for drift detection:**

| File | Purpose |
|------|---------|
| `dags/utils/__init__.py` | Makes `utils` a Python package |
| `dags/utils/drift_detection.py` | Contains `extract_features()`, `run_drift_analysis()`, `upload_report()` |
| `scripts/generate_reference_features.py` | One-time script to create reference dataset from training images |

---

## CI/CD Pipeline

This repository includes a GitHub Actions CI/CD pipeline that runs automatically on every push and pull request. It ensures code quality before changes reach production.

### Pipeline Overview

```
git push
    │
    ▼
┌─────────────────────────────────────┐
│  Job 1: Lint                        │
│                                     │
│  • Checks Python syntax errors      │
│  • Detects undefined variables      │
│  • Finds unused imports             │
│  • Enforces code style              │
│                                     │
│  Tool: ruff (fast Python linter)    │
└──────────────┬──────────────────────┘
               │ Pass
               ▼
┌─────────────────────────────────────┐
│  Job 2: Validate DAGs               │
│                                     │
│  • Installs Airflow + Evidently     │
│  • Parses all DAG files             │
│  • Detects import errors            │
│  • Validates DAG configuration      │
│                                     │
│  Command: airflow dags list         │
└──────────────┬──────────────────────┘
               │ Pass
               ▼
         Merge allowed
```

### What is Linting?

**Linting** is automatic code analysis that detects errors without executing the code:

| Error Type | Example | Without Linting |
|------------|---------|-----------------|
| Syntax errors | `if x = 5:` instead of `if x == 5:` | Crash at runtime |
| Undefined variables | `print(user_name)` when variable is `username` | Crash at runtime |
| Unused imports | `import pandas` but never used | Bloated code |
| Style issues | 500-character lines, inconsistent indentation | Hard to read |

Think of it as a **spell-checker for code** — it catches typos and mistakes before they cause problems.

### Viewing CI Results

After pushing code, go to your GitHub repository → **Actions** tab:

**Success:**
```
✓ CI Pipeline
   ├── ✓ Lint Python Code (32s)
   └── ✓ Validate Airflow DAGs (1m 12s)
```

**Failure:**
```
✗ CI Pipeline
   ├── ✗ Lint Python Code (8s)
   │      └── Error: dags/dag_monitoring.py:42 — undefined name 'mlflwo'
   └── ○ Validate Airflow DAGs (skipped)
```

### Running Lint Locally

Before pushing, you can run the linter locally to catch errors early:

```bash
# Install ruff
pip install ruff

# Run linter on DAGs
ruff check dags/

# Auto-fix simple issues
ruff check dags/ --fix
```

---

## Prerequisites

- **Docker** and **Docker Compose** (recommended)
- Python 3.11+ (for local development without Docker)
- Apache Airflow 3.1.3
- AWS account with access to S3 and EC2
- MLflow tracking server running on HuggingFace Spaces
- HuggingFace account with a Space for the Diagnostic API

---

## Environment Variables

Copy `.env.template` to `.env` and fill in your values:

```bash
cp .env.template .env
```

### Core Variables

| Variable | Description | Example |
|---|---|---|
| `AIRFLOW__CORE__FERNET_KEY` | Encryption key for sensitive data | (generate with Python) |
| `AWS_ACCESS_KEY_ID` | AWS credentials for S3 access | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `...` |
| `AWS_DEFAULT_REGION` | AWS region | `eu-west-3` |
| `VITISCAN_S3_BUCKET` | S3 bucket name | `vitiscanpro-bucket` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `https://mouniat-vitiscanpro-hf.hf.space` |
| `VITISCAN_API_DIAGNO_URL` | Diagnostic API URL | `https://mouniat-vitiscanpro-diagno-api.hf.space` |
| `HF_TOKEN` | HuggingFace API token for deployment | `hf_...` |

### Drift Detection Variables

| Variable | Default | Description |
|---|---|---|
| `VITISCAN_DRIFT_DETECTION_ENABLED` | `true` | Enable/disable drift detection |
| `VITISCAN_DRIFT_THRESHOLD` | `0.3` | Alert threshold (0.3 = 30% of features) |
| `VITISCAN_MIN_IMAGES_FOR_DRIFT` | `50` | Minimum images required for analysis |

**Generate a Fernet key:**
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## Setup & Installation

### Option 1: Docker (Recommended)

**1. Clone the repository**
```bash
git clone https://github.com/mouniat/vitiscan-airflow.git
cd vitiscan-airflow
```

**2. Create your environment file**
```bash
cp .env.template .env
# Edit .env with your values (especially FERNET_KEY and AWS credentials)
```

**3. Start Airflow**
```bash
docker-compose build
docker-compose up -d
```

**4. Initialize drift detection reference (one-time setup)**
```bash
# Enter the scheduler container
docker-compose exec airflow-scheduler bash

# Verify Evidently is installed
python -c "import evidently; print(f'Evidently version: {evidently.__version__}')"

# Generate reference features from training dataset
cd /opt/airflow/scripts
python generate_reference_features.py

# Expected output:
# ======================================================================
# VITISCAN — Reference Features Generator
# ======================================================================
# Listing training images...
# - Found 2450 images
# - Extracting features...
# - Uploading to s3://vitiscanpro-bucket/monitoring/reference_features.csv...
# - Successfully uploaded reference features!

# Exit container
exit
```

**5. Access Airflow UI**
- Open http://localhost:8081
- Login with credentials from `.env` (default: airflow/airflow)
- Unpause `dag_monitoring` to start the pipeline

**6. Check service status**
```bash
docker-compose ps
docker-compose logs -f airflow-scheduler
```

### Option 2: Local Installation (Development)

**1. Clone and create virtual environment**
```bash
git clone https://github.com/mouniat/vitiscan-airflow.git
cd vitiscan-airflow
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install apache-airflow==3.1.3
pip install -r requirements.txt
```

**3. Initialize Airflow**
```bash
export AIRFLOW_HOME=$(pwd)
airflow db init
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@vitiscan.com
```

**4. Set environment variables**
```bash
export AIRFLOW__CORE__FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=eu-west-3
export MLFLOW_TRACKING_URI=https://mouniat-vitiscanpro-hf.hf.space
export VITISCAN_DRIFT_DETECTION_ENABLED=true
```

**5. Generate reference features**
```bash
python scripts/generate_reference_features.py
```

**6. Start Airflow**
```bash
airflow scheduler &
airflow webserver --port 8080
```

---

## Running the DAGs

**Automatic execution:**
Once `dag_monitoring` is unpaused in the Airflow UI, it runs automatically every week. The downstream DAGs are triggered automatically when conditions are met.

**Manual trigger (for testing):**
```bash
# Trigger the full pipeline manually
airflow dags trigger dag_monitoring

# Trigger ingestion directly (skipping monitoring checks)
airflow dags trigger dag_data_ingestion

# Trigger retraining directly (skipping ingestion)
airflow dags trigger dag_retraining
```

**Checking DAG status:**
```bash
airflow dags list
airflow dags list-runs --dag-id dag_monitoring
```

**View drift reports:**
```bash
# List generated reports
aws s3 ls s3://vitiscanpro-bucket/monitoring/evidently/reports/

# Download latest HTML report for viewing
aws s3 cp s3://vitiscanpro-bucket/monitoring/evidently/reports/drift_report_YYYYMMDD_HHMMSS.html ./
open drift_report_*.html  # macOS
```

---

## Infrastructure Overview

| Component | Tool | Purpose |
|---|---|---|
| Pipeline orchestration | Apache Airflow 3.1.3 | Schedules and runs the 3 DAGs |
| **Data drift detection** | **Evidently 0.4.33** | **Statistical analysis of image feature distributions** |
| Data storage | AWS S3 (`vitiscanpro-bucket`) | Stores images, metadata, and drift reports |
| Model training | AWS EC2 (`p3.2xlarge`) | GPU instance for ResNet18 training (simulated) |
| Experiment tracking | MLflow (HuggingFace Spaces) | Logs metrics and manages model versions |
| Model registry | MLflow Model Registry | Manages model stages (Staging/Production) |
| Model serving | HuggingFace Spaces | Hosts the Diagnostic API |
| CI/CD (code) | GitHub Actions | Lints code and validates DAGs on push |

**S3 bucket structure:**
```
vitiscanpro-bucket/
├── new-images/
│   └── <class_name>/                     ← new labeled images uploaded by experts
├── datasets/
│   ├── combined/
│   │   ├── <class_name>/                 ← current training dataset
│   │   └── last_training_metadata.json
│   └── archived/
│       └── <class_name>/                 ← excess images (never deleted)
├── monitoring/
│   ├── reference_features.csv            ← baseline features from training set
│   └── evidently/
│       └── reports/
│           ├── drift_report_YYYYMMDD_HHMMSS.html
│           └── drift_results_YYYYMMDD_HHMMSS.json
└── mlflow-artifacts/                     ← model weights and artifacts
```

---

## Disease Classes

The pipeline handles exactly ** 7 classes**:

| Class (scientific name) | Common name |
|---|---|
| `colomerus_vitis` | Grape erineum mite |
| `elsinoe_ampelina` | Grape anthracnose |
| `erysiphe_necator` | Powdery mildew |
| `guignardia_bidwellii` | Black rot |
| `healthy` | Healthy leaf |
| `phaeomoniella_chlamydospora` | Esca disease |
| `plasmopara_viticola` | Downy mildew |

Images uploaded to `new-images/` must be placed in a subfolder named exactly after one of these classes. Any other subfolder name will be rejected during the validation step.

---

## Configuration

All pipeline configuration is centralized in `dags/config.py`. Values can be overridden via environment variables:

### Retraining Triggers

| Config | Environment Variable | Default |
|---|---|---|
| S3 bucket | `VITISCAN_S3_BUCKET` | `vitiscanpro-bucket` |
| Min new images trigger | `VITISCAN_MIN_NEW_IMAGES` | `200` |
| Max days without training | `VITISCAN_MAX_DAYS` | `60` |
| F1 threshold | `VITISCAN_F1_THRESHOLD` | `0.90` |
| Recall threshold | `VITISCAN_RECALL_THRESHOLD` | `0.90` |
| Images per class target | `VITISCAN_TARGET_PER_CLASS` | `350` |
| Min F1 improvement | `VITISCAN_MIN_F1_IMPROVEMENT` | `0.01` |

### Drift Detection

| Config | Environment Variable | Default | Description |
|---|---|---|---|
| Enabled | `VITISCAN_DRIFT_DETECTION_ENABLED` | `true` | Enable/disable drift detection |
| Threshold | `VITISCAN_DRIFT_THRESHOLD` | `0.3` | Alert if > 30% of features drift |
| Min images | `VITISCAN_MIN_IMAGES_FOR_DRIFT` | `50` | Skip analysis if fewer images |

**Adjusting drift sensitivity:**

```bash
# More sensitive (alert on 20% drift) — use for critical deployments
VITISCAN_DRIFT_THRESHOLD=0.2

# Less sensitive (alert on 50% drift) — use if too many false positives
VITISCAN_DRIFT_THRESHOLD=0.5

# Disable drift detection entirely
VITISCAN_DRIFT_DETECTION_ENABLED=false
```

See `dags/config.py` for the complete list of configuration options.

---

## Troubleshooting

### Drift Detection Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `No reference features found` | `reference_features.csv` missing on S3 | Run `scripts/generate_reference_features.py` |
| `Not enough images for drift analysis` | < 50 images in `new-images/` | Normal — drift detection skipped |
| `ModuleNotFoundError: evidently` | Docker image not rebuilt | `docker-compose build --no-cache` |
| Reports not generated | Feature extraction error | Check logs: `docker-compose logs airflow-scheduler \| grep drift` |

### General Airflow Issues

| Error | Cause | Solution |
|-------|-------|----------|
| DAG not visible in UI | Syntax error in DAG file | Check `docker-compose logs airflow-scheduler` |
| Task stuck in "queued" | Scheduler not running | `docker-compose restart airflow-scheduler` |
| S3 access denied | Wrong AWS credentials | Check `.env` variables |
| MLflow connection error | Wrong URI or server down | Verify `MLFLOW_TRACKING_URI` |

### Viewing Logs

```bash
# All scheduler logs
docker-compose logs -f airflow-scheduler

# Filter for drift-related logs
docker-compose logs airflow-scheduler | grep -i drift

# Filter for specific DAG
docker-compose logs airflow-scheduler | grep dag_monitoring
```

---

## Author

**Mounia Tonazzini** — Agronomist Engineer & Data Scientist and Data Engineer

- HuggingFace: [huggingface.co/MouniaT](https://huggingface.co/MouniaT)
- LinkedIn: [www.linkedin.com/in/mounia-tonazzini](https://www.linkedin.com/in/mounia-tonazzini)
- GitHub: [github/Mounia-Agronomist-Datascientist](https://github.com/Mounia-Agronomist-Datascientist)
- Email: mounia.tonazzini@gmail.com