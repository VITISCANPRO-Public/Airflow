# Vitiscan — Airflow MLOps Pipeline

Automated MLOps pipeline for the Vitiscan grape leaf disease classification system.
This repository contains three Apache Airflow DAGs that handle the full machine learning lifecycle: data monitoring, dataset preparation, and model retraining with automated deployment.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [DAGs Description](#dags-description)
  - [dag_monitoring](#dag_monitoring)
  - [dag_data_ingestion](#dag_data_ingestion)
  - [dag_retraining](#dag_retraining)
- [How the DAGs Work Together](#how-the-dags-work-together)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Setup & Installation](#setup--installation)
- [Running the DAGs](#running-the-dags)
- [Infrastructure Overview](#infrastructure-overview)
- [Disease Classes](#disease-classes)

---

## Overview

Vitiscan is an AI-powered application that diagnoses grape vine leaf diseases from photographs. A farmer takes a photo of a suspicious leaf and receives an instant diagnosis among 7 disease categories, along with a treatment plan.

This Airflow repository automates the **machine learning lifecycle** — ensuring the model is always trained on the latest labeled data, deployed only when it genuinely improves on the current production model, and continuously monitored for performance degradation.

> **Note:** This pipeline manages data and model lifecycle only.
> Application code quality and deployment is handled separately by GitHub Actions (see the `Diagnostic-API` repository).

---

## Pipeline Architecture

```
Every Monday (scheduled)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   dag_monitoring                    │
│                                                     │
│  Trigger 1: New images ≥ 200 in S3 new-images/?     │
│  Trigger 2: Last training > 60 days ago?            │
│  Performance check: F1 or Recall < 0.90?            │
└──────────────┬──────────────────────┬───────────────┘
               │ YES (data trigger)   │ NO (performance check only)
               ▼                      ▼
┌──────────────────────┐     ┌─────────────────────┐
│  dag_data_ingestion  │     │   Send alert /      │
│                      │     │   No action needed  │
│  1. List new images  │     └─────────────────────┘
│  2. Validate classes │
│  3. Integrate to S3  │
│  4. Balance dataset  │
│  5. Update metadata  │
└──────────┬───────────┘
           │ triggers
           ▼
┌──────────────────────────────────────────────────────┐
│                   dag_retraining                     │
│                                                      │
│  1. Provision EC2 GPU instance                       │
│  2. Train ResNet18 on new dataset                    │
│  3. Register model in MLflow Model Registry          │
│  4. Compare new vs production model (F1 score)       │
│     ├── New model better → deploy to pre-production  │
│     │       ├── Tests pass → promote to Production   │
│     │       └── Tests fail → rollback                │
│     └── No improvement → keep current model          │
│  5. Terminate EC2 instance (always)                  │
└──────────────────────────────────────────────────────┘
```

---

## DAGs Description

### dag_monitoring

**Schedule:** Every week (`@weekly`)  
**Trigger:** Automatic — no external trigger required

This is the entry point of the pipeline. It acts as a watchdog that runs weekly and decides whether action is needed.

It checks two independent conditions:

**Retraining triggers** — if either condition is met, `dag_data_ingestion` is triggered:
- **Volume trigger:** ≥ 200 new labeled images are available in `s3://vitiscanpro-bucket/new-images/`
- **Delay trigger:** The last model training happened more than 60 days ago

**Performance check** — if no retraining is needed, the model metrics are inspected:
- Fetches the Production model metrics from MLflow Model Registry
- Sends an alert if `test_f1_macro < 0.90` or `test_recall_macro < 0.90`

| Task | Type | Description |
|---|---|---|
| `check_retraining_triggers` | BranchPythonOperator | Checks volume and delay triggers |
| `trigger_ingestion` | TriggerDagRunOperator | Triggers dag_data_ingestion |
| `check_model_performance` | BranchPythonOperator | Checks F1 and recall thresholds |
| `send_alert` | PythonOperator | Sends performance alert |
| `no_action` | EmptyOperator | Terminal state when all metrics are healthy |

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
- Images must be in a subfolder matching one of the 7 INRAE class names
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
├── dags/
│   ├── config.py                # Centralized configuration (env vars + defaults)
│   ├── dag_monitoring.py        # Weekly watchdog DAG
│   ├── dag_data_ingestion.py    # Dataset preparation DAG
│   └── dag_retraining.py        # Model training and deployment DAG
├── config/
│   └── airflow.cfg              # Airflow configuration overrides
├── logs/                        # Airflow task logs (auto-generated)
├── plugins/                     # Custom Airflow plugins (empty)
├── .env.template                # Environment variables template
├── docker-compose.yaml          # Docker Compose configuration
├── Dockerfile                   # Custom Airflow image with dependencies
├── requirements.txt             # Python dependencies
└── README.md
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
docker-compose up -d
```

**4. Access Airflow UI**
- Open http://localhost:8081
- Login with credentials from `.env` (default: airflow/airflow)
- Unpause `dag_monitoring` to start the pipeline

**5. Check service status**
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
```

**5. Start Airflow**
```bash
airflow scheduler &
airflow api-server --port 8080
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

---

## Infrastructure Overview

| Component | Tool | Purpose |
|---|---|---|
| Pipeline orchestration | Apache Airflow 3.1.3 | Schedules and runs the 3 DAGs |
| Data storage | AWS S3 (`vitiscanpro-bucket`) | Stores images and dataset metadata |
| Model training | AWS EC2 (`p3.2xlarge`) | GPU instance for ResNet18 training (simulated) |
| Experiment tracking | MLflow (HuggingFace Spaces) | Logs metrics and manages model versions |
| Model registry | MLflow Model Registry | Manages model stages (Staging/Production) |
| Model serving | HuggingFace Spaces | Hosts the Diagnostic API |
| CI/CD (code) | GitHub Actions | Tests and deploys API code on push |

**S3 bucket structure:**
```
vitiscanpro-bucket/
├── new-images/
│   └── <class_name>/          ← new labeled images uploaded by experts
├── datasets/
│   ├── combined/
│   │   ├── <class_name>/      ← current training dataset
│   │   └── last_training_metadata.json
│   └── archived/
│       └── <class_name>/      ← excess images (never deleted)
└── mlflow-artifacts/          ← model weights and artifacts
```

---

## Disease Classes

The pipeline handles exactly **7 INRAE grape disease classes**:

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

| Config | Environment Variable | Default |
|---|---|---|
| S3 bucket | `VITISCAN_S3_BUCKET` | `vitiscanpro-bucket` |
| Min new images trigger | `VITISCAN_MIN_NEW_IMAGES` | `200` |
| Max days without training | `VITISCAN_MAX_DAYS` | `60` |
| F1 threshold | `VITISCAN_F1_THRESHOLD` | `0.90` |
| Recall threshold | `VITISCAN_RECALL_THRESHOLD` | `0.90` |
| Images per class target | `VITISCAN_TARGET_PER_CLASS` | `350` |
| Min F1 improvement | `VITISCAN_MIN_F1_IMPROVEMENT` | `0.01` |

See `dags/config.py` for the complete list.

## Author

**Mounia Tonazzini** — Agronomist Engineer & Data Scientist and Data Engineer

- HuggingFace: [huggingface.co/MouniaT](https://huggingface.co/MouniaT)
- LinkedIn: [www.linkedin.com/in/mounia-tonazzini](www.linkedin.com/in/mounia-tonazzini)
- GitHub: [github/Mounia-Agronomist-Datascientist](https://github.com/Mounia-Agronomist-Datascientist)
- Email : mounia.tonazzini@gmail.com