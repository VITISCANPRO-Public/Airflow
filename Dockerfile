# ══════════════════════════════════════════════════════════════════════════════
# VITISCAN AIRFLOW — CUSTOM DOCKER IMAGE
# ══════════════════════════════════════════════════════════════════════════════
#
# This Dockerfile extends the official Apache Airflow image with our
# Python dependencies (boto3, mlflow, requests).
#
# Build: docker-compose build
# Run:   docker-compose up -d
#
# ══════════════════════════════════════════════════════════════════════════════

# Base image — must match the version in docker-compose.yaml
FROM apache/airflow:2.10.4-python3.11


# Copy and install Python dependencies
COPY requirements.txt /requirements.txt

# Switch to airflow user (non-root) for security
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt

