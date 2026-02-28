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
FROM apache/airflow:3.1.3

# Switch to airflow user (non-root) for security
USER airflow

# Copy and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt