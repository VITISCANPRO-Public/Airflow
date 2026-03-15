#!/bin/bash

# ============================================================================
# S3 Initialization Script for Vitiscan Airflow
# ============================================================================
# 
# This script creates the folder structure required on S3 so that
# the Airflow DAGs can run correctly.
#
# Prerequisites :
# - AWS CLI configured with the correct credentials
# - BUCKET_NAME variable defined
# ============================================================================

set -e  # Stop on error

BUCKET_NAME="vitiscanpro-bucket"

echo "========================================================================"
echo "Initializing S3 structure for Vitiscan"
echo "========================================================================"
echo ""
echo "Target bucket : s3://${BUCKET_NAME}"
echo ""

# ============================================================================
# 1. Creating the folder structure
# ============================================================================
echo "Step 1/4 : Creating folder structure..."

# Root folders
aws s3api put-object --bucket ${BUCKET_NAME} --key new-images/
aws s3api put-object --bucket ${BUCKET_NAME} --key datasets/
aws s3api put-object --bucket ${BUCKET_NAME} --key datasets/combined/
aws s3api put-object --bucket ${BUCKET_NAME} --key datasets/archived/
aws s3api put-object --bucket ${BUCKET_NAME} --key monitoring/
aws s3api put-object --bucket ${BUCKET_NAME} --key monitoring/evidently/
aws s3api put-object --bucket ${BUCKET_NAME} --key monitoring/evidently/reports/

# Classes in new-images/ only (to test the DAG later)
for class in colomerus_vitis elsinoe_ampelina erysiphe_necator guignardia_bidwellii healthy phaeomoniella_chlamydospora plasmopara_viticola; do
    aws s3api put-object --bucket ${BUCKET_NAME} --key new-images/${class}/
done


echo "Folder structure created"
echo ""

# ============================================================================
# 2. Verifying the structure
# ============================================================================
echo "Step 2/4 : Verifying structure..."

echo "Root folders :"
aws s3 ls s3://${BUCKET_NAME}/

echo ""
echo "Folders in new-images/ :"
aws s3 ls s3://${BUCKET_NAME}/new-images/

echo ""
echo "Folders in datasets/combined/ :"
aws s3 ls s3://${BUCKET_NAME}/datasets/combined/

echo ""

# ============================================================================
# 3. Instructions for dataset upload
# ============================================================================
echo "Step 3/4 : Dataset upload"
echo ""
echo "IMPORTANT : You must now upload your combined dataset"
echo "            (INRAE + Kaggle) to S3."
echo ""
echo "Recommended command :"
echo ""
echo "  cd Model_CNN"
echo "  aws s3 sync data/combined/ s3://${BUCKET_NAME}/datasets/combined/ \\"
echo "      --exclude '.*' \\"
echo "      --exclude '__pycache__/*'"
echo ""
echo "Press Enter when the upload is complete..."
read

# ============================================================================
# 4. Verifying uploaded dataset
# ============================================================================
echo "Step 4/4 : Verifying uploaded dataset..."
echo ""

for class in colomerus_vitis elsinoe_ampelina erysiphe_necator guignardia_bidwellii healthy phaeomoniella_chlamydospora plasmopara_viticola; do
    count=$(aws s3 ls s3://${BUCKET_NAME}/datasets/combined/train/${class}/ | grep -c ".jpg" || echo "0")
    echo "${class}: ${count} images"
done

echo ""
echo "========================================================================"
echo "S3 initialization complete!"
echo "========================================================================"
echo ""
echo "Next steps :"
echo ""
echo "1. Generate the Evidently reference file :"
echo "   docker-compose exec airflow-scheduler bash"
echo "   cd /opt/airflow/scripts"
echo "   python generate_reference_features.py"
echo ""
echo "2. Upload some test images to new-images/ :"
echo "   aws s3 cp test-images/ s3://${BUCKET_NAME}/new-images/ --recursive"
echo ""
echo "3. Trigger the monitoring DAG in the Airflow UI :"
echo "   http://localhost:8081"
echo ""