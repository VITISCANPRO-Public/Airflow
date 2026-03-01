#!/usr/bin/env python3
"""
generate_reference_features.py — Initialize reference dataset for drift detection.

THIS SCRIPT MUST BE RUN ONCE before enabling drift detection in the pipeline.

It extracts statistical features from all images in the training dataset
(datasets/combined/) and saves them as a CSV file on S3. This CSV becomes
the "reference" against which all new images will be compared.

Usage:
    # From the Airflow directory
    python scripts/generate_reference_features.py

    # Or with custom bucket
    python scripts/generate_reference_features.py --bucket my-bucket

    # Dry run (don't upload to S3)
    python scripts/generate_reference_features.py --dry-run

What it does:
    1. Lists all images in s3://bucket/datasets/combined/
    2. Downloads each image and extracts features (brightness, contrast, etc.)
    3. Saves features as CSV to s3://bucket/monitoring/reference_features.csv

When to re-run:
    - After significant changes to the training dataset
    - After rebalancing classes
    - If drift detection gives too many false positives

Example output (reference_features.csv):
    image_path,brightness,contrast,aspect_ratio,file_size_kb,red_mean,green_mean,blue_mean,class
    datasets/combined/healthy/img_001.jpg,0.45,0.21,1.333,45.2,0.42,0.48,0.44,healthy
    datasets/combined/plasmopara/img_002.jpg,0.38,0.19,1.0,52.1,0.35,0.41,0.38,plasmopara
    ...
"""

import argparse
import io
import os
import sys
from datetime import datetime

import boto3
import pandas as pd
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dags"))

from config import (
    S3_BUCKET,
    S3_COMBINED_DIR,
    S3_REFERENCE_FEATURES_KEY,
    VALID_EXTENSIONS,
    VALID_CLASSES,
)


def extract_features_from_s3_image(s3_client, bucket: str, key: str) -> dict:
    """
    Download an image from S3 and extract its features.
    
    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        Dictionary of features, or None if extraction failed
    """
    try:
        # Download image
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response["Body"].read()
        file_size_kb = len(image_data) / 1024
        
        # Open with PIL
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Convert to numpy array (normalized to 0-1)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Extract class from path
        # Expected: datasets/combined/<class_name>/<filename>
        parts = key.split("/")
        class_name = parts[-2] if len(parts) >= 2 else "unknown"
        
        features = {
            "image_path": key,
            "brightness": float(np.mean(img_array)),
            "contrast": float(np.std(img_array)),
            "width": img.width,
            "height": img.height,
            "aspect_ratio": round(img.width / img.height, 3),
            "file_size_kb": round(file_size_kb, 2),
            "red_mean": float(np.mean(img_array[:, :, 0])),
            "green_mean": float(np.mean(img_array[:, :, 1])),
            "blue_mean": float(np.mean(img_array[:, :, 2])),
            "class": class_name,
        }
        
        return features
        
    except Exception as e:
        print(f"  ✗ Error processing {key}: {e}")
        return None


def list_all_training_images(s3_client, bucket: str, prefix: str) -> list[str]:
    """
    List all images in the training dataset using pagination.
    
    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., "datasets/combined/")
        
    Returns:
        List of S3 keys for all images
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    images = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(VALID_EXTENSIONS):
                images.append(key)
    
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference features for Evidently drift detection"
    )
    parser.add_argument(
        "--bucket",
        default=S3_BUCKET,
        help=f"S3 bucket name (default: {S3_BUCKET})"
    )
    parser.add_argument(
        "--prefix",
        default=S3_COMBINED_DIR,
        help=f"S3 prefix for training images (default: {S3_COMBINED_DIR})"
    )
    parser.add_argument(
        "--output-key",
        default=S3_REFERENCE_FEATURES_KEY,
        help=f"S3 key for output CSV (default: {S3_REFERENCE_FEATURES_KEY})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract features but don't upload to S3"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process N images (for testing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VITISCAN — Reference Features Generator")
    print("=" * 70)
    print(f"Bucket      : {args.bucket}")
    print(f"Prefix      : {args.prefix}")
    print(f"Output key  : {args.output_key}")
    print(f"Dry run     : {args.dry_run}")
    print("=" * 70)
    
    # Initialize S3 client
    s3 = boto3.client("s3")
    
    # List all training images
    print("\nListing training images...")
    all_images = list_all_training_images(s3, args.bucket, args.prefix)
    print(f"   Found {len(all_images)} images")
    
    if not all_images:
        print("No images found! Check your bucket and prefix.")
        sys.exit(1)
    
    # Show class distribution
    class_counts = {}
    for img in all_images:
        parts = img.split("/")
        class_name = parts[-2] if len(parts) >= 2 else "unknown"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nClass distribution:")
    for cls in VALID_CLASSES:
        count = class_counts.get(cls, 0)
        print(f"   {cls}: {count} images")
    
    # Sample if requested
    if args.sample:
        import random
        all_images = random.sample(all_images, min(args.sample, len(all_images)))
        print(f"\nSampling {len(all_images)} images for testing")
    
    # Extract features from each image
    print("\nExtracting features...")
    all_features = []
    
    for i, image_key in enumerate(all_images):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"   Processing image {i + 1}/{len(all_images)}...")
        
        features = extract_features_from_s3_image(s3, args.bucket, image_key)
        if features:
            all_features.append(features)
    
    print(f"\n✓ Successfully extracted features from {len(all_features)} images")
    
    if not all_features:
        print("No features extracted! Check your images.")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Show summary statistics
    print("\nFeature statistics:")
    numeric_cols = ["brightness", "contrast", "aspect_ratio", "file_size_kb", 
                    "red_mean", "green_mean", "blue_mean"]
    for col in numeric_cols:
        if col in df.columns:
            print(f"   {col:15s}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
    
    # Save to S3 (or local if dry run)
    if args.dry_run:
        local_path = "reference_features_test.csv"
        df.to_csv(local_path, index=False)
        print(f"\nDry run: saved to {local_path}")
        print(f"   Shape: {df.shape}")
    else:
        print(f"\nUploading to s3://{args.bucket}/{args.output_key}...")
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3.put_object(
            Bucket=args.bucket,
            Key=args.output_key,
            Body=csv_buffer.getvalue().encode("utf-8"),
            ContentType="text/csv",
            Metadata={
                "generated_at": datetime.now().isoformat(),
                "image_count": str(len(df)),
                "generator": "generate_reference_features.py",
            }
        )
        
        print(f"Successfully uploaded reference features!")
        print(f"   Location: s3://{args.bucket}/{args.output_key}")
        print(f"   Images: {len(df)}")
        print(f"   Features: {len(numeric_cols)}")
    
    print("\n" + "=" * 70)
    print("Done! You can now enable drift detection in the pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()