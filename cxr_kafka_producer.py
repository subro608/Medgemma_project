#!/usr/bin/env python3
"""
cxr_kafka_producer.py

Kafka producer for chest X-ray requests (Kafka = request stream, S3 = data lake).

What it does:
1) Scans the local organized dataset under:
     /scratch/<USER>/bigdata_project/data/organized/{NORMAL,PNEUMONIA}

2) Uploads each image to S3 (optional, but recommended for Kafka+S3 architecture)

3) Produces JSON messages to a Kafka topic, each with:
    {
        "file_path": "s3a://<bucket>/<key>",   # S3 URI (preferred)
        "true_label": "NORMAL" or "PNEUMONIA", # optional / for reference
        "request_id": "req_00000001",
        "ingest_ts": "2025-12-09T12:34:56.789012"
    }

Usage examples:
  # Dry-run: show what would be sent (and what S3 keys would be used)
  python cxr_kafka_producer.py --dry-run --limit 10

  # Real run: upload to S3 + produce to Kafka
  python cxr_kafka_producer.py --s3-upload --limit 200

Notes:
- This script intentionally sends *pointers* (S3 URIs) in Kafka, not raw bytes.
- For production, prefer IAM roles on the cluster. Access/secret env vars are supported.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from dotenv import load_dotenv
load_dotenv() 
# -------------------- CONFIG (EDIT THESE!) --------------------

USER = os.environ.get("USER", "sd5963")

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", f"/scratch/{USER}/bigdata_project")
DATA_DIR = f"{SCRATCH_DIR}/data"
DATA_ROOT = Path(DATA_DIR) / "organized"   # expects NORMAL/ and PNEUMONIA/ inside

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RAW = os.environ.get("KAFKA_TOPIC_RAW", "cxr_raw_requests")

# S3 configuration (bucket name only; no s3:// prefix)
S3_BUCKET_IMAGES = os.environ.get("S3_BUCKET_IMAGES", "medgemma-images")
S3_PREFIX_IMAGES = os.environ.get("S3_PREFIX_IMAGES", "cxr")  # optional folder prefix inside bucket

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# -------------------- DATASET SCAN --------------------


def scan_images_local(data_root: Path) -> List[Tuple[str, str]]:
    """
    Scan the organized dataset and return a list of (file_path, label).

    Expected layout:
        data_root/
            NORMAL/
                *.jpeg|*.jpg|*.png
            PNEUMONIA/
                *.jpeg|*.jpg|*.png
    """
    data: List[Tuple[str, str]] = []

    for label in ["NORMAL", "PNEUMONIA"]:
        label_path = data_root / label
        if not label_path.exists():
            print(f"⚠️  Label directory not found: {label_path}")
            continue

        images = [
            f for f in label_path.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        images_sorted = sorted(images)
        data.extend([(str(img), label) for img in images_sorted])
        print(f"  {label}: {len(images_sorted):,} images found")

    print(f"✅ Total images discovered: {len(data):,}\n")
    return data


# -------------------- S3 UPLOAD --------------------


def _make_s3_client():
    """
    Create an S3 client. If AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set,
    boto3 can still work via IAM roles / default credential chain (recommended).
    """
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 is not installed. Install with: pip install boto3")

    kwargs = {"region_name": AWS_REGION}

    # Only set explicit keys if present; otherwise let boto3 use IAM/default chain.
    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY
        kwargs["aws_secret_access_key"] = AWS_SECRET_KEY

    return boto3.client("s3", **kwargs)


def upload_to_s3(local_path: str, bucket: str, key: str) -> None:
    """
    Upload local file to S3 bucket/key.
    """
    s3 = _make_s3_client()
    s3.upload_file(local_path, bucket, key)


def build_s3_key(prefix: str, label: str, request_id: str, local_path: str) -> str:
    """
    Construct a deterministic key so you can trace an S3 object back to request_id.
    Example:
        cxr/NORMAL/req_00000042_train_img123.jpeg
    """
    fname = Path(local_path).name
    prefix = (prefix or "").strip("/")
    parts = [p for p in [prefix, label, f"{request_id}_{fname}"] if p]
    return "/".join(parts)


def to_s3a_uri(bucket: str, key: str) -> str:
    return f"s3a://{bucket}/{key}"


# -------------------- KAFKA PRODUCER --------------------


def kafka_produce_from_dataset(
    data_root: Path,
    bootstrap_servers: str,
    topic: str,
    limit: Optional[int] = None,
    dry_run: bool = False,
    s3_upload: bool = False,
    s3_bucket: str = S3_BUCKET_IMAGES,
    s3_prefix: str = S3_PREFIX_IMAGES,
) -> None:
    """
    Produce messages into Kafka from the chest X-ray dataset.

    If s3_upload=True:
      - uploads each local image to S3
      - sends Kafka message with file_path = s3a://bucket/key

    If s3_upload=False:
      - sends Kafka message with file_path = local filesystem path
      (works only if Spark workers can access those local paths)

    Message format:
        {
            "file_path": "<s3a://... OR local path>",
            "true_label": "NORMAL" or "PNEUMONIA",
            "request_id": "req_00000042",
            "ingest_ts": "2025-12-09T12:34:56.789012"
        }
    """
    images = scan_images_local(data_root)

    if limit is not None:
        images = images[:limit]
        print(f"⚠️  Limiting to first {len(images):,} images for this run\n")

    if not images:
        print("⚠️  No images found; nothing to produce.")
        return

    if dry_run:
        mode = "S3+Kafka" if s3_upload else "LocalPath+Kafka"
        print(f"🧪 DRY RUN MODE — NOT sending to Kafka. Mode={mode}")
        print(f"Would produce {len(images):,} messages to topic '{topic}'.\n")

        for idx, (path, label) in enumerate(images):
            request_id = f"req_{idx:08d}"
            ingest_ts = datetime.utcnow().isoformat()

            if s3_upload:
                key = build_s3_key(s3_prefix, label, request_id, path)
                file_path_out = to_s3a_uri(s3_bucket, key)
            else:
                file_path_out = path

            msg = {
                "file_path": file_path_out,
                "true_label": label,
                "request_id": request_id,
                "ingest_ts": ingest_ts,
            }
            print(json.dumps(msg))
            if idx >= 9:
                print("... (showing only first 10 messages)")
                break

        print("✅ Dry run complete.\n")
        return

    # Real Kafka production path
    try:
        from kafka import KafkaProducer
    except ImportError:
        print("❌ kafka-python is not installed. Install with:")
        print("   pip install kafka-python")
        return

    # If S3 upload enabled, validate boto3 early to fail fast
    if s3_upload:
        try:
            _ = _make_s3_client()
        except Exception as e:
            print(f"❌ S3 upload requested but S3 client init failed: {e}")
            return

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    mode = "S3+Kafka" if s3_upload else "LocalPath+Kafka"
    print(f"🚀 Producing {len(images):,} messages to Kafka topic '{topic}' ... Mode={mode}")

    for idx, (path, label) in enumerate(images):
        request_id = f"req_{idx:08d}"
        ingest_ts = datetime.utcnow().isoformat()

        try:
            if s3_upload:
                key = build_s3_key(s3_prefix, label, request_id, path)
                upload_to_s3(local_path=path, bucket=s3_bucket, key=key)
                file_path_out = to_s3a_uri(s3_bucket, key)
            else:
                file_path_out = path

            msg = {
                "file_path": file_path_out,
                "true_label": label,
                "request_id": request_id,
                "ingest_ts": ingest_ts,
            }
            producer.send(topic, value=msg)

        except Exception as e:
            print(f"❌ Failed for {path}: {type(e).__name__}: {e}")

        # Optional: small progress log every N messages
        if (idx + 1) % 500 == 0:
            print(f"  Produced {idx + 1:,} messages...")

    producer.flush()
    producer.close()
    print("✅ Kafka production complete\n")


# -------------------- CLI ENTRYPOINT --------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Produce chest X-ray requests into Kafka from the local dataset (optionally uploading to S3)."
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=KAFKA_BOOTSTRAP_SERVERS,
        help=f"Kafka bootstrap servers (default: {KAFKA_BOOTSTRAP_SERVERS})",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=KAFKA_TOPIC_RAW,
        help=f"Kafka topic name (default: {KAFKA_TOPIC_RAW})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images to send (for testing).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DATA_ROOT),
        help=f"Root directory of organized dataset (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not send to Kafka; just print messages that would be sent.",
    )

    # S3 options
    parser.add_argument(
        "--s3-upload",
        action="store_true",
        help="If set, upload images to S3 and send Kafka messages with s3a:// URIs.",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=S3_BUCKET_IMAGES,
        help=f"S3 bucket name for images (default: {S3_BUCKET_IMAGES})",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default=S3_PREFIX_IMAGES,
        help=f"S3 key prefix inside the bucket (default: {S3_PREFIX_IMAGES})",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ Data root does not exist: {data_root}")
        raise SystemExit(1)

    print("======== CXR Kafka Producer ========")
    print(f"Data root          : {data_root}")
    print(f"Bootstrap servers  : {args.bootstrap_servers}")
    print(f"Topic              : {args.topic}")
    print(f"Limit              : {args.limit}")
    print(f"Dry run            : {args.dry_run}")
    print(f"S3 upload          : {args.s3_upload}")
    if args.s3_upload:
        print(f"S3 bucket          : {args.s3_bucket}")
        print(f"S3 prefix          : {args.s3_prefix}")
        print(f"AWS region         : {AWS_REGION}")
    print("====================================\n")

    kafka_produce_from_dataset(
        data_root=data_root,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        limit=args.limit,
        dry_run=args.dry_run,
        s3_upload=args.s3_upload,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
    )
