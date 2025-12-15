#!/usr/bin/env python3
"""
cxr_kafka_producer.py

Kafka producer for chest X-ray requests.

- Scans the local chest X-ray dataset under:
    /scratch/<USER>/bigdata_project/data/organized/{NORMAL,PNEUMONIA}

- Produces JSON messages to a Kafka topic, each with:
    {
        "file_path": "<absolute path to image>",
        "true_label": "NORMAL" or "PNEUMONIA",
        "request_id": "req_00000001",
        "ingest_ts": "2025-12-09T12:34:56.789012"
    }

Later you can import:

    from cxr_kafka_producer import kafka_produce_from_dataset

and call `kafka_produce_from_dataset(...)` from your main Spark pipeline.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# -------------------- CONFIG (EDIT THESE!) --------------------

USER = "sd5963"

SCRATCH_DIR = f"/scratch/{USER}/bigdata_project"
DATA_DIR = f"{SCRATCH_DIR}/data"
DATA_ROOT = Path(DATA_DIR) / "organized"   # expects NORMAL/ and PNEUMONIA/ inside

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"   # EDIT to your Kafka cluster
KAFKA_TOPIC_RAW = "cxr_raw_requests"        # topic to produce into

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


# -------------------- KAFKA PRODUCER --------------------


def kafka_produce_from_dataset(
    data_root: Path,
    bootstrap_servers: str,
    topic: str,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    """
    Produce messages into Kafka from the chest X-ray dataset.

    Parameters
    ----------
    data_root : Path
        Root directory containing NORMAL/ and PNEUMONIA/ folders.
    bootstrap_servers : str
        Kafka bootstrap servers (e.g., "localhost:9092").
    topic : str
        Kafka topic to send messages to.
    limit : Optional[int]
        Optional cap on number of images to send (for testing).
    dry_run : bool
        If True, do not actually send to Kafka; just print messages
        that *would* be sent. Useful for testing this script only.

    Message format:
        {
            "file_path": "<absolute or full path to image>",
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
        print("🧪 DRY RUN MODE — NOT sending to Kafka.")
        print(f"Would produce {len(images):,} messages to topic '{topic}'.\n")
        for idx, (path, label) in enumerate(images):
            msg = {
                "file_path": path,
                "true_label": label,
                "request_id": f"req_{idx:08d}",
                "ingest_ts": datetime.utcnow().isoformat(),
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

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"🚀 Producing {len(images):,} messages to Kafka topic '{topic}' ...")

    for idx, (path, label) in enumerate(images):
        msg = {
            "file_path": path,
            "true_label": label,
            "request_id": f"req_{idx:08d}",
            "ingest_ts": datetime.utcnow().isoformat(),
        }
        producer.send(topic, value=msg)

        # Optional: small progress log every N messages
        if (idx + 1) % 1000 == 0:
            print(f"  Produced {idx + 1:,} messages...")

    producer.flush()
    producer.close()
    print("✅ Kafka production complete\n")


# -------------------- CLI ENTRYPOINT --------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Produce chest X-ray requests into Kafka from the local dataset."
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
    print("====================================\n")

    kafka_produce_from_dataset(
        data_root=data_root,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        limit=args.limit,
        dry_run=args.dry_run,
    )
