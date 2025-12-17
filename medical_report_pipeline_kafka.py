#!/usr/bin/env python3
"""
FINAL FIXED PIPELINE — Kafka (request stream) + S3 (data lake)
==============================================================

Goal (strong systems story):
- Kafka provides: request_id, ingest_ts, file_path (S3 pointer), optional true_label
- S3 provides: image bytes (Spark binaryFile)
- Pipeline measures per request_id:
    (a) queue_wait_ms  = dequeue_ts - ingest_ts
    (b) inference_time_ms (per image)
    (c) end_to_end_ms  = queue_wait_ms + inference_time_ms
- Writes predictions + metrics (+ optional viz) locally and (optionally) to S3.

BEFORE RUNNING:
1) Accept model terms: https://huggingface.co/google/medgemma-4b-it
2) Set HF_TOKEN, KAGGLE creds (only needed for local dataset mode)
3) Configure AWS creds or IAM role (for S3 access)
4) Ensure Spark has kafka + hadoop-aws deps (cluster submit-time packages)
"""

from dotenv import load_dotenv
load_dotenv(".env")  # loads variables into os.environ

# ============================================================================
# CELL 1: SETUP + CONFIG
# ============================================================================

import os
import time
import json
import shutil
import logging
import warnings
import multiprocessing
import sys
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator, List, Tuple, Dict, Any, Optional

warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, BinaryType
from pyspark.sql.functions import col, from_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ Imports successful\n")

# -------------------- CONFIGURATION --------------------

USER = os.environ.get("USER", "sd5963")
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", f"/scratch/{USER}/big_data")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")

DATA_DIR = f"{SCRATCH_DIR}/data"
OUTPUT_DIR = f"{SCRATCH_DIR}/outputs"

# -------------------- AWS / S3 CONFIG --------------------

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# S3 prefixes (data lake + results)
S3_RAW_IMAGES_PREFIX = os.environ.get("S3_RAW_IMAGES_PREFIX", "s3a://medgemma-images")
S3_RESULTS_PREFIX = os.environ.get("S3_RESULTS_PREFIX", "s3a://medgemma-results")

# Image extensions we will accept from S3
S3_IMAGE_EXTS = [".png", ".jpg", ".jpeg"]

# -------------------- MODE SELECTION --------------------
# Recommended grading setup:
# - Kafka = request stream
# - S3   = data lake
USE_KAFKA = True
USE_S3_DATALAKE = True       # read image bytes from S3 via Spark binaryFile
USE_S3_RESULTS = True        # mirror outputs to S3

# If you want local-only fallback (no S3), set USE_S3_DATALAKE=False and USE_S3_RESULTS=False.

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "cxr_raw_requests")

# LOAD-TEST CONFIG (batch-mode loop like your original script)
TEST_MODE = True
TEST_IMAGE_SIZES = [20, 400, 800, 1600]   # capped by available Kafka+S3 joinable messages

# -------------------- AUTO-TUNING + BATCHING CONFIG --------------------

MODEL_SIZE_GB = 16.0
DESIRED_THROUGHPUT_IMG_PER_SEC = 10.0
TARGET_IMGS_PER_WORKER = 200
GPU_BATCH_SIZE = None

# Queue simulation (kept from your original)
REQUEST_SIZE_IMAGES = 4
CONCURRENT_REQUEST_LEVELS = [100, 1000, 10000]

# ============================================================================
# DIRECTORIES + LOGGING
# ============================================================================

for directory in [SCRATCH_DIR, OUTPUT_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Only create DATA_DIR if you will use Kaggle/local mode
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = Path(OUTPUT_DIR) / "results"
METRICS_ROOT = Path(OUTPUT_DIR) / "metrics"
VIZ_ROOT = Path(OUTPUT_DIR) / "visualizations"
HF_CACHE_DIR = Path(SCRATCH_DIR) / "hf_cache"

for d in [RESULTS_ROOT, METRICS_ROOT, VIZ_ROOT, HF_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(f"{OUTPUT_DIR}/pipeline.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

print("✅ Directories created\n")

# ============================================================================
# CREDENTIALS / CACHE
# ============================================================================

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_TOKEN"] = HF_TOKEN

# Kaggle creds only needed if you do local dataset mode (kept for compatibility)
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)
with open(kaggle_dir / "kaggle.json", "w") as f:
    json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
(kaggle_dir / "kaggle.json").chmod(0o600)

print("✅ Credentials configured\n")

# ============================================================================
# GPU INFO + AUTO-TUNER HELPERS
# ============================================================================

print("=" * 70)
print("GPU INFORMATION")
print("=" * 70)

gpu_info = {
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": 0,
    "gpus": [],
}

if torch.cuda.is_available():
    gpu_info["gpu_count"] = torch.cuda.device_count()
    for i in range(gpu_info["gpu_count"]):
        gpu_name = torch.cuda.get_device_name(i)
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        gpu_info["gpus"].append({"id": i, "name": gpu_name, "memory_total_gb": mem_total})
        print(f"  GPU {i}: {gpu_name} ({mem_total:.1f} GB)")
else:
    print("⚠️  No GPU")

print("=" * 70 + "\n")


def estimate_max_partitions_per_gpu(
    gpu_info: Dict[str, Any],
    model_size_gb: float = MODEL_SIZE_GB,
    safety_margin: float = 1.2,
) -> int:
    """Heuristic: maximum concurrent model replicas per GPU given VRAM."""
    if not gpu_info.get("cuda_available") or gpu_info.get("gpu_count", 0) == 0:
        return 1
    min_mem = min(g["memory_total_gb"] for g in gpu_info["gpus"])
    usable = min_mem / safety_margin
    max_workers = max(1, int(usable // model_size_gb))
    return max_workers if max_workers > 0 else 1


def auto_select_gpu_batch_size(gpu_info: Dict[str, Any], model_size_gb: float = MODEL_SIZE_GB) -> int:
    """Simple heuristic for GPU batch size based on VRAM."""
    if not gpu_info.get("cuda_available") or gpu_info.get("gpu_count", 0) == 0:
        return 1
    mem = gpu_info["gpus"][0]["memory_total_gb"]
    if mem >= 80:
        return 8
    elif mem >= 40:
        return 4
    elif mem >= 24:
        return 2
    else:
        return 1


def auto_tune_spark_partitions(
    num_images: int,
    base_partitions: int,
    desired_throughput: float = DESIRED_THROUGHPUT_IMG_PER_SEC,
    default_target_per_worker: int = TARGET_IMGS_PER_WORKER,
) -> int:
    """Auto-tune Spark partitions based on dataset size and soft throughput target."""
    if num_images <= 0:
        return 1

    if desired_throughput and desired_throughput > 0:
        target_per_worker = max(50, int(default_target_per_worker * (10.0 / desired_throughput)))
    else:
        target_per_worker = default_target_per_worker

    ideal_parts = max(1, math.ceil(num_images / float(target_per_worker)))
    return max(1, min(base_partitions, ideal_parts))


max_partitions_per_gpu = estimate_max_partitions_per_gpu(gpu_info, MODEL_SIZE_GB)
if max_partitions_per_gpu < 1:
    max_partitions_per_gpu = 1

if GPU_BATCH_SIZE is None:
    GPU_BATCH_SIZE = auto_select_gpu_batch_size(gpu_info, MODEL_SIZE_GB)

os.environ["GPU_BATCH_SIZE"] = str(GPU_BATCH_SIZE)

print(f"⚙  Auto-tuner: max_partitions_per_gpu = {max_partitions_per_gpu}")
print(f"⚙  Auto-tuner: GPU_BATCH_SIZE = {GPU_BATCH_SIZE}")
print("=" * 70 + "\n")

print("✅ SETUP COMPLETE")
print("⚠️  IMPORTANT: Accept model terms at https://huggingface.co/google/medgemma-4b-it\n")

# ============================================================================
# CELL 2: SPARK INIT
# ============================================================================

print("\n" + "=" * 70)
print("STARTING PIPELINE")
print("=" * 70)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70 + "\n")

pipeline_start_time = time.time()

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
print(f"✅ Spark will use: {sys.executable}\n")

print("Initializing Spark...")
n_cores = multiprocessing.cpu_count()
cpu_limit = min(n_cores, 32)

gpu_count = gpu_info.get("gpu_count", 0)
if gpu_count <= 0:
    base_partitions = cpu_limit
else:
    max_partitions_total = max_partitions_per_gpu * gpu_count
    base_partitions = max(1, min(cpu_limit, max_partitions_total))

print(f"Using base_partitions = {base_partitions} (cpu_limit={cpu_limit}, gpu_count={gpu_count})")

try:
    existing = SparkSession.getActiveSession()
    if existing:
        existing.stop()
except Exception:
    pass

builder = (
    SparkSession.builder
    .appName("MedGEMMA_Kafka_S3_LoadTest_ReportGen")
    .master(f"local[{base_partitions}]")
    .config("spark.driver.memory", "16g")
    .config("spark.executor.memory", "16g")
    .config("spark.sql.shuffle.partitions", str(max(4, base_partitions)))
    .config("spark.default.parallelism", str(base_partitions))
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
)


# -----------------------------------------------------------------------------------
# S3 integration for datalake + results
if USE_S3_DATALAKE or USE_S3_RESULTS:
    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        builder = (
            builder.config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)
            .config("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_REGION}.amazonaws.com")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
            .config("spark.hadoop.fs.s3a.fast.upload", "true")
        )
    else:
        print("⚠️  S3 is enabled but AWS creds are missing. If running on AWS, prefer IAM roles.")

spark = builder.getOrCreate()
print(f"✅ Spark v{spark.version} initialized\n")

keys = [
    "fs.s3a.connection.timeout",
    "fs.s3a.connection.establish.timeout",
    "fs.s3a.socket.timeout",
    "spark.hadoop.fs.s3a.connection.timeout",
    "spark.hadoop.fs.s3a.connection.establish.timeout",
    "spark.hadoop.fs.s3a.socket.timeout",
]
hc = spark.sparkContext._jsc.hadoopConfiguration()
for k in keys:
    v = hc.get(k)
    print(f"[S3A CONF] {k} = {v}")
# ============================================================================
# INPUT: Kafka requests + S3 data lake join
# ============================================================================

print("Building input DataFrame...")

kafka_schema = StructType([
    StructField("file_path",  StringType(), False),
    StructField("true_label", StringType(), True),
    StructField("request_id", StringType(), True),
    StructField("ingest_ts",  StringType(), True),
])

base_df = None

if USE_KAFKA:
    print(f"⚙  Using Kafka request stream: {KAFKA_BOOTSTRAP_SERVERS}, topic={KAFKA_TOPIC}")

    kafka_raw_df = (
        spark.read
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("endingOffsets", "latest")
        .load()
    )

    requests_df = (
        kafka_raw_df
        .selectExpr("CAST(value AS STRING) AS json_str")
        .select(from_json(col("json_str"), kafka_schema).alias("data"))
        .select(
            col("data.file_path").alias("file_path"),
            col("data.true_label").alias("true_label"),
            col("data.request_id").alias("request_id"),
            col("data.ingest_ts").alias("ingest_ts"),
        )
        .filter(
            col("file_path").isNotNull()
            & col("request_id").isNotNull()
            & col("ingest_ts").isNotNull()
        )
    )

    if USE_S3_DATALAKE:
        if not S3_RAW_IMAGES_PREFIX:
            print("❌ USE_S3_DATALAKE is True but S3_RAW_IMAGES_PREFIX is empty.")
            spark.stop()
            sys.exit(1)

        images_path = f"{S3_RAW_IMAGES_PREFIX}/"
        print(f"⚙  Using S3 data lake (binaryFile): {images_path}")

        # Load all files, then filter extensions (binaryFile supports only one glob filter reliably)
        s3_images_all = (
            spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            .load(images_path)
            .select(col("path").alias("file_path"), col("content").alias("content"))
        )

        # Filter by extension in Spark (safe for mixed jpg/png)
        ext_cond = None
        for ext in S3_IMAGE_EXTS:
            c = col("file_path").endswith(ext)
            ext_cond = c if ext_cond is None else (ext_cond | c)
        s3_images_df = s3_images_all.filter(ext_cond)

        # Join Kafka requests to S3 content
        base_df = requests_df.join(s3_images_df, on="file_path", how="inner")

        total_available = base_df.count()
        print(f"✅ Kafka requests joined to S3 content: {total_available:,} rows\n")
    else:
        # Local file paths must be readable by workers; content is not available
        base_df = requests_df
        total_available = base_df.count()
        print(f"✅ Kafka DataFrame built (no S3 join): {total_available:,} rows\n")

else:
    # If you ever want non-Kafka mode, you can add your filesystem scan here.
    print("❌ This script is intended for USE_KAFKA=True (Kafka+S3 story).")
    spark.stop()
    sys.exit(1)

# Decide which image counts to test
if TEST_MODE:
    test_sizes = sorted([n for n in TEST_IMAGE_SIZES if n <= total_available])
    if not test_sizes:
        test_sizes = [min(20, total_available)] if total_available > 0 else []
    print(f"⚠️  LOAD-TEST MODE: will run experiments with image counts: {test_sizes}\n")
else:
    test_sizes = [total_available]
    print(f"⚠️  FULL-RUN MODE: single experiment with all {total_available} rows\n")

if total_available == 0 or not test_sizes:
    print("❌ No joinable requests/images available. Exiting.")
    spark.stop()
    sys.exit(0)

# ============================================================================
# OUTPUT SCHEMA
# ============================================================================

prediction_schema = StructType([
    StructField("request_id", StringType(), False),
    StructField("file_path", StringType(), False),

    StructField("ingest_ts", StringType(), False),
    StructField("dequeue_ts", StringType(), False),
    StructField("queue_wait_ms", FloatType(), False),

    StructField("true_label", StringType(), True),

    StructField("report_text", StringType(), False),
    StructField("inference_time_ms", FloatType(), False),
    StructField("end_to_end_ms", FloatType(), False),

    StructField("batch_size", IntegerType(), False),
    StructField("timestamp", StringType(), False),
    StructField("gpu_used", StringType(), True),
    StructField("error_message", StringType(), True),
])

# ============================================================================
# INFERENCE FUNCTION (uses S3 content bytes if present)
# ============================================================================
def parse_iso(ts: str) -> datetime:
    """Parse ISO timestamps and always return a UTC-aware datetime."""
    ts = (ts or "").strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return datetime.now(timezone.utc)

    # If timestamp is naive, assume it's UTC and attach tzinfo.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def predict_batch(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    import os
    import time
    import torch
    import pandas as pd
    from datetime import datetime, timezone
    from io import BytesIO
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    MODEL_ID = "google/medgemma-4b-it"
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    worker_id = os.getpid()

    try:
        gpu_batch_size = int(os.environ.get("GPU_BATCH_SIZE", "1"))
    except Exception:
        gpu_batch_size = 1

    print(f"\n[Worker-{worker_id}] Starting on {device} ({gpu_name}), GPU_BATCH_SIZE={gpu_batch_size}")
    print(f"[Worker-{worker_id}] Loading model...")

    load_start = time.time()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

        if device == "cuda":
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                dtype=torch.bfloat16,
                device_map="auto",
                token=HF_TOKEN,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                token=HF_TOKEN,
            ).to(device)

        print(f"[Worker-{worker_id}] ✅ Model loaded in {time.time() - load_start:.2f}s\n")

    except Exception as e:
        error_msg = f"Model load failed: {str(e)}"
        print(f"[Worker-{worker_id}] ❌ {error_msg}\n")

        for batch in iterator:
            n = len(batch)
            now = datetime.now(timezone.utc).isoformat()
            yield pd.DataFrame({
                "request_id": batch.get("request_id", pd.Series([""] * n)).tolist(),
                "file_path": batch.get("file_path", pd.Series([""] * n)).tolist(),
                "ingest_ts": batch.get("ingest_ts", pd.Series([now] * n)).tolist(),
                "dequeue_ts": [now] * n,
                "queue_wait_ms": [0.0] * n,
                "true_label": batch.get("true_label", pd.Series([None] * n)).tolist(),
                "report_text": [""] * n,
                "inference_time_ms": [0.0] * n,
                "end_to_end_ms": [0.0] * n,
                "batch_size": [n] * n,
                "timestamp": [now] * n,
                "gpu_used": [gpu_name] * n,
                "error_message": [error_msg] * n,
            })
        return

    SYSTEM_PROMPT = (
        "You are an expert thoracic radiologist. You are given a single chest X-ray.\n"
        "Write a detailed, structured radiology report including:\n"
        "- Study type and quality\n"
        "- Findings (lungs, pleura, heart and mediastinum, bones, upper abdomen)\n"
        "- Any signs of pneumonia or other pathology\n"
        "- Impression (concise summary and diagnosis/differential)\n"
        "Use clear headings and bullet points where appropriate."
    )

    USER_TEXT = (
        "Here is a chest X-ray. Carefully analyze it and generate the full radiology "
        "report as described in the instructions."
    )

    batch_count = 0

    for batch in iterator:
        batch_count += 1
        batch_start = time.time()

        request_ids = batch["request_id"].tolist()
        file_paths = batch["file_path"].tolist()
        ingest_tss = batch["ingest_ts"].tolist()
        true_labels = batch.get("true_label", pd.Series([None] * len(batch))).tolist()

        # We expect bytes in 'content' when using S3 binaryFile join.
        contents = batch.get("content", None)
        has_bytes = contents is not None

        batch_size = len(request_ids)
        dequeue_ts_str = datetime.now(timezone.utc).isoformat()

        # queue wait per image
        queue_waits_ms = []
        for its in ingest_tss:
            qw = (parse_iso(dequeue_ts_str) - parse_iso(its)).total_seconds() * 1000.0
            queue_waits_ms.append(float(qw))

        reports: List[str] = []
        inference_times: List[float] = []
        end_to_end_times: List[float] = []
        error_messages: List[Optional[str]] = []

        print(f"[Worker-{worker_id}] Batch {batch_count}: {batch_size} images")

        # Process in micro-batches
        for mb_start in range(0, batch_size, gpu_batch_size):
            mb_end = min(batch_size, mb_start + gpu_batch_size)
            micro_n = mb_end - mb_start
            if micro_n <= 0:
                continue

            mb_t0 = time.time()
            try:
                # Load images (bytes preferred; fallback to local filesystem paths)
                if has_bytes:
                    mb_bytes = contents.iloc[mb_start:mb_end].tolist()
                    mb_images = [Image.open(BytesIO(b)).convert("RGB") for b in mb_bytes]
                else:
                    mb_paths = file_paths[mb_start:mb_end]
                    mb_images = [Image.open(p).convert("RGB") for p in mb_paths]

                # Build batched chat messages
                messages_batch = []
                for img in mb_images:
                    messages_batch.append([
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": [{"type": "text", "text": USER_TEXT}, {"type": "image", "image": img}]},
                    ])

                inputs = processor.apply_chat_template(
                    messages_batch,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                attention_mask = inputs.get("attention_mask", None)

                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                    )

                if attention_mask is not None:
                    input_lens = attention_mask.sum(dim=1)
                else:
                    input_lens = torch.full(
                        (generated.shape[0],),
                        fill_value=max(0, generated.shape[1] - 512),
                        dtype=torch.long,
                        device=generated.device,
                    )

                mb_time = time.time() - mb_t0
                per_image_ms = (mb_time * 1000.0) / float(micro_n)

                for i in range(micro_n):
                    seq = generated[i]
                    input_len_i = int(input_lens[i].item())
                    decoded = processor.decode(seq[input_len_i:], skip_special_tokens=True).strip()

                    reports.append(decoded)
                    inference_times.append(float(per_image_ms))
                    error_messages.append(None)

                print(f"[Worker-{worker_id}]   {mb_end}/{batch_size}")

            except Exception as e:
                mb_time = time.time() - mb_t0
                per_image_ms = (mb_time * 1000.0) / max(1.0, float(micro_n))
                err_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[Worker-{worker_id}] ❌ Micro-batch error: {err_msg}")

                for _ in range(micro_n):
                    reports.append("")
                    inference_times.append(float(per_image_ms))
                    error_messages.append(err_msg)

        # end-to-end per image = queue wait + inference time
        for i in range(batch_size):
            end_to_end_times.append(float(queue_waits_ms[i]) + float(inference_times[i]))

        print(f"[Worker-{worker_id}] ✅ Batch {batch_count} done in {time.time() - batch_start:.2f}s\n")

        now_ts = datetime.now(timezone.utc).isoformat()
        yield pd.DataFrame({
            "request_id": request_ids,
            "file_path": file_paths,
            "ingest_ts": ingest_tss,
            "dequeue_ts": [dequeue_ts_str] * batch_size,
            "queue_wait_ms": queue_waits_ms,
            "true_label": true_labels,
            "report_text": reports,
            "inference_time_ms": inference_times,
            "end_to_end_ms": end_to_end_times,
            "batch_size": [batch_size] * batch_size,
            "timestamp": [now_ts] * batch_size,
            "gpu_used": [gpu_name] * batch_size,
            "error_message": error_messages,
        })


# ============================================================================
# LOAD-TEST LOOP
# ============================================================================

load_test_records: List[Dict[str, Any]] = []

for idx, size in enumerate(test_sizes):
    run_label = f"run_{idx+1:03d}_{size}reqs"
    print("\n" + "=" * 70)
    print(f"RUN {idx+1}/{len(test_sizes)} — {size} REQUESTS  ({run_label})")
    print("=" * 70 + "\n")

    # Subset requests (batch-style)
    requests_df = base_df.limit(size)

    # Auto-tune partitions
    n_partitions = auto_tune_spark_partitions(
        num_images=size,
        base_partitions=base_partitions,
        desired_throughput=DESIRED_THROUGHPUT_IMG_PER_SEC,
        default_target_per_worker=TARGET_IMGS_PER_WORKER,
    )

    requests_df = requests_df.repartition(n_partitions).cache()
    actual_n = requests_df.count()

    print(f"✅ DataFrame for {run_label}: {actual_n:,} rows, {n_partitions} partitions\n")

    RESULTS_DIR = RESULTS_ROOT / run_label
    METRICS_DIR = METRICS_ROOT / run_label
    VIZ_DIR = VIZ_ROOT / run_label
    for d in [RESULTS_DIR, METRICS_DIR, VIZ_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"RUNNING INFERENCE — {run_label}")
    print("=" * 70)
    print(f"Requests: {actual_n:,} | GPUs: {gpu_info['gpu_count']} | Partitions: {n_partitions}")
    print("=" * 70 + "\n")

    inference_start = time.time()
    results_df = requests_df.mapInPandas(predict_batch, schema=prediction_schema).cache()
    total_results = results_df.count()
    inference_elapsed = time.time() - inference_start

    print("\n" + "=" * 70)
    print(f"✅ INFERENCE DONE — {run_label}")
    print("=" * 70)
    print(f"Processed: {total_results:,}")
    print(f"Time: {inference_elapsed:.2f}s ({inference_elapsed/60:.2f} min)")
    print(f"Throughput (wall-clock): {total_results / inference_elapsed:.2f} req/s")
    print("=" * 70 + "\n")

    # -------------------- SAVE RAW PREDICTIONS --------------------

    print(f"Saving predictions for {run_label}...")

    results_df.write.mode("overwrite").parquet(str(RESULTS_DIR / "predictions.parquet"))
    pdf = results_df.toPandas()
    pdf.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    pdf.to_json(RESULTS_DIR / "predictions.json", orient="records", lines=True)
    print("✅ Local results saved")

    if USE_S3_RESULTS and S3_RESULTS_PREFIX:
        s3_run_base = f"{S3_RESULTS_PREFIX}/{run_label}"
        s3_parquet_path = f"{s3_run_base}/predictions_parquet/"
        s3_csv_path = f"{s3_run_base}/predictions_csv/"

        print(f"Saving predictions to S3 (Parquet): {s3_parquet_path}")
        results_df.write.mode("overwrite").parquet(s3_parquet_path)

        print(f"Saving predictions to S3 (CSV): {s3_csv_path}")
        (
            results_df.coalesce(1)
            .write.mode("overwrite")
            .option("header", "true")
            .csv(s3_csv_path)
        )
        print("✅ S3 results saved\n")

    # -------------------- ANALYTICS --------------------

    print(f"Computing analytics for {run_label}...")

    results_df.createOrReplaceTempView("predictions")
    analytics: Dict[str, pd.DataFrame] = {}

    # Inference latency
    analytics["inference_latency_stats"] = spark.sql("""
        SELECT ROUND(AVG(inference_time_ms), 2) as mean_ms,
               ROUND(MIN(inference_time_ms), 2) as min_ms,
               ROUND(MAX(inference_time_ms), 2) as max_ms,
               ROUND(PERCENTILE_APPROX(inference_time_ms, 0.50), 2) as p50_ms,
               ROUND(PERCENTILE_APPROX(inference_time_ms, 0.95), 2) as p95_ms,
               ROUND(PERCENTILE_APPROX(inference_time_ms, 0.99), 2) as p99_ms
        FROM predictions WHERE error_message IS NULL
    """).toPandas()

    # Queue wait
    analytics["queue_wait_stats"] = spark.sql("""
        SELECT ROUND(AVG(queue_wait_ms), 2) as mean_ms,
               ROUND(MIN(queue_wait_ms), 2) as min_ms,
               ROUND(MAX(queue_wait_ms), 2) as max_ms,
               ROUND(PERCENTILE_APPROX(queue_wait_ms, 0.50), 2) as p50_ms,
               ROUND(PERCENTILE_APPROX(queue_wait_ms, 0.95), 2) as p95_ms,
               ROUND(PERCENTILE_APPROX(queue_wait_ms, 0.99), 2) as p99_ms
        FROM predictions WHERE error_message IS NULL
    """).toPandas()

    # End-to-end latency
    analytics["end_to_end_stats"] = spark.sql("""
        SELECT ROUND(AVG(end_to_end_ms), 2) as mean_ms,
               ROUND(MIN(end_to_end_ms), 2) as min_ms,
               ROUND(MAX(end_to_end_ms), 2) as max_ms,
               ROUND(PERCENTILE_APPROX(end_to_end_ms, 0.50), 2) as p50_ms,
               ROUND(PERCENTILE_APPROX(end_to_end_ms, 0.95), 2) as p95_ms,
               ROUND(PERCENTILE_APPROX(end_to_end_ms, 0.99), 2) as p99_ms
        FROM predictions WHERE error_message IS NULL
    """).toPandas()

    # Throughput based on per-request inference times (not wall clock)
    analytics["throughput_metrics"] = spark.sql("""
        SELECT COUNT(*) as total_requests,
               COUNT(DISTINCT gpu_used) as num_gpus,
               ROUND(SUM(inference_time_ms) / 1000, 2) as summed_inference_time_sec,
               ROUND(COUNT(*) / NULLIF(SUM(inference_time_ms) / 1000, 0), 2) as throughput_req_per_sec
        FROM predictions WHERE error_message IS NULL
    """).toPandas()

    # Error rate
    analytics["error_statistics"] = spark.sql("""
        SELECT CASE WHEN error_message IS NULL THEN 'Success' ELSE 'Error' END as status,
               COUNT(*) as count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM predictions GROUP BY status
    """).toPandas()

    # Report length
    analytics["report_length_statistics"] = spark.sql("""
        SELECT
            ROUND(AVG(LENGTH(report_text)), 2) as mean_chars,
            ROUND(MIN(LENGTH(report_text)), 2) as min_chars,
            ROUND(MAX(LENGTH(report_text)), 2) as max_chars,
            ROUND(PERCENTILE_APPROX(LENGTH(report_text), 0.50), 2) as p50_chars,
            ROUND(PERCENTILE_APPROX(LENGTH(report_text), 0.95), 2) as p95_chars
        FROM predictions WHERE error_message IS NULL
    """).toPandas()

    for name, df in analytics.items():
        df.to_csv(METRICS_DIR / f"{name}.csv", index=False)

    print(f"✅ {len(analytics)} analytics saved for {run_label}")

    if USE_S3_RESULTS and S3_RESULTS_PREFIX:
        s3_run_base = f"{S3_RESULTS_PREFIX}/{run_label}"
        s3_metrics_base = f"{s3_run_base}/metrics/"
        print(f"Saving metrics CSVs to S3 under: {s3_metrics_base}")
        for name in analytics.keys():
            local_path = METRICS_DIR / f"{name}.csv"
            metrics_df = spark.read.option("header", "true").csv(str(local_path))
            metrics_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
                f"{s3_metrics_base}{name}/"
            )
        print("✅ S3 metrics saved\n")

    # -------------------- VISUALIZATIONS --------------------

    print(f"Creating visualizations for {run_label}...")
    sns.set_style("whitegrid")

    # End-to-end latency bar chart (mean/p50/p95/p99)
    e2e = analytics["end_to_end_stats"]
    if len(e2e) > 0 and e2e["mean_ms"].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))
        cols = ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]
        labels = ["Mean", "P50", "P95", "P99"]
        values = [float(e2e[c].values[0]) for c in cols]
        ax.bar(labels, values, edgecolor="black")
        ax.set_title("End-to-End Latency Percentiles (per request)")
        ax.set_ylabel("Milliseconds")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.1f}ms", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "end_to_end_latency_percentiles.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Report length histogram
    rep_len = analytics["report_length_statistics"]
    if len(rep_len) > 0 and rep_len["mean_chars"].notna().any():
        pdf_len = results_df.select("report_text", "error_message").toPandas()
        pdf_len = pdf_len[pdf_len["error_message"].isna()]
        pdf_len["report_len_chars"] = pdf_len["report_text"].str.len()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(pdf_len["report_len_chars"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Report Length (characters)")
        ax.set_xlabel("Characters")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "report_length_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"✅ Visualizations saved for {run_label}")

    # -------------------- REPORT (PER-RUN) --------------------

    inf = analytics["inference_latency_stats"]
    qw = analytics["queue_wait_stats"]
    e2e = analytics["end_to_end_stats"]
    err = analytics["error_statistics"]

    def safe_val(df: pd.DataFrame, key: str) -> Optional[float]:
        try:
            v = df[key].values[0]
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    inf_mean = safe_val(inf, "mean_ms")
    inf_p95 = safe_val(inf, "p95_ms")
    qw_mean = safe_val(qw, "mean_ms")
    qw_p95 = safe_val(qw, "p95_ms")
    e2e_mean = safe_val(e2e, "mean_ms")
    e2e_p95 = safe_val(e2e, "p95_ms")

    success_count = 0
    error_count = 0
    if len(err) > 0:
        for _, r in err.iterrows():
            if r["status"] == "Success":
                success_count = int(r["count"])
            elif r["status"] == "Error":
                error_count = int(r["count"])

    report = f"""
{'='*80}
MEDGEMMA REPORT GENERATION — KAFKA+S3 SYSTEMS RUN SUMMARY
RUN LABEL: {run_label}
{'='*80}

EXECUTION:
  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Duration (wall-clock for inference): {inference_elapsed/60:.2f} minutes
  Requests processed: {total_results:,}
  Test Mode: {'Yes' if TEST_MODE else 'No'}

RESULTS:
  Successful reports: {success_count:,}
  Failed (errors):    {error_count:,}

LATENCY (PER REQUEST):
  Queue wait (mean / p95): {qw_mean if qw_mean is not None else 'NA'} ms / {qw_p95 if qw_p95 is not None else 'NA'} ms
  Inference (mean / p95):  {inf_mean if inf_mean is not None else 'NA'} ms / {inf_p95 if inf_p95 is not None else 'NA'} ms
  End-to-end (mean / p95): {e2e_mean if e2e_mean is not None else 'NA'} ms / {e2e_p95 if e2e_p95 is not None else 'NA'} ms

INFRASTRUCTURE:
  CPUs (local parallelism): {base_partitions}
  GPUs: {gpu_info['gpu_count']}
  Spark: {spark.version}

OUTPUT:
  Local Results: {RESULTS_DIR}
  Local Metrics: {METRICS_DIR}
  Local Charts:  {VIZ_DIR}
  S3 Results Prefix: {S3_RESULTS_PREFIX if USE_S3_RESULTS else 'Disabled'}

{'='*80}
"""

    with open(RESULTS_DIR / "pipeline_summary_report.txt", "w") as f:
        f.write(report)

    print(report)

    # -------------------- ADD TO GLOBAL SUMMARY --------------------

    wall_tp = total_results / inference_elapsed if inference_elapsed > 0 else None

    load_test_records.append({
        "run_label": run_label,
        "num_requests": int(total_results),
        "wall_time_sec": float(inference_elapsed),
        "wall_throughput_req_per_sec": float(wall_tp) if wall_tp is not None else None,
        "queue_wait_mean_ms": qw_mean,
        "queue_wait_p95_ms": qw_p95,
        "inference_mean_ms": inf_mean,
        "inference_p95_ms": inf_p95,
        "end_to_end_mean_ms": e2e_mean,
        "end_to_end_p95_ms": e2e_p95,
    })

# ============================================================================
# GLOBAL SUMMARY + QUEUE SIMULATION
# ============================================================================

summary_df = pd.DataFrame(load_test_records)
summary_path = Path(OUTPUT_DIR) / "load_test_summary.csv"
summary_df.to_csv(summary_path, index=False)

queue_records = []
for _, row in summary_df.iterrows():
    tp = row.get("wall_throughput_req_per_sec", None)
    if tp is None or tp <= 0:
        continue

    for level in CONCURRENT_REQUEST_LEVELS:
        total_requests = int(level)
        total_images_for_level = total_requests * REQUEST_SIZE_IMAGES
        # NOTE: tp is requests/sec; convert to images/sec if you model requests as multi-image
        images_per_sec = tp * REQUEST_SIZE_IMAGES
        time_to_clear_sec = total_images_for_level / images_per_sec if images_per_sec > 0 else None

        queue_records.append({
            "run_label": row["run_label"],
            "concurrent_requests": total_requests,
            "request_size_images": REQUEST_SIZE_IMAGES,
            "total_images": total_images_for_level,
            "wall_throughput_req_per_sec": tp,
            "estimated_images_per_sec": images_per_sec,
            "time_to_clear_sec": time_to_clear_sec,
            "time_to_clear_min": (time_to_clear_sec / 60.0) if time_to_clear_sec is not None else None,
        })

if queue_records:
    queue_df = pd.DataFrame(queue_records)
    queue_path = Path(OUTPUT_DIR) / "queue_simulation_summary.csv"
    queue_df.to_csv(queue_path, index=False)
    print(f"\n📊 Queue simulation written to: {queue_path}")
    print(queue_df.head())

print("=" * 70)
print("✅✅✅ LOAD-TEST COMPLETE! ✅✅✅")
print("=" * 70)
print(f"Total time (all runs): {(time.time() - pipeline_start_time)/60:.2f} minutes")
print(f"Per-run summary saved to: {summary_path}")
print(f"Outputs root: {OUTPUT_DIR}")
print("=" * 70)

try:
    spark.stop()
except Exception:
    pass
