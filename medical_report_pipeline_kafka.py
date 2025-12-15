"""
FINAL FIXED PIPELINE - LOAD TEST VERSION (AUTO-TUNED + GPU BATCHING)
--------------------------------------------------------------------

Report-generation version:
- Uses Med-GEMMA to generate full chest X-ray radiology reports
  (no NORMAL/PNEUMONIA classification).
- Keeps auto-tuning, GPU batching, Spark load tests, and queue simulation.

This version has been extended to optionally:
- Read input images from AWS S3 (via Spark `binaryFile`)
- Write predictions to AWS S3 (Parquet/CSV), similar to the S3 notebook

BEFORE RUNNING:
1. Accept Med-GEMMA terms: https://huggingface.co/google/medgemma-4b-it
2. Get HF token: https://huggingface.co/settings/tokens
3. Get Kaggle credentials: https://www.kaggle.com/settings
4. (Optional, for S3 mode) Configure AWS credentials and S3 bucket paths
"""
from dotenv import load_dotenv
load_dotenv(".env")  # loads variables into os.environ
# ============================================================================
# CELL 1: SETUP + DATA DOWNLOAD
# ============================================================================

# -------------------- CONFIGURATION (EDIT THESE!) --------------------

import os

USER = os.environ.get("USER", "sd5963")
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", f"/scratch/{USER}/big_data")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")

DATA_DIR = f"{SCRATCH_DIR}/data"
OUTPUT_DIR = f"{SCRATCH_DIR}/outputs"

# -------------------- OPTIONAL S3 CONFIG --------------------
# Load AWS credentials from environment variables (set in .env file)
# For production, prefer IAM roles or environment variables instead of hardcoding.

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")  # Or use IAM roles
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")  # Or use IAM roles
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")  # Change to your region

# Buckets / prefixes (equivalent to RAW_IMAGES_BUCKET and RESULTS_BUCKET in the notebook)
S3_RAW_IMAGES_PREFIX = "s3a://medgemma-images"
S3_RESULTS_PREFIX = "s3a://medgemma-results"

# Hardcode S3 usage flags so the pipeline always uses S3 for input + results
USE_S3_SOURCE = True
USE_S3_RESULTS = True

# LOAD-TEST CONFIG
TEST_MODE = True
# In TEST_MODE, we will run one experiment per size below (capped by available images)
TEST_IMAGE_SIZES = [20, 400, 800, 1600]

GPU_CONFIG = "auto"  # currently unused, but kept for clarity

# -------------------- KAFKA CONFIG --------------------
# If True: read image requests from Kafka instead of scanning filesystem/S3.
USE_KAFKA = True
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "cxr_raw_requests"  # must match your producer

# If S3 is selected as the source, override Kafka-based input.
if USE_S3_SOURCE:
    USE_KAFKA = False

# -------------------- AUTO-TUNING + BATCHING CONFIG --------------------

# Approximate GPU-side model footprint for google/medgemma-4b-it (in GB).
# This is just for capacity heuristics, not an exact measurement.
MODEL_SIZE_GB = 16.0

# Soft throughput target used to pick how many images to feed per worker.
DESIRED_THROUGHPUT_IMG_PER_SEC = 10.0  # used as a heuristic

# Default target images per worker (before adjusting by throughput).
TARGET_IMGS_PER_WORKER = 200

# If None, GPU_BATCH_SIZE will be auto-picked from GPU VRAM.
GPU_BATCH_SIZE = None

# "Real" load-test simulation: interpret each request as this many images.
REQUEST_SIZE_IMAGES = 4
CONCURRENT_REQUEST_LEVELS = [100, 1000, 10000]

# ------------------------ IMPORTS ------------------------

import os, time, json, shutil, logging, warnings, zipfile, multiprocessing, sys, math
from pathlib import Path
from datetime import datetime
from typing import Iterator, List, Tuple, Dict, Any

warnings.filterwarnings('ignore')

import torch, pandas as pd, numpy as np
from PIL import Image
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, BooleanType
from pyspark.sql.functions import (
    col,
    count,
    avg,
    stddev,
    percentile_approx,
    sum as spark_sum,
    when,
    from_json,
    split,
    size,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ Imports successful\n")

# ------------------------ CREATE DIRECTORIES ------------------------

for directory in [SCRATCH_DIR, DATA_DIR, OUTPUT_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = Path(OUTPUT_DIR) / "results"
METRICS_ROOT = Path(OUTPUT_DIR) / "metrics"
VIZ_ROOT = Path(OUTPUT_DIR) / "visualizations"
HF_CACHE_DIR = Path(SCRATCH_DIR) / "hf_cache"

for d in [RESULTS_ROOT, METRICS_ROOT, VIZ_ROOT, HF_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(f'{OUTPUT_DIR}/pipeline.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print(f"✅ Directories created\n")

# ------------------------ SETUP CREDENTIALS ------------------------

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_TOKEN"] = HF_TOKEN

kaggle_dir = Path.home() / '.kaggle'
kaggle_dir.mkdir(exist_ok=True)
with open(kaggle_dir / 'kaggle.json', 'w') as f:
    json.dump({'username': KAGGLE_USERNAME, 'key': KAGGLE_KEY}, f)
(kaggle_dir / 'kaggle.json').chmod(0o600)

print("✅ Credentials configured\n")

# ------------------------ DOWNLOAD & ORGANIZE DATASET (LOCAL MODE) ------------------------

print("="*70)
print("DATASET SETUP")
print("="*70)

if USE_S3_SOURCE:
    # In S3 mode we assume images already live in S3 and we do NOT download Kaggle.
    print("⚙  S3 source is enabled — skipping local Kaggle download/organization.")
    DATA_ROOT = None
else:
    raw_data_dir = Path(DATA_DIR) / "raw"
    raw_data_dir.mkdir(exist_ok=True)

    if not (raw_data_dir / "chest_xray").exists():
        print("Downloading from Kaggle...")
        print("Dataset URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "paultimothymooney/chest-xray-pneumonia",
            path=str(raw_data_dir),
            unzip=True,
        )
        print("✅ Downloaded\n")
    else:
        print("✅ Already downloaded\n")

    organized_dir = Path(DATA_DIR) / "organized"
    (organized_dir / "NORMAL").mkdir(parents=True, exist_ok=True)
    (organized_dir / "PNEUMONIA").mkdir(parents=True, exist_ok=True)

    source_dir = raw_data_dir / "chest_xray"

    for split in ["train", "test", "val"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            src = source_dir / split / label
            dst = organized_dir / label
            if src.exists():
                for ext in ["*.jpeg", "*.jpg", "*.png"]:
                    for img in src.glob(ext):
                        dest_file = dst / f"{split}_{img.name}"
                        if not dest_file.exists():
                            shutil.copy2(img, dest_file)

    normal_count = len(list((organized_dir / "NORMAL").glob("*")))
    pneumonia_count = len(list((organized_dir / "PNEUMONIA").glob("*")))

    print("✅ Dataset organized:")
    print(f"   NORMAL: {normal_count:,}")
    print(f"   PNEUMONIA: {pneumonia_count:,}")
    print(f"   TOTAL: {normal_count + pneumonia_count:,}\n")

    DATA_ROOT = organized_dir

# ------------------------ GPU CHECK ------------------------

print("="*70)
print("GPU INFORMATION")
print("="*70)

gpu_info = {
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": 0,
    "gpus": []
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

print("="*70 + "\n")

# ------------------------ AUTO-TUNER HELPERS ------------------------

def estimate_max_partitions_per_gpu(gpu_info: Dict[str, Any],
                                    model_size_gb: float = MODEL_SIZE_GB,
                                    safety_margin: float = 1.2) -> int:
    """
    Heuristic: maximum concurrent model replicas per GPU given VRAM and approximate model size.
    Uses the *smallest* GPU VRAM to be conservative.
    """
    if not gpu_info.get("cuda_available") or gpu_info.get("gpu_count", 0) == 0:
        return 1
    min_mem = min(g["memory_total_gb"] for g in gpu_info["gpus"])
    usable = min_mem / safety_margin
    max_workers = max(1, int(usable // model_size_gb))
    return max_workers if max_workers > 0 else 1


def auto_select_gpu_batch_size(gpu_info: Dict[str, Any],
                               model_size_gb: float = MODEL_SIZE_GB) -> int:
    """
    Simple heuristic for GPU batch size based on VRAM.
    """
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


def auto_tune_spark_partitions(num_images: int,
                               base_partitions: int,
                               desired_throughput: float = DESIRED_THROUGHPUT_IMG_PER_SEC,
                               default_target_per_worker: int = TARGET_IMGS_PER_WORKER) -> int:
    """
    Auto-tune Spark partitions based on dataset size and a soft throughput target.
    - More images → more partitions, up to base_partitions.
    - desired_throughput controls how aggressive we are in splitting work.
    """
    if num_images <= 0:
        return 1

    # Adjust target images per worker based on desired throughput.
    # Higher desired throughput → smaller chunks per worker.
    if desired_throughput and desired_throughput > 0:
        target_per_worker = max(50, int(default_target_per_worker * (10.0 / desired_throughput)))
    else:
        target_per_worker = default_target_per_worker

    ideal_parts = max(1, math.ceil(num_images / float(target_per_worker)))
    n_parts = max(1, min(base_partitions, ideal_parts))
    return n_parts


# Decide MAX_PARTITIONS_PER_GPU and GPU_BATCH_SIZE now
max_partitions_per_gpu = estimate_max_partitions_per_gpu(gpu_info, MODEL_SIZE_GB)
if max_partitions_per_gpu < 1:
    max_partitions_per_gpu = 1

if GPU_BATCH_SIZE is None:
    GPU_BATCH_SIZE = auto_select_gpu_batch_size(gpu_info, MODEL_SIZE_GB)

os.environ["GPU_BATCH_SIZE"] = str(GPU_BATCH_SIZE)

print(f"⚙  Auto-tuner: max_partitions_per_gpu = {max_partitions_per_gpu}")
print(f"⚙  Auto-tuner: GPU_BATCH_SIZE = {GPU_BATCH_SIZE}")
print("="*70 + "\n")
print("✅ SETUP COMPLETE\n")
print("⚠️  IMPORTANT: Accept model terms at https://huggingface.co/google/medgemma-4b-it\n")
print("📋 Next: pipeline will run load-test experiments\n")


# ============================================================================
# CELL 2: COMPLETE PIPELINE (LOAD-TEST ENABLED)
# ============================================================================

print("\n" + "="*70)
print("STARTING PIPELINE")
print("="*70)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")

pipeline_start_time = time.time()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
print(f"✅ Spark will use: {sys.executable}\n")

# -------------------- SPARK INIT --------------------

print("Initializing Spark...")
n_cores = multiprocessing.cpu_count()

# CPU-side cap to avoid oversubscription.
cpu_limit = min(n_cores, 32)

gpu_count = gpu_info.get("gpu_count", 0)
if gpu_count <= 0:
    # CPU-only: use up to cpu_limit partitions
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
    .appName("MedicalImagePipeline_LoadTest_AutoTuned_ReportGen_Kafka_S3")
    .master(f"local[{base_partitions}]")
    .config("spark.driver.memory", "16g")
    .config("spark.executor.memory", "16g")
    .config("spark.sql.shuffle.partitions", str(max(4, base_partitions)))
    .config("spark.default.parallelism", str(base_partitions))
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
)

# If S3 integration is requested, configure Hadoop s3a connector.
if USE_S3_SOURCE or USE_S3_RESULTS:
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
        print("⚠️  USE_S3_SOURCE/USE_S3_RESULTS is enabled but AWS credentials are missing.")
        print("    Make sure AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION are set.")

spark = builder.getOrCreate()


print(f"✅ Spark v{spark.version} initialized\n")

# -------------------- BUILD BASE DATAFRAME (KAFKA OR FILESYSTEM) --------------------

print("Building input DataFrame...")

base_df = None

if USE_S3_SOURCE:
    # ------------------------------------------------------------------
    # S3 SOURCE MODE
    # ------------------------------------------------------------------
    if not S3_RAW_IMAGES_PREFIX:
        print("❌ USE_S3_SOURCE is True but S3_RAW_IMAGES_PREFIX is not set.")
        spark.stop()
        sys.exit(1)

    images_path = f"{S3_RAW_IMAGES_PREFIX}/"
    print(f"⚙  Using S3 source: {images_path}")

    # We mirror the notebook's S3 image loading pattern, but only keep path.
    # true_label is left as NULL since S3 paths may not encode labels.
    images_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.png")
        .option("recursiveFileLookup", "true")
        .load(images_path)
        .selectExpr(
            "path as file_path",
            "NULL as true_label",
        )
    )

    base_df = images_df
    total_available = base_df.count()
    print(f"✅ S3 DataFrame built with {total_available:,} images\n")

elif USE_KAFKA:
    print(f"⚙  Using Kafka source: {KAFKA_BOOTSTRAP_SERVERS}, topic={KAFKA_TOPIC}")

    kafka_schema = StructType([
        StructField("file_path",  StringType(), False),
        StructField("true_label", StringType(), True),
        StructField("request_id", StringType(), True),
        StructField("ingest_ts",  StringType(), True),
    ])

    # Batch read from Kafka (earliest → latest)
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
        .select("data.file_path", "data.true_label")
    )

    # Drop any null paths just in case
    base_df = requests_df.filter(col("file_path").isNotNull())

    total_available = base_df.count()
    print(f"✅ Kafka DataFrame built with {total_available:,} image requests\n")

else:
    print("⚙  Using filesystem scan from organized dataset")

    def scan_images(data_root: Path):
        data = []
        for label in ["NORMAL", "PNEUMONIA"]:
            label_path = data_root / label
            if label_path.exists():
                images = [
                    f
                    for f in label_path.iterdir()
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]
                data.extend([(str(img), label) for img in images])
                print(f"  {label}: {len(images):,} images")
        return data

    all_images = scan_images(DATA_ROOT)
    total_available = len(all_images)
    print(f"✅ Total: {total_available:,} images discovered\n")

    rows = [Row(file_path=path, true_label=label) for path, label in all_images]
    base_df = spark.createDataFrame(rows)

# Decide which image counts to test
if TEST_MODE:
    # Filter to only sizes that are <= total_available
    test_sizes = sorted([n for n in TEST_IMAGE_SIZES if n <= total_available])
    if not test_sizes:
        test_sizes = [min(20, total_available)] if total_available > 0 else []
    print(f"⚠️  LOAD-TEST MODE: will run experiments with image counts: {test_sizes}\n")
else:
    test_sizes = [total_available]
    print(f"⚠️  FULL-RUN MODE: single experiment with all {total_available} images\n")

if total_available == 0 or not test_sizes:
    print("❌ No images available from input source (Kafka or filesystem). Exiting.")
    spark.stop()
    sys.exit(0)

# -------------------- SCHEMA --------------------

prediction_schema = StructType([
    StructField("file_path", StringType(), False),
    # Keep the original dataset label for reference, but it's no longer used for accuracy.
    StructField("true_label", StringType(), True),
    StructField("report_text", StringType(), False),
    StructField("inference_time_ms", FloatType(), False),
    StructField("batch_size", IntegerType(), False),
    StructField("timestamp", StringType(), False),
    StructField("gpu_used", StringType(), True),
    StructField("error_message", StringType(), True),
])

# -------------------- INFERENCE FUNCTION (GPU-AWARE BATCHING, REPORT GENERATION) --------------------

def predict_batch(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    import torch, time, os, traceback, pandas as pd
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    from datetime import datetime

    MODEL_ID = "google/medgemma-4b-it"
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    worker_id = os.getpid()

    # Read GPU batch size from env set by driver
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

        # If model fails to load, mark everything in this worker as error
        for batch in iterator:
            yield pd.DataFrame({
                "file_path": batch["file_path"].tolist(),
                "true_label": batch.get("true_label", pd.Series([None] * len(batch))).tolist(),
                "report_text": [""] * len(batch),
                "inference_time_ms": [0.0] * len(batch),
                "batch_size": [len(batch)] * len(batch),
                "timestamp": [datetime.now().isoformat()] * len(batch),
                "gpu_used": [gpu_name] * len(batch),
                "error_message": [error_msg] * len(batch),
            })
        return

    # -------------------- BUILD PROMPT TEMPLATE --------------------

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

        file_paths = batch["file_path"].tolist()
        true_labels = batch.get("true_label", pd.Series([None] * len(batch))).tolist()
        batch_size = len(batch)

        reports, inference_times, error_messages = [], [], []

        print(f"[Worker-{worker_id}] Batch {batch_count}: {batch_size} images")

        # Process in GPU micro-batches
        for mb_start in range(0, batch_size, gpu_batch_size):
            mb_paths = file_paths[mb_start:mb_start + gpu_batch_size]
            micro_batch_size = len(mb_paths)
            if micro_batch_size == 0:
                continue

            mb_t0 = time.time()
            try:
                # Load images
                mb_images = [Image.open(p).convert("RGB") for p in mb_paths]

                # Build batched chat messages for report generation
                messages_batch = []
                for img in mb_images:
                    messages_batch.append([
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": SYSTEM_PROMPT}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": USER_TEXT},
                                {"type": "image", "image": img},
                            ],
                        },
                    ])

                # Apply chat template in batch
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

                # Compute per-sample prompt lengths from attention_mask
                if attention_mask is not None:
                    input_lens = attention_mask.sum(dim=1)
                else:
                    # Fallback: assume last 512 tokens are the report (rough heuristic)
                    input_lens = torch.full(
                        (generated.shape[0],),
                        fill_value=max(0, generated.shape[1] - 512),
                        dtype=torch.long,
                        device=generated.device,
                    )

                mb_time = time.time() - mb_t0
                per_image_ms = (mb_time * 1000.0) / float(micro_batch_size)

                for i in range(micro_batch_size):
                    seq = generated[i]
                    input_len_i = int(input_lens[i].item())
                    decoded = processor.decode(
                        seq[input_len_i:],
                        skip_special_tokens=True,
                    ).strip()

                    # Store full report text
                    reports.append(decoded)
                    error_messages.append(None)
                    inference_times.append(per_image_ms)

                if (mb_start + micro_batch_size) % max(1, gpu_batch_size) == 0:
                    print(f"[Worker-{worker_id}]   {mb_start + micro_batch_size}/{batch_size}")

            except Exception as e:
                # If something goes wrong for the micro-batch, mark them as ERROR
                mb_time = time.time() - mb_t0
                per_image_ms = (mb_time * 1000.0) / max(1, float(micro_batch_size))
                err_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[Worker-{worker_id}] ❌ Micro-batch error: {err_msg}")

                for _ in mb_paths:
                    reports.append("")
                    error_messages.append(err_msg)
                    inference_times.append(per_image_ms)

        print(f"[Worker-{worker_id}] ✅ Batch {batch_count} done in {time.time() - batch_start:.2f}s\n")

        yield pd.DataFrame({
            "file_path": file_paths,
            "true_label": true_labels,
            "report_text": reports,
            "inference_time_ms": inference_times,
            "batch_size": [batch_size] * batch_size,
            "timestamp": [datetime.now().isoformat()] * batch_size,
            "gpu_used": [gpu_name] * batch_size,
            "error_message": error_messages,
        })


# -------------------- LOAD-TEST LOOP --------------------

load_test_records = []  # will store per-run summary rows

for idx, size in enumerate(test_sizes):
    run_label = f"run_{idx+1:03d}_{size}imgs"
    print("\n" + "="*70)
    print(f"RUN {idx+1}/{len(test_sizes)} — {size} IMAGES  ({run_label})")
    print("="*70 + "\n")

    # Subset images from base_df (Kafka or filesystem)
    images_df = base_df.limit(size)

    # Auto-tune partitions for this run
    n_partitions = auto_tune_spark_partitions(
        num_images=size,
        base_partitions=base_partitions,
        desired_throughput=DESIRED_THROUGHPUT_IMG_PER_SEC,
        default_target_per_worker=TARGET_IMGS_PER_WORKER
    )

    images_df = images_df.repartition(n_partitions).cache()

    print(f"✅ DataFrame for {run_label}: {images_df.count():,} images, {n_partitions} partitions\n")

    # Per-run output dirs
    RESULTS_DIR = RESULTS_ROOT / run_label
    METRICS_DIR = METRICS_ROOT / run_label
    VIZ_DIR = VIZ_ROOT / run_label
    for d in [RESULTS_DIR, METRICS_DIR, VIZ_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------- RUN INFERENCE --------------------

    print("="*70)
    print(f"RUNNING INFERENCE — {run_label}")
    print("="*70)
    print(f"Images: {images_df.count():,} | GPUs: {gpu_info['gpu_count']} | Partitions: {n_partitions}")
    print("="*70 + "\n")

    inference_start = time.time()
    results_df = images_df.mapInPandas(predict_batch, schema=prediction_schema).cache()
    total_results = results_df.count()
    inference_elapsed = time.time() - inference_start

    print("\n" + "="*70)
    print(f"✅ INFERENCE DONE — {run_label}")
    print("="*70)
    print(f"Processed: {total_results:,}")
    print(f"Time: {inference_elapsed:.2f}s ({inference_elapsed/60:.2f} min)")
    print(f"Throughput (wall-clock): {total_results/inference_elapsed:.2f} img/s")
    print("="*70 + "\n")

    # -------------------- SAVE RAW PREDICTIONS --------------------

    print(f"Saving predictions (reports) for {run_label}...")

    # Always save locally under SCRATCH_DIR / outputs
    results_df.write.mode("overwrite").parquet(str(RESULTS_DIR / "predictions.parquet"))
    pdf = results_df.toPandas()
    pdf.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    pdf.to_json(RESULTS_DIR / "predictions.json", orient="records", lines=True)
    print("✅ Local results saved")

    # Optionally mirror predictions to S3, similar to the S3 notebook.
    if USE_S3_RESULTS:
        if not S3_RESULTS_PREFIX:
            print("⚠️  USE_S3_RESULTS is True but S3_RESULTS_PREFIX is not set; skipping S3 write.")
        else:
            # E.g. s3a://bucket/medgemma-results/run_001_20imgs/predictions_parquet/
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
    else:
        print("ℹ️  USE_S3_RESULTS is False — skipping S3 results write.\n")

    # -------------------- ANALYTICS --------------------

    print(f"Computing analytics for {run_label}...")

    results_df.createOrReplaceTempView("predictions")
    analytics: Dict[str, pd.DataFrame] = {}

    # Latency statistics per image
    analytics['latency_statistics'] = spark.sql("""
        SELECT ROUND(AVG(inference_time_ms), 2) as mean_ms,
               ROUND(MIN(inference_time_ms), 2) as min_ms,
               ROUND(MAX(inference_time_ms), 2) as max_ms,
               ROUND(PERCENTILE_APPROX(inference_time_ms, 0.50), 2) as p50_ms,
               ROUND(PERCENTILE_APPROX(inference_time_ms, 0.95), 2) as p95_ms,
               ROUND(PERCENTILE_APPROX(inference_time_ms, 0.99), 2) as p99_ms
        FROM predictions WHERE error_message IS NULL
    """).toPandas()

    # Throughput metrics based on per-image latencies
    analytics['throughput_metrics'] = spark.sql("""
        SELECT COUNT(*) as total_images,
               COUNT(DISTINCT gpu_used) as num_gpus,
               ROUND(SUM(inference_time_ms) / 1000, 2) as total_time_sec,
               ROUND(COUNT(*) / NULLIF(SUM(inference_time_ms) / 1000, 0), 2) as throughput_img_per_sec
        FROM predictions WHERE error_message IS NULL
    """).toPandas()

    # Error vs success rate
    analytics['error_statistics'] = spark.sql("""
        SELECT CASE WHEN error_message IS NULL THEN 'Success' ELSE 'Error' END as status,
               COUNT(*) as count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM predictions GROUP BY status
    """).toPandas()

    # Simple report length stats (characters)
    analytics['report_length_statistics'] = spark.sql("""
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

    # Optionally mirror metrics to S3 alongside predictions.
    if USE_S3_RESULTS:
        if not S3_RESULTS_PREFIX:
            print("⚠️  USE_S3_RESULTS is True but S3_RESULTS_PREFIX is not set; skipping S3 metrics write.")
        else:
            s3_run_base = f"{S3_RESULTS_PREFIX}/{run_label}"
            s3_metrics_base = f"{s3_run_base}/metrics/"
            print(f"Saving metrics CSVs to S3 under: {s3_metrics_base}")
            for name in analytics.keys():
                local_path = METRICS_DIR / f"{name}.csv"
                # Use Spark to write each metric CSV to S3 for simplicity.
                metrics_df = spark.read.option("header", "true").csv(str(local_path))
                metrics_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
                    f"{s3_metrics_base}{name}/"
                )
            print("✅ S3 metrics saved\n")
    else:
        print("ℹ️  USE_S3_RESULTS is False — skipping S3 metrics write.\n")

    # -------------------- VISUALIZATIONS --------------------

    print(f"Creating visualizations for {run_label}...")

    sns.set_style("whitegrid")

    # Latency chart
    lat = analytics['latency_statistics']

    if len(lat) > 0 and lat['mean_ms'].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))

        percentiles = ['mean_ms', 'p50_ms', 'p95_ms', 'p99_ms']
        labels = ['Mean', 'P50', 'P95', 'P99']
        values = [float(lat[p].values[0]) for p in percentiles if lat[p].notna().any()]
        colors_lat = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']

        if len(values) == 4:
            bars = ax.bar(labels, values, color=colors_lat, edgecolor='black')
            ax.set_title('Inference Latency Percentiles (per image)', fontsize=15, fontweight='bold')
            ax.set_ylabel('Milliseconds')

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}ms',
                        ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.savefig(VIZ_DIR / 'latency_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print("⚠️  Skipping latency chart (no successful predictions)")

    # Report length distribution (characters)
    rep_len = analytics['report_length_statistics']
    if len(rep_len) > 0 and rep_len['mean_chars'].notna().any():
        pdf_len = results_df.select("report_text", "error_message").toPandas()
        pdf_len = pdf_len[pdf_len["error_message"].isna()]
        pdf_len["report_len_chars"] = pdf_len["report_text"].str.len()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(pdf_len["report_len_chars"], bins=30, kde=True, ax=ax)
        ax.set_title('Distribution of Report Length (characters)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Characters')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'report_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("⚠️  Skipping report length chart (no successful reports)")

    print(f"✅ Visualizations saved for {run_label}")

    # Optionally mirror visualizations to S3 as well (upload PNGs).
    if USE_S3_RESULTS:
        if not S3_RESULTS_PREFIX:
            print("⚠️  USE_S3_RESULTS is True but S3_RESULTS_PREFIX is not set; skipping S3 visualization write.")
        else:
            import boto3
            from botocore.exceptions import ClientError

            s3_run_base = f"{S3_RESULTS_PREFIX}/{run_label}"
            s3_viz_base = f"{s3_run_base}/visualizations/"

            # Parse bucket and prefix from s3a://bucket/prefix
            bucket = s3_viz_base.replace("s3a://", "").split("/", 1)[0]
            base_prefix = s3_viz_base.replace("s3a://", "").split("/", 1)[1] if "/" in s3_viz_base.replace("s3a://", "") else ""

            print(f"Saving visualizations to S3 under: s3://{bucket}/{base_prefix}")

            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=AWS_REGION,
            )

            # Upload each PNG in the local visualization directory
            for file_path in VIZ_DIR.glob("*.png"):
                key = f"{base_prefix}{file_path.name}" if base_prefix.endswith("/") or base_prefix == "" else f"{base_prefix}/{file_path.name}"
                try:
                    s3_client.upload_file(str(file_path), bucket, key)
                    print(f"  ✅ Uploaded {file_path.name} -> s3://{bucket}/{key}")
                except ClientError as e:
                    print(f"  ❌ Failed to upload {file_path.name}: {e}")

            print("✅ S3 visualizations upload complete\n")
    else:
        print("ℹ️  USE_S3_RESULTS is False — skipping S3 visualization write.\n")

    # -------------------- REPORT (PER-RUN) --------------------

    print(f"Generating report for {run_label}...")

    lat = analytics['latency_statistics']
    tp = analytics['throughput_metrics']
    err = analytics['error_statistics']

    mean_ms = lat['mean_ms'].values[0] if len(lat) > 0 and lat['mean_ms'].notna().any() else None
    p95_ms = lat['p95_ms'].values[0] if len(lat) > 0 and lat['p95_ms'].notna().any() else None

    throughput_val = None
    total_images_metric = None
    if len(tp) > 0 and tp['throughput_img_per_sec'].notna().any():
        throughput_val = tp['throughput_img_per_sec'].values[0]
        total_images_metric = int(tp['total_images'].values[0])

    success_count = 0
    error_count = 0
    if len(err) > 0:
        for _, row in err.iterrows():
            if row['status'] == 'Success':
                success_count = int(row['count'])
            elif row['status'] == 'Error':
                error_count = int(row['count'])

    report = f"""
{'='*80}
MEDICAL IMAGE REPORT GENERATION PIPELINE - SUMMARY REPORT
RUN LABEL: {run_label}
{'='*80}

EXECUTION:
  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Duration (wall-clock for inference): {inference_elapsed/60:.2f} minutes
  Image Count: {size}
  Test Mode: {'Yes' if TEST_MODE else 'No'}

DATASET:
  Location: {DATA_ROOT}
  Total Images in this run: {total_results:,}

RESULTS:
  Successful reports: {success_count:,}
  Failed (errors):    {error_count:,}
"""

    if mean_ms is not None and p95_ms is not None:
        report += f"""
PERFORMANCE (PER-IMAGE LATENCY):
  Mean Latency: {mean_ms:.2f} ms
  P95 Latency:  {p95_ms:.2f} ms
"""

    if throughput_val is not None:
        report += f"""
  Throughput (per-image latency-based): {throughput_val:.2f} images/sec
  Daily Capacity (theoretical): {throughput_val * 86400:,.0f} images
"""

    report += f"""
INFRASTRUCTURE:
  CPUs used: {base_partitions}
  GPUs: {gpu_info['gpu_count']}
  Spark: {spark.version}

OUTPUT:
  Results (reports): {RESULTS_DIR}
  Metrics: {METRICS_DIR}
  Charts: {VIZ_DIR}

{'='*80}
PIPELINE RUN COMPLETED SUCCESSFULLY
{'='*80}
"""

    with open(Path(RESULTS_DIR) / 'pipeline_summary_report.txt', 'w') as f:
        f.write(report)

    print(report)

    # -------------------- ADD TO GLOBAL SUMMARY --------------------

    lat = analytics['latency_statistics']
    tp_wall = total_results / inference_elapsed if inference_elapsed > 0 else None

    mean_ms_val = float(lat['mean_ms'].values[0]) if len(lat) > 0 and lat['mean_ms'].notna().any() else None
    p95_ms_val = float(lat['p95_ms'].values[0]) if len(lat) > 0 and lat['p95_ms'].notna().any() else None

    load_test_records.append({
        "run_label": run_label,
        "num_images": size,
        "wall_time_sec": inference_elapsed,
        "wall_throughput_img_per_sec": tp_wall,
        "mean_latency_ms": mean_ms_val,
        "p95_latency_ms": p95_ms_val,
        "overall_accuracy_pct": None,  # no classification accuracy for report generation
    })

# -------------------- GLOBAL SUMMARY + QUEUE SIMULATION --------------------

summary_df = pd.DataFrame(load_test_records)
summary_path = Path(OUTPUT_DIR) / "load_test_summary.csv"
summary_df.to_csv(summary_path, index=False)

# Simple queue simulation for different concurrent request levels
queue_records = []
for _, row in summary_df.iterrows():
    run_label = row["run_label"]
    tp = row["wall_throughput_img_per_sec"]
    if tp is None or tp <= 0:
        continue

    for level in CONCURRENT_REQUEST_LEVELS:
        total_requests = level
        total_images_for_level = total_requests * REQUEST_SIZE_IMAGES
        time_to_clear_sec = total_images_for_level / tp  # all arrive at once, single GPU
        queue_records.append({
            "run_label": run_label,
            "concurrent_requests": level,
            "request_size_images": REQUEST_SIZE_IMAGES,
            "total_images": total_images_for_level,
            "throughput_img_per_sec": tp,
            "time_to_clear_sec": time_to_clear_sec,
            "time_to_clear_min": time_to_clear_sec / 60.0,
        })

if queue_records:
    queue_df = pd.DataFrame(queue_records)
    queue_path = Path(OUTPUT_DIR) / "queue_simulation_summary.csv"
    queue_df.to_csv(queue_path, index=False)
    print(f"\n📊 Queue simulation written to: {queue_path}")
    print(queue_df.head())

print("="*70)
print("✅✅✅ LOAD-TEST COMPLETE! ✅✅✅")
print("="*70)
print(f"Total time (all runs): {(time.time() - pipeline_start_time)/60:.2f} minutes")
print(f"Per-run summary saved to: {summary_path}")
print(f"Outputs root: {OUTPUT_DIR}")
print("="*70)
