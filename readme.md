# Medical Chest X-ray Report Generation Pipeline (Spark + Med-GEMMA + Kafka)

## 1. Overview

This project implements a **scalable, GPU-accelerated medical image pipeline** for **chest X-ray radiology report generation** using:

* **Med-GEMMA 4B (google/medgemma-4b-it)** for full radiology reports
* **PySpark** for distributed data processing and load testing
* **(Optional) Apache Kafka** for streaming request ingestion and queue-style workloads

Two modes of operation:

1. **Dataset / Load-Test Mode** (no Kafka):

   * Uses the Kaggle chest X-ray dataset
   * Runs load tests over increasing image counts (20, 400, 800, 1600, etc.)
   * Measures latency, throughput, and capacity

2. **Kafka-Integrated Mode**:

   * Reads incoming CXR requests from a Kafka topic
   * Generates full radiology reports via Med-GEMMA
   * Optionally writes results to an output Kafka topic and/or local storage
   * Still supports Spark-based load-test analytics and queue simulation

The core script (example name):

```text
medical_report_pipeline_kafka.py
```

If you saved it under a different name, adjust commands accordingly.

---

## 2. Features

* Downloads and organizes **Kaggle chest X-ray pneumonia dataset**
* Auto-detects GPU configuration and **auto-tunes**:

  * Spark parallelism / number of partitions
  * GPU batch size per worker
* Uses **Spark + mapInPandas** to distribute Med-GEMMA inference
* Generates **structured radiology reports** (no classification)
* Collects and saves:

  * Per-image **latency statistics** (mean, p50, p95, p99)
  * **Throughput** (images/sec)
  * **Report length distribution**
  * **Success / error rates**
* Creates plots:

  * Latency percentile bar plot
  * Report-length histogram
* Writes multi-run **load-test summary** and **queue simulation** for different concurrent request levels
* (Optional) Integrates **Kafka** for streaming ingestion of CXR requests

---

## 3. Requirements

### 3.1. System

* Linux environment with:

  * Python 3.8+
  * CUDA-capable GPU (recommended)
* Sufficient GPU VRAM for `google/medgemma-4b-it`

  * Script assumes ~16GB model footprint per replica

### 3.2. Python Packages

Install at least:

```bash
pip install torch transformers accelerate \
            pyspark pandas numpy pillow matplotlib seaborn \
            kaggle
```

If using Kafka via Spark:

```bash
# Spark Kafka connector is loaded via --packages (see Run section)
# No extra pip needed for Spark's Kafka integration itself.
```

### 3.3. External Services

1. **Hugging Face**

   * Accept Med-GEMMA terms:
     [https://huggingface.co/google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
   * Create an HF token:
     [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

2. **Kaggle**

   * Get API credentials from:
     [https://www.kaggle.com/settings](https://www.kaggle.com/settings)

3. **Kafka (Optional)**

   * A running Kafka broker, reachable at `KAFKA_BOOTSTRAP_SERVERS`
   * Input topic: e.g. `cxr_raw_requests`
   * Output topic (optional): e.g. `cxr_reports`

---

## 4. Directory Layout

The script uses a scratch area under:

```text
SCRATCH_DIR = /scratch/<USER>/bigdata_project
```

Inside this, it creates:

```text
data/
  raw/            # raw Kaggle download (chest_xray/)
  organized/      # flattened NORMAL / PNEUMONIA folders

outputs/
  results/        # per-run predictions (parquet, CSV, JSON, reports)
  metrics/        # per-run metric CSVs (latency, throughput, etc.)
  visualizations/ # per-run plots (PNG)
  load_test_summary.csv
  queue_simulation_summary.csv
  pipeline.log
hf_cache/         # Hugging Face cache (models, etc.)
```

You can customize `SCRATCH_DIR` in the script header.

---

## 5. Configuration

All key configuration is at the top of the script.

### 5.1. User / Credentials

```python
USER = "sd5963"

HF_TOKEN = "YOUR_HF_TOKEN_HERE"           # Replace with your HF token
KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY = "your_kaggle_key"

SCRATCH_DIR = f"/scratch/{USER}/bigdata_project"
DATA_DIR = f"{SCRATCH_DIR}/data"
OUTPUT_DIR = f"{SCRATCH_DIR}/outputs"
```

The script:

* Sets Hugging Face cache env vars to `HF_CACHE_DIR = SCRATCH_DIR/hf_cache`
* Writes `~/.kaggle/kaggle.json` using `KAGGLE_USERNAME` and `KAGGLE_KEY`

### 5.2. Modes

```python
TEST_MODE = True  # True = run only the test sizes; False = full dataset
TEST_IMAGE_SIZES = [20, 400, 800, 1600]
```

* `TEST_MODE = True`:

  * Runs separate experiments with N images in `TEST_IMAGE_SIZES` (but ≤ total available)
* `TEST_MODE = False`:

  * Runs a single experiment on all available images

### 5.3. GPU Auto-Tuning

```python
MODEL_SIZE_GB = 16.0
DESIRED_THROUGHPUT_IMG_PER_SEC = 10.0
TARGET_IMGS_PER_WORKER = 200
GPU_BATCH_SIZE = None  # None = auto-select based on VRAM
```

* Script queries GPU(s) via `torch.cuda`:

  * Estimates `max_partitions_per_gpu`
  * Picks a `GPU_BATCH_SIZE` (1, 2, 4, or 8) based on VRAM

### 5.4. Kafka Integration

(If your version of the script includes Kafka switches.)

Typical flags:

```python
USE_KAFKA = False  # Set True to read from Kafka instead of local dataset

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_IN = "cxr_raw_requests"
KAFKA_TOPIC_OUT = "cxr_reports"  # optional
```

When `USE_KAFKA = True`, a Kafka DataFrame is used as the **source** instead of the Kaggle `organized/` directory, with schema derived from Kafka messages (see Section 6.2).

---

## 6. Input Data

### 6.1. Dataset Mode (No Kafka)

The script downloads and organizes the Kaggle dataset:

* Source: `paultimothymooney/chest-xray-pneumonia`
* Directory after extraction:
  `data/raw/chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}`

It then flattens into:

```text
data/organized/NORMAL
  train_NORMAL2-IM-0588-0001.jpeg
  test_NORMAL2-IM-0001-0001.jpeg
  ...
data/organized/PNEUMONIA
  train_bacteria_1234.jpeg
  ...
```

In Spark, each image becomes a row:

```text
file_path: string (absolute path to image)
true_label: string ("NORMAL" or "PNEUMONIA")
```

### 6.2. Kafka Mode (Optional)

Kafka messages are expected to be **JSON** with fields like:

```json
{
  "file_path": "/scratch/.../data/organized/NORMAL/train_NORMAL2-IM-0588-0001.jpeg",
  "true_label": "NORMAL",
  "request_id": "req_00000989",
  "ingest_ts": "2025-12-10T04:35:45.410891"
}
```

Minimal required field for inference is:

* `file_path`: path accessible from each Spark worker

The script can be extended to write reports back to Kafka on `KAFKA_TOPIC_OUT`, for example:

```json
{
  "file_path": "...",
  "true_label": "NORMAL",
  "request_id": "req_00000989",
  "report_text": "...full radiology report...",
  "inference_time_ms": 345.41,
  "timestamp": "2025-12-10T05:12:34.123456",
  "gpu_used": "NVIDIA A40"
}
```

---

## 7. Model & Inference

Each Spark worker:

1. Detects device: `cuda` or `cpu`

2. Loads **Med-GEMMA 4B**:

   ```python
   MODEL_ID = "google/medgemma-4b-it"
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
   ```

3. Uses a **structured prompt**:

   * System: “You are an expert thoracic radiologist… write detailed report with Findings, Impression, etc.”
   * User: “Here is a chest X-ray… generate the full radiology report…”

4. Batches images in micro-batches of `GPU_BATCH_SIZE`, builds chat templates, and calls:

   ```python
   generated = model.generate(
       **inputs,
       max_new_tokens=512,
       do_sample=False,
   )
   ```

5. Splits off the **generated** part from prompt tokens and decodes the output as `report_text`.

Each output row (per image) includes:

* `file_path`
* `true_label` (carried from dataset or Kafka, not used for metrics)
* `report_text`
* `inference_time_ms`
* `batch_size`
* `timestamp`
* `gpu_used`
* `error_message` (if any)

---

## 8. Running the Script

### 8.1. Basic Run (No Kafka, Local Dataset)

1. Ensure environment is active (conda/venv).
2. Edit the header in `medical_report_pipeline_kafka.py`:

   * `USER`, `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`
   * `TEST_MODE` and `TEST_IMAGE_SIZES` as desired
3. Run:

```bash
python medical_report_pipeline_kafka.py
```

This will:

* Download Kaggle dataset (first run only)
* Organize images
* Initialize Spark
* Auto-tune partitions and GPU batch size
* Run load-test experiments over selected image counts
* Save predictions, metrics, and plots

### 8.2. Running With Kafka (Spark Structured Streaming)

If your version uses Spark’s Kafka connector, you should run via `spark-submit` to include Kafka packages:

```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  medical_report_pipeline_kafka.py
```

Before that:

* Set `USE_KAFKA = True` in the script
* Ensure Kafka broker is up and reachable at `KAFKA_BOOTSTRAP_SERVERS`
* Create topics:

  * `cxr_raw_requests`
  * `cxr_reports` (optional, for outputs)
* Make sure produced messages contain `file_path` pointing to images visible to Spark workers

---

## 9. Outputs

For each run label, e.g. `run_001_20imgs`:

### 9.1. Results

```text
outputs/results/run_001_20imgs/
  predictions.parquet
  predictions.csv
  predictions.json
  pipeline_summary_report.txt
```

`predictions.*` include per-image report rows.
`pipeline_summary_report.txt` is a human-readable summary report.

### 9.2. Metrics

```text
outputs/metrics/run_001_20imgs/
  latency_statistics.csv
  throughput_metrics.csv
  error_statistics.csv
  report_length_statistics.csv
```

### 9.3. Visualizations

```text
outputs/visualizations/run_001_20imgs/
  latency_analysis.png
  report_length_distribution.png
```

### 9.4. Global Summaries

```text
outputs/load_test_summary.csv
outputs/queue_simulation_summary.csv
outputs/pipeline.log
```

* `load_test_summary.csv`: one row per run (image count, mean latency, p95, throughput, etc.)
* `queue_simulation_summary.csv`: estimated time to drain queues at different **concurrent request levels** (e.g., 100, 1000, 10000) given pipeline throughput.

---

## 10. Queue Simulation (Conceptual)

For each run:

* Let `tp = wall_throughput_img_per_sec` (images/second)
* For each concurrent request level `L` with `REQUEST_SIZE_IMAGES = 4`:

```text
total_images = L * REQUEST_SIZE_IMAGES
time_to_clear = total_images / tp
```

This approximates **how long it takes to clear a burst of L requests** arriving at once, assuming current throughput and a single pipeline instance.

These results are summarized in `queue_simulation_summary.csv`.

---

## 11. Troubleshooting

1. **Model download / HF errors**

   * Check `HF_TOKEN`
   * Verify you accepted Med-GEMMA terms on Hugging Face
   * Confirm HF cache directory (`hf_cache`) is writable

2. **Kaggle download errors**

   * Confirm `KAGGLE_USERNAME` and `KAGGLE_KEY`
   * Ensure `~/.kaggle/kaggle.json` has correct permissions (`chmod 600`)

3. **GPU out-of-memory**

   * Reduce `GPU_BATCH_SIZE` at the top of the script (e.g., set it to `1` explicitly)
   * Ensure no other large processes use the same GPU

4. **Spark / Kafka connector error**

   * Use `spark-submit` with the Kafka package:

     ```bash
     spark-submit \
       --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
       medical_report_pipeline_kafka.py
     ```
   * Ensure Spark version matches the connector version

5. **File path not found (Kafka mode)**

   * Ensure `file_path` in Kafka messages points to a path visible from Spark workers
   * If running on an HPC cluster, ensure shared storage paths are consistent

---

## 12. Quick Start Checklist

1. Edit script header:

   * `USER`, `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`
   * `USE_KAFKA` as needed
2. (Optional) Start Kafka and create topics.
3. Activate Python environment with required packages.
4. Run:

   * **Without Kafka**:

     ```bash
     python medical_report_pipeline_kafka.py
     ```
   * **With Kafka**:

     ```bash
     spark-submit \
       --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
       medical_report_pipeline_kafka.py
     ```
5. Inspect:

   * `outputs/results/*/pipeline_summary_report.txt`
   * `outputs/metrics/*/*.csv`
   * `outputs/visualizations/*/*.png`
   * `outputs/load_test_summary.csv`
   * `outputs/queue_simulation_summary.csv`

zookeper running Terminal 1
cd /scratch/sd5963/big_data/kafka_2.13-3.8.0   # adjust if your path is different
export KAFKA_HOME=$(pwd)
export PATH=$KAFKA_HOME/bin:$PATH
echo $KAFKA_HOME
bin/zookeeper-server-start.sh config/zookeeper.properties

kafka Terminal 2
bin/kafka-server-start.sh config/server.properties

Watch messages Terminal 3
bin/kafka-console-consumer.sh \
  --topic cxr_raw_requests \
  --bootstrap-server localhost:9092 \
  --from-beginning

Main command Terminal 4
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,org.apache.spark:spark-token-provider-kafka-0-10_2.13:4.0.1 \
  medical_report_pipeline_kafka.py

