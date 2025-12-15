#!/usr/bin/env bash
set -euo pipefail

############################################
# CONFIG
############################################

USER="sd5963"
BASE_DIR="/scratch/${USER}/big_data"
KAFKA_DIR="${BASE_DIR}/kafka_2.13-3.8.0"
ENV_DIR="${BASE_DIR}/medgemma_env"

TOPIC="cxr_raw_requests"
PARTITIONS=4
REPLICATION=1

# How many image requests you want to send via Kafka
NUM_MESSAGES=200

SPARK_KAFKA_PACKAGES="org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,org.apache.spark:spark-token-provider-kafka-0-10_2.13:4.0.1"

export KAFKA_HOME="${KAFKA_DIR}"

mkdir -p "${BASE_DIR}/logs"

echo ">>> Using BASE_DIR=${BASE_DIR}"
echo ">>> Using KAFKA_DIR=${KAFKA_DIR}"
echo ">>> Using ENV_DIR=${ENV_DIR}"
echo ">>> Topic = ${TOPIC}"
echo

############################################
# HELPERS
############################################

wait_for_kafka() {
  local retries=30
  local i
  echo ">>> Waiting for Kafka broker to be ready on localhost:9092..."
  for i in $(seq 1 "${retries}"); do
    if "${KAFKA_HOME}/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
      echo "    Kafka is up (attempt ${i}/${retries})."
      return 0
    fi
    echo "    Kafka not ready yet, sleeping 2s (attempt ${i}/${retries})..."
    sleep 2
  done
  echo "!!! Kafka did not become ready in time. Check ${BASE_DIR}/logs/kafka.log"
  return 1
}

############################################
# START ZOOKEEPER
############################################

echo ">>> Starting ZooKeeper..."
"${KAFKA_HOME}/bin/zookeeper-server-start.sh" \
  "${KAFKA_HOME}/config/zookeeper.properties" \
  > "${BASE_DIR}/logs/zookeeper.log" 2>&1 &

ZK_PID=$!
echo "    ZooKeeper PID = ${ZK_PID}"
sleep 5   # small delay just to get it going

############################################
# START KAFKA BROKER
############################################

echo ">>> Starting Kafka broker..."
"${KAFKA_HOME}/bin/kafka-server-start.sh" \
  "${KAFKA_HOME}/config/server.properties" \
  > "${BASE_DIR}/logs/kafka.log" 2>&1 &

KAFKA_PID=$!
echo "    Kafka PID = ${KAFKA_PID}"

# Wait for broker to be actually ready
wait_for_kafka

############################################
# CREATE TOPIC (IF NOT EXISTS)
############################################

echo ">>> Creating topic '${TOPIC}' (if not exists)..."
"${KAFKA_HOME}/bin/kafka-topics.sh" \
  --create \
  --if-not-exists \
  --topic "${TOPIC}" \
  --bootstrap-server localhost:9092 \
  --partitions "${PARTITIONS}" \
  --replication-factor "${REPLICATION}"

echo ">>> Topic setup complete."
echo

############################################
# ACTIVATE PYTHON ENV
############################################

echo ">>> Activating Python env: ${ENV_DIR}"
# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"

############################################
# PRODUCE MESSAGES TO KAFKA
############################################

echo ">>> Running Kafka producer: pushing ${NUM_MESSAGES} messages to '${TOPIC}'..."
python3 "${BASE_DIR}/cxr_kafka_producer.py" \
  --bootstrap-servers localhost:9092 \
  --topic "${TOPIC}" \
  --limit "${NUM_MESSAGES}"

echo ">>> Producer finished."
echo

############################################
# RUN SPARK PIPELINE
############################################

echo ">>> Running Spark pipeline (medical_report_pipeline_kafka.py)..."

spark-submit \
  --packages "${SPARK_KAFKA_PACKAGES}" \
  "${BASE_DIR}/medical_report_pipeline_kafka.py"

PIPE_STATUS=$?
echo ">>> Spark job finished with status: ${PIPE_STATUS}"

############################################
# STOP KAFKA / ZK
############################################

echo ">>> Stopping Kafka and ZooKeeper..."
kill "${KAFKA_PID}" || true
kill "${ZK_PID}" || true

echo ">>> All done."
exit "${PIPE_STATUS}"
