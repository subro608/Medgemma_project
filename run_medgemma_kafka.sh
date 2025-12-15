#!/usr/bin/env bash
set -euo pipefail

############################################
# CONFIG
############################################
USER="sd5963"
BASE_DIR="/scratch/${USER}/big_data"
KAFKA_DIR="${BASE_DIR}/kafka_2.13-3.8.0"
ENV_DIR="${BASE_DIR}/medgemma_env"
LOG_DIR="${BASE_DIR}/logs"

# Data + scripts
DATA_ROOT="/scratch/${USER}/bigdata_project/data/organized"
PRODUCER_PY="${BASE_DIR}/cxr_kafka_producer.py"
PIPELINE_PY="${BASE_DIR}/medical_report_pipeline_kafka.py"

# Kafka
TOPIC="cxr_raw_requests"
PARTITIONS=4
REPLICATION=1
NUM_MESSAGES=200

# Spark + Kafka connector packages (Spark 4.0.1, Scala 2.13)
SPARK_KAFKA_PACKAGES="org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,org.apache.spark:spark-token-provider-kafka-0-10_2.13:4.0.1"

# Runtime toggles
KEEP_KAFKA_RUNNING="${KEEP_KAFKA_RUNNING:-0}"   # set to 1 to keep Kafka/ZK alive after script exits
DRY_RUN_PRODUCER="${DRY_RUN_PRODUCER:-0}"       # set to 1 to not actually publish messages
LIMIT_MESSAGES="${LIMIT_MESSAGES:-${NUM_MESSAGES}}"

export KAFKA_HOME="${KAFKA_DIR}"
mkdir -p "${LOG_DIR}"

ZK_LOG="${LOG_DIR}/zookeeper.log"
KAFKA_LOG="${LOG_DIR}/kafka.log"
ZK_PIDFILE="${LOG_DIR}/zookeeper.pid"
KAFKA_PIDFILE="${LOG_DIR}/kafka.pid"

# Per-run state dirs to avoid stale metadata
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
STATE_DIR="${BASE_DIR}/kafka_state/${RUN_TAG}"
ZK_DATADIR="${STATE_DIR}/zookeeper"
KAFKA_LOGDIR="${STATE_DIR}/kafka-logs"
mkdir -p "${ZK_DATADIR}" "${KAFKA_LOGDIR}"

ZK_PROPS_SRC="${KAFKA_DIR}/config/zookeeper.properties"
KAFKA_PROPS_SRC="${KAFKA_DIR}/config/server.properties"
ZK_PROPS="${STATE_DIR}/zookeeper.properties"
KAFKA_PROPS="${STATE_DIR}/server.properties"

# Host + ports
LISTEN_HOST="127.0.0.1"
ZK_PORT=2181
KAFKA_PORT=9092

############################################
# HELPERS
############################################
log() { echo "[$(date '+%F %T')] $*"; }

is_listening() {
  local port="$1"
  ss -ltn "( sport = :${port} )" | grep -q LISTEN
}

kill_pidfile() {
  local pidfile="$1"
  if [[ -f "${pidfile}" ]]; then
    local pid
    pid="$(cat "${pidfile}" || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      log "Killing PID ${pid} from ${pidfile}"
      kill "${pid}" 2>/dev/null || true
      sleep 1
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${pidfile}" || true
  fi
}

cleanup() {
  if [[ "${KEEP_KAFKA_RUNNING}" == "1" ]]; then
    log "KEEP_KAFKA_RUNNING=1 -> leaving Kafka/ZooKeeper running."
    return 0
  fi
  log "Cleanup: stopping Kafka/ZooKeeper started by this script (if any)."
  kill_pidfile "${KAFKA_PIDFILE}"
  kill_pidfile "${ZK_PIDFILE}"
}

trap cleanup EXIT

wait_for_port() {
  local port="$1"
  local name="$2"
  local attempts="${3:-60}"
  local sleep_s="${4:-1}"

  for i in $(seq 1 "${attempts}"); do
    if is_listening "${port}"; then
      log "${name} is listening on :${port}"
      return 0
    fi
    log "${name} not ready yet on :${port} (attempt ${i}/${attempts})..."
    sleep "${sleep_s}"
  done
  return 1
}

############################################
# START
############################################
log "BASE_DIR=${BASE_DIR}"
log "KAFKA_DIR=${KAFKA_DIR}"
log "ENV_DIR=${ENV_DIR}"
log "LOG_DIR=${LOG_DIR}"
log "STATE_DIR=${STATE_DIR}"
log "Topic=${TOPIC}"
log "DATA_ROOT=${DATA_ROOT}"
log "PRODUCER_PY=${PRODUCER_PY}"
log "PIPELINE_PY=${PIPELINE_PY}"
echo

# Activate venv
if [[ -f "${ENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_DIR}/bin/activate"
fi

# Sanity checks
if [[ ! -d "${KAFKA_DIR}" ]]; then
  log "ERROR: Kafka dir not found: ${KAFKA_DIR}"
  exit 1
fi
if [[ ! -f "${PRODUCER_PY}" ]]; then
  log "ERROR: Producer script not found: ${PRODUCER_PY}"
  exit 1
fi
if [[ ! -f "${PIPELINE_PY}" ]]; then
  log "ERROR: Pipeline script not found: ${PIPELINE_PY}"
  log "Fix PIPELINE_PY to: ${BASE_DIR}/medical_report_pipeline_kafka.py (you said this is correct)"
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  log "ERROR: DATA_ROOT not found: ${DATA_ROOT}"
  exit 1
fi

# Kill any previous instances started by pidfiles (only)
kill_pidfile "${KAFKA_PIDFILE}"
kill_pidfile "${ZK_PIDFILE}"

# If ports are still held, refuse (avoids killing other people's services)
if is_listening "${ZK_PORT}" || is_listening "${KAFKA_PORT}"; then
  log "Ports already in use. Current listeners:"
  ss -ltnp | egrep ":${ZK_PORT}|:${KAFKA_PORT}" || true
  log "Refusing to proceed to avoid colliding with another running broker."
  log "If this is yours, kill those PIDs and rerun."
  exit 1
fi

############################################
# ZooKeeper (per-run props)
############################################
log "Writing ZooKeeper properties to ${ZK_PROPS}"
cp "${ZK_PROPS_SRC}" "${ZK_PROPS}"
sed -i "s|^dataDir=.*|dataDir=${ZK_DATADIR}|g" "${ZK_PROPS}"
if grep -q "^clientPort=" "${ZK_PROPS}"; then
  sed -i "s|^clientPort=.*|clientPort=${ZK_PORT}|g" "${ZK_PROPS}"
else
  echo "clientPort=${ZK_PORT}" >> "${ZK_PROPS}"
fi

log "Starting ZooKeeper..."
"${KAFKA_DIR}/bin/zookeeper-server-start.sh" "${ZK_PROPS}" > "${ZK_LOG}" 2>&1 &
echo $! > "${ZK_PIDFILE}"
wait_for_port "${ZK_PORT}" "ZooKeeper" 60 1

############################################
# Kafka broker (per-run props)
############################################
log "Writing Kafka broker properties to ${KAFKA_PROPS}"
cp "${KAFKA_PROPS_SRC}" "${KAFKA_PROPS}"

# zookeeper.connect
if grep -q "^zookeeper.connect=" "${KAFKA_PROPS}"; then
  sed -i "s|^zookeeper.connect=.*|zookeeper.connect=${LISTEN_HOST}:${ZK_PORT}|g" "${KAFKA_PROPS}"
else
  echo "zookeeper.connect=${LISTEN_HOST}:${ZK_PORT}" >> "${KAFKA_PROPS}"
fi

# log.dirs
if grep -q "^log.dirs=" "${KAFKA_PROPS}"; then
  sed -i "s|^log.dirs=.*|log.dirs=${KAFKA_LOGDIR}|g" "${KAFKA_PROPS}"
else
  echo "log.dirs=${KAFKA_LOGDIR}" >> "${KAFKA_PROPS}"
fi

# listeners + advertised.listeners on localhost
if grep -q "^listeners=" "${KAFKA_PROPS}"; then
  sed -i "s|^listeners=.*|listeners=PLAINTEXT://${LISTEN_HOST}:${KAFKA_PORT}|g" "${KAFKA_PROPS}"
else
  echo "listeners=PLAINTEXT://${LISTEN_HOST}:${KAFKA_PORT}" >> "${KAFKA_PROPS}"
fi
if grep -q "^advertised.listeners=" "${KAFKA_PROPS}"; then
  sed -i "s|^advertised.listeners=.*|advertised.listeners=PLAINTEXT://${LISTEN_HOST}:${KAFKA_PORT}|g" "${KAFKA_PROPS}"
else
  echo "advertised.listeners=PLAINTEXT://${LISTEN_HOST}:${KAFKA_PORT}" >> "${KAFKA_PROPS}"
fi

# Start Kafka
log "Starting Kafka broker..."
: > "${KAFKA_LOG}"
"${KAFKA_DIR}/bin/kafka-server-start.sh" "${KAFKA_PROPS}" >> "${KAFKA_LOG}" 2>&1 &
echo $! > "${KAFKA_PIDFILE}"

wait_for_port "${KAFKA_PORT}" "Kafka" 60 1

# Strong readiness check
log "Checking broker API versions..."
"${KAFKA_DIR}/bin/kafka-broker-api-versions.sh" --bootstrap-server "${LISTEN_HOST}:${KAFKA_PORT}" >/dev/null

############################################
# Topic ensure
############################################
log "Ensuring topic exists: ${TOPIC}"
"${KAFKA_DIR}/bin/kafka-topics.sh" \
  --bootstrap-server "${LISTEN_HOST}:${KAFKA_PORT}" \
  --create --if-not-exists \
  --topic "${TOPIC}" \
  --partitions "${PARTITIONS}" \
  --replication-factor "${REPLICATION}" >/dev/null

log "Topic description:"
"${KAFKA_DIR}/bin/kafka-topics.sh" \
  --bootstrap-server "${LISTEN_HOST}:${KAFKA_PORT}" \
  --describe --topic "${TOPIC}" || true

############################################
# Producer
############################################
log "Kafka is up. Running producer..."

PRODUCER_ARGS=(--data-root "${DATA_ROOT}" --bootstrap "${LISTEN_HOST}:${KAFKA_PORT}" --topic "${TOPIC}" --limit "${LIMIT_MESSAGES}")
if [[ "${DRY_RUN_PRODUCER}" == "1" ]]; then
  PRODUCER_ARGS+=(--dry-run)
fi

python "${PRODUCER_PY}" "${PRODUCER_ARGS[@]}"

log "Producer done."

############################################
# Spark pipeline
############################################
log "Running Spark pipeline: ${PIPELINE_PY}"

# If you’re on NYU HPC and need a specific Spark, ensure spark-submit is in PATH.
# Otherwise call the full path to spark-submit.
spark-submit \
  --packages "${SPARK_KAFKA_PACKAGES}" \
  "${PIPELINE_PY}" \
  --use-kafka \
  --bootstrap "${LISTEN_HOST}:${KAFKA_PORT}" \
  --topic "${TOPIC}"

log "Pipeline finished successfully."
log "Done."
