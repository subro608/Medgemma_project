import os
from pyspark.sql import SparkSession

print(">>> STARTING TEST SCRIPT <<<")

USER = "sd5963"
SCRATCH_DIR = f"/scratch/{USER}/bigdata_project"

# Use a dedicated ivy cache for this job
SPARK_IVY_DIR = os.path.join(SCRATCH_DIR, "spark_ivy_cache_4_0_1")
os.makedirs(SPARK_IVY_DIR, exist_ok=True)
print("Using Ivy cache:", SPARK_IVY_DIR)

base_partitions = 4  # just for testing

spark = (
    SparkSession.builder
    .appName("KafkaJarTest")
    .master(f"local[{base_partitions}]")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.jars.ivy", SPARK_IVY_DIR)
    .config("spark.jars", "")  # force-clear default jars
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,"
        "org.apache.spark:spark-token-provider-kafka-0-10_2.13:4.0.1"
    )
    .getOrCreate()
)

print(">>> Spark version:", spark.version)
print(">>> spark.jars.packages:", spark.conf.get("spark.jars.packages"))

# simple operation to confirm session works
df = spark.range(5)
df.show()

print(">>> TEST COMPLETE <<<")
