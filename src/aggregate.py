# src/aggregate.py
import io
import os
from datetime import datetime, timezone

import boto3
import pyarrow.parquet as pq
import pyarrow as pa
from botocore.client import Config
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

load_dotenv()

#MinIO config
ENDPOINT   = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "annu")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "annu1234")
BUCKET     = os.getenv("MINIO_BUCKET", "weather-raw")



# WMO Weather Code -> human readable category
# Open-Meteo uses WMO codes: 0=Clear, 1-3=Cloudy, 51-67=Rain, 71-77=Snow
# Full table: https://open-meteo.com/en/docs#weathervariables

def wmo_code_to_category(code: int) -> str:
    if code == 0:
        return "Clear"
    elif code in (1, 2, 3):
        return "Cloudy"
    elif code in range(51, 68):
        return "Rain"
    elif code in range(71, 78):
        return "Snow"
    elif code in range(95, 100):
        return "Thunderstorm"
    else:
        return "Other"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def read_parquet_from_minio(s3_client, date_str: str) -> bytes:
    """
    Download the processed Parquet file from MinIO into memory.
    Same pattern as transform.py but in reverse — get_object → read bytes.
    """
    key = f"processed/weather_{date_str}.parquet"
    print(f"Reading: s3://{BUCKET}/{key}")
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    return response["Body"].read()


def create_spark_session() -> SparkSession:
    """
    Create a local SparkSession.

    local[*] = run on this machine using all CPU cores.
    In production you'd replace this with:
        .master("spark://cluster-host:7077")
    or submit via: spark-submit --master yarn

    We lower the shuffle partitions from 200 (default) to 5
    because we only have 5 cities — 200 empty partitions would
    waste time during the shuffle/reduce phase.
    """
    return (
        SparkSession.builder
        .appName("WeatherETL-Aggregate")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "5")
  
        .config("spark.driver.extraJavaOptions", "-Dlog4j.rootCategory=WARN")
        .getOrCreate()
    )


def bytes_to_spark_df(spark: SparkSession, parquet_bytes: bytes):
    """
    Load Parquet bytes into a Spark DataFrame.

    Strategy:
    1. Write bytes to a local temp file   <- Spark needs a file path or HDFS URI
    2. spark.read.parquet(path)           <- Spark reads it in parallel partitions
    3. Delete temp file                   <- cleanup

    Why not read directly from MinIO?
    We *could* configure Spark with hadoop-aws + MinIO credentials,
    but that requires extra JARs and config. For this project, the
    in-memory -> temp file approach keeps setup simple while teaching
    the same Spark concepts.
    """
    tmp_path = "/tmp/weather_input.parquet"
    with open(tmp_path, "wb") as f:
        f.write(parquet_bytes)

    df = spark.read.parquet(tmp_path)
    return df

# MAP step — add a derived column to each row independently.

#apply a transformation to every record without any grouping or aggregation yet.
# F.udf = User Defined Function: wraps a plain Python function so
# Spark can apply it across all partitions in parallel.

def map_add_weather_category(df):
    """
    Add a 'weather_category' column derived from 'weather_code'.

    Why UDF here instead of a native Spark function?
    The WMO code -> category mapping is custom logic with range checks.
    Native Spark functions (F.when, F.otherwise) could do it but would
    be very verbose. UDFs are the escape hatch for arbitrary Python logic.

    Trade-off: UDFs are slower than native Spark functions because
    data must be serialized between the JVM and Python interpreter.
    For small data like ours this doesn't matter.
    """
    category_udf = F.udf(wmo_code_to_category, StringType())
    return df.withColumn("weather_category", category_udf(F.col("weather_code")))




 #after the shuffle groups all rows by city, we collapse each group into a single summary row.

def reduce_aggregate_by_city(df):
    """
    Group by city and compute summary statistics.

    F.round(..., 2)    -> round to 2 decimal places
    F.avg(...)         -> mean across all rows in the group
    F.min / F.max      -> min / max in the group
    F.first(...)       -> take the first value (for non-aggregatable cols)
    F.count(...)       -> how many rows went into this group

    .alias(...)        -> rename the output column

    The result: one row per city with all stats collapsed.
    """
    return df.groupBy("city").agg(
        F.round(F.avg("temperature_c"),  2).alias("avg_temp_c"),
        F.round(F.min("temperature_c"),  2).alias("min_temp_c"),
        F.round(F.max("temperature_c"),  2).alias("max_temp_c"),
        F.round(F.avg("feels_like_c"),   2).alias("avg_feels_like_c"),
        F.round(F.avg("humidity_pct"),   2).alias("avg_humidity_pct"),
        F.round(F.avg("wind_speed_kmh"), 2).alias("avg_wind_speed_kmh"),
        F.first("weather_category")        .alias("weather_category"),
        F.first("timezone")                .alias("timezone"),
        F.count("city")                    .alias("record_count"),
    )


def spark_df_to_parquet_bytes(df) -> bytes:
    """
    Convert Spark DataFrame → Parquet bytes for MinIO upload.

    Spark normally writes Parquet as a folder of part files
    (part-00000.parquet, part-00001.parquet, ...) — one per partition.
    That's great for distributed storage but awkward for a small result.

    Strategy:
    1. df.toPandas()   -> pull entire result to driver as pandas DataFrame
    2. Convert to PyArrow Table
    3. Write to BytesIO buffer as a single Parquet file
    4. Return bytes

    This is fine for small aggregation results (5 rows).
    For large outputs you'd write directly to HDFS/S3 as part files.
    """
    pandas_df = df.toPandas()
    table = pa.Table.from_pandas(pandas_df)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")
    buffer.seek(0)
    return buffer.read()


def save_aggregated_to_minio(s3_client, parquet_bytes: bytes, date_str: str) -> str:
    """Save the aggregated summary Parquet to MinIO under aggregated/ prefix."""
    key = f"aggregated/summary_{date_str}.parquet"
    s3_client.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=parquet_bytes,
        ContentType="application/octet-stream",
    )
    print(f" Aggregated saved → s3://{BUCKET}/{key}")
    return key


def run_aggregate(date_str: str = None) -> dict:
    """
    Main entry point for the aggregation phase.
    Reads processed Parquet -> MapReduce via Spark → saves summary.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    print(f"Running PySpark aggregation for date: {date_str}")

    s3 = get_s3_client()

    # 1. Pull Parquet from MinIO
    parquet_bytes = read_parquet_from_minio(s3, date_str)

    # 2. Boot Spark
    spark = create_spark_session()
    print(f"Spark version: {spark.version}")

    # 3. Load into DataFrame
    df = bytes_to_spark_df(spark, parquet_bytes)
    print(f"Schema:")
    df.printSchema()
    print(f"Total records: {df.count()}")

    # 4. MAP — add weather category column
    df_mapped = map_add_weather_category(df)

    # 5. REDUCE — group by city and aggregate
    df_aggregated = reduce_aggregate_by_city(df_mapped)

    # 6. Show results in terminal (great for debugging)
    print("\nAggregation Results:")
    df_aggregated.show(truncate=False)

    # 7. Serialize and upload
    result_bytes = spark_df_to_parquet_bytes(df_aggregated)
    key = save_aggregated_to_minio(s3, result_bytes, date_str)

    # 8. Stop Spark (releases all JVM resources)
    spark.stop()

    return {
        "date": date_str,
        "aggregated_key": key,
        "status": "ok",
    }


if __name__ == "__main__":
    result = run_aggregate()
    print(f"\nDone: {result}")