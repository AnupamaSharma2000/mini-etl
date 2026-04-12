# src/transform.py
import io
import json
import os
from datetime import datetime, timezone

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()

#MinIO config
ENDPOINT   = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "annu")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "annu1234")
BUCKET     = os.getenv("MINIO_BUCKET", "weather-raw")

# Output goes to a separate prefix (just like a separate "folder" in S3)
PROCESSED_PREFIX = "processed"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def list_raw_keys(s3_client, date_str: str) -> list[str]:
    """
    List all raw JSON keys for a given date.

    S3/MinIO doesn't have real folders — everything is a flat key.
    The '/' in key names is just a naming convention.
    list_objects_v2 with a Prefix acts like 'find all files in this folder'.

    date_str format: YYYYMMDD  e.g. "20240412"
    """
    keys = []

    # We use a paginator because list_objects_v2 returns max 1000 keys at a time.
    # A paginator automatically handles the continuation token for us.
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"raw/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Filter to only today's date partition
            if date_str in key and key.endswith(".json"):
                keys.append(key)

    return keys


def read_json_from_minio(s3_client, key: str) -> dict:
    """
    Download a single JSON object from MinIO and parse it.

    get_object returns a streaming body — we read it all at once
    since our files are small. For very large files we would stream.
    """
    response = s3_client.get_object(Bucket=BUCKET, Key=key)
    raw_bytes = response["Body"].read()
    return json.loads(raw_bytes)


def flatten_record(raw: dict) -> dict:
    """
    Flatten Open-Meteo's nested JSON into a single-level dict.

    Input shape:
    {
        "latitude": 51.5,
        "longitude": -0.12,
        "timezone": "Europe/London",
        "current": {
            "time": "2024-04-12T14:00",
            "temperature_2m": 12.3,
            "relative_humidity_2m": 78,
            "wind_speed_10m": 14.2,
            "weather_code": 3,
            "apparent_temperature": 10.1
        },
        "_city_name": "London",
        "_fetched_at": "2024-04-12T14:30:00+00:00"
    }

    Output shape (flat):
    {
        "city":               "London",
        "latitude":           51.5,
        "longitude":          -0.12,
        "timezone":           "Europe/London",
        "observation_time":   "2024-04-12T14:00",
        "temperature_c":      12.3,
        "humidity_pct":       78,
        "wind_speed_kmh":     14.2,
        "weather_code":       3,
        "feels_like_c":       10.1,
        "fetched_at":         "2024-04-12T14:30:00+00:00"
    }
    """
    current = raw.get("current", {})

    return {
        "city":             raw.get("_city_name", "unknown"),
        "latitude":         float(raw.get("latitude", 0.0)),
        "longitude":        float(raw.get("longitude", 0.0)),
        "timezone":         raw.get("timezone", ""),
        "observation_time": current.get("time", ""),
        "temperature_c":    float(current.get("temperature_2m", 0.0)),
        "humidity_pct":     int(current.get("relative_humidity_2m", 0)),
        "wind_speed_kmh":   float(current.get("wind_speed_10m", 0.0)),
        "weather_code":     int(current.get("weather_code", 0)),
        "feels_like_c":     float(current.get("apparent_temperature", 0.0)),
        "fetched_at":       raw.get("_fetched_at", ""),
    }


# Explicit PyArrow schema
# Why define this manually instead of letting PyArrow infer it?
# 1. Inference can guess wrong (e.g. treat an int column as float)
# 2. Explicit schema = contract. If upstream data changes shape,
#    we get a clear error here instead of silent corruption downstream.
PARQUET_SCHEMA = pa.schema([
    pa.field("city",             pa.string()),
    pa.field("latitude",         pa.float64()),
    pa.field("longitude",        pa.float64()),
    pa.field("timezone",         pa.string()),
    pa.field("observation_time", pa.string()),
    pa.field("temperature_c",    pa.float64()),
    pa.field("humidity_pct",     pa.int32()),
    pa.field("wind_speed_kmh",   pa.float64()),
    pa.field("weather_code",     pa.int32()),
    pa.field("feels_like_c",     pa.float64()),
    pa.field("fetched_at",       pa.string()),
])


def records_to_parquet_bytes(records: list[dict]) -> bytes:
    """
    Convert a list of flat dicts-> PyArrow Table -> Parquet bytes in memory.

    We write to an in-memory buffer (io.BytesIO) instead of a file
    so we can upload directly to MinIO without touching local disk.

    Why in-memory?
    - Avoids temp file cleanup
    - Works in serverless / container environments with no writable disk
    - Slightly faster (no filesystem round-trip)
    """
    # Step 1: Build column arrays from the list of dicts
    # PyArrow works column-first — we give it each column as an array
    table = pa.Table.from_pylist(records, schema=PARQUET_SCHEMA)

    # Step 2: Write to in-memory bytes buffer
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        compression="snappy",   # Snappy = fast compress/decompress, decent ratio
                                 # Alternative: "gzip" (smaller file, slower)
    )

    # Step 3: Return raw bytes (ready to upload)
    buffer.seek(0)
    return buffer.read()


def save_parquet_to_minio(s3_client, parquet_bytes: bytes, date_str: str) -> str:
    """
    Upload the Parquet file to MinIO under processed/ prefix.
    One file per pipeline run, named by date.
    """
    key = f"{PROCESSED_PREFIX}/weather_{date_str}.parquet"

    s3_client.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=parquet_bytes,
        ContentType="application/octet-stream",
    )
    print(f"Parquet saved → s3://{BUCKET}/{key}")
    return key


def run_transform(date_str: str = None) -> dict:
    """
    Main entry point.
    Reads all raw JSONs for a date, flattens them, writes one Parquet file.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    print(f"Transforming data for date: {date_str}")

    s3 = get_s3_client()

    # 1. Find all raw JSON files for today
    keys = list_raw_keys(s3, date_str)
    if not keys:
        raise ValueError(f"No raw JSON files found for date {date_str}")
    print(f"Found {len(keys)} raw file(s)")

    # 2. Read and flatten each one
    records = []
    for key in keys:
        raw = read_json_from_minio(s3, key)
        flat = flatten_record(raw)
        records.append(flat)
        print(f"Flattened: {flat['city']}")

    # 3. Convert to Parquet bytes
    parquet_bytes = records_to_parquet_bytes(records)
    print(f" Parquet size: {len(parquet_bytes):,} bytes")

    # 4. Upload to MinIO
    key = save_parquet_to_minio(s3, parquet_bytes, date_str)

    return {
        "date": date_str,
        "records": len(records),
        "parquet_key": key,
        "status": "ok",
    }


if __name__ == "__main__":
    result = run_transform()
    print(f"\nDone: {result}")