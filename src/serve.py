# src/serve.py
import io
import os
from datetime import datetime, timezone
from typing import Optional

import boto3
import pyarrow.parquet as pq
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

#MinIO config
ENDPOINT   = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "annu")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "annu1234")
BUCKET     = os.getenv("MINIO_BUCKET", "weather-raw")


# FastAPI app instance
app = FastAPI(
    title="Weather ETL API",
    description="Serves aggregated weather data processed by the Mini ETL Pipeline.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS Middleware
#
# Allows browsers from any origin to call this API.
# In production we would restrict allow_origins to our frontend domain.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# Pydantic Response Models
# These define the exact shape of our API responses.
# FastAPI reads these at startup and builds the /docs schema from them.
# If our code returns a dict missing a required field--> FastAPI raises
# a clear validation error instead of sending malformed JSON silently.


class CityWeather(BaseModel):
    city: str
    avg_temp_c: float
    min_temp_c: float
    max_temp_c: float
    avg_feels_like_c: float
    avg_humidity_pct: float
    avg_wind_speed_kmh: float
    weather_category: str
    timezone: str
    record_count: int


class SummaryResponse(BaseModel):
    date: str
    city_count: int
    generated_at: str
    cities: list[CityWeather]


class HealthResponse(BaseModel):
    status: str
    minio_connected: bool
    timestamp: str



# S3 Client — module-level singleton
#
# We create one client when the app starts, reuse it across requests.
# Creating a new boto3 client per request would be wasteful (TCP handshake
# + credential resolution on every call).

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

s3_client = get_s3_client()



# Helper: read aggregated Parquet from MinIO--> pandas DataFrame
#
# Why pandas here instead of PySpark?
# The aggregated result is tiny (5 rows × 9 columns).
# Spinning up a SparkSession just to read 5 rows would add 20s latency
# to every API request. Pandas reads it in milliseconds.


def load_summary(date_str: str) -> pd.DataFrame:
    """
    Download aggregated Parquet for a given date and return as DataFrame.
    Raises HTTPException(404) if the file doesn't exist in MinIO.
    """
    key = f"aggregated/summary_{date_str}.parquet"

    try:
        response = s3_client.get_object(Bucket=BUCKET, Key=key)
        parquet_bytes = response["Body"].read()
    except ClientError as e:
        # ClientError code "NoSuchKey" means the file doesn't exist
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchKey":
            raise HTTPException(
                status_code=404,
                detail=f"No aggregated data found for date {date_str}. "
                       f"Run the ETL pipeline first."
            )
        # Any other S3 error → 500
        raise HTTPException(status_code=500, detail=f"MinIO error: {str(e)}")

    # Parse Parquet bytes → pandas DataFrame
    buffer = io.BytesIO(parquet_bytes)
    return pq.read_table(buffer).to_pandas()


def find_latest_date(s3_client) -> Optional[str]:
    """
    Scan the aggregated/ prefix and return the most recent date string.

    Since our keys look like: aggregated/summary_20240412.parquet
    we extract the date part and return the lexicographically largest one
    (which equals the most recent date in YYYYMMDD format).
    """
    try:
        response = s3_client.list_objects_v2(
            Bucket=BUCKET,
            Prefix="aggregated/summary_"
        )
        keys = [obj["Key"] for obj in response.get("Contents", [])]
        if not keys:
            return None

        # Extract date strings: "aggregated/summary_20240412.parquet" → "20240412"
        dates = [k.split("_")[1].replace(".parquet", "") for k in keys]
        return sorted(dates)[-1]   # latest = lexicographically largest YYYYMMDD

    except ClientError:
        return None


def df_to_city_list(df: pd.DataFrame) -> list[CityWeather]:
    """
    Convert a pandas DataFrame to a list of CityWeather Pydantic models.

    df.to_dict("records") -> list of dicts, one per row.
    We then unpack each dict into the Pydantic model.
    Pydantic validates types — if avg_temp_c is somehow a string,
    it either coerces it or raises a clear validation error.
    """
    records = df.to_dict("records")
    return [CityWeather(**row) for row in records]



# Routes


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Liveness check endpoint.
    Tries to list MinIO buckets to confirm connectivity.
    Returns 200 if healthy, still returns 200 but minio_connected=False
    if MinIO is unreachable (so load balancers don't kill the pod).
    """
    try:
        s3_client.list_buckets()
        minio_ok = True
    except Exception:
        minio_ok = False

    return HealthResponse(
        status="ok",
        minio_connected=minio_ok,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/summary/{date}", response_model=SummaryResponse)
async def get_summary(date: str):
    """
    Return aggregated weather summary for all cities on a given date.

    date format: YYYYMMDD  e.g. /summary/20240412

    FastAPI automatically:
    - Extracts {date} from the URL path
    - Validates it's a string
    - Documents this parameter in /docs
    """
    # Basic input validation — YYYYMMDD should be exactly 8 digits
    if not date.isdigit() or len(date) != 8:
        raise HTTPException(
            status_code=400,
            detail="Date must be in YYYYMMDD format e.g. 20240412"
        )

    df = load_summary(date)

    return SummaryResponse(
        date=date,
        city_count=len(df),
        generated_at=datetime.now(timezone.utc).isoformat(),
        cities=df_to_city_list(df),
    )


@app.get("/summary/{date}/city/{city_name}", response_model=CityWeather)
async def get_city_summary(date: str, city_name: str):
    """
    Return aggregated weather for a single city on a given date.

    city_name is case-insensitive: 'london', 'London', 'LONDON' all work.
    """
    if not date.isdigit() or len(date) != 8:
        raise HTTPException(status_code=400, detail="Date must be YYYYMMDD")

    df = load_summary(date)

    # Case-insensitive city match
    mask = df["city"].str.lower() == city_name.lower()
    matched = df[mask]

    if matched.empty:
        available = df["city"].tolist()
        raise HTTPException(
            status_code=404,
            detail=f"City '{city_name}' not found. Available: {available}"
        )

    return CityWeather(**matched.iloc[0].to_dict())


@app.get("/latest", response_model=SummaryResponse)
async def get_latest():
    """
    Convenience endpoint — returns the most recent aggregation available.
    No need to know the date. Useful for dashboards.
    """
    latest_date = find_latest_date(s3_client)

    if not latest_date:
        raise HTTPException(
            status_code=404,
            detail="No aggregated data found. Run the ETL pipeline first."
        )

    df = load_summary(latest_date)

    return SummaryResponse(
        date=latest_date,
        city_count=len(df),
        generated_at=datetime.now(timezone.utc).isoformat(),
        cities=df_to_city_list(df),
    )