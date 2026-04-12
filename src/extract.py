# src/extract.py
import json
import os
import boto3
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from botocore.client import Config

load_dotenv()

# MinIO Config
ENDPOINT   = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "annu")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "annu1234")
BUCKET     = os.getenv("MINIO_BUCKET", "weather-raw")


# Open-Meteo needs lat/lon

CITIES = [
    {"name": "London",    "lat": 51.5085, "lon": -0.1257},
    {"name": "Tokyo",     "lat": 35.6895, "lon": 139.6917},
    {"name": "New York",  "lat": 40.7143, "lon": -74.0060},
    {"name": "Mumbai",    "lat": 19.0728, "lon": 72.8826},
    {"name": "Sao Paulo", "lat": -23.5475, "lon": -46.6361},
]

#What current weather fields we want ---
CURRENT_FIELDS = [
    "temperature_2m",         # air temp at 2m height in °C
    "relative_humidity_2m",   # humidity %
    "wind_speed_10m",         # wind speed at 10m in km/h
    "weather_code",           # WMO weather code (0=clear, 61=rain, etc.)
    "apparent_temperature",   # "feels like" temp in °C
]


def get_s3_client():
    """
    boto3 S3 client pointed at MinIO.
    endpoint_url is the key — redirects all AWS calls to localhost.
    """
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(s3_client, bucket: str):
    """Create bucket if it doesn't exist yet."""
    existing = [b["Name"] for b in s3_client.list_buckets()["Buckets"]]
    if bucket not in existing:
        s3_client.create_bucket(Bucket=bucket)
        print(f"Created bucket: {bucket}")


def fetch_weather(city: dict) -> dict:
    """
    Call Open-Meteo for a single city.
    
    We attach the city name + fetch timestamp ourselves since
    Open-Meteo's response only contains coordinates, not names.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "current": ",".join(CURRENT_FIELDS),
        "timezone": "auto",   # returns time in the city's local timezone
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Enrich with metadata we control
    data["_city_name"] = city["name"]
    data["_fetched_at"] = datetime.now(timezone.utc).isoformat()

    return data


def save_to_minio(s3_client, data: dict, city_name: str) -> str:
    """
    Upload JSON to MinIO under a partitioned path:
        raw/{city}/{YYYYMMDD}/{timestamp}.json

    Why partition by date?
    In real data lakes, date partitioning lets you query
    'all data from 2024-03-01' without scanning everything.
    """
    now = datetime.now(timezone.utc)
    city_slug = city_name.replace(" ", "_").lower()
    date_partition = now.strftime("%Y%m%d")
    timestamp = now.strftime("%H%M%S")

    key = f"raw/{city_slug}/{date_partition}/{timestamp}.json"

    s3_client.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json",
    )
    print(f"Saved to s3://{BUCKET}/{key}")
    return key


def run_extract() -> list[dict]:
    """
    Main entry point.
    Fetches all cities and stores raw JSON in MinIO.
    Returns a list of result dicts (used by Airflow later).
    """
    s3 = get_s3_client()
    ensure_bucket(s3, BUCKET)

    results = []
    for city in CITIES:
        print(f"Fetching: {city['name']}")
        try:
            data = fetch_weather(city)
            key  = save_to_minio(s3, data, city["name"])
            results.append({"city": city["name"], "key": key, "status": "ok"})
        except Exception as e:
            print(f"Failed for {city['name']}: {e}")
            results.append({"city": city["name"], "status": "error", "error": str(e)})

    return results


if __name__ == "__main__":
    run_extract()