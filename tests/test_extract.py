import json
import pytest
from moto import mock_aws
from unittest.mock import patch, MagicMock
from src.extract import ensure_bucket, save_to_minio, fetch_weather

BUCKET = "weather-raw"


def test_ensure_bucket_creates_if_missing(s3):
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    assert BUCKET in buckets


def test_ensure_bucket_idempotent(s3):
    from src.extract import ensure_bucket
    # calling twice should not raise
    ensure_bucket(s3, BUCKET)
    ensure_bucket(s3, BUCKET)
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    assert buckets.count(BUCKET) == 1


def test_save_to_minio_uploads_valid_json(s3):
    data = {"_city_name": "London", "current": {"temperature_2m": 12.3}}
    key = save_to_minio(s3, data, "London")

    response = s3.get_object(Bucket=BUCKET, Key=key)
    uploaded = json.loads(response["Body"].read())
    assert uploaded["_city_name"] == "London"
    assert "raw/london/" in key
    assert key.endswith(".json")


def test_fetch_weather_returns_enriched_data():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "latitude": 51.5,
        "longitude": -0.12,
        "current": {"temperature_2m": 12.3},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("src.extract.requests.get", return_value=mock_response):
        city = {"name": "London", "lat": 51.5, "lon": -0.12}
        result = fetch_weather(city)

    assert result["_city_name"] == "London"
    assert "_fetched_at" in result