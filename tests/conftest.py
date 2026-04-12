import json
import boto3
import pytest
from moto import mock_aws
from botocore.client import Config

BUCKET = "weather-raw"
ENDPOINT = "http://localhost:9000"

SAMPLE_RAW = {
    "latitude": 51.5085,
    "longitude": -0.1257,
    "timezone": "Europe/London",
    "current": {
        "time": "2024-04-12T14:00",
        "temperature_2m": 12.3,
        "relative_humidity_2m": 78,
        "wind_speed_10m": 14.2,
        "weather_code": 3,
        "apparent_temperature": 10.1,
    },
    "_city_name": "London",
    "_fetched_at": "2024-04-12T14:30:00+00:00",
}

@pytest.fixture
def aws_credentials(monkeypatch):
    # moto requires these to be set, values don't matter
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test")
    monkeypatch.setenv("MINIO_BUCKET", BUCKET)


@pytest.fixture
def s3(aws_credentials):
    # mock_aws intercepts all boto3 calls for the duration of the test
    with mock_aws():
        client = boto3.client(
            "s3",
            region_name="us-east-1",
        )
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture
def s3_with_raw(s3):
    # pre-populate bucket with one raw JSON file
    s3.put_object(
        Bucket=BUCKET,
        Key="raw/london/20240412/120000.json",
        Body=json.dumps(SAMPLE_RAW),
        ContentType="application/json",
    )
    yield s3