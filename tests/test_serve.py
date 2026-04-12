import io
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

SAMPLE_AGGREGATED = [
    {
        "city": "London",
        "avg_temp_c": 12.3,
        "min_temp_c": 12.3,
        "max_temp_c": 12.3,
        "avg_feels_like_c": 10.1,
        "avg_humidity_pct": 78.0,
        "avg_wind_speed_kmh": 14.2,
        "weather_category": "Cloudy",
        "timezone": "Europe/London",
        "record_count": 1,
    }
]


def make_parquet_bytes(records: list[dict]) -> bytes:
    table = pa.Table.from_pylist(records)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def client():
    from src.serve import app
    return TestClient(app)


@pytest.fixture
def mock_s3_with_data():
    parquet_bytes = make_parquet_bytes(SAMPLE_AGGREGATED)
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": io.BytesIO(parquet_bytes)
    }
    mock_s3.list_buckets.return_value = {"Buckets": []}
    mock_s3.list_objects_v2.return_value = {
        "Contents": [{"Key": "aggregated/summary_20240412.parquet"}]
    }
    return mock_s3


def test_health_returns_200(client, mock_s3_with_data):
    with patch("src.serve.s3_client", mock_s3_with_data):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_summary_valid_date(client, mock_s3_with_data):
    with patch("src.serve.s3_client", mock_s3_with_data):
        response = client.get("/summary/20240412")
    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "20240412"
    assert data["city_count"] == 1
    assert data["cities"][0]["city"] == "London"


def test_summary_invalid_date_format(client):
    response = client.get("/summary/april12")
    assert response.status_code == 400


def test_city_summary_found(client, mock_s3_with_data):
    with patch("src.serve.s3_client", mock_s3_with_data):
        response = client.get("/summary/20240412/city/london")
    assert response.status_code == 200
    assert response.json()["city"] == "London"


def test_city_summary_not_found(client, mock_s3_with_data):
    with patch("src.serve.s3_client", mock_s3_with_data):
        response = client.get("/summary/20240412/city/berlin")
    assert response.status_code == 404


def test_latest_returns_most_recent(client, mock_s3_with_data):
    with patch("src.serve.s3_client", mock_s3_with_data):
        response = client.get("/latest")
    assert response.status_code == 200
    assert response.json()["date"] == "20240412"