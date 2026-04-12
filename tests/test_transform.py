import io
import json
import pytest
import pyarrow.parquet as pq
from src.transform import flatten_record, records_to_parquet_bytes, list_raw_keys

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


def test_flatten_record_structure():
    flat = flatten_record(SAMPLE_RAW)
    expected_keys = {
        "city", "latitude", "longitude", "timezone",
        "observation_time", "temperature_c", "humidity_pct",
        "wind_speed_kmh", "weather_code", "feels_like_c", "fetched_at",
    }
    assert set(flat.keys()) == expected_keys


def test_flatten_record_values():
    flat = flatten_record(SAMPLE_RAW)
    assert flat["city"] == "London"
    assert flat["temperature_c"] == 12.3
    assert flat["humidity_pct"] == 78
    assert flat["wind_speed_kmh"] == 14.2


def test_flatten_record_types():
    flat = flatten_record(SAMPLE_RAW)
    assert isinstance(flat["temperature_c"], float)
    assert isinstance(flat["humidity_pct"], int)
    assert isinstance(flat["city"], str)


def test_records_to_parquet_bytes_valid():
    flat = flatten_record(SAMPLE_RAW)
    parquet_bytes = records_to_parquet_bytes([flat])
    assert isinstance(parquet_bytes, bytes)
    assert len(parquet_bytes) > 0

    # verify it's actually readable Parquet
    table = pq.read_table(io.BytesIO(parquet_bytes))
    assert table.num_rows == 1
    assert "city" in table.schema.names
    assert "temperature_c" in table.schema.names


def test_list_raw_keys_filters_by_date(s3_with_raw):
    from unittest.mock import patch
    import src.transform as transform_module

    with patch.object(transform_module, "get_s3_client", return_value=s3_with_raw):
        keys = list_raw_keys(s3_with_raw, "20240412")

    assert len(keys) == 1
    assert "20240412" in keys[0]
    assert keys[0].endswith(".json")