"""Tests for the dataset download script's path handling (no network)."""
from pathlib import Path

from data import download_dataset


def test_outputs_are_anchored_to_the_data_directory():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    assert download_dataset.DATA_DIR == data_dir
    assert download_dataset.CSV_PATH.parent == data_dir
    assert download_dataset.PARQUET_PATH.parent == data_dir / "parquet_cache"
    assert download_dataset.CSV_PATH.is_absolute()
