"""S3 Express One Zone Parquet reader for the 90K waveguide dataset."""
import os
from typing import Optional

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

from mcp_server.config import (
    S3_BUCKET_NAME,
    S3_DATASET_KEY,
    S3_REGION,
    PARQUET_LOCAL_CACHE_DIR,
)


class S3DatasetReader:
    """Reads the photonic waveguide Parquet dataset from S3 Express One Zone."""

    def __init__(
        self,
        bucket: str = S3_BUCKET_NAME,
        key: str = S3_DATASET_KEY,
        region: str = S3_REGION,
        local_fallback: str = "data/SiN_Photonic_Waveguide_Loss_Efficiency.csv",
    ):
        self.s3_uri = f"s3://{bucket}/{key}"
        self.region = region
        self.local_fallback = local_fallback
        self._table_cache: Optional[pa.Table] = None

    def _try_s3_read(self, columns: Optional[list[str]] = None) -> Optional[pa.Table]:
        try:
            import s3fs
            fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": self.region})
            return pq.read_table(self.s3_uri, columns=columns, filesystem=fs)
        except Exception as e:
            print(f"[S3DatasetReader] S3 read failed ({e}). Falling back to local.")
            return None

    def _try_local_read(self, columns: Optional[list[str]] = None) -> pa.Table:
        local_parquet = os.path.join(PARQUET_LOCAL_CACHE_DIR, "dataset.parquet")
        if os.path.exists(local_parquet):
            return pq.read_table(local_parquet, columns=columns)
        if os.path.exists(self.local_fallback):
            df = pd.read_csv(self.local_fallback)
            if columns:
                df = df[[c for c in columns if c in df.columns]]
            return pa.Table.from_pandas(df)
        raise FileNotFoundError(
            f"No dataset found. Run 'python data/download_dataset.py' first."
        )

    def read_columns(self, columns: list[str]) -> pa.Table:
        """Read specific columns from the dataset (columnar read)."""
        table = self._try_s3_read(columns=columns)
        if table is None:
            table = self._try_local_read(columns=columns)
        return table

    def read_full(self) -> pa.Table:
        """Read the entire dataset."""
        if self._table_cache is not None:
            return self._table_cache
        table = self._try_s3_read()
        if table is None:
            table = self._try_local_read()
        self._table_cache = table
        return table

    def to_pandas(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Convenience method: read dataset as a pandas DataFrame."""
        if columns:
            return self.read_columns(columns).to_pandas()
        return self.read_full().to_pandas()

    def save_local_parquet(self, output_dir: str = PARQUET_LOCAL_CACHE_DIR) -> str:
        """Convert the local CSV to Parquet for faster subsequent reads."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "dataset.parquet")
        table = self.read_full()
        pq.write_table(table, output_path, compression="snappy")
        print(f"Parquet cache saved to {output_path}")
        return output_path
