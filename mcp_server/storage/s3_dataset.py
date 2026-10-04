"""S3 Express One Zone Parquet reader for the 90K waveguide dataset."""
import logging
import os
import time
from typing import Optional

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd

from mcp_server.config import (
    S3_BUCKET_NAME,
    S3_DATASET_KEY,
    S3_REGION,
    PARQUET_LOCAL_CACHE_DIR,
    DATASET_PATH,
)

logger = logging.getLogger(__name__)

# After a failed S3 attempt, go straight to local data for this long instead
# of paying the credential/connection timeout on every call.
S3_RETRY_COOLDOWN_S = 300.0


class S3DatasetReader:
    """Reads the photonic waveguide Parquet dataset from S3 Express One Zone.

    Falls back to the local Parquet cache, then to the CSV.
    """

    def __init__(
        self,
        bucket: str = S3_BUCKET_NAME,
        key: str = S3_DATASET_KEY,
        region: str = S3_REGION,
        local_fallback: str = DATASET_PATH,
    ):
        self.s3_uri = f"s3://{bucket}/{key}"
        self.region = region
        self.local_fallback = local_fallback
        self._table_cache: Optional[pa.Table] = None
        self._s3_fs = None
        self._s3_available: Optional[bool] = None  # unknown until the first attempt
        self._s3_retry_after = 0.0

    def _s3_filesystem(self):
        """Return the shared S3 filesystem, or None while S3 is known to be unavailable."""
        if self._s3_available is False and time.monotonic() < self._s3_retry_after:
            return None
        if self._s3_fs is None:
            try:
                import s3fs
                self._s3_fs = s3fs.S3FileSystem(
                    anon=False, client_kwargs={"region_name": self.region}
                )
            except Exception as e:
                self._mark_s3_unavailable(e)
                return None
        return self._s3_fs

    def _mark_s3_unavailable(self, error: Exception) -> None:
        logger.warning(
            "[S3DatasetReader] S3 unavailable (%s). Using local data for the next %.0fs.",
            error, S3_RETRY_COOLDOWN_S,
        )
        self._s3_available = False
        self._s3_retry_after = time.monotonic() + S3_RETRY_COOLDOWN_S

    def _try_s3_read(
        self, columns: Optional[list[str]] = None, limit: Optional[int] = None
    ) -> Optional[pa.Table]:
        fs = self._s3_filesystem()
        if fs is None:
            return None
        try:
            if limit is None:
                table = pq.read_table(self.s3_uri, columns=columns, filesystem=fs)
            else:
                scanner = ds.dataset(self.s3_uri, format="parquet", filesystem=fs).scanner(columns=columns)
                table = scanner.head(limit)
            self._s3_available = True
            return table
        except Exception as e:
            self._mark_s3_unavailable(e)
            return None

    def _try_local_read(
        self, columns: Optional[list[str]] = None, limit: Optional[int] = None
    ) -> pa.Table:
        local_parquet = os.path.join(PARQUET_LOCAL_CACHE_DIR, "dataset.parquet")
        if os.path.exists(local_parquet):
            if limit is None:
                return pq.read_table(local_parquet, columns=columns)
            return ds.dataset(local_parquet, format="parquet").scanner(columns=columns).head(limit)
        if os.path.exists(self.local_fallback):
            usecols = None
            if columns:
                header = pd.read_csv(self.local_fallback, nrows=0).columns
                usecols = [c for c in columns if c in header]
            df = pd.read_csv(self.local_fallback, usecols=usecols, nrows=limit)
            if usecols is not None:
                df = df[usecols]
            return pa.Table.from_pandas(df, preserve_index=False)
        raise FileNotFoundError(
            "No dataset found. Run 'python data/download_dataset.py' first."
        )

    def read_columns(self, columns: list[str], limit: Optional[int] = None) -> pa.Table:
        """Read specific columns (columnar read), optionally only the first `limit` rows."""
        if self._table_cache is not None:
            present = [c for c in columns if c in self._table_cache.column_names]
            table = self._table_cache.select(present)
            return table if limit is None else table.slice(0, limit)
        table = self._try_s3_read(columns=columns, limit=limit)
        if table is None:
            table = self._try_local_read(columns=columns, limit=limit)
        return table

    def read_full(self) -> pa.Table:
        """Read the entire dataset (cached for the life of the reader)."""
        if self._table_cache is not None:
            return self._table_cache
        table = self._try_s3_read()
        if table is None:
            table = self._try_local_read()
        self._table_cache = table
        return table

    def to_pandas(
        self, columns: Optional[list[str]] = None, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Convenience method: read dataset as a pandas DataFrame."""
        if columns:
            return self.read_columns(columns, limit=limit).to_pandas()
        table = self.read_full()
        if limit is not None:
            table = table.slice(0, limit)
        return table.to_pandas()

    def save_local_parquet(self, output_dir: str = PARQUET_LOCAL_CACHE_DIR) -> str:
        """Convert the local CSV to Parquet for faster subsequent reads."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "dataset.parquet")
        table = self.read_full()
        pq.write_table(table, output_path, compression="snappy")
        print(f"Parquet cache saved to {output_path}")
        return output_path
