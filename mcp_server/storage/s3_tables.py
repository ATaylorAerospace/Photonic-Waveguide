"""S3 Tables SQL query interface for the waveguide dataset."""
import logging
import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from mcp_server.config import (
    DATASET_PATH,
    PARQUET_LOCAL_CACHE_DIR,
    S3_BUCKET_NAME,
    S3_REGION,
)

logger = logging.getLogger(__name__)

S3_RETRY_COOLDOWN_S = 300.0


class S3TablesQuery:
    """SQL-like query interface over Parquet data in S3 Express One Zone."""

    def __init__(
        self,
        bucket: str = S3_BUCKET_NAME,
        region: str = S3_REGION,
        local_fallback: str = DATASET_PATH,
    ):
        self.s3_uri = f"s3://{bucket}/"
        self.region = region
        self.local_fallback = local_fallback
        self._dataset: ds.Dataset | None = None
        self._s3_available: bool | None = None
        self._s3_retry_after = 0.0

    def _get_dataset(self) -> ds.Dataset:
        if self._dataset is not None:
            return self._dataset
        if self._s3_available is not False or time.monotonic() >= self._s3_retry_after:
            try:
                import s3fs
                fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": self.region})
                self._dataset = ds.dataset(self.s3_uri, format="parquet", filesystem=fs)
                self._s3_available = True
                return self._dataset
            except Exception as e:
                logger.warning(
                    "[S3TablesQuery] S3 unavailable (%s). Using local data for the next %.0fs.",
                    e, S3_RETRY_COOLDOWN_S,
                )
                self._s3_available = False
                self._s3_retry_after = time.monotonic() + S3_RETRY_COOLDOWN_S
        local_parquet = os.path.join(PARQUET_LOCAL_CACHE_DIR, "dataset.parquet")
        if os.path.exists(local_parquet):
            self._dataset = ds.dataset(local_parquet, format="parquet")
        elif os.path.exists(self.local_fallback):
            df = pd.read_csv(self.local_fallback)
            table = pa.Table.from_pandas(df, preserve_index=False)
            self._dataset = ds.dataset(table)
        else:
            raise FileNotFoundError("No dataset available for S3 Tables queries.")
        return self._dataset

    def query(
        self,
        filter_expr: ds.Expression | None = None,
        columns: list[str] | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Execute a filtered query with predicate pushdown.

        Uses a scanner so the read stops as soon as `limit` rows are
        collected, instead of materializing every matching row first.
        """
        dataset = self._get_dataset()
        selected_columns = None
        if columns is not None:
            schema_names = set(dataset.schema.names)
            selected_columns = [c for c in columns if c in schema_names]
        scanner = dataset.scanner(filter=filter_expr, columns=selected_columns)
        return scanner.head(limit).to_pandas()

    def query_low_loss(
        self,
        max_loss_db_cm: float = 2.0,
        polarization: str = "TE",
        columns: list[str] | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Convenience: find waveguides below a loss threshold."""
        filter_expr = (
            (ds.field("propagation_loss_dB_cm") < max_loss_db_cm) &
            (ds.field("polarization") == polarization)
        )
        return self.query(filter_expr=filter_expr, columns=columns, limit=limit)

    def query_by_geometry(
        self,
        width_um: float,
        height_nm: float,
        tolerance_um: float = 0.05,
        tolerance_nm: float = 10.0,
    ) -> pd.DataFrame:
        """Find dataset entries near a specific waveguide geometry."""
        filter_expr = (
            (ds.field("width_um") >= width_um - tolerance_um) &
            (ds.field("width_um") <= width_um + tolerance_um) &
            (ds.field("height_nm") >= height_nm - tolerance_nm) &
            (ds.field("height_nm") <= height_nm + tolerance_nm)
        )
        return self.query(filter_expr=filter_expr)

    def aggregate(
        self,
        group_by: str,
        value_column: str,
        agg: str = "mean",
        filter_expr: ds.Expression | None = None,
    ) -> pd.DataFrame:
        """Group-by aggregation with optional pre-filtering.

        Aggregates natively in Arrow instead of materializing up to 100K
        rows into a pandas DataFrame first.
        """
        dataset = self._get_dataset()
        table = dataset.to_table(filter=filter_expr, columns=[group_by, value_column])
        aggregated = table.group_by(group_by).aggregate([(value_column, agg)])
        df = aggregated.to_pandas()
        return df.rename(columns={f"{value_column}_{agg}": value_column})
