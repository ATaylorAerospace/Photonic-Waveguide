"""S3 Tables SQL query interface for the waveguide dataset."""
from typing import Optional

import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd

from mcp_server.config import S3_BUCKET_NAME, S3_REGION


class S3TablesQuery:
    """SQL-like query interface over Parquet data in S3 Express One Zone."""

    def __init__(
        self,
        bucket: str = S3_BUCKET_NAME,
        region: str = S3_REGION,
        local_fallback: str = "data/SiN_Photonic_Waveguide_Loss_Efficiency.csv",
    ):
        self.s3_uri = f"s3://{bucket}/"
        self.region = region
        self.local_fallback = local_fallback
        self._dataset: Optional[ds.Dataset] = None

    def _get_dataset(self) -> ds.Dataset:
        if self._dataset is not None:
            return self._dataset
        try:
            import s3fs
            fs = s3fs.S3FileSystem(anon=False, client_kwargs={"region_name": self.region})
            self._dataset = ds.dataset(self.s3_uri, format="parquet", filesystem=fs)
            return self._dataset
        except Exception:
            pass
        import os
        local_parquet = "data/parquet_cache/dataset.parquet"
        if os.path.exists(local_parquet):
            self._dataset = ds.dataset(local_parquet, format="parquet")
        elif os.path.exists(self.local_fallback):
            df = pd.read_csv(self.local_fallback)
            table = pa.Table.from_pandas(df)
            self._dataset = ds.dataset(table)
        else:
            raise FileNotFoundError("No dataset available for S3 Tables queries.")
        return self._dataset

    def query(
        self,
        filter_expr: Optional[ds.Expression] = None,
        columns: Optional[list[str]] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Execute a filtered query with predicate pushdown."""
        dataset = self._get_dataset()
        table = dataset.to_table(filter=filter_expr, columns=columns)
        if len(table) > limit:
            table = table.slice(0, limit)
        return table.to_pandas()

    def query_low_loss(
        self,
        max_loss_db_cm: float = 2.0,
        polarization: str = "TE",
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Convenience: find waveguides below a loss threshold."""
        filter_expr = (
            (ds.field("propagation_loss_dB_cm") < max_loss_db_cm) &
            (ds.field("polarization") == polarization)
        )
        return self.query(filter_expr=filter_expr, columns=columns)

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
        filter_expr: Optional[ds.Expression] = None,
    ) -> pd.DataFrame:
        """Group-by aggregation with optional pre-filtering."""
        df = self.query(filter_expr=filter_expr, columns=[group_by, value_column], limit=100000)
        return df.groupby(group_by)[value_column].agg(agg).reset_index()
