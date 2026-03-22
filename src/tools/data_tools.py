"""Dataset loading, filtering, and query tools."""
import os
import pandas as pd
from strands import tool
from mcp_server.storage.s3_dataset import S3DatasetReader
from mcp_server.storage.s3_tables import S3TablesQuery

_dataset_reader = S3DatasetReader()
_s3_tables = S3TablesQuery()
_df_cache = None


def _load_dataset() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        try:
            _df_cache = _dataset_reader.to_pandas()
        except FileNotFoundError:
            csv_path = os.getenv("DATASET_PATH", "data/SiN_Photonic_Waveguide_Loss_Efficiency.csv")
            _df_cache = pd.read_csv(csv_path)
    return _df_cache


@tool
def query_dataset(
    filters: dict | None = None, columns: list[str] | None = None,
    group_by: str | None = None, agg_func: str = "mean", limit: int = 50,
) -> dict:
    """Query the 90K-row waveguide dataset with optional filtering and aggregation.

    Uses S3 Tables with predicate pushdown when available — filters are evaluated
    at the storage layer so only matching rows are read into memory.
    """
    try:
        import pyarrow.dataset as ds
        filter_expr = None
        if filters:
            exprs = []
            for col, val in filters.items():
                if isinstance(val, (int, float)):
                    exprs.append(ds.field(col) == val)
                else:
                    exprs.append(ds.field(col) == str(val))
            if exprs:
                filter_expr = exprs[0]
                for e in exprs[1:]:
                    filter_expr = filter_expr & e
        df = _s3_tables.query(filter_expr=filter_expr, columns=columns, limit=limit)
        if group_by and group_by in df.columns:
            numeric_cols = df.select_dtypes(include="number").columns
            df = df.groupby(group_by)[numeric_cols].agg(agg_func).reset_index()
        return df.head(limit).to_dict(orient="records")
    except Exception:
        pass
    df = _load_dataset()
    if filters:
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    if group_by and group_by in df.columns:
        numeric_cols = df.select_dtypes(include="number").columns
        df = df.groupby(group_by)[numeric_cols].agg(agg_func).reset_index()
    return df.head(limit).to_dict(orient="records")


@tool
def query_low_loss_waveguides(
    max_loss_db_cm: float = 2.0,
    polarization: str = "TE",
    columns: list[str] | None = None,
    limit: int = 50,
) -> dict:
    """Find waveguides below a propagation loss threshold."""
    try:
        df = _s3_tables.query_low_loss(
            max_loss_db_cm=max_loss_db_cm,
            polarization=polarization,
            columns=columns,
        )
        return df.head(limit).to_dict(orient="records")
    except Exception:
        df = _load_dataset()
        mask = (df["propagation_loss_dB_cm"] < max_loss_db_cm)
        if "polarization" in df.columns:
            mask = mask & (df["polarization"] == polarization)
        df = df[mask]
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        return df.head(limit).to_dict(orient="records")


@tool
def read_dataset_columns(columns: list[str]) -> dict:
    """Read specific columns from the dataset using Parquet columnar reads."""
    try:
        table = _dataset_reader.read_columns(columns)
        return table.to_pandas().to_dict(orient="records")
    except Exception:
        df = _load_dataset()
        cols = [c for c in columns if c in df.columns]
        return df[cols].to_dict(orient="records")
