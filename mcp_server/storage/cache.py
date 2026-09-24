"""DynamoDB simulation cache for the MCP physics server.

Provides sub-10ms point lookups to prevent redundant eigenmode expansions,
inverse design runs, and mask generation. Before running an expensive physics
calculation, the MCP server checks this cache for a matching geometry hash.

Schema:
    PK: geometry_hash (SHA-256 of sorted, canonical parameter dict)
    SK: tool_key ("MODE#1550nm#TE" | "OPTIM#prop_loss#0.2" | "GDS#edge_coupler")
    Attributes: result JSON, created_at, ttl
"""
import hashlib
import json
import logging
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from mcp_server.config import DYNAMODB_TABLE_NAME, DYNAMODB_REGION, CACHE_TTL_DAYS

logger = logging.getLogger(__name__)


class SimulationCache:
    """DynamoDB-backed simulation result cache."""

    def __init__(
        self,
        table_name: str = DYNAMODB_TABLE_NAME,
        region: str = DYNAMODB_REGION,
        ttl_days: int = CACHE_TTL_DAYS,
    ):
        self.table_name = table_name
        self.region = region
        self.ttl_days = ttl_days
        self._enabled = True
        self._table = None

    def _get_table(self):
        """Connect lazily on first use.

        The cache is constructed at server import time; probing DynamoDB
        there would stall startup for the full boto3 timeout/retry cycle
        whenever AWS is unreachable.
        """
        if not self._enabled:
            return None
        if self._table is None:
            try:
                dynamodb = boto3.resource("dynamodb", region_name=self.region)
                table = dynamodb.Table(self.table_name)
                table.table_status  # probe — raises if unreachable or missing
                self._table = table
            except Exception as e:
                logger.warning(
                    "[SimulationCache] DynamoDB unavailable (%s). Running without cache.", e
                )
                self._enabled = False
                return None
        return self._table

    @staticmethod
    def _hash_params(params: dict) -> str:
        """Create a deterministic SHA-256 hash from a parameter dict."""
        canonical = {}
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, float):
                v = round(v, 6)
            canonical[k] = v
        raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _build_sort_key(tool_name: str, params: dict) -> str:
        """Build a human-readable sort key from tool name and key params."""
        if tool_name == "solve_waveguide_mode":
            wl = params.get("wavelength_nm", 1550.0)
            pol = params.get("polarization", "TE")
            return f"MODE#{wl}nm#{pol}"
        elif tool_name == "optimize_waveguide":
            metric = params.get("target_metric", "unknown")
            value = params.get("target_value", 0)
            return f"OPTIM#{metric}#{value}"
        elif tool_name == "generate_mask":
            io = params.get("io_type", "edge_coupler")
            return f"GDS#{io}"
        return f"OTHER#{tool_name}"

    def get(self, tool_name: str, params: dict) -> Optional[dict]:
        """Look up a cached simulation result."""
        table = self._get_table()
        if table is None:
            return None
        geometry_hash = self._hash_params(params)
        sort_key = self._build_sort_key(tool_name, params)
        try:
            response = table.get_item(
                Key={"geometry_hash": geometry_hash, "sort_key": sort_key}
            )
            item = response.get("Item")
            if item and "result" in item:
                return json.loads(item["result"])
        except ClientError as e:
            logger.warning("[SimulationCache] Cache read failed: %s", e)
        return None

    def put(self, tool_name: str, params: dict, result: dict) -> None:
        """Store a simulation result in the cache."""
        table = self._get_table()
        if table is None:
            return
        geometry_hash = self._hash_params(params)
        sort_key = self._build_sort_key(tool_name, params)
        now = int(time.time())
        ttl = now + (self.ttl_days * 86400)
        try:
            table.put_item(
                Item={
                    "geometry_hash": geometry_hash,
                    "sort_key": sort_key,
                    "result": json.dumps(result),
                    "params": json.dumps(params),
                    "created_at": now,
                    "ttl": ttl,
                }
            )
        except ClientError as e:
            logger.warning("[SimulationCache] Failed to write cache entry: %s", e)

    def invalidate(self, tool_name: str, params: dict) -> None:
        """Remove a specific cache entry."""
        table = self._get_table()
        if table is None:
            return
        geometry_hash = self._hash_params(params)
        sort_key = self._build_sort_key(tool_name, params)
        try:
            table.delete_item(
                Key={"geometry_hash": geometry_hash, "sort_key": sort_key}
            )
        except ClientError as e:
            logger.warning("[SimulationCache] Failed to invalidate cache entry: %s", e)
