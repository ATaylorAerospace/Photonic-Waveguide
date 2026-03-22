"""Agent configuration parameters."""
import os

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
DATASET_PATH = os.getenv("DATASET_PATH", "data/SiN_Photonic_Waveguide_Loss_Efficiency.csv")
MODEL_ARTIFACTS_PATH = os.getenv("MODEL_ARTIFACTS_PATH", "models/")

# Hybrid Storage Architecture
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "photonic-waveguide-data--usw2-az1--x-s3")
S3_DATASET_KEY = os.getenv("S3_DATASET_KEY", "dataset.parquet")
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "photonic-simulation-cache")
S3_TABLES_DATABASE = os.getenv("S3_TABLES_DATABASE", "photonic_waveguide_db")
S3_TABLES_TABLE = os.getenv("S3_TABLES_TABLE", "waveguide_measurements")
