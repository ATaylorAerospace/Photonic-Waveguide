"""Configuration constants for the MCP physics server."""
import os

# Material refractive index library (at ~1550nm)
MATERIAL_INDEX = {
    "SiN": 2.0,
    "Si3N4": 2.0,
    "SiO2": 1.44,
    "Air": 1.0,
    "Si": 3.48,
    "SiON": 1.65,
}

# Server defaults
DEFAULT_WAVELENGTH_NM = 1550.0
DEFAULT_POLARIZATION = "TE"
MCP_SERVER_HOST = "0.0.0.0"
MCP_SERVER_PORT = 8000

# Solver grid defaults
DEFAULT_X_STEP_UM = 0.02
DEFAULT_Y_STEP_UM = 0.02
DEFAULT_BOUNDARY_UM = 2.0

# Optimization defaults
DEFAULT_MAX_ITERATIONS = 200
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WIDTH_RANGE_UM = (0.3, 5.0)
DEFAULT_HEIGHT_RANGE_NM = (100.0, 800.0)

# Mask generation defaults
DEFAULT_BEND_RADIUS_UM = 50.0
DEFAULT_TAPER_LENGTH_UM = 200.0
DEFAULT_LAYER = (1, 0)
GDS_OUTPUT_DIR = "output/gds"

# --- Hybrid Storage Architecture ---
# Layer 1: S3 Express One Zone (Bulk Parquet Data)
S3_BUCKET_NAME = os.getenv(
    "S3_BUCKET_NAME",
    "photonic-waveguide-data--usw2-az1--x-s3"
)
S3_DATASET_KEY = os.getenv("S3_DATASET_KEY", "dataset.parquet")
S3_REGION = os.getenv("AWS_REGION", "us-west-2")
PARQUET_LOCAL_CACHE_DIR = "data/parquet_cache"

# Layer 2: DynamoDB (Simulation Cache)
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "photonic-simulation-cache")
DYNAMODB_REGION = os.getenv("AWS_REGION", "us-west-2")
CACHE_TTL_DAYS = 30  # Cache entries expire after 30 days

# Layer 3: S3 Tables (SQL over Parquet)
S3_TABLES_BUCKET = S3_BUCKET_NAME  # Same bucket, managed table layer
S3_TABLES_DATABASE = os.getenv("S3_TABLES_DATABASE", "photonic_waveguide_db")
S3_TABLES_TABLE = os.getenv("S3_TABLES_TABLE", "waveguide_measurements")
