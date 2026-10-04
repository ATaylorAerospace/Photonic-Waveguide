"""Agent-side configuration (the MCP server has its own in mcp_server/config.py)."""
import os

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
DATASET_PATH = os.getenv("DATASET_PATH", "data/SiN_Photonic_Waveguide_Loss_Efficiency.csv")
MODEL_ARTIFACTS_PATH = os.getenv("MODEL_ARTIFACTS_PATH", "models/")
