"""AWS resource configuration."""
import os

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
AWS_PROFILE = os.getenv("AWS_PROFILE", "photonic-agent-dev")
# Cross-region inference profile ID — bare model IDs without the -v1:0 suffix
# are rejected by Bedrock with a ValidationException.
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
