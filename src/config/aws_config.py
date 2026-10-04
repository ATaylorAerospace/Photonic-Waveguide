"""AWS resource configuration.

AWS_PROFILE needs no constant here: boto3 reads that environment variable
itself when it builds a session.
"""
import os

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
# Cross-region inference profile ID — bare model IDs without the -v1:0 suffix
# are rejected by Bedrock with a ValidationException.
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
