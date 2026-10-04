"""Shared Bedrock model factory so every agent runs on the configured model."""
from strands.models import BedrockModel

from src.config.aws_config import AWS_REGION, BEDROCK_MODEL_ID


def create_bedrock_model() -> BedrockModel:
    return BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)
