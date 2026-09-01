#!/bin/bash
# AWS resource provisioning for the Photonic Waveguide MCP Agent
set -euo pipefail

ENV="${1:-dev}"
REGION="${AWS_REGION:-us-west-2}"
# S3 Express One Zone availability zone ID — must belong to $REGION.
EXPRESS_AZ_ID="${EXPRESS_AZ_ID:-usw2-az1}"

echo "=========================================="
echo " Photonic Waveguide MCP — AWS Setup"
echo " Environment: $ENV"
echo " Region: $REGION"
echo "=========================================="

echo ""
echo "[1/4] Deploying storage stack (S3 bucket, DynamoDB cache, Glue catalog)..."
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/storage-stack.yaml \
  --stack-name "photonic-waveguide-storage-${ENV}" \
  --parameter-overrides "EnvironmentName=${ENV}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$REGION"

echo ""
echo "[2/4] Creating S3 Express One Zone directory bucket..."
EXPRESS_BUCKET="photonic-waveguide-data--${EXPRESS_AZ_ID}--x-s3"
if aws s3api head-bucket --bucket "$EXPRESS_BUCKET" --region "$REGION" 2>/dev/null; then
  echo "  (bucket already exists — skipping)"
else
  aws s3api create-bucket \
    --bucket "$EXPRESS_BUCKET" \
    --region "$REGION" \
    --create-bucket-configuration \
      "Location={Type=AvailabilityZone,Name=${EXPRESS_AZ_ID}},Bucket={DataRedundancy=SingleAvailabilityZone,Type=Directory}"
fi

echo ""
echo "[3/4] Uploading dataset to S3 Express One Zone..."
PARQUET_FILE="data/parquet_cache/dataset.parquet"
if [ -f "$PARQUET_FILE" ]; then
  aws s3 cp "$PARQUET_FILE" "s3://${EXPRESS_BUCKET}/dataset.parquet"
  echo "  Uploaded $PARQUET_FILE → s3://${EXPRESS_BUCKET}/dataset.parquet"
else
  echo "  WARNING: $PARQUET_FILE not found."
  echo "  Run 'python data/download_dataset.py' first to create it."
fi

echo ""
echo "[4/4] Deploying AgentCore and Bedrock Agent stacks..."
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/agentcore-stack.yaml \
  --stack-name "photonic-waveguide-agentcore-${ENV}" \
  --region "$REGION" \
  2>/dev/null || echo "  (agentcore stack — placeholder, skipping)"

echo ""
echo "=========================================="
echo " Setup complete!"
echo ""
echo " Storage resources:"
echo "   S3 Express:  s3://${EXPRESS_BUCKET}/"
echo "   DynamoDB:    photonic-simulation-cache-${ENV}"
echo "   Glue DB:     photonic_waveguide_db_${ENV}"
echo ""
echo " Next steps:"
echo "   1. python data/download_dataset.py"
echo "   2. python -m mcp_server.server"
echo "   3. python -m src.agents.photonic_agent"
echo "=========================================="
