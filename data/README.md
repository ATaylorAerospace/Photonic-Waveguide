# 📦 Dataset

## SiN Photonic Waveguide Loss & Efficiency Dataset

- **Source:** [HuggingFace — Taylor658/SiN-photonic-waveguide-loss-efficiency](https://huggingface.co/datasets/Taylor658/SiN-photonic-waveguide-loss-efficiency)
- **Size:** 90,000 rows × 25 columns
- **Type:** Synthetic

### Download

```bash
python download_dataset.py
```

This creates both:
- `SiN_Photonic_Waveguide_Loss_Efficiency.csv` — human-readable, local dev fallback
- `parquet_cache/dataset.parquet` — columnar format for S3 Express One Zone / fast reads

### Hybrid Storage Architecture

This dataset is served through a three-layer storage system:

| Layer | Technology | Purpose |
|---|---|---|
| Bulk Data | S3 Express One Zone (Parquet) | 10x faster reads, columnar access, 50% lower request costs |
| Simulation Cache | Amazon DynamoDB | Sub-10ms lookups to prevent redundant physics calculations |
| SQL Queries | Amazon S3 Tables | Predicate pushdown — filter 90K rows without loading into RAM |

### Upload to S3 Express One Zone

```bash
# Create S3 Express One Zone directory bucket (one-time)
aws s3api create-bucket \
  --bucket photonic-waveguide-data--usw2-az1--x-s3 \
  --region us-west-2 \
  --create-bucket-configuration \
    'Location={Type=AvailabilityZone,Name=usw2-az1},Bucket={DataRedundancy=SingleAvailabilityZone,Type=Directory}'

# Upload Parquet file
aws s3 cp parquet_cache/dataset.parquet \
  s3://photonic-waveguide-data--usw2-az1--x-s3/dataset.parquet
```

### Columns

The dataset covers waveguide geometry, fabrication parameters, and performance metrics.
Refer to the HuggingFace dataset card for the full data dictionary.

> ⚠️ This dataset is synthetic — predictions may differ from real foundry results.
