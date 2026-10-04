"""Download the SiN photonic waveguide dataset from HuggingFace."""
import os
from pathlib import Path

# Anchor outputs to this file so the script works from any working directory
# (data/README.md runs it from inside data/, the top-level README from the root).
DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "SiN_Photonic_Waveguide_Loss_Efficiency.csv"
PARQUET_PATH = DATA_DIR / "parquet_cache" / "dataset.parquet"


def main():
    from datasets import load_dataset

    print("Downloading SiN Photonic Waveguide dataset from HuggingFace...")
    ds = load_dataset("Taylor658/SiN-photonic-waveguide-loss-efficiency")
    df = ds["train"].to_pandas()

    df.to_csv(CSV_PATH, index=False)
    print(f"CSV saved to {CSV_PATH} ({len(df):,} rows)")

    os.makedirs(PARQUET_PATH.parent, exist_ok=True)
    df.to_parquet(PARQUET_PATH, engine="pyarrow", compression="snappy", index=False)
    print(f"Parquet saved to {PARQUET_PATH} ({len(df):,} rows, snappy compression)")

    print("\nTo upload to S3 Express One Zone:")
    print(f"  aws s3 cp {PARQUET_PATH} s3://photonic-waveguide-data--usw2-az1--x-s3/dataset.parquet")


if __name__ == "__main__":
    main()
