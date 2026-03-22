"""Download the SiN photonic waveguide dataset from HuggingFace."""
import os
from datasets import load_dataset


def main():
    print("Downloading SiN Photonic Waveguide dataset from HuggingFace...")
    ds = load_dataset("Taylor658/SiN-photonic-waveguide-loss-efficiency")
    df = ds["train"].to_pandas()

    csv_path = "data/SiN_Photonic_Waveguide_Loss_Efficiency.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path} ({len(df):,} rows)")

    parquet_dir = "data/parquet_cache"
    os.makedirs(parquet_dir, exist_ok=True)
    parquet_path = os.path.join(parquet_dir, "dataset.parquet")
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Parquet saved to {parquet_path} ({len(df):,} rows, snappy compression)")

    print("\nTo upload to S3 Express One Zone:")
    print(f"  aws s3 cp {parquet_path} s3://photonic-waveguide-data--usw2-az1--x-s3/dataset.parquet")


if __name__ == "__main__":
    main()
