"""Batch inference from trusted local artifacts."""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_pipeline.core import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = joblib.load(ROOT / "fraud_model_pipeline.joblib")
    calibrated = joblib.load(ROOT / "calibrated_model.joblib")
    data = pd.read_csv(args.input)
    result = predict(data, base["model"], calibrated["calibrator"], calibrated["policy"], calibrated["cost"], base["version"])
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} transactions to {args.output}")


if __name__ == "__main__":
    main()
