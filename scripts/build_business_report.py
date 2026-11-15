"""Rebuild the final management report from existing measured evidence."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_pipeline.business_report import build_business_report

if __name__ == "__main__":
    print(build_business_report(ROOT))
