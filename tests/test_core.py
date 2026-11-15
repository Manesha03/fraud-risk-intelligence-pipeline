import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib
import nbformat
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud_pipeline.core import Policy, ReferenceFeatures, TrainingUnderSampler, calibration_diagnostics, features, make_model, operational_decisions, policy_stats, predict, realized_cost

ROOT = Path(__file__).resolve().parents[1]


def config():
    return {"fraud_loss_multiplier":1,"review_cost":3,"legitimate_review_friction":2,"review_success":.8,"fraud_block_cost":1,"legitimate_block_cost":25,"missing_amount_fraud_loss":100}


def test_threshold_boundaries():
    assert Policy(.2,.8).decide([.199,.2,.799,.8]).tolist() == ["APPROVE","REVIEW","REVIEW","BLOCK"]
    with pytest.raises(ValueError):
        Policy(.8,.2)
    with pytest.raises(ValueError):
        Policy(.2,.8).decide([np.nan])


def test_cost_accounting():
    actual=realized_cost([1,1,0,0],[100,100,100,100],np.array(["APPROVE","REVIEW","REVIEW","BLOCK"]),config())
    assert actual == pytest.approx([100,23,5,25])


def test_missing_amount_has_explicit_loss_and_review_route():
    assert realized_cost([1],[np.nan],np.array(["APPROVE"]),config())[0] == 100
    assert operational_decisions([.01],[np.nan],Policy(.2,.8)).tolist() == ["REVIEW"]
    with pytest.raises(ValueError):
        realized_cost([1],[-1],np.array(["APPROVE"]),config())


def test_features_are_row_local():
    frame = pd.DataFrame({
        "transaction_id":["a","b"],"fraud":[0,1],"transaction_amount":[10.,20.],
        "merchant_risk_score":[.2,.5],"distance_from_usual_location_km":[1.,2.],
        "new_device":[0,1],"transactions_last_1h":[1,2],"transactions_last_24h":[3,4],
        "account_age_days":[100,10],"device_age_days":[100,10],"hour":[12,2],
        "day_of_week":[2,6],"authentication_level":["Strong","Standard"],
        "channel":["POS","E-commerce"],"login_to_transaction_minutes":[10.,20.],
        "prior_chargebacks":[0,1],
    })
    first = features(frame)
    reversed_rows = features(frame.iloc[::-1]).iloc[::-1]
    pd.testing.assert_frame_equal(first,reversed_rows)
    assert "fraud" not in first and "transaction_id" not in first


def test_trusted_artifacts_and_inference_schema():
    base = joblib.load(ROOT / "fraud_model_pipeline.joblib")
    calibrated = joblib.load(ROOT / "calibrated_model.joblib")
    frame = pd.read_csv(ROOT / "data/raw/transaction_fraud_risk_advanced_dataset.csv").head(2)
    frame.loc[frame.index[0],"transaction_amount"] = np.nan
    scored = predict(frame,base["model"],calibrated["calibrator"],calibrated["policy"],calibrated["cost"])
    assert set(["transaction_id","scored_at_utc","fraud_probability","decision","reason_code","model_version","data_warning","expected_cost_estimate"]).issubset(scored)
    assert scored.iloc[0].decision in {"REVIEW","BLOCK"}
    assert scored.fraud_probability.between(0,1).all()
    with pytest.raises(ValueError):
        predict(frame.drop(columns=["transaction_amount"]),base["model"],calibrated["calibrator"],calibrated["policy"],calibrated["cost"])
    with pytest.raises(ValueError,match="Missing model fields"):
        predict(frame.drop(columns=["country"]),base["model"],calibrated["calibrator"],calibrated["policy"],calibrated["cost"])


def test_generated_pdf_page_counts():
    card = (ROOT / "model_card.pdf").read_bytes()
    report = (ROOT / "business_fraud_strategy.pdf").read_bytes()
    assert len(re.findall(rb"/Type\s*/Page\b",card)) == 1
    assert 8 <= len(re.findall(rb"/Type\s*/Page\b",report)) <= 12


def test_notebook_has_executed_outputs():
    notebook = nbformat.read(ROOT / "fraud_risk_analysis.ipynb",as_version=4)
    nbformat.validate(notebook)
    sections = [cell for cell in notebook.cells if cell.cell_type == "markdown" and cell.source.startswith("## ")]
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(sections) == 20
    assert code and all(cell.execution_count is not None and cell.outputs for cell in code)


def test_calibration_diagnostics_validate_probabilities():
    summary = calibration_diagnostics([0,1,0,1],[.1,.9,.2,.8])
    assert abs(summary["calibration_in_large"]) < 1e-12
    assert 0 <= summary["expected_calibration_error"] <= 1
    with pytest.raises(ValueError):
        calibration_diagnostics([0,1],[.2,np.nan])
    assert calibration_diagnostics([0, 1], [.1, .1])["expected_calibration_error"] == pytest.approx(.4)


def test_stress_missing_input_and_financial_exposure_are_separate():
    cfg = {**config(), "review_capacity_fraction": .15}
    stats = policy_stats(np.array([1, 0]), np.array([.01, .01]), np.array([np.nan, 50]),
                         Policy(.2, .8), cfg, exposure_amount=np.array([200, 50]))
    assert stats["review_count"] == 1
    assert stats["approve_count"] == 1
    assert stats["total_expected_cost"] == pytest.approx(43)


def test_reference_features_never_learn_from_inference_rows():
    train = pd.DataFrame({"transaction_amount": [10., 30.], "merchant_category": ["A", "A"],
                          "channel": ["web", "web"], "distance_from_usual_location_km": [2., 6.],
                          "amount_to_merchant_median": [np.nan]*2,
                          "amount_merchant_deviation": [np.nan]*2,
                          "distance_channel_deviation": [np.nan]*2})
    reference = ReferenceFeatures().fit(train)
    scored = train.iloc[:1].copy()
    scored["transaction_amount"] = 1000.
    assert reference.transform(scored).amount_to_merchant_median.iloc[0] == 50
    scored["merchant_category"] = "never seen"
    assert reference.transform(scored).amount_to_merchant_median.iloc[0] == 50
    assert reference.merchant_["A"] == 20


def test_sampling_fits_preprocessing_only_on_sampled_training_rows():
    frame = pd.DataFrame({"value": np.arange(20, dtype=float)})
    labels = np.array([1]*3 + [0]*17)
    estimator = TrainingUnderSampler(make_model(frame, "logistic"), majority_ratio=2).fit(frame, labels)
    assert len(estimator.sample_indices_) == 9
    imputer = estimator.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    assert imputer.statistics_[0] == np.median(frame.iloc[estimator.sample_indices_].value)
    before = estimator.sample_indices_.copy()
    assert estimator.predict_proba(frame).shape == (20, 2)
    np.testing.assert_array_equal(before, estimator.sample_indices_)


def test_single_artifact_scores_raw_transactions_and_explains_every_flag():
    engine = joblib.load(ROOT / "calibrated_model.joblib")["engine"]
    frame = pd.read_csv(ROOT / "data/raw/transaction_fraud_risk_advanced_dataset.csv").head(100)
    frame = frame.drop(columns=["fraud"])
    result = engine.predict(frame)
    np.testing.assert_allclose(engine.predict_proba(frame)[:, 1], result.fraud_probability)
    flagged = result.decision.ne("APPROVE")
    assert flagged.any()
    assert result.loc[flagged, "explanation"].str.len().gt(30).all()
    raw_pipeline = joblib.load(ROOT / "fraud_model_pipeline.joblib")["pipeline"]
    assert raw_pipeline.predict_proba(frame).shape == (100, 2)


def test_partition_manifest_is_disjoint_and_reference_fit_is_train_only():
    manifest = pd.read_csv(ROOT / "reports/split_manifest.csv")
    assert not manifest.transaction_id.duplicated().any()
    assert set(manifest.partition) == {"train", "selection", "calibration_fit", "policy_selection", "test"}
    raw = pd.read_csv(ROOT / "data/raw/transaction_fraud_risk_advanced_dataset.csv")
    assert set(manifest.transaction_id) == set(raw.transaction_id)
    train = raw[raw.transaction_id.isin(manifest.loc[manifest.partition.eq("train"), "transaction_id"])]
    fitted = ReferenceFeatures().fit(features(train))
    expected = train.groupby("merchant_category").transaction_amount.median().to_dict()
    assert fitted.merchant_ == expected


def test_reported_missingness_stress_decisions_match_operational_engine():
    manifest = pd.read_csv(ROOT / "reports/split_manifest.csv")
    ids = manifest.loc[manifest.partition.eq("test"), "transaction_id"]
    raw = pd.read_csv(ROOT / "data/raw/transaction_fraud_risk_advanced_dataset.csv").set_index("transaction_id", drop=False)
    changed = raw.loc[ids].copy()
    selected = np.random.default_rng(42).choice(changed.index, size=len(changed)//10, replace=False)
    changed.loc[selected, "transaction_amount"] = np.nan
    engine = joblib.load(ROOT / "calibrated_model.joblib")["engine"]
    actual = engine.predict(changed).decision.value_counts()
    reported = pd.read_csv(ROOT / "reports/robustness.csv").set_index("scenario").loc["missing_amount_10pct"]
    for decision in ("APPROVE", "REVIEW", "BLOCK"):
        assert actual.get(decision, 0) == reported[f"{decision.lower()}_count"]
