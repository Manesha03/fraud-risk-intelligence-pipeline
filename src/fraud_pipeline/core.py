"""Reusable training, cost and inference functions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "fraud"
ID = "transaction_id"
ENGINEERED = ["log_amount", "amount_risk", "device_distance", "distance_risk", "velocity_ratio", "velocity_amount", "new_account", "young_device", "night_transaction", "weekend", "auth_channel", "login_speed", "chargeback_amount"]
REFERENCE_FEATURES = ["amount_to_merchant_median", "amount_merchant_deviation", "distance_channel_deviation"]
ENGINEERED += ["velocity_burst", "combined_risk"] + REFERENCE_FEATURES


class ReferenceFeatures(TransformerMixin, BaseEstimator):
    """Learn population references only on each fit partition, never on scored rows."""

    def fit(self, x, y=None):
        self.enabled_ = all(c in x for c in REFERENCE_FEATURES)
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        if self.enabled_:
            self.amount_median_ = float(x.transaction_amount.median())
            self.distance_median_ = float(x.distance_from_usual_location_km.median())
            self.merchant_ = x.groupby("merchant_category").transaction_amount.median().to_dict()
            self.channel_ = x.groupby("channel").distance_from_usual_location_km.median().to_dict()
        return self

    def transform(self, x):
        out = x.copy()
        if self.enabled_:
            reference = x.merchant_category.map(self.merchant_).fillna(self.amount_median_).clip(lower=1)
            out["amount_to_merchant_median"] = x.transaction_amount / reference
            out["amount_merchant_deviation"] = (x.transaction_amount - reference) / reference
            distance = x.channel.map(self.channel_).fillna(self.distance_median_)
            out["distance_channel_deviation"] = x.distance_from_usual_location_km - distance
        return out


class TrainingUnderSampler(ClassifierMixin, BaseEstimator):
    """Fit-only sampling around the entire preprocessing/model pipeline; CV-cloneable."""

    def __init__(self, estimator, majority_ratio=3, random_state=42):
        self.estimator = estimator
        self.majority_ratio = majority_ratio
        self.random_state = random_state

    def fit(self, x, y):
        labels = np.asarray(y)
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        rng = np.random.default_rng(self.random_state)
        selected = np.r_[positive, rng.choice(negative, min(len(negative), self.majority_ratio * len(positive)), replace=False)]
        rng.shuffle(selected)
        self.sample_indices_ = selected
        self.estimator_ = clone(self.estimator).fit(x.iloc[selected], labels[selected])
        self.classes_ = self.estimator_.classes_
        self.feature_names_in_ = self.estimator_.feature_names_in_
        return self

    @property
    def named_steps(self):
        return self.estimator_.named_steps

    def predict_proba(self, x):
        return self.estimator_.predict_proba(x)

    def predict(self, x):
        return self.estimator_.predict(x)


def read_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {TARGET, ID, "transaction_amount", "merchant_risk_score"}
    absent = needed - set(df)
    if absent:
        raise ValueError(f"Missing required columns: {sorted(absent)}")
    if df[ID].duplicated().any():
        raise ValueError("Duplicate transaction identifiers")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary")
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Create row-local authorization-time candidates; no future rows are consulted."""
    x = df.drop(columns=[TARGET, ID], errors="ignore").copy()
    required = {"transaction_amount", "merchant_risk_score", "distance_from_usual_location_km", "new_device", "transactions_last_1h", "transactions_last_24h", "account_age_days", "device_age_days", "hour", "day_of_week", "authentication_level", "channel", "login_to_transaction_minutes", "prior_chargebacks"}
    absent = required - set(x)
    if absent:
        raise ValueError(f"Missing inference fields: {sorted(absent)}")
    a = pd.to_numeric(x.transaction_amount, errors="coerce").clip(lower=0)
    d = pd.to_numeric(x.distance_from_usual_location_km, errors="coerce").clip(lower=0)
    x["log_amount"] = np.log1p(a)
    x["amount_risk"] = a * x.merchant_risk_score
    x["device_distance"] = x.new_device * d
    x["distance_risk"] = d * x.merchant_risk_score
    x["velocity_ratio"] = (x.transactions_last_1h + 1) / (x.transactions_last_24h + 1)
    x["velocity_amount"] = x.transactions_last_1h * a
    x["new_account"] = (x.account_age_days < 30).astype(int)
    x["young_device"] = (x.device_age_days < 30).astype(int)
    x["night_transaction"] = x.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    x["weekend"] = x.day_of_week.isin([5, 6]).astype(int)
    x["auth_channel"] = x.authentication_level.fillna("Unknown").astype(str) + " / " + x.channel.fillna("Unknown").astype(str)
    x["login_speed"] = 1 / (1 + x.login_to_transaction_minutes.clip(lower=0))
    x["chargeback_amount"] = x.prior_chargebacks * a
    x["velocity_burst"] = 24 * (x.transactions_last_1h + 1) / (x.transactions_last_24h + 24)
    x["combined_risk"] = x.merchant_risk_score * x.get("ip_risk_score", np.nan)
    for column in REFERENCE_FEATURES:
        x[column] = np.nan  # Filled by ReferenceFeatures fitted inside the model pipeline.
    return x.replace([np.inf, -np.inf], np.nan)


def make_model(x: pd.DataFrame, kind: str, weighted: bool = False, seed: int = 42) -> Pipeline:
    num = list(x.select_dtypes(include="number").columns)
    cat = [c for c in x if c not in num]
    transform = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    weight = "balanced" if weighted else None
    if kind == "logistic":
        model = LogisticRegression(max_iter=1000, class_weight=weight, random_state=seed)
    elif kind == "forest":
        model = RandomForestClassifier(n_estimators=120, min_samples_leaf=5, max_features="sqrt", class_weight=weight, random_state=seed, n_jobs=-1)
    elif kind == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=5, max_features="sqrt", class_weight=weight, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(kind)
    return Pipeline([("reference", ReferenceFeatures()), ("prep", transform), ("model", model)])


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-8, 1-1e-8)))}


def calibration_diagnostics(y: np.ndarray, p: np.ndarray) -> dict:
    """Probability-scale bias, logistic slope, and quantile-bin calibration error."""
    actual = np.asarray(y, dtype=int)
    probability = np.asarray(p, dtype=float)
    if len(actual) != len(probability) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid calibration inputs")
    clipped = np.clip(probability,1e-6,1-1e-6)
    logit = np.log(clipped / (1-clipped)).reshape(-1,1)
    slope = float(LogisticRegression(C=1e6,max_iter=1000).fit(logit,actual).coef_[0,0]) if len(np.unique(actual)) == 2 else None
    bands = pd.DataFrame({"actual":actual,"probability":probability})
    bands["band"] = pd.qcut(bands.probability,8,duplicates="drop")
    grouped = bands.groupby("band",observed=True).agg(count=("actual","size"),observed=("actual","mean"),predicted=("probability","mean"))
    ece = float((grouped["count"] * (grouped["observed"]-grouped["predicted"]).abs()).sum()/len(actual)) if len(grouped) else float(abs(actual.mean() - probability.mean()))
    return {"calibration_in_large":float(actual.mean()-probability.mean()),"calibration_slope":slope,"expected_calibration_error":ece}


class SigmoidCalibrator:
    """Importable sigmoid probability mapping for serialized inference."""

    def __init__(self):
        self.model = LogisticRegression(random_state=42)

    def fit(self, p: np.ndarray, y: np.ndarray) -> SigmoidCalibrator:
        self.model.fit(np.asarray(p).reshape(-1, 1), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


@dataclass(frozen=True)
class Policy:
    review: float
    block: float

    def __post_init__(self) -> None:
        if not 0 <= self.review <= self.block <= 1:
            raise ValueError("Require 0 <= review <= block <= 1")

    def decide(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid probability")
        return np.where(p >= self.block, "BLOCK", np.where(p >= self.review, "REVIEW", "APPROVE"))


def realized_cost(y: np.ndarray, amount: np.ndarray, decisions: np.ndarray, cfg: dict) -> np.ndarray:
    """Counterfactual expected cost conditional on observed label; all money assumptions are hypothetical."""
    y = np.asarray(y, dtype=int)
    raw_amount = np.asarray(amount, dtype=float)
    if np.isinf(raw_amount).any() or (raw_amount < 0).any():
        raise ValueError("Transaction amounts must be nonnegative or missing")
    a = np.where(np.isnan(raw_amount), cfg["missing_amount_fraud_loss"], raw_amount)
    approve = y * a * cfg["fraud_loss_multiplier"]
    review = cfg["review_cost"] + (1-y)*cfg["legitimate_review_friction"] + y*(1-cfg["review_success"])*a*cfg["fraud_loss_multiplier"]
    block = y*cfg["fraud_block_cost"] + (1-y)*cfg["legitimate_block_cost"]
    return np.select([decisions == "APPROVE", decisions == "REVIEW", decisions == "BLOCK"], [approve, review, block], default=np.nan)


def operational_decisions(p: np.ndarray, amount: np.ndarray, policy: Policy) -> np.ndarray:
    """Route an otherwise approved transaction with unknown amount to review."""
    decisions = policy.decide(p)
    missing_amount = np.isnan(np.asarray(amount, dtype=float))
    return np.where(missing_amount & (decisions == "APPROVE"), "REVIEW", decisions)


def policy_stats(y: np.ndarray, p: np.ndarray, amount: np.ndarray, policy: Policy, cfg: dict, force_missing_review: bool = True, exposure_amount=None) -> dict:
    decisions = operational_decisions(p, amount, policy) if force_missing_review else policy.decide(p)
    fraud = np.asarray(y) == 1
    legitimate = ~fraud
    caught = fraud & (decisions == "BLOCK")
    reviewed_fraud = fraud & (decisions == "REVIEW")
    n = len(y)
    costs = realized_cost(y, amount if exposure_amount is None else exposure_amount, decisions, cfg)
    return {"review_threshold":policy.review,"block_threshold":policy.block,"total_expected_cost":float(costs.sum()),"cost_per_transaction":float(costs.mean()),"review_count":int((decisions=="REVIEW").sum()),"block_count":int((decisions=="BLOCK").sum()),"approve_count":int((decisions=="APPROVE").sum()),"legitimate_block_count":int((legitimate & (decisions=="BLOCK")).sum()),"false_positive_count":int((legitimate & (decisions!="APPROVE")).sum()),"fraud_approved_count":int((fraud & (decisions=="APPROVE")).sum()),"fraud_capture_rate":float((caught.sum()+cfg["review_success"]*reviewed_fraud.sum())/max(fraud.sum(),1)),"precision":float(fraud[decisions!="APPROVE"].mean()) if (decisions!="APPROVE").any() else 0.0,"recall":float((decisions[fraud]!="APPROVE").mean()) if fraud.any() else 0.0,"review_capacity_utilization":float((decisions=="REVIEW").sum()/max(1,n*cfg["review_capacity_fraction"]))}


def search_policy(y: np.ndarray, p: np.ndarray, amount: np.ndarray, cfg: dict) -> pd.DataFrame:
    levels = np.unique(np.r_[0, np.quantile(p, np.linspace(0.05, 0.98, 30)), 1.0])
    rows = []
    for review in levels:
        for block in levels[levels >= review]:
            row = policy_stats(y, p, amount, Policy(float(review), float(block)), cfg)
            row["capacity_ok"] = row["review_capacity_utilization"] <= 1
            rows.append(row)
    return pd.DataFrame(rows)


def predict(df: pd.DataFrame, model, calibrator, policy: Policy, cfg: dict, version: str = "1.0") -> pd.DataFrame:
    x = features(df)
    missing_model_fields = set(model.feature_names_in_) - set(x.columns)
    if missing_model_fields:
        raise ValueError(f"Missing model fields: {sorted(missing_model_fields)}")
    raw = model.predict_proba(x)[:, 1]
    probability = calibrator.predict(raw)
    amount = df.transaction_amount.to_numpy(dtype=float)
    if np.isinf(amount).any() or (amount < 0).any():
        raise ValueError("Transaction amounts must be nonnegative or missing")
    decisions = operational_decisions(probability, amount, policy)
    a = np.where(np.isnan(amount), cfg["missing_amount_fraud_loss"], amount)
    expected = np.where(decisions == "APPROVE", probability*a*cfg["fraud_loss_multiplier"], np.where(decisions == "REVIEW", cfg["review_cost"]+(1-probability)*cfg["legitimate_review_friction"]+probability*(1-cfg["review_success"])*a*cfg["fraud_loss_multiplier"], probability*cfg["fraud_block_cost"]+(1-probability)*cfg["legitimate_block_cost"]))
    warnings = np.where(np.isnan(amount), "missing_amount_review_required", np.where(df.isna().any(axis=1), "missing_input_imputed", ""))
    reasons = np.where(np.isnan(amount) & (decisions == "REVIEW"), "MISSING_TRANSACTION_AMOUNT", np.where(decisions == "APPROVE", "", "MODEL_RISK_THRESHOLD"))
    explanations = explain_rows(x, model, decisions != "APPROVE")
    scored_at = datetime.now(timezone.utc).isoformat()
    return pd.DataFrame({"transaction_id":df[ID].values,"scored_at_utc":scored_at,"fraud_probability":probability,"decision":decisions,"reason_code":reasons,"explanation":explanations,"model_version":version,"data_warning":warnings,"expected_cost_estimate":expected})


def explain_rows(x, model, mask):
    """Exact additive base-logit terms for logistic; local ablation for trees.

    Ablations hold other fields fixed and are not causal or additive explanations.
    """
    output = np.full(len(x), "", dtype=object)
    positions = np.flatnonzero(mask)
    if not len(positions):
        return output
    rows = x.iloc[positions][list(model.feature_names_in_)]
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        prepared = model.named_steps["reference"].transform(rows)
        matrix = model.named_steps["prep"].transform(prepared)
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        contributions = matrix * estimator.coef_[0]
        names = model.named_steps["prep"].get_feature_names_out()
        for i, position in enumerate(positions):
            ranked = np.argsort(-np.abs(contributions[i]))[:5]
            terms = "; ".join(f"{names[j]}={contributions[i,j]:+.4f}" for j in ranked)
            output[position] = f"Base logit intercept={estimator.intercept_[0]:+.4f}; largest signed terms: {terms}. Not causal; not calibrated-probability contributions."
    else:
        base = model.predict_proba(rows)[:, 1]
        changes = []
        for field in rows:
            changed = rows.copy()
            changed[field] = np.nan if pd.api.types.is_numeric_dtype(rows[field]) else "Unknown"
            changes.append(base - model.predict_proba(changed)[:, 1])
        differences = np.asarray(changes).T
        for i, position in enumerate(positions):
            ranked = np.argsort(-np.abs(differences[i]))[:5]
            output[position] = "Base probability change on single-field ablation: " + "; ".join(f"{rows.columns[j]}={differences[i,j]:+.4f}" for j in ranked) + ". Not causal; terms need not add up."
    return output


@dataclass
class FraudDecisionEngine:
    """Single trusted artifact accepting raw transaction frames."""
    model: object
    calibrator: object
    policy: Policy
    cost: dict
    version: str = "2.0"

    def predict_proba(self, frame):
        p = self.calibrator.predict(self.model.predict_proba(features(frame))[:, 1])
        return np.column_stack([1 - p, p])

    def predict(self, frame):
        return predict(frame, self.model, self.calibrator, self.policy, self.cost, self.version)


@dataclass
class FraudModelPipeline:
    """Raw-frame feature engineering plus fitted preprocessing and base classifier."""
    model: object

    def predict_proba(self, frame):
        return self.model.predict_proba(features(frame))

    def predict(self, frame):
        return self.predict_proba(frame).argmax(axis=1)
