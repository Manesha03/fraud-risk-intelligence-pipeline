"""Development-only evidence and plots used by the reproducible report builder."""
from __future__ import annotations

import json
import platform
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score

from .core import Policy, SigmoidCalibrator, features, metrics, policy_stats, search_policy


def development_economics(candidates, xcal, ycal, xpolicy, ypolicy, amount, scenarios, reports):
    """All candidate calibration and policy fits use development partitions only."""
    rows = []
    for candidate in candidates:
        model, columns = candidate["model"], candidate["columns"]
        mapping = SigmoidCalibrator().fit(model.predict_proba(xcal[columns])[:, 1], ycal)
        probability = mapping.predict(model.predict_proba(xpolicy[columns])[:, 1])
        for scenario, setting in scenarios.items():
            grid = search_policy(ypolicy, probability, amount, setting)
            feasible = grid[grid.capacity_ok]
            if feasible.empty:
                continue
            best = feasible.nsmallest(1, "total_expected_cost").iloc[0].to_dict()
            rows.append({"model": candidate["name"], "cost_scenario": scenario,
                         "selection_ap": candidate["selection"]["average_precision"],
                         "selection_roc_auc": candidate["selection"]["roc_auc"],
                         "calibration_method": "sigmoid", **best})
    table = pd.DataFrame(rows)
    table.to_csv(reports / "model_cost_comparison.csv", index=False)
    return table


def fold_evidence(candidates, x, y, folds, reports):
    rows, importance = [], []
    for candidate in candidates:
        cols = candidate["columns"]
        for fold, (fit, validation) in enumerate(folds.split(x, y), 1):
            estimator = clone(candidate["model"]).fit(x.iloc[fit][cols], y[fit])
            probability = estimator.predict_proba(x.iloc[validation][cols])[:, 1]
            rows.append({"model": candidate["name"], "fold": fold,
                         **metrics(y[validation], probability),
                         "precision_050": precision_score(y[validation], probability >= .5, zero_division=0),
                         "recall_050": recall_score(y[validation], probability >= .5, zero_division=0),
                         "f1_050": f1_score(y[validation], probability >= .5, zero_division=0)})
            if candidate["name"] == "logistic_plain_base":
                scores = permutation_importance(estimator, x.iloc[validation][cols], y[validation],
                                                scoring="average_precision", n_repeats=5, random_state=42)
                importance.extend({"fold": fold, "feature": c, "ap_drop": float(value)}
                                  for c, value in zip(cols, scores.importances_mean))
    table = pd.DataFrame(rows)
    table.to_csv(reports / "candidate_cv_folds.csv", index=False)
    table.groupby("model").agg(ap_mean=("average_precision", "mean"), ap_sd=("average_precision", "std"),
                               roc_mean=("roc_auc", "mean"), roc_sd=("roc_auc", "std")).to_csv(reports / "candidate_cv_summary.csv")
    pd.DataFrame(importance).to_csv(reports / "selected_baseline_importance_folds.csv", index=False)


def extended_plots(train, select, model, calibrators, xpolicy, ypolicy, raw_policy, reports):
    """Training associations and development model responses, never test-led feature choice."""
    paths, effects, interactions = [], [], []

    def save(fig, name):
        fig.tight_layout()
        path = reports / name
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path.name)

    for field in ["transaction_amount", "transactions_last_1h", "transactions_last_24h",
                  "distance_from_usual_location_km", "account_age_days", "merchant_risk_score", "ip_risk_score", "hour"]:
        data = train[[field, "fraud"]].dropna().copy()
        data["bin"] = pd.qcut(data[field], 8, duplicates="drop")
        grouped = data.groupby("bin", observed=True).agg(count=("fraud", "size"), rate=("fraud", "mean"), midpoint=(field, "median"))
        se = np.sqrt(grouped.rate * (1 - grouped.rate) / grouped["count"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(grouped.midpoint, grouped.rate, yerr=1.96 * se, marker="o", capsize=3)
        for _, row in grouped.iterrows():
            ax.annotate(f"n={int(row['count'])}", (row.midpoint, row.rate), fontsize=7, xytext=(0, 8), textcoords="offset points")
        ax.set(xlabel=field, ylabel="Training fraud rate", title=f"Nonlinear association: {field} (approx. 95% intervals)")
        save(fig, f"nonlinear_{field}.png")
        grouped.assign(feature=field).to_csv(reports / f"nonlinear_{field}.csv")

    # Recompute engineered terms from raw interventions so linked fields stay consistent.
    for field in ["transaction_amount", "distance_from_usual_location_km", "merchant_risk_score", "transactions_last_1h"]:
        levels = np.unique(train[field].quantile(np.linspace(.05, .95, 15)).dropna())
        response = []
        for value in levels:
            changed = select.copy()
            changed[field] = value
            response.append(float(model.predict_proba(features(changed))[:, 1].mean()))
            effects.append({"feature": field, "value": float(value), "mean_base_probability": response[-1]})
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(levels, response, "o-")
        ax.set(xlabel=field, ylabel="Mean base-model fraud probability", title="Development partial dependence; associations, not causal effects")
        save(fig, f"dependence_{field}.png")
    pd.DataFrame(effects).to_csv(reports / "dependence.csv", index=False)

    for left, right in [("new_device", "distance_from_usual_location_km"),
                        ("merchant_risk_score", "transaction_amount"), ("authentication_level", "channel")]:
        data = train[[left, right, "fraud"]].copy()
        for field in (left, right):
            if pd.api.types.is_numeric_dtype(data[field]) and data[field].nunique() > 4:
                data[field] = pd.qcut(data[field], 3, duplicates="drop").astype(str)
            else:
                data[field] = data[field].fillna("Missing").astype(str)
        grouped = data.groupby([left, right], observed=True).fraud.agg(["mean", "size"])
        grouped.to_csv(reports / f"interaction_{left}_{right}.csv")
        pivot = grouped["mean"].unstack()
        counts = grouped["size"].unstack().reindex_like(pivot)
        fig, ax = plt.subplots(figsize=(9, 5))
        image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=max(.3, float(pivot.max().max())), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=20, ha="right")
        ax.set_yticks(range(len(pivot.index)), [str(c) for c in pivot.index])
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if pd.notna(pivot.iloc[i, j]):
                    ax.text(j, i, f"{pivot.iloc[i,j]:.1%}\nn={int(counts.iloc[i,j])}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="Training fraud rate")
        ax.set(xlabel=right, ylabel=left, title="Observed interaction; small cells are uncertain")
        save(fig, f"interaction_{left}_{right}.png")
        reliable = grouped[grouped["size"] >= 30]["mean"]
        interactions.append({"interaction": f"{left} x {right}", "eligible_cells": len(reliable),
                             "max_minus_min_rate": float(reliable.max() - reliable.min()) if len(reliable) else np.nan})
    pd.DataFrame(interactions).to_csv(reports / "interaction_effects.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probability in [("raw", raw_policy)] + [(name, mapping.predict(raw_policy)) for name, mapping in calibrators.items()]:
        observed, predicted = calibration_curve(ypolicy, probability, n_bins=6, strategy="quantile")
        ax.plot(predicted, observed, "o-", label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.legend()
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraud rate", title="Development calibration comparison")
    save(fig, "calibration_methods.png")
    return paths


def evidence_answers(economics, importance, sensitivity, grids, scenarios, ypolicy, ppolicy, amounts, reports):
    lines = ["# Evidence-based assessment", "", "All model/cost comparisons below use development data. They are exploratory optimized costs, not unbiased savings estimates.", ""]
    top = importance.head(5)
    lines += ["## Predictors and stability", top.to_string(index=False), "",
              "Fold estimates are in selected_baseline_importance_folds.csv. Negative permutation importance means the shuffled feature improved AP in that sample; correlated inputs and small samples can make rankings unstable. No causal claim follows.", ""]
    fold_importance = pd.read_csv(reports / "selected_baseline_importance_folds.csv")
    stability = fold_importance.groupby("feature").ap_drop.agg(["mean", "std", "min", "max"])
    stability["positive_folds"] = fold_importance.groupby("feature").ap_drop.apply(lambda values: int((values > 0).sum()))
    stability = stability.sort_values("mean", ascending=False)
    stability.to_csv(reports / "feature_stability_summary.csv")
    lines += ["Measured baseline stability (3 stratified folds):", stability.head(8).round(4).to_string(),
              "A feature positive in all three folds has more consistent direction in this experiment; overlapping uncertainty, only three folds, and unknown chronology prevent a claim of long-term stability."]
    winners = economics.loc[economics.groupby("cost_scenario").cost_per_transaction.idxmin()]
    lines += ["## Model economics", winners[["cost_scenario", "model", "selection_ap", "cost_per_transaction"]].to_string(index=False), ""]
    highest_roc = economics.loc[economics.selection_roc_auc.idxmax(), "model"]
    lines.append(f"Highest selection ROC-AUC candidate: {highest_roc}. Compare its cost below with each scenario winner:")
    lines.append(economics[economics.model.eq(highest_roc)][["cost_scenario", "cost_per_transaction"]].to_string(index=False))
    inversions = []
    for scenario, group in economics.groupby("cost_scenario"):
        for _, a in group.iterrows():
            for _, b in group.iterrows():
                if a.selection_ap < b.selection_ap and a.cost_per_transaction < b.cost_per_transaction:
                    inversions.append({"scenario": scenario, "lower_ap_model": a.model, "lower_ap": a.selection_ap,
                                       "lower_ap_cost": a.cost_per_transaction, "higher_ap_model": b.model,
                                       "higher_ap": b.selection_ap, "higher_ap_cost": b.cost_per_transaction,
                                       "cost_advantage": b.cost_per_transaction - a.cost_per_transaction})
    contrasts = pd.DataFrame(inversions)
    contrasts.to_csv(reports / "operational_model_tradeoffs.csv", index=False)
    if len(contrasts):
        lines += ["Measured examples where lower AP has lower optimized development cost:", contrasts.sort_values("cost_advantage", ascending=False).head(5).to_string(index=False)]
    else:
        lines.append("No lower-AP/lower-cost example was measured under these assumptions; do not assert one exists.")
    lines += ["", "## Calibration", "Calibration maps scores to empirical fraud frequencies on separate fit rows. A score of 0.10 should imply approximately 10 frauds per 100 comparable transactions only if validated calibration persists. Small calibration partitions and distribution shifts weaken this interpretation. See raw/sigmoid/isotonic Brier scores and calibration_methods.png.", "",
              "## Threshold sensitivity", sensitivity[["cost_scenario", "review_threshold", "block_threshold", "review_count", "cost_per_transaction"]].to_string(index=False)]
    workload = []
    for threshold in np.unique(np.r_[0, np.quantile(ppolicy, np.linspace(0, 1, 20)), 1]):
        workload.append(policy_stats(ypolicy, ppolicy, amounts, Policy(float(threshold), 1), scenarios["base"]))
    workload = pd.DataFrame(workload)
    workload.to_csv(reports / "recall_review_workload.csv", index=False)
    lines += ["", "## Recall and review workload", "Review-only threshold sweep (no model-driven blocking):", workload[["review_threshold", "recall", "review_count", "review_capacity_utilization"]].iloc[::4].to_string(index=False),
              "Higher recall requires more flagged transactions in this sweep; missing-amount review supplies a fixed workload floor. Capacity violations remain visible.", "",
              "## Interactions", pd.read_csv(reports / "interaction_effects.csv").to_string(index=False),
              "These are observed cell contrasts, not isolated interaction coefficients or causal effects. Counts and labeled heatmaps are supplied.", "",
              "## Drift priorities", "Prioritize the leading predictors above, transaction amount, velocity, authentication missingness, merchant/IP risk scores, channel mix, and output probabilities. Monitor point-in-time feature lineage and mature-label calibration. Upstream score changes, new merchants/devices, fraud adaptation, label delays, and changed review success can degrade performance."]
    dependence = pd.read_csv(reports / "dependence.csv")
    lines += ["", "## Counterintuitive explanations", "Measured base-model dependence endpoints on development rows:",
              dependence.groupby("feature").mean_base_probability.agg(["first", "last", "min", "max"]).round(5).to_string(),
              "A decreasing or flat amount/risk response can reflect weak signals, correlated fields, regularization, or sampling noise. These responses are model behavior, not evidence that increasing an amount or upstream risk score prevents fraud. Signed local terms are relative to encoded/scaled zero, not to an individual customer's normal behavior. A missing-input flag can be predictive of collection practices rather than fraud.",
              "", "## Validation boundaries", "The final model remains the average-precision-selected candidate; the economics table is a development comparison, not a retrospective switch based on the test. Every cost-table candidate uses the same sigmoid method for comparability, whereas the chosen candidate also has an isotonic comparison. Method choice and threshold optimization share the policy-selection sample, so optimized development cost is optimistic. CV scores for tuned candidates reuse training-selected hyperparameters and are descriptive, not nested-CV estimates. The final test had already been inspected before this revision; repeated reporting is regression evidence, not a new untouched holdout."]
    (reports / "assessment_answers.md").write_text("\n\n".join(lines), encoding="utf-8")
    return winners


def environment_manifest(root, partitions, data_path):
    import hashlib
    manifest = {"python": platform.python_version(), "packages": {name: version(name) for name in
                ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "nbformat", "pytest"]},
                "seeds": [42, 43, 44, 45], "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "validation": "Previously inspected stratified holdout; no fresh or temporal test available"}
    (root / "reports/environment.json").write_text(json.dumps(manifest, indent=2))
    (root / "requirements-lock.txt").write_text("\n".join(f"{name}=={value}" for name, value in manifest["packages"].items()) + "\n")
    pd.concat([frame[["transaction_id"]].assign(partition=name) for name, frame in partitions.items()]).to_csv(root / "reports/split_manifest.csv", index=False)
