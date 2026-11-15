"""Execute the reproducible experiment and write its eight deliverables."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import shutil
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nbformat
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_pipeline.core import ENGINEERED, Policy, SigmoidCalibrator, calibration_diagnostics, features, make_model, metrics, operational_decisions, policy_stats, predict, read_data, search_policy
from fraud_pipeline.core import FraudDecisionEngine, FraudModelPipeline, ReferenceFeatures, TrainingUnderSampler
from fraud_pipeline.assessment import development_economics, extended_plots, fold_evidence, evidence_answers, environment_manifest


def figure(title, fn):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fn(ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "transaction_fraud_risk_advanced_dataset.csv"))
    parser.add_argument("--skip-notebook", action="store_true", help="Used when a notebook rebuilds missing analysis outputs")
    args = parser.parse_args()
    df = read_data(args.data)
    cfg = json.loads((ROOT / "configs" / "cost.json").read_text())
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    # No event timestamp exists: IDs and hour/weekday do not establish chronology.
    dev, test = train_test_split(df, test_size=.2, stratify=df.fraud, random_state=42)
    train, remainder = train_test_split(dev, test_size=.375, stratify=dev.fraud, random_state=43)
    select, calibration = train_test_split(remainder, test_size=.5, stratify=remainder.fraud, random_state=44)
    cal_fit, policy_set = train_test_split(calibration, test_size=.5, stratify=calibration.fraud, random_state=45)
    partitions = {"train":train,"selection":select,"calibration_fit":cal_fit,"policy_selection":policy_set,"test":test}
    environment_manifest(ROOT, partitions, Path(args.data))
    audit = {"source":str(Path(args.data).name),"shape":list(df.shape),"fraud_counts":df.fraud.value_counts().to_dict(),"missing":df.isna().sum().to_dict(),"duplicates":int(df.duplicated().sum()),"duplicate_ids":int(df.transaction_id.duplicated().sum()),"partitions":{k:{"rows":len(v),"fraud":int(v.fraud.sum())} for k,v in partitions.items()},"validation_limit":"No reliable timestamp or customer identifier; stratified random partitions are not temporal evidence.","availability_questions":["prior_chargebacks","transactions_last_1h","transactions_last_24h","merchant_risk_score","ip_risk_score","distance_from_usual_location_km"]}
    audit["dtypes"] = {column:str(dtype) for column,dtype in df.dtypes.items()}
    audit["cardinality"] = {column:int(count) for column,count in df.nunique(dropna=True).items()}
    audit["high_cardinality"] = [column for column,count in audit["cardinality"].items() if count > 50]
    audit["numeric_quantiles_train"] = {column:{str(q):float(value) for q,value in train[column].quantile([.01,.5,.99]).items()} for column in train.select_dtypes(include="number") if column != "fraud"}
    audit["missingness_vs_fraud_train"] = {}
    for column in train:
        missing = train[column].isna()
        if missing.any():
            audit["missingness_vs_fraud_train"][column] = {"missing_count":int(missing.sum()),"fraud_rate_missing":float(train.loc[missing,"fraud"].mean()),"fraud_rate_present":float(train.loc[~missing,"fraud"].mean())}
    audit["impossible_or_inconsistent"] = {"negative_amount":int(df.transaction_amount.lt(0).sum()),"negative_distance":int(df.distance_from_usual_location_km.lt(0).sum()),"invalid_hour":int((~df.hour.between(0,23)).sum()),"invalid_weekday":int((~df.day_of_week.between(0,6)).sum()),"hourly_count_exceeds_daily":int(df.transactions_last_1h.gt(df.transactions_last_24h).sum())}
    for column in ["account_age_days", "device_age_days", "email_age_days", "transactions_last_1h", "transactions_last_24h", "failed_logins_24h", "prior_chargebacks", "login_to_transaction_minutes"]:
        audit["impossible_or_inconsistent"][f"negative_{column}"] = int(df[column].lt(0).sum())
    for column in ["merchant_risk_score", "ip_risk_score", "new_device", "billing_address_mismatch"]:
        audit["impossible_or_inconsistent"][f"outside_0_1_{column}"] = int((df[column].notna() & ~df[column].between(0, 1)).sum())
    df.select_dtypes(include="number").describe(percentiles=[.01,.05,.5,.95,.99]).to_csv(reports / "numeric_distributions.csv")
    pd.concat([df[c].value_counts(dropna=False).rename_axis("value").reset_index(name="count").assign(feature=c) for c in df.select_dtypes(exclude="number")]).to_csv(reports / "categorical_distributions.csv", index=False)
    audit["cleaning_decisions"] = ["No rows excluded for extreme values.","Training-fold numeric medians and categorical modes impute model inputs.","Missing transaction amount routes an otherwise approved transaction to review and uses the configured hypothetical fallback loss for cost accounting."]
    audit["leakage_review"] = {"prohibited_inputs":["fraud","transaction_id"],"availability_unverified":audit["availability_questions"],"label_delay":"Unknown: no label timestamp supplied","timezone":"Unknown: hour and weekday have no timezone metadata"}
    (reports / "data_audit.json").write_text(json.dumps(audit, indent=2))
    xtrain, xselect, xcal, xpolicy, xtest = [features(partitions[k]) for k in partitions]
    ytrain, yselect, ycal, ypolicy, ytest = [partitions[k].fraud.to_numpy() for k in partitions]
    candidates = []
    for engineered in (False, True):
        cols = list(xtrain.columns) if engineered else [c for c in xtrain if c not in ENGINEERED]
        for kind in ("logistic", "forest", "extra_trees"):
            for weighted in (False, True):
                model = make_model(xtrain[cols], kind, weighted)
                model.fit(xtrain[cols], ytrain)
                p = model.predict_proba(xselect[cols])[:, 1]
                candidates.append({"name":f"{kind}_{'weighted' if weighted else 'plain'}_{'engineered' if engineered else 'base'}","model":model,"columns":cols,"selection":metrics(yselect,p)})
    base_cols = [c for c in xtrain if c not in ENGINEERED]
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        make_model(xtrain[base_cols], "logistic"),
        {"model__C":[0.03,0.1,0.3,1.0,3.0,10.0]},
        n_iter=6, scoring="average_precision", cv=folds, refit=True,
        random_state=42, n_jobs=1, return_train_score=False,
    )
    search.fit(xtrain[base_cols], ytrain)
    cv_table = pd.DataFrame(search.cv_results_)[["param_model__C","mean_test_score","std_test_score","split0_test_score","split1_test_score","split2_test_score"]]
    cv_table.to_csv(reports / "stratified_cv_tuning.csv", index=False)
    fold_importance = []
    for fold_number,(fit_idx,validation_idx) in enumerate(folds.split(xtrain[base_cols],ytrain),1):
        fold_model = clone(search.best_estimator_)
        fold_model.fit(xtrain.iloc[fit_idx][base_cols],ytrain[fit_idx])
        scores = permutation_importance(fold_model,xtrain.iloc[validation_idx][base_cols],ytrain[validation_idx],n_repeats=2,random_state=42,scoring="average_precision")
        fold_importance.extend({"fold":fold_number,"feature":feature,"ap_drop":float(value)} for feature,value in zip(base_cols,scores.importances_mean))
    pd.DataFrame(fold_importance).to_csv(reports / "feature_importance_folds.csv",index=False)
    candidates.append({"name":"logistic_tuned_base","model":search.best_estimator_,"columns":base_cols,"selection":metrics(yselect,search.predict_proba(xselect[base_cols])[:,1])})
    undersampled_model = TrainingUnderSampler(make_model(xtrain[base_cols],"logistic"))
    undersampled_model.fit(xtrain[base_cols], ytrain)
    candidates.append({"name":"logistic_undersampled_base","model":undersampled_model,"columns":base_cols,"selection":metrics(yselect,undersampled_model.predict_proba(xselect[base_cols])[:,1])})
    for kind in ("forest", "extra_trees"):
        nonlinear_search = RandomizedSearchCV(make_model(xtrain, kind),
            {"model__min_samples_leaf": [3, 8, 15], "model__max_features": ["sqrt", .7]},
            n_iter=3, scoring="average_precision", cv=folds, random_state=42, n_jobs=1)
        nonlinear_search.fit(xtrain, ytrain)
        pd.DataFrame(nonlinear_search.cv_results_).to_csv(reports / f"{kind}_tuning.csv", index=False)
        candidates.append({"name": f"{kind}_tuned_engineered", "model": nonlinear_search.best_estimator_,
                           "columns": list(xtrain), "selection": metrics(yselect, nonlinear_search.predict_proba(xselect)[:,1])})
    fold_evidence(candidates, xtrain, ytrain, folds, reports)
    table = pd.DataFrame([{"name":c["name"],**c["selection"]} for c in candidates]).sort_values("average_precision", ascending=False)
    table.to_csv(reports / "model_comparison.csv", index=False)
    table[table.name.isin(["logistic_plain_base","logistic_weighted_base","logistic_undersampled_base"])].to_csv(reports / "imbalance_comparison.csv",index=False)
    best_name = table.iloc[0]["name"]
    best = next(c for c in candidates if c["name"] == best_name)
    model, cols = best["model"], best["columns"]
    raw_cal = model.predict_proba(xcal[cols])[:,1]
    calibrators = {"sigmoid":SigmoidCalibrator().fit(raw_cal,ycal),"isotonic":IsotonicRegression(out_of_bounds="clip").fit(raw_cal,ycal)}
    raw_policy = model.predict_proba(xpolicy[cols])[:,1]
    cal_scores = pd.DataFrame([{"method":name,**metrics(ypolicy,cal.predict(raw_policy)),**calibration_diagnostics(ypolicy,cal.predict(raw_policy))} for name,cal in calibrators.items()]).sort_values("brier")
    raw_diagnostics = {"method": "uncalibrated", **metrics(ypolicy, raw_policy), **calibration_diagnostics(ypolicy, raw_policy)}
    pd.concat([pd.DataFrame([raw_diagnostics]), cal_scores]).to_csv(reports / "calibration_before_after.csv", index=False)
    cal_scores.to_csv(reports / "calibration_comparison.csv", index=False)
    cal_name = cal_scores.iloc[0]["method"]
    calibrator = calibrators[cal_name]
    pcal = calibrator.predict(raw_policy)
    calibration_bands = pd.DataFrame({"observed":ypolicy,"probability":pcal})
    calibration_bands["risk_band"] = pd.cut(calibration_bands.probability,bins=[0,.05,.1,.15,.2,.3,.5,1],include_lowest=True)
    calibration_bands.groupby("risk_band",observed=True).agg(count=("observed","size"),fraud_rate=("observed","mean"),mean_probability=("probability","mean")).to_csv(reports / "calibration_risk_bands.csv")
    amount_cal = policy_set.transaction_amount.to_numpy(dtype=float)
    scenarios = {"base":cfg,"higher_fraud_loss":{**cfg,"fraud_loss_multiplier":2},"higher_block_friction":{**cfg,"legitimate_block_cost":50},"lower_review_success":{**cfg,"review_success":.6},"higher_review_cost":{**cfg,"review_cost":6}}
    scenarios.update({"higher_review_friction": {**cfg, "legitimate_review_friction": 5},
                      "higher_fraud_block_cost": {**cfg, "fraud_block_cost": 5},
                      "higher_missing_loss": {**cfg, "missing_amount_fraud_loss": 250},
                      "tighter_review_capacity": {**cfg, "review_capacity_fraction": .10}})
    (reports / "cost_assumptions.json").write_text(json.dumps(scenarios, indent=2))
    economics = development_economics(candidates, xcal, ycal, xpolicy, ypolicy, amount_cal, scenarios, reports)
    grids = []
    for name,setting in scenarios.items():
        grid = search_policy(ypolicy,pcal,amount_cal,setting)
        grid.insert(0,"cost_scenario",name)
        grid.insert(0,"calibration_method",cal_name)
        grid.insert(0,"model",best_name)
        grids.append(grid)
    threshold_table = pd.concat(grids,ignore_index=True)
    threshold_table.to_csv(ROOT / "threshold_cost_analysis.csv",index=False)
    sensitivity = pd.concat([grid[grid.capacity_ok].nsmallest(1,"total_expected_cost") for grid in grids],ignore_index=True)
    sensitivity.to_csv(reports / "cost_sensitivity.csv",index=False)
    valid = grids[0].query("capacity_ok").sort_values("total_expected_cost")
    chosen = valid.iloc[0]
    policy = Policy(float(chosen.review_threshold),float(chosen.block_threshold))
    joblib.dump({"model":model,"pipeline":FraudModelPipeline(model),"columns":cols,"version":"1.0","selection_metrics":best["selection"]}, ROOT / "fraud_model_pipeline.joblib")
    engine = FraudDecisionEngine(model, calibrator, policy, cfg)
    joblib.dump({"calibrator":calibrator,"method":cal_name,"policy":policy,"cost":cfg,"engine":engine},ROOT / "calibrated_model.joblib")
    ptest = calibrator.predict(model.predict_proba(xtest[cols])[:,1])
    test_metrics = metrics(ytest,ptest)
    test_metrics.update(calibration_diagnostics(ytest,ptest))
    test_stats = policy_stats(ytest,ptest,test.transaction_amount.to_numpy(dtype=float),policy,cfg)
    test_decisions = operational_decisions(ptest,test.transaction_amount.to_numpy(dtype=float),policy)
    test_metrics.update({"precision_at_review":float(precision_score(ytest,test_decisions!="APPROVE",zero_division=0)),"recall_at_review":float(recall_score(ytest,test_decisions!="APPROVE",zero_division=0)),"f1_at_review":float(f1_score(ytest,test_decisions!="APPROVE",zero_division=0)),"confusion_at_review":confusion_matrix(ytest,test_decisions!="APPROVE").tolist()})
    baseline = policy_stats(ytest,ptest,test.transaction_amount.to_numpy(dtype=float),Policy(1,1),cfg,force_missing_review=False)
    binary = policy_stats(ytest,ptest,test.transaction_amount.to_numpy(dtype=float),Policy(.5,.5),cfg,force_missing_review=False)
    review_only = policy_stats(ytest,ptest,test.transaction_amount.to_numpy(dtype=float),Policy(policy.review,1),cfg)
    summary = {"selected_model":best_name,"calibration":cal_name,"policy":{"review":policy.review,"block":policy.block},"selection_metrics":best["selection"],"calibration_metrics":cal_scores.iloc[0].to_dict(),"stratified_cv_best_C":float(search.best_params_["model__C"]),"stratified_cv_ap_mean":float(search.best_score_),"test_metrics":test_metrics,"test_policy":test_stats,"approve_all":baseline,"binary_050":binary,"review_only":review_only,"limitations":audit["validation_limit"]}
    (reports / "results.json").write_text(json.dumps(summary,indent=2))
    processed_features = ReferenceFeatures().fit(xtrain).transform(features(df))
    processed = df.drop(columns=["fraud"]).join(processed_features[ENGINEERED]).assign(fraud=df.fraud)
    processed.to_csv(ROOT / "processed_dataset.csv",index=False)
    prediction = predict(test,model,calibrator,policy,cfg)
    prediction.to_csv(reports / "test_predictions.csv",index=False)
    if hasattr(model.named_steps["model"], "coef_"):
        flagged=np.flatnonzero(prediction.decision.to_numpy()!="APPROVE")[:5]
        names=model.named_steps["prep"].get_feature_names_out()
        transformed=model.named_steps["prep"].transform(model.named_steps["reference"].transform(xtest.iloc[flagged][cols]))
        if hasattr(transformed,"toarray"):
            transformed=transformed.toarray()
        contributions=np.asarray(transformed) * model.named_steps["model"].coef_[0]
        local=[]
        for pos,row in enumerate(flagged):
            ranked=np.argsort(-contributions[pos])[:5]
            local.append({"transaction_id":str(test.iloc[row].transaction_id),"calibrated_probability":float(ptest[row]),"decision":str(test_decisions[row]),"top_positive_base_logit_contributions":"; ".join(f"{names[j]}={contributions[pos,j]:.3f}" for j in ranked if contributions[pos,j]>0),"limitation":"Base logistic log-odds contributions; sigmoid calibration changes the probability scale; associations are not causal."})
        pd.DataFrame(local).to_csv(reports/"local_explanations.csv",index=False)
    prediction.loc[prediction.decision.ne("APPROVE"), ["transaction_id", "fraud_probability", "decision", "explanation"]].head(5).to_csv(reports / "local_explanations.csv", index=False)
    importance = permutation_importance(model,xselect[cols],yselect,n_repeats=3,random_state=42,scoring="average_precision")
    importance_table = pd.DataFrame({"feature":cols,"ap_drop_mean":importance.importances_mean,"ap_drop_sd":importance.importances_std}).sort_values("ap_drop_mean",ascending=False)
    importance_table.to_csv(reports/"feature_importance.csv",index=False)
    extra_paths = extended_plots(train, select, model, calibrators, xpolicy, ypolicy, raw_policy, reports)
    winners = evidence_answers(economics, importance_table, sensitivity, grids, scenarios, ypolicy, pcal, amount_cal, reports)
    stress_rows=[]
    for name,changed in [
        ("base",test.copy()),
        ("amount_plus_50pct",test.assign(transaction_amount=test.transaction_amount*1.5)),
        ("velocity_plus_2",test.assign(transactions_last_1h=test.transactions_last_1h+2)),
        ("missing_amount_10pct",test.copy()),
        ("unseen_channel",test.assign(channel="Novel channel")),
        ("ecommerce_channel_mix",test.assign(channel="E-commerce")),
        ("extreme_top_amounts",test.copy()),
    ]:
        if name=="missing_amount_10pct":
            chosen_rows=np.random.default_rng(42).choice(changed.index,size=max(1,len(changed)//10),replace=False)
            changed.loc[chosen_rows,"transaction_amount"]=np.nan
        if name=="extreme_top_amounts":
            chosen_rows=changed.transaction_amount.nlargest(max(1,len(changed)//100)).index
            changed.loc[chosen_rows,"transaction_amount"] *= 10
        try:
            score=calibrator.predict(model.predict_proba(features(changed)[cols])[:,1])
            stat=policy_stats(ytest,score,changed.transaction_amount.to_numpy(dtype=float),policy,cfg,
                              exposure_amount=test.transaction_amount.to_numpy(dtype=float))
            stress_rows.append({"scenario":name,**metrics(ytest,score),**stat,"failed":0})
        except Exception as error:
            stress_rows.append({"scenario":name,"failed":len(changed),"error":str(error)})
    weights = np.where(ytest == 1,2.0,1.0)
    selected_rows = np.random.default_rng(42).choice(len(test),size=len(test),replace=True,p=weights/weights.sum())
    shifted = test.iloc[selected_rows]
    shifted_y = shifted.fraud.to_numpy()
    shifted_score = calibrator.predict(model.predict_proba(features(shifted)[cols])[:,1])
    stress_rows.append({"scenario":"fraud_prevalence_weight_2x",**metrics(shifted_y,shifted_score),**policy_stats(shifted_y,shifted_score,shifted.transaction_amount.to_numpy(dtype=float),policy,cfg),"failed":0})
    stress_table = pd.DataFrame(stress_rows)
    stress_table.to_csv(reports/"robustness.csv",index=False)
    figures=[]
    figures.append(figure("Class counts, full supplied data",lambda ax:df.fraud.value_counts().sort_index().plot.bar(ax=ax)))
    figures.append(figure("Missing values by field",lambda ax:df.isna().sum().sort_values().tail(12).plot.barh(ax=ax)))
    figures.append(figure("Transaction amount, training",lambda ax:train.transaction_amount.clip(upper=train.transaction_amount.quantile(.99)).hist(ax=ax,bins=30)))
    for field in ["channel","merchant_category","authentication_level","device_type","customer_tier","day_of_week"]:
        figures.append(figure(f"Training fraud rate by {field}",lambda ax,field=field:train.groupby(field,dropna=False).fraud.agg(["mean","count"]).sort_values("mean")["mean"].plot.barh(ax=ax)))
    for field,other in [("new_device","distance_from_usual_location_km"),("merchant_risk_score","transaction_amount"),("authentication_level","channel")]:
        tmp=train.copy()
        if pd.api.types.is_numeric_dtype(tmp[other]): tmp[other]=pd.qcut(tmp[other],3,duplicates="drop")
        if pd.api.types.is_numeric_dtype(tmp[field]) and tmp[field].nunique()>4: tmp[field]=pd.qcut(tmp[field],3,duplicates="drop")
        pivot=tmp.pivot_table(index=field,columns=other,values="fraud",aggfunc="mean",observed=True)
        figures.append(figure(f"Training interaction: {field} x {other}",lambda ax,pivot=pivot: ax.imshow(pivot.fillna(0).to_numpy(),aspect="auto")))
    figures.append(figure("Test precision-recall curve",lambda ax:PrecisionRecallDisplay.from_predictions(ytest,ptest,ax=ax)))
    frac,mean=calibration_curve(ytest,ptest,n_bins=8,strategy="quantile")
    figures.append(figure("Test calibration",lambda ax:(ax.plot(mean,frac,"o-"),ax.plot([0,1],[0,1],"--"),ax.set_xlim(0,1),ax.set_ylim(0,1))))
    figures.append(figure("Calibration cost against review threshold",lambda ax:valid.groupby("review_threshold").total_expected_cost.min().plot(ax=ax)))
    for i,fig in enumerate(figures,1):
        fig.savefig(reports/f"figure_{i:02d}.png",dpi=120)
        plt.close(fig)
    for number, name in enumerate(["interaction_new_device_distance_from_usual_location_km.png",
                                   "interaction_merchant_risk_score_transaction_amount.png",
                                   "interaction_authentication_level_channel.png"], 10):
        shutil.copyfile(reports / name, reports / f"figure_{number:02d}.png")
    from fraud_pipeline.business_report import build_business_report
    build_business_report(ROOT)
    with PdfPages(ROOT/"model_card.pdf") as pdf:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.94,"Fraud model card | version 1.0",fontsize=18,weight="bold")
        lines=[f"Source: supplied transaction CSV; {len(df)} rows; target: fraud.","Intended use: research and offline decision simulation only.","Out of scope: live authorization without timestamp/lineage validation.",f"Model: {best_name}; calibration: {cal_name}.",f"Test ROC-AUC {test_metrics['roc_auc']:.3f}; AP {test_metrics['average_precision']:.3f}; Brier {test_metrics['brier']:.3f}.",f"Thresholds: review {policy.review:.4f}; block {policy.block:.4f}.",f"Hypothetical test cost/transaction: {test_stats['cost_per_transaction']:.2f}.",f"Missing amount fallback fraud loss: {cfg['missing_amount_fraud_loss']}; otherwise review.","Split: stratified train/selection/calibration/test; no timestamp supplied.","Leakage concern: chargebacks, velocity and risk scores need point-in-time proof.","Missingness: median/mode imputation learned on train only.","Limitations: no temporal, external, fairness or real-world cost validation.","Human oversight: REVIEW cases require analyst disposition.","Monitor input quality, drift, decisions, mature-label precision and calibration.","Rollback on severe data failures or verified performance loss.","Joblib artifacts must only be loaded from trusted sources."]
        fig.text(.08,.86,"\n\n".join(lines),fontsize=10,va="top",wrap=True);pdf.savefig(fig);plt.close(fig)
    if not args.skip_notebook:
        cells = []
        def section(title, description, code=None):
            cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n{description}"))
            if code:
                cells.append(nbformat.v4.new_code_cell(code))

        cells.append(nbformat.v4.new_markdown_cell("# Fraud risk analysis\nOffline assessment on supplied data. Costs and review outcomes are assumptions, not observed business results."))
        section("1. Objective and assumptions", "Score fraud probability, simulate three decisions, and minimize hypothetical cost subject to review capacity. No event timestamps or customer keys are available.")
        section("2. Environment and reproducibility", "Seeds 42-45 control splitting and models. Execute from the repository root; raw CSV belongs in `data/raw/`.", "from pathlib import Path\nimport json, pandas as pd, subprocess, sys\nprint('Python:', sys.version.split()[0])")
        section("3. Dataset loading", "Running this notebook rebuilds analysis outputs from source, without regenerating the open notebook. Existing outputs are not silently reused.", "source = " + repr(str(Path(args.data).resolve())) + "\nif not globals().get('_pipeline_build_complete', False):\n    subprocess.run([sys.executable, 'scripts/run_analysis.py', '--skip-notebook', '--data', source], check=True)\ndata = pd.read_csv(source)\nprint('Shape:', data.shape, 'Fraud:', int(data.fraud.sum()))")
        section("4. Data audit", "Schema, missingness, duplicate identifiers, and partition counts are machine-readable in `reports/data_audit.json`.", "audit = json.loads(Path('reports/data_audit.json').read_text())\nprint(pd.Series(audit['missing']).sort_values(ascending=False).head(12).to_string())\nprint(audit['partitions'])")
        section("5. Leakage assessment", "`fraud` and transaction IDs are excluded. Chargebacks, velocity, distance, and upstream scores require point-in-time lineage proof before deployment.")
        section("6. Exploratory analysis", "Charts use training rows for fraud patterns. Full-data quality charts are descriptive. Interaction cells may have small sample sizes.\n\n![Class distribution](reports/figure_01.png)\n\n![Missingness](reports/figure_02.png)\n\n![Channel fraud rates](reports/figure_04.png)\n\n![Device-distance interaction](reports/figure_10.png)\n\n![Merchant amount interaction](reports/figure_11.png)")
        section("7. Feature engineering", "Eighteen derived features include training-fold merchant amount normalization and channel distance deviation. Population references are fitted inside each training pipeline; they are not personal customer histories.", "print(pd.read_csv('processed_dataset.csv', nrows=1).columns.tolist())")
        section("8. Validation", "Stratified train, model-selection, calibrator-fit, policy-selection, and final-test partitions are disjoint. They are not temporal validation.", "print(pd.DataFrame(audit['partitions']).T.to_string())")
        section("9. Baseline modeling", "Unweighted logistic regression with imputation, scaling, and one-hot encoding is the reference model.", "comparison = pd.read_csv('reports/model_comparison.csv')\nprint(comparison[comparison.name.eq('logistic_plain_base')].to_string(index=False))")
        section("10. Nonlinear models", "Random forest and extra trees are compared under the same partition design.", "print(comparison[comparison.name.str.startswith(('forest','extra_trees'))].head(6).to_string(index=False))")
        section("11. Hyperparameter tuning", "Logistic C is searched in three stratified training folds. These folds do not establish temporal stability.", "print(pd.read_csv('reports/stratified_cv_tuning.csv').to_string(index=False))")
        section("12. Class imbalance", "Unweighted, class-weighted, and training-only undersampled logistic models are compared on untouched selection rows.", "print(pd.read_csv('reports/imbalance_comparison.csv').to_string(index=False))")
        section("13. Calibration", "Sigmoid and isotonic mappings fit on calibration-fit rows and are compared on policy-selection rows.\n\n![Calibration](reports/figure_14.png)", "print(pd.read_csv('reports/calibration_comparison.csv').to_string(index=False))\nprint(pd.read_csv('reports/calibration_risk_bands.csv').to_string(index=False))")
        section("14. Cost and threshold selection", "Joint threshold search uses hypothetical costs; missing amounts have an explicit fallback loss and cannot be automatically approved.\n\n![Cost by review threshold](reports/figure_15.png)", "thresholds = pd.read_csv('threshold_cost_analysis.csv')\nvalid = thresholds[thresholds.capacity_ok]\nprint(valid.loc[valid.groupby('cost_scenario').total_expected_cost.idxmin()][['cost_scenario','review_threshold','block_threshold','total_expected_cost']].to_string(index=False))")
        section("15. Explainability", "Permutation importance and local base-model log-odds contributions are associations, not calibrated-probability or causal contributions. Fold importance is stratified, not temporal.", "print(pd.read_csv('reports/feature_importance.csv').head(10).to_string(index=False))\nprint(pd.read_csv('reports/feature_importance_folds.csv').groupby('feature').ap_drop.agg(['mean','std']).sort_values('mean',ascending=False).head(10).to_string())\nprint(pd.read_csv('reports/local_explanations.csv').head().to_string(index=False))")
        section("16. Fraud operations", "Review success and financial impacts are simulated; compare approve-all, 0.50 binary, review-only, and the selected policy.", "results = json.loads(Path('reports/results.json').read_text())\nprint({k:round(results[k]['cost_per_transaction'],3) for k in ['approve_all','binary_050','review_only','test_policy']})")
        section("17. Robustness", "Controlled perturbations retain labels and do not reproduce adaptive fraud.", "print(pd.read_csv('reports/robustness.csv')[['scenario','average_precision','cost_per_transaction','review_count','failed']].to_string(index=False))")
        section("18. Final test", "The final partition was evaluated after model, calibration, and policy selection.\n\n![Precision-recall curve](reports/figure_13.png)", "print(json.dumps(results['test_metrics'], indent=2))\nprint(json.dumps(results['test_policy'], indent=2))")
        section("19. Monitoring and governance", "Monitor missingness, unseen categories, score and decision rates, review capacity, mature-label precision, calibration, and realized loss. Human approval and rollback procedures are required before live use.")
        section("20. Conclusions and limits", "The dataset lacks event and label timestamps, customer keys, feature lineage, actual review outcomes, verified costs, and a fairness audit. No future-period or production-performance claim is supported.")
        cells.append(nbformat.v4.new_markdown_cell("### Added evidence and required business questions\n" + (reports / "assessment_answers.md").read_text(encoding="utf-8") + "\n\n" + "\n\n".join(f"![{path}](reports/{path})" for path in extra_paths)))
        cells.append(nbformat.v4.new_code_cell("print(pd.read_csv('reports/calibration_before_after.csv').to_string(index=False))\nprint(pd.read_csv('reports/candidate_cv_summary.csv').to_string(index=False))"))
        notebook = nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
        scope = {"_pipeline_build_complete": True}
        execution_count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(compile(cell.source,"fraud_risk_analysis.ipynb","exec"),scope)
            cell.execution_count = execution_count
            cell.outputs = [nbformat.v4.new_output("stream",name="stdout",text=output.getvalue())]
        nbformat.validate(notebook)
        nbformat.write(notebook,ROOT/"fraud_risk_analysis.ipynb")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
