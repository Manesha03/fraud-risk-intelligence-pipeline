"""Execute the reproducible experiment and write its eight deliverables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nbformat
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_pipeline.core import ENGINEERED, Policy, SigmoidCalibrator, features, make_model, metrics, policy_stats, predict, read_data, search_policy


def figure(title, fn):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fn(ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "transaction_fraud_risk_advanced_dataset.csv"))
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
    audit = {"source":str(Path(args.data).name),"shape":list(df.shape),"fraud_counts":df.fraud.value_counts().to_dict(),"missing":df.isna().sum().to_dict(),"duplicates":int(df.duplicated().sum()),"duplicate_ids":int(df.transaction_id.duplicated().sum()),"partitions":{k:{"rows":len(v),"fraud":int(v.fraud.sum())} for k,v in partitions.items()},"validation_limit":"No reliable timestamp or customer identifier; stratified random partitions are not temporal evidence.","availability_questions":["prior_chargebacks","transactions_last_1h","transactions_last_24h","merchant_risk_score","ip_risk_score","distance_from_usual_location_km"]}
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
    table = pd.DataFrame([{"name":c["name"],**c["selection"]} for c in candidates]).sort_values("average_precision", ascending=False)
    table.to_csv(reports / "model_comparison.csv", index=False)
    best_name = table.iloc[0]["name"]
    best = next(c for c in candidates if c["name"] == best_name)
    model, cols = best["model"], best["columns"]
    raw_cal = model.predict_proba(xcal[cols])[:,1]
    calibrators = {"sigmoid":SigmoidCalibrator().fit(raw_cal,ycal),"isotonic":IsotonicRegression(out_of_bounds="clip").fit(raw_cal,ycal)}
    raw_policy = model.predict_proba(xpolicy[cols])[:,1]
    cal_scores = pd.DataFrame([{"method":name,**metrics(ypolicy,cal.predict(raw_policy))} for name,cal in calibrators.items()]).sort_values("brier")
    cal_scores.to_csv(reports / "calibration_comparison.csv", index=False)
    cal_name = cal_scores.iloc[0]["method"]
    calibrator = calibrators[cal_name]
    pcal = calibrator.predict(raw_policy)
    amount_cal = policy_set.transaction_amount.to_numpy(dtype=float)
    scenarios = {"base":cfg,"higher_fraud_loss":{**cfg,"fraud_loss_multiplier":2},"higher_block_friction":{**cfg,"legitimate_block_cost":50},"lower_review_success":{**cfg,"review_success":.6},"higher_review_cost":{**cfg,"review_cost":6}}
    grids = []
    for name,setting in scenarios.items():
        grid = search_policy(ypolicy,pcal,amount_cal,setting)
        grid.insert(0,"cost_scenario",name)
        grid.insert(0,"calibration_method",cal_name)
        grid.insert(0,"model",best_name)
        grids.append(grid)
    threshold_table = pd.concat(grids,ignore_index=True)
    threshold_table.to_csv(ROOT / "threshold_cost_analysis.csv",index=False)
    valid = grids[0].query("capacity_ok").sort_values("total_expected_cost")
    chosen = valid.iloc[0]
    policy = Policy(float(chosen.review_threshold),float(chosen.block_threshold))
    joblib.dump({"model":model,"columns":cols,"version":"1.0","selection_metrics":best["selection"]}, ROOT / "fraud_model_pipeline.joblib")
    joblib.dump({"calibrator":calibrator,"method":cal_name,"policy":policy,"cost":cfg},ROOT / "calibrated_model.joblib")
    ptest = calibrator.predict(model.predict_proba(xtest[cols])[:,1])
    test_metrics = metrics(ytest,ptest)
    test_stats = policy_stats(ytest,ptest,test.transaction_amount.to_numpy(dtype=float),policy,cfg)
    test_decisions = policy.decide(ptest)
    test_metrics.update({"precision_at_review":float(precision_score(ytest,test_decisions!="APPROVE",zero_division=0)),"recall_at_review":float(recall_score(ytest,test_decisions!="APPROVE",zero_division=0)),"f1_at_review":float(f1_score(ytest,test_decisions!="APPROVE",zero_division=0)),"confusion_at_review":confusion_matrix(ytest,test_decisions!="APPROVE").tolist()})
    baseline = policy_stats(ytest,ptest,test.transaction_amount.to_numpy(dtype=float),Policy(1,1),cfg)
    binary = policy_stats(ytest,ptest,test.transaction_amount.to_numpy(dtype=float),Policy(.5,.5),cfg)
    summary = {"selected_model":best_name,"calibration":cal_name,"policy":{"review":policy.review,"block":policy.block},"selection_metrics":best["selection"],"calibration_metrics":cal_scores.iloc[0].to_dict(),"test_metrics":test_metrics,"test_policy":test_stats,"approve_all":baseline,"binary_050":binary,"limitations":audit["validation_limit"]}
    (reports / "results.json").write_text(json.dumps(summary,indent=2))
    processed = df.drop(columns=["fraud"]).join(features(df)[ENGINEERED]).assign(fraud=df.fraud)
    processed.to_csv(ROOT / "processed_dataset.csv",index=False)
    prediction = predict(test,model,calibrator,policy,cfg)
    prediction.to_csv(reports / "test_predictions.csv",index=False)
    if hasattr(model.named_steps["model"], "coef_"):
        flagged=np.flatnonzero(prediction.decision.to_numpy()!="APPROVE")[:5]
        names=model.named_steps["prep"].get_feature_names_out()
        transformed=model.named_steps["prep"].transform(xtest.iloc[flagged][cols])
        if hasattr(transformed,"toarray"):
            transformed=transformed.toarray()
        contributions=np.asarray(transformed) * model.named_steps["model"].coef_[0]
        local=[]
        for pos,row in enumerate(flagged):
            ranked=np.argsort(-contributions[pos])[:5]
            local.append({"transaction_id":str(test.iloc[row].transaction_id),"calibrated_probability":float(ptest[row]),"decision":str(test_decisions[row]),"top_positive_base_logit_contributions":"; ".join(f"{names[j]}={contributions[pos,j]:.3f}" for j in ranked if contributions[pos,j]>0),"limitation":"Base logistic log-odds contributions; sigmoid calibration changes the probability scale; associations are not causal."})
        pd.DataFrame(local).to_csv(reports/"local_explanations.csv",index=False)
    importance = permutation_importance(model,xselect[cols],yselect,n_repeats=3,random_state=42,scoring="average_precision")
    importance_table = pd.DataFrame({"feature":cols,"ap_drop_mean":importance.importances_mean,"ap_drop_sd":importance.importances_std}).sort_values("ap_drop_mean",ascending=False)
    importance_table.to_csv(reports/"feature_importance.csv",index=False)
    stress_rows=[]
    for name,changed in [
        ("base",test.copy()),
        ("amount_plus_50pct",test.assign(transaction_amount=test.transaction_amount*1.5)),
        ("velocity_plus_2",test.assign(transactions_last_1h=test.transactions_last_1h+2)),
        ("missing_amount_10pct",test.copy()),
        ("unseen_channel",test.assign(channel="Novel channel")),
    ]:
        if name=="missing_amount_10pct":
            chosen_rows=np.random.default_rng(42).choice(changed.index,size=max(1,len(changed)//10),replace=False)
            changed.loc[chosen_rows,"transaction_amount"]=np.nan
        try:
            score=calibrator.predict(model.predict_proba(features(changed)[cols])[:,1])
            stat=policy_stats(ytest,score,test.transaction_amount.to_numpy(dtype=float),policy,cfg)
            stress_rows.append({"scenario":name,**metrics(ytest,score),**stat,"failed":0})
        except Exception:
            stress_rows.append({"scenario":name,"failed":len(changed)})
    pd.DataFrame(stress_rows).to_csv(reports/"robustness.csv",index=False)
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
    # Each page contains measured results or explicitly labeled assumptions.
    pages=[("Executive summary",[f"Supplied dataset: {len(df):,} transactions; fraud prevalence {df.fraud.mean():.1%}.",f"Selected: {best_name}; {cal_name} calibration.",f"Test AP {test_metrics['average_precision']:.3f}; ROC-AUC {test_metrics['roc_auc']:.3f}.","Costs and review outcomes below are hypothetical, not observed business outcomes."],0),
    ("Data and quality",[f"Missing authentication: {df.authentication_level.isna().sum()}; amount: {df.transaction_amount.isna().sum()}.","No event timestamp or customer key: temporal and historical evaluation unavailable.","Prior chargebacks and upstream risk scores require authorization-time verification."],1),
    ("Fraud patterns",["Charts show training-partition associations, not causal effects.","Small cells and missing categories require caution."],3),
    ("Model comparison",[table.head(6).round(3).to_string(index=False),"Selection used a disjoint development partition; no temporal claim."],None),
    ("Calibration",[cal_scores.round(4).to_string(index=False),"Calibrators were fitted on a separate calibration-fit partition.","Method comparison and policy search share the policy-selection partition."],12),
    ("Cost policy",[f"Review >= {policy.review:.4f}; block >= {policy.block:.4f}.",f"Review labor {cfg['review_cost']}; legitimate block {cfg['legitimate_block_cost']}; review success {cfg['review_success']:.0%}.","Threshold search uses calibration labels; cost estimates are conditional simulation."],13),
    ("Operational test simulation",[f"Review {test_stats['review_count']}; block {test_stats['block_count']}; legitimate blocks {test_stats['legitimate_block_count']}.",f"Policy cost {test_stats['total_expected_cost']:.2f}; approve-all {baseline['total_expected_cost']:.2f}; 0.50 binary {binary['total_expected_cost']:.2f}.",f"Fraud capture assumption-adjusted: {test_stats['fraud_capture_rate']:.1%}.","Review capacity is a per-batch fraction, not a staffed SLA."],None),
    ("Explainability",["Permutation importance and five local base-logit explanations are saved in reports/.","Sigmoid calibration changes the probability scale; logit contributions are not calibrated-probability contributions.","Neither importance nor local contributions establish causality."],None),
    ("Robustness and monitoring",["No temporal or external validation is possible with supplied fields.","Monitor missingness, score drift, decision rates, review capacity and mature-label performance.","Stress results require timestamps, deployment data and independent fraud outcomes."],None),
    ("Recommendations",["Collect event timestamp, label timestamp, customer key and feature-availability lineage.","Validate review success and customer-friction economics with operations.","Run prospective shadow evaluation and governance approval before integration."],None)]
    with PdfPages(ROOT/"business_fraud_strategy.pdf") as pdf:
        for title,lines,chart in pages:
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.94,title,fontsize=20,weight="bold")
            fig.text(.08,.85,"\n\n".join(lines),fontsize=10,va="top",family="monospace",wrap=True)
            if chart is not None:
                from matplotlib.image import imread
                ax=fig.add_axes([.1,.1,.8,.38]);ax.imshow(imread(reports/f"figure_{chart+1:02d}.png"));ax.axis("off")
            pdf.savefig(fig);plt.close(fig)
    with PdfPages(ROOT/"model_card.pdf") as pdf:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.94,"Fraud model card | version 1.0",fontsize=18,weight="bold")
        lines=[f"Source: supplied transaction CSV; {len(df)} rows; target: fraud.","Intended use: research and offline decision simulation only.","Out of scope: live authorization without timestamp/lineage validation.",f"Model: {best_name}; calibration: {cal_name}.",f"Test ROC-AUC {test_metrics['roc_auc']:.3f}; AP {test_metrics['average_precision']:.3f}; Brier {test_metrics['brier']:.3f}.",f"Thresholds: review {policy.review:.4f}; block {policy.block:.4f}.",f"Hypothetical test cost/transaction: {test_stats['cost_per_transaction']:.2f}.","Split: stratified train/selection/calibration/test; no timestamp supplied.","Leakage concern: chargebacks, velocity and risk scores need point-in-time proof.","Missingness: median/mode imputation learned on train only.","Limitations: no temporal, external, fairness or real-world cost validation.","Human oversight: REVIEW cases require analyst disposition.","Monitor input quality, drift, decisions, mature-label precision and calibration.","Rollback on severe data failures or verified performance loss.","Joblib artifacts must only be loaded from trusted sources."]
        fig.text(.08,.86,"\n\n".join(lines),fontsize=10,va="top",wrap=True);pdf.savefig(fig);plt.close(fig)
    cells=[nbformat.v4.new_markdown_cell("# Fraud risk analysis\nSupplied data; hypothetical cost assumptions. No valid transaction timestamp exists, so results are not temporal validation."),
           nbformat.v4.new_markdown_cell("## 1. Environment and reproducibility\nRun from the repository root with dependencies in `requirements.txt`. The experiment uses seed 42 plus documented split seeds 43-45."),
           nbformat.v4.new_code_cell("%run scripts/run_analysis.py --data data/raw/transaction_fraud_risk_advanced_dataset.csv"),
           nbformat.v4.new_code_cell("import json, pandas as pd\nfrom IPython.display import display, Image\naudit=json.load(open('reports/data_audit.json'))\nresults=json.load(open('reports/results.json'))\naudit"),
           nbformat.v4.new_markdown_cell("## 2. Data audit and leakage\nSee the machine-readable audit above. Chargebacks, velocity, distance and upstream scores need proof of availability at authorization time. Transaction IDs and the target are excluded from fitting."),
           nbformat.v4.new_code_cell("display(pd.DataFrame({'missing':audit['missing']}))\nfor i in range(1,4): display(Image(filename=f'reports/figure_{i:02d}.png'))"),
           nbformat.v4.new_markdown_cell("## 3. Training-only exploratory analysis\nAssociations are descriptive. Interaction charts have small cells and should not be read causally."),
           nbformat.v4.new_code_cell("for i in range(4,13): display(Image(filename=f'reports/figure_{i:02d}.png'))"),
           nbformat.v4.new_markdown_cell("## 4. Feature engineering and validation\nRow-local features are defined in `fraud_pipeline.core.features`. No customer history is reconstructed without customer IDs and timestamps. Stratified train, selection, calibration-fit, policy-selection and test partitions are disjoint."),
           nbformat.v4.new_code_cell("display(pd.DataFrame(audit['partitions']).T)"),
           nbformat.v4.new_markdown_cell("## 5. Baseline, nonlinear models and imbalance\nUnweighted and class-weighted logistic, random forest and extra trees models are compared with baseline and engineered features. Selection uses average precision."),
           nbformat.v4.new_code_cell("display(pd.read_csv('reports/model_comparison.csv'))\ndisplay(pd.read_csv('reports/feature_importance.csv').head(12))"),
           nbformat.v4.new_markdown_cell("## 6. Calibration\nSigmoid and isotonic calibrators fit on calibration-fit rows and are compared on policy-selection rows. These methods change probability interpretation, while ROC-AUC chiefly measures ranking."),
           nbformat.v4.new_code_cell("display(pd.read_csv('reports/calibration_comparison.csv'))\ndisplay(Image(filename='reports/figure_14.png'))"),
           nbformat.v4.new_markdown_cell("## 7. Cost optimization and operations\nCosts are assumptions from `configs/cost.json`. Policy search is capacity constrained. The final test simulates outcomes after policy selection."),
           nbformat.v4.new_code_cell("display(pd.read_csv('threshold_cost_analysis.csv').groupby('cost_scenario').apply(lambda x:x[x.capacity_ok].nsmallest(1,'total_expected_cost')).reset_index(drop=True))\ndisplay(results['test_policy'])"),
           nbformat.v4.new_markdown_cell("## 8. Explainability and robustness\nPermutation importance describes model dependence, not causality. Local reason codes denote threshold routing only. Controlled perturbations retain original labels and do not emulate adaptive fraud."),
           nbformat.v4.new_code_cell("display(pd.read_csv('reports/robustness.csv'))\ndisplay(pd.read_csv('reports/local_explanations.csv'))"),
           nbformat.v4.new_markdown_cell("## 9. Final test, monitoring and limitations\nMonitor input quality, unknown categories, score and decision rates, mature-label precision and calibration, and review capacity. There is no defensible claim of temporal stability, causal effects, real financial savings, fairness or deployment readiness without new data."),
           nbformat.v4.new_code_cell("display(results['test_metrics'])\ndisplay(results['approve_all'])")]
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    nbformat.write(notebook,ROOT/"fraud_risk_analysis.ipynb")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
