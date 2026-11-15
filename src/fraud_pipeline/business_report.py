"""Standalone, evidence-led management report; no model fitting or policy selection."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

NAVY = "#132D46"
TEAL = "#087F83"
BLUE = "#387AA1"
INK = "#24384A"
MUTED = "#5D6F7D"
PALE = "#EDF4F6"
LINE = "#DCE5EB"
AMBER = "#B66B18"
RED = "#AD4B4B"
W, H = 595.28, 841.89


def label(value):
    names = {"logistic_plain_base": "Logistic / original", "logistic_weighted_base": "Logistic / weighted",
             "logistic_plain_engineered": "Logistic / engineered", "logistic_tuned_base": "Logistic / tuned",
             "logistic_undersampled_base": "Logistic / undersampled", "forest_tuned_engineered": "Forest / tuned + engineered",
             "extra_trees_plain_engineered": "Extra trees / engineered", "forest_weighted_engineered": "Forest / weighted + engineered"}
    return names.get(value, value.replace("_", " ").capitalize())


class Page:
    """Point-based layout with measured text wrapping and overflow checks."""
    def __init__(self, number, title, subtitle, source):
        self.number = number
        self.fig = plt.figure(figsize=(W / 72, H / 72), dpi=100, facecolor="white")
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()
        self.rect(0, 0, W, 8, TEAL)
        self.text(42, 28, "FRAUD RISK INTELLIGENCE  /  FINAL PROJECT REPORT", 8, color=MUTED, weight="bold")
        self.text(42, 68, title, 25, width=512, weight="bold", color=NAVY, leading=1.13)
        self.text(42, 111, subtitle, 10, width=510, color=MUTED)
        self.rect(42, 779, 511, .7, LINE)
        self.text(42, 791, source, 6.6, width=475, color=MUTED, footer=True)
        self.text(42, 820, "Offline research assessment | Costs and intervention outcomes are hypothetical", 6.4, color=MUTED, footer=True)
        self.text(540, 816, f"{number:02d} / 12", 7.5, color=NAVY, footer=True)

    def rect(self, x, y, w, h, color, radius=0):
        if radius:
            patch = FancyBboxPatch((x/W, 1-(y+h)/H), w/W, h/H, boxstyle=f"round,pad=0,rounding_size={radius/W}",
                                   transform=self.fig.transFigure, facecolor=color, edgecolor="none")
        else:
            patch = Rectangle((x/W, 1-(y+h)/H), w/W, h/H, transform=self.fig.transFigure, facecolor=color, edgecolor="none")
        self.fig.add_artist(patch)

    def wrap(self, content, width, size, weight="normal"):
        font = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        lines = []
        for paragraph in str(content).split("\n"):
            line = ""
            for word in paragraph.split():
                candidate = f"{line} {word}".strip()
                pixels = self.renderer.get_text_width_height_descent(candidate, font, False)[0]
                if line and pixels * 72 / self.fig.dpi > width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        return lines

    def text(self, x, y, content, size=10, width=511, color=INK, weight="normal", leading=1.42, footer=False):
        lines = self.wrap(content, width, size, weight)
        bottom = y + len(lines) * size * leading
        if not footer and bottom > 770:
            raise ValueError(f"Report page {self.number}: text overflows at {bottom:.1f}: {str(content)[:65]}")
        for i, line in enumerate(lines):
            self.fig.text(x/W, 1-(y+i*size*leading)/H, line, fontsize=size, fontfamily="DejaVu Sans",
                          weight=weight, color=color, va="top")
        return bottom

    def heading(self, y, content):
        return self.text(42, y, content, 12, weight="bold", color=NAVY) + 8

    def paragraph(self, y, content, size=10):
        return self.text(42, y, content, size) + 12

    def callout(self, y, title, content, color=TEAL):
        lines = self.wrap(content, 477, 9.4)
        height = 42 + len(lines)*13.4
        if y + height > 770:
            raise ValueError(f"Report callout overflows on page {self.number}")
        self.rect(42, y, 511, height, PALE, radius=5)
        self.rect(42, y, 3, height, color)
        self.text(56, y+12, title, 10, width=477, weight="bold", color=color)
        self.text(56, y+32, content, 9.4, width=477)
        return y + height + 14

    def table(self, y, headers, rows, widths, size=9, row_min=31):
        x0 = 42
        def row(values, top, header=False, alternate=False):
            line_counts = [len(self.wrap(str(value), width-16, size, "bold" if header else "normal")) for value, width in zip(values, widths)]
            height = max(row_min, max(line_counts)*size*1.35 + 16)
            if top + height > 770:
                raise ValueError(f"Report table overflows on page {self.number}")
            self.rect(x0, top, sum(widths), height, NAVY if header else (PALE if alternate else "white"))
            x = x0
            for value, width in zip(values, widths):
                self.text(x+8, top+8, str(value), size, width=width-16,
                          color="white" if header else INK, weight="bold" if header else "normal", leading=1.35)
                x += width
            if not header:
                self.rect(x0, top+height-.4, sum(widths), .4, LINE)
            return top + height
        y = row(headers, y, header=True)
        for index, values in enumerate(rows):
            y = row(values, y, alternate=index % 2 == 0)
        return y + 14

    def metric(self, x, y, value, caption, note, width=163):
        self.rect(x, y, width, 89, PALE, radius=5)
        self.text(x+12, y+12, value, 24, width=width-24, color=TEAL, weight="bold")
        self.text(x+12, y+46, caption, 9.3, width=width-24, weight="bold")
        self.text(x+12, y+65, note, 8, width=width-24, color=MUTED)

    def axes(self, x, y, w, h):
        ax = self.fig.add_axes([x/W, 1-(y+h)/H, w/W, h/H])
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(LINE)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.set_axisbelow(True)
        return ax


def build_business_report(root: str | Path):
    root = Path(root)
    reports = root / "reports"
    load = lambda name: pd.read_csv(reports / name)
    results = json.loads((reports / "results.json").read_text())
    audit = json.loads((reports / "data_audit.json").read_text())
    config = json.loads((root / "configs/cost.json").read_text())
    environment = json.loads((reports / "environment.json").read_text())
    models, cv = load("model_comparison.csv"), load("candidate_cv_summary.csv")
    calibration, sensitivity = load("calibration_before_after.csv"), load("cost_sensitivity.csv")
    economics, stress = load("model_cost_comparison.csv"), load("robustness.csv")
    importance, stability = load("feature_importance.csv"), load("feature_stability_summary.csv")
    local = load("local_explanations.csv")
    test, policy = results["test_metrics"], results["test_policy"]
    n = audit["partitions"]["test"]["rows"]
    fraud = audit["partitions"]["test"]["fraud"]
    prevalence = fraud/n
    benefit = results["approve_all"]["total_expected_cost"] - policy["total_expected_cost"]
    reduction = benefit / results["approve_all"]["total_expected_cost"]
    pages = []
    source = lambda files: "Evidence: " + files + ". Unless stated otherwise, paths are under reports/."

    # 1. A cover that already explains the project and its decision boundary.
    p = Page(1, "Fraud Risk Intelligence", "Business strategy and final project assessment", source("results.json; data_audit.json; environment.json"))
    p.rect(42, 154, 511, 208, NAVY, radius=7)
    p.text(62, 177, "FROM TRANSACTION DATA\nTO OPERATIONAL DECISIONS", 25, width=467, color="white", weight="bold", leading=1.18)
    p.text(62, 257, "A reproducible pipeline for fraud scoring, probability calibration, explainable decisions and cost-aware evaluation.", 12, width=455, color="#D7E7EF")
    p.text(62, 324, "FINAL REPORT  |  OFFLINE ASSESSMENT  |  NOVEMBER 2026", 8.5, color="#A9D9DB", weight="bold")
    p.metric(42, 382, f"{audit['shape'][0]:,}", "Transactions assessed", "Supplied dataset")
    p.metric(216, 382, f"{int(audit['fraud_counts']['1'])/audit['shape'][0]:.2%}", "Fraud prevalence", f"{int(audit['fraud_counts']['1'])} labeled frauds")
    p.metric(390, 382, f"{reduction:.1%}", "Simulated cost reduction", "Versus approve-all; test")
    y = p.heading(500, "The business question")
    y = p.paragraph(y, "How can a financial platform detect suspicious transactions while limiting avoidable customer friction, analyst workload and fraud losses? This project treats fraud detection as a decision system: each transaction receives a probability, an approve/review/block action and an explanation when flagged.")
    p.callout(y+3, "Recommendation: advance to controlled validation", "Retain the transparent logistic model as the offline reference candidate. Validate feature timing, real costs, calibration and customer impact before any live blocking. The measured economic benefit is a simulation, not realized savings.")
    pages.append(p)

    # 2. Executive decision and navigation.
    p = Page(2, "Executive assessment", "The pipeline is complete as an offline prototype; deployment evidence remains incomplete.", source("results.json; model_comparison.csv; model_cost_comparison.csv"))
    y = p.heading(155, "What the project demonstrates")
    y = p.paragraph(y, "The workflow audits data, creates 18 derived features, compares 16 model candidates, evaluates two calibration methods and optimizes two decision thresholds against explicit costs and review capacity. A saved scoring engine supports batch prediction and transaction-level explanations.")
    y = p.table(y, ["Final-test indicator", "Measured result", "Meaning"], [
        ["Ranking quality", f"ROC-AUC {test['roc_auc']:.3f}; AP {test['average_precision']:.3f}", "Useful signal, with substantial errors remaining"],
        ["Flagged fraud", f"34 of {fraud} ({test['recall_at_review']:.1%})", "39 labeled frauds are approved"],
        ["Customer burden", f"{policy['false_positive_count']} legitimate flags; {policy['legitimate_block_count']} legitimate blocks", "False positives remain operationally material"],
        ["Review workload", f"{policy['review_count']} of {n} ({policy['review_count']/n:.1%})", "Within the 15% base batch allowance"],
        ["Cost simulation", f"{policy['cost_per_transaction']:.2f} vs {results['approve_all']['cost_per_transaction']:.2f} units/transaction", "Conditional on hypothetical costs and review success"]
    ], [120, 170, 221], size=9)
    y = p.callout(y, "Management decision", "Approve further data collection and a prospective shadow evaluation. Keep automatic declines outside the approved scope until mature labels, calibrated risk bands and legitimate-customer impact meet agreed acceptance criteria.")
    y = p.heading(y, "How to read this report")
    p.table(y, ["Project foundations", "Decision evidence", "Operational readiness"], [
        ["03  Data and quality\n04  Validation and features\n05  Observed fraud patterns", "06  Model comparison\n07  Calibration\n08  Policy and assumptions", "09  Financial impact\n10  Explanations\n11  Stress tests\n12  Governance and roadmap"]
    ], [170, 170, 171], size=8.6)
    pages.append(p)

    # 3. Data forensics and concrete treatments.
    p = Page(3, "Data quality and trust", "The strongest operational uncertainty is whether source fields were available at authorization time.", source("data_audit.json; numeric_distributions.csv; categorical_distributions.csv"))
    y = p.paragraph(157, f"The supplied CSV contains {audit['shape'][0]:,} rows and {audit['shape'][1]} columns, including the target and transaction identifier. There are {audit['duplicates']} duplicate rows and {audit['duplicate_ids']} duplicate IDs. The implemented range and consistency checks recorded no violations; these checks do not establish source authenticity.")
    y = p.table(y, ["Field with missing values", "Count / share", "Treatment and operational implication"], [
        ["Authentication level", f"{audit['missing']['authentication_level']} / {audit['missing']['authentication_level']/3000:.1%}", "Training-fit categorical mode; monitor source missingness"],
        ["Transaction amount", f"{audit['missing']['transaction_amount']} / {audit['missing']['transaction_amount']/3000:.1%}", "Median for model input; otherwise-approved cases go to review"],
        ["Distance from usual location", "75 / 2.5%", "Training median and numeric missingness indicator"],
        ["Merchant risk / IP risk", "60 / 45", "Training medians and numeric missingness indicators"],
        ["Login-to-transaction time", "60 / 2.0%", "Training median and numeric missingness indicator"]
    ], [151, 90, 270], size=9)
    missing = audit["missingness_vs_fraud_train"]["authentication_level"]
    y = p.callout(y, "Missingness carries information", f"In training data, fraud prevalence is {missing['fraud_rate_missing']:.1%} when authentication is missing versus {missing['fraud_rate_present']:.1%} when present. This is an association and may reflect collection practices. Mode imputation can suppress this signal; explicitly testing an authentication-missing flag is a future development task.")
    y = p.heading(y, "Treatment and leakage controls")
    y = p.paragraph(y, "No difficult or extreme observations were removed to improve metrics. Numeric medians, category encoding, scaling and population references are learned only inside the relevant fit partition. The target and transaction ID are excluded from predictors.", size=9.6)
    p.paragraph(y, "Chargeback history, velocity counts, merchant/IP scores and location history need point-in-time lineage. Event timestamps, label timestamps, customer IDs, source provenance and timezone rules are unavailable. Their absence prevents chronological validation and reconstructed personal histories.", size=9.6)
    pages.append(p)

    # 4. Training boundaries and engineering.
    p = Page(4, "Validation and feature design", "Separate model fitting, calibration and policy selection protect the evaluation boundary.", source("split_manifest.csv; environment.json; ../docs/feature_dictionary.md"))
    y = p.table(157, ["Partition", "Rows / fraud", "Permitted use"], [
        ["Training", "1,500 / 182", "Fit preprocessing, candidate models and training-only CV"],
        ["Model selection", "450 / 55", "Compare candidate average precision"],
        ["Calibration fit", "225 / 27", "Fit sigmoid and isotonic mappings"],
        ["Policy selection", "225 / 27", "Compare calibration; optimize review/block thresholds"],
        ["Final test", "600 / 73", "Report frozen-model discrimination and policy outcomes"]
    ], [122, 100, 289], size=9, row_min=28)
    y = p.heading(y, "Eighteen derived features, with explicit availability assumptions")
    y = p.table(y, ["Feature family", "Derived signals"], [
        ["Amount and combined risk (5)", "Log amount; amount x merchant risk; distance x merchant risk; merchant x IP risk; chargebacks x amount"],
        ["Device and age (3)", "New device x distance; account age below 30 days; device age below 30 days"],
        ["Velocity and session (4)", "Hourly/daily count ratio; count x amount; smoothed velocity burst; inverse login interval"],
        ["Time and context (3)", "Night; weekend; authentication x channel"],
        ["Population deviation (3)", "Amount/merchant median; relative amount deviation; distance minus channel median"]
    ], [152, 359], size=8.6, row_min=29)
    p.callout(y, "Interpret the validation evidence conservatively", "Splits are stratified, not chronological. Calibration-method choice and threshold search share a policy sample. The test was inspected before the revision; repeated reporting is regression evidence. Seeds 42-45 and the data hash support reproduction, not future-period validity.", color=AMBER)
    pages.append(p)

    # 5. Readable nonlinear and interaction charts with denominator information.
    p = Page(5, "Observed fraud patterns", "Associations guide investigation; they do not establish why a transaction is fraudulent.", source("nonlinear_transaction_amount.csv; interaction_authentication_level_channel.csv; interaction_effects.csv"))
    y = p.paragraph(155, "Fraud rates were compared across channels, merchant categories, authentication levels, customer tiers, devices and available time fields. Nonlinear analyses cover amount, velocity, distance, account age and upstream risk scores. All risk-pattern charts use training rows.", size=9.6)
    p.text(42, 226, "Amount alone does not show a simple monotonic risk pattern", 11, weight="bold", color=NAVY)
    amount = load("nonlinear_transaction_amount.csv")
    ax = p.axes(85, 260, 440, 142)
    se = np.sqrt(amount.rate*(1-amount.rate)/amount["count"])
    ax.errorbar(amount.midpoint, amount.rate, yerr=1.96*se, color=TEAL, marker="o", capsize=3, linewidth=1.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Median transaction amount within each of 8 bins", fontsize=8, color=MUTED)
    ax.set_ylabel("Fraud rate", fontsize=8, color=MUTED)
    ax.grid(axis="y", color=LINE, linewidth=.6)
    p.text(85, 432, f"Each bin contains {int(amount['count'].min())}-{int(amount['count'].max())} observed amounts. Error bars: approximate 95% intervals.", 7.5, width=440, color=MUTED)
    p.text(42, 470, "Authentication and channel create distinct observed risk segments", 11, weight="bold", color=NAVY)
    data = load("interaction_authentication_level_channel.csv")
    pivot = data.pivot(index="authentication_level", columns="channel", values="mean")
    counts = data.pivot(index="authentication_level", columns="channel", values="size").reindex_like(pivot)
    ax = p.axes(122, 507, 391, 136)
    ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0, vmax=.35)
    ax.set_xticks(range(len(pivot.columns)), list(pivot.columns), fontsize=7.3)
    ax.set_yticks(range(len(pivot.index)), list(pivot.index), fontsize=8)
    ax.tick_params(length=0)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i,j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1%}\nn={int(counts.iloc[i,j])}", ha="center", va="center", fontsize=8, color="white" if value>.19 else NAVY)
    p.text(42, 681, "Among cells with at least 30 rows, the authentication/channel fraud-rate spread is 19.2 percentage points; new-device/distance is 16.1 points and merchant-risk/amount is 10.2 points. These are joint cell contrasts, not isolated interaction effects. Small cells and missing categories need caution.", 9.3)
    pages.append(p)

    # 6. Candidate comparison, imbalance and uncertainty.
    p = Page(6, "Model selection and performance", "A transparent baseline won the predefined average-precision comparison.", source("model_comparison.csv; candidate_cv_summary.csv; imbalance_comparison.csv; results.json"))
    y = p.paragraph(156, "Logistic regression, random forest and extra trees were compared with original and engineered inputs, class weights, fit-only undersampling and randomized hyperparameter search. Average precision (AP) emphasizes ranking among scarce fraud cases; accuracy alone would reward approving the majority class.", size=9.6)
    chosen_names = ["logistic_plain_base", "logistic_weighted_base", "logistic_plain_engineered", "extra_trees_plain_engineered", "forest_tuned_engineered", "logistic_undersampled_base"]
    selected = models.set_index("name").loc[chosen_names]
    y = p.table(y, ["Candidate / selection sample", "ROC-AUC", "AP"], [[label(name), f"{row.roc_auc:.3f}", f"{row.average_precision:.3f}"] for name, row in selected.iterrows()], [325, 93, 93], size=9)
    baseline_cv = cv.set_index("model").loc["logistic_plain_base"]
    y = p.paragraph(y, f"The original-feature logistic model achieved the highest selection AP ({results['selection_metrics']['average_precision']:.3f}) and remains the reference candidate. Its three training-fold AP scores have mean {baseline_cv.ap_mean:.3f} and standard deviation {baseline_cv.ap_sd:.3f}. The engineered logistic model has higher selection ROC-AUC but lower AP; engineered features therefore did not automatically improve the selection objective.", size=9.5)
    y = p.table(y, ["Final test: 600 transactions", "Result", "Reference / interpretation"], [
        ["Average precision", f"{test['average_precision']:.3f}", f"No-skill prevalence reference: {prevalence:.3f}"],
        ["ROC-AUC", f"{test['roc_auc']:.3f}", "0.5 is chance ranking"],
        ["Flag precision / recall / F1", f"{test['precision_at_review']:.3f} / {test['recall_at_review']:.3f} / {test['f1_at_review']:.3f}", "Fraud share of flags / frauds flagged / harmonic mean"]
    ], [200, 115, 196], size=8.6)
    p.callout(y, "Selection quality is not business optimality", "Class weighting and undersampling did not beat the unweighted baseline on selection AP. Cost rankings can still change with loss and friction assumptions (page 9). Three-fold variability is descriptive; tuned candidates reuse training-selected parameters rather than nested CV.")
    pages.append(p)

    # 7. Calibration with correct sample labels.
    p = Page(7, "Probability calibration", "Probability reliability must be checked separately from ranking performance.", source("calibration_before_after.csv; calibration_risk_bands.csv; results.json"))
    y = p.paragraph(156, "A reliable score of 10% should correspond to about 10 frauds per 100 comparable transactions. This interpretation makes probability-based loss estimates meaningful. A model can rank transactions well while producing poor probability estimates.", size=9.6)
    y = p.table(y, ["Policy-selection method", "Brier score", "Log loss", "Calibration slope"], [
        [row.method.capitalize(), f"{row.brier:.4f}", f"{row.log_loss:.4f}", f"{row.calibration_slope:.2f}"] for row in calibration.itertuples()
    ], [205, 100, 100, 106], size=9)
    y = p.paragraph(y, "Both mappings were fitted on a separate 225-row partition containing 27 frauds. Sigmoid was chosen by the lowest Brier score on policy-selection rows. It is a regularized logistic mapping of base probabilities; isotonic is a more flexible monotonic mapping. Lower Brier score and log loss are preferable.", size=9.5)
    p.text(42, y+2, "Selected model: final-test calibration by score band", 11, weight="bold", color=NAVY)
    predictions = load("test_predictions.csv")
    # Reconstruct test labels from the supplied source, respecting saved transaction order.
    raw_path = root / "data/raw" / audit["source"]
    if raw_path.exists():
        raw = pd.read_csv(raw_path).set_index("transaction_id")
        observed = raw.loc[predictions.transaction_id, "fraud"].to_numpy()
        from sklearn.calibration import calibration_curve
        actual, predicted = calibration_curve(observed, predictions.fraud_probability, n_bins=8, strategy="quantile")
        ax = p.axes(92, y+44, 415, 166)
        ax.plot([0,.3], [0,.3], "--", color=MUTED, linewidth=1, label="Ideal")
        ax.plot(predicted, actual, "o-", color=TEAL, linewidth=2, label="Observed bands")
        ax.set(xlim=(0,.3), ylim=(0,.3))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Mean predicted probability", fontsize=8)
        ax.set_ylabel("Observed fraud rate", fontsize=8)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(color=LINE, linewidth=.6)
    else:
        p.text(65, y+65, "The source CSV is required to redraw the final-test calibration curve. The saved diagnostics below remain available.", 10, width=450)
    p.callout(y+262, "Calibration remains a deployment blocker", f"Final-test Brier score is {test['brier']:.4f}; slope is {test['calibration_slope']:.2f} versus an ideal of 1; eight-band calibration error is {test['expected_calibration_error']:.3f}. The mapping improves development Brier score but does not establish reliable future probabilities. Validate on larger, later-period, mature-label samples.", color=AMBER)
    pages.append(p)

    # 8. Explain the action contract and every money assumption.
    p = Page(8, "Decision policy and assumptions", "The model score becomes an action only through an explicit business policy.", source("../configs/cost.json; cost_assumptions.json; results.json; ../threshold_cost_analysis.csv"))
    review, block = results["policy"]["review"], results["policy"]["block"]
    y = p.table(156, ["Action", "Selected probability rule", "Operational meaning"], [
        ["APPROVE", f"Below {review:.2%}", "Continue under the simulated policy"],
        ["MANUAL REVIEW", f"{review:.2%} to below {block:.2%}", "Queue for analyst disposition"],
        ["BLOCK", f"At or above {block:.2%}", "Simulated decline, subject to deployment approval"]
    ], [110, 170, 231], size=9)
    y = p.paragraph(y, "The two thresholds were jointly searched on development data to minimize expected cost, with reviews capped at 15% of transactions. An otherwise approved record with missing amount is routed to review. Thresholds are fitted policy parameters, not universal fraud cutoffs.", size=9.5)
    y = p.table(y, ["Hypothetical economic assumption", "Base value", "How it enters cost"], [
        ["Approved fraud loss", "1 x amount", "Full assumed transaction exposure"],
        ["Review labor / legitimate friction", "3 / 2 units", "Every review / legitimate reviews only"],
        ["Review interception success", "80%", "20% residual fraud exposure after review"],
        ["Legitimate block / fraud block", "25 / 1 units", "Customer friction / residual block cost"],
        ["Unknown fraud-loss amount", "100 units", "Accounting fallback when amount is missing"],
        ["Review allowance", "15% per batch", "Feasibility constraint; not an analyst SLA"]
    ], [225, 99, 187], size=8.7)
    p.callout(y, "Cost is counted once per transaction", "Approve: fraud loss only. Review: labor plus legitimate friction or residual fraud loss. Block: legitimate-block friction or fraud-block cost. Amounts and costs have no verified currency; review effectiveness and customer friction have not been measured in live operations.")
    pages.append(p)

    # 9. Business impact and alternative rankings, all with sample boundary labels.
    p = Page(9, "Financial and workload tradeoffs", "The policy lowers hypothetical cost while leaving a substantial customer-impact burden.", source("results.json; model_cost_comparison.csv; cost_sensitivity.csv; recall_review_workload.csv"))
    y = p.table(156, ["Final-test policy", "Cost / transaction", "Review / block", "Capacity"], [
        ["Approve all", f"{results['approve_all']['cost_per_transaction']:.2f}", "0 / 0", "Within"],
        ["Binary threshold 0.50", f"{results['binary_050']['cost_per_transaction']:.2f}", "0 / 0", "Within"],
        ["Selected three-action policy", f"{policy['cost_per_transaction']:.2f}", f"{policy['review_count']} / {policy['block_count']}", "Within"],
        ["Review-only at selected cutoff", f"{results['review_only']['cost_per_transaction']:.2f}", f"{results['review_only']['review_count']} / 0", "Exceeds: 1.90x"]
    ], [215, 110, 100, 86], size=8.8)
    y = p.paragraph(y, f"The selected policy costs {policy['total_expected_cost']:,.2f} units versus {results['approve_all']['total_expected_cost']:,.2f} for approve-all: a simulated difference of {benefit:,.2f} units ({reduction:.1%}) across {n} test transactions. Review-only is cheaper but requires 171 reviews against a 90-review allowance, so it is infeasible under the base capacity assumption.", size=9.5)
    y = p.callout(y, "Customer impact constrains the economic recommendation", f"The policy approves {policy['approve_count']} transactions, reviews {policy['review_count']} and blocks {policy['block_count']}. Of the blocks, {policy['legitimate_block_count']} ({policy['legitimate_block_count']/policy['block_count']:.1%}) are legitimate. It flags 34 of 73 frauds; assumed review success reduces effective fraud capture to {policy['fraud_capture_rate']:.1%}. Real customer harm could outweigh simulated savings.", color=AMBER)
    y = p.heading(y, "A lower-AP model can be economically preferable")
    doubled = economics[economics.cost_scenario.eq("higher_fraud_loss")].set_index("model")
    forest, logistic = doubled.loc["forest_weighted_engineered"], doubled.loc["logistic_plain_base"]
    y = p.table(y, ["Doubled fraud loss: development only", "Selection AP", "Optimized cost"], [
        ["Selected original-feature logistic", f"{logistic.selection_ap:.3f}", f"{logistic.cost_per_transaction:.2f}"],
        ["Weighted engineered random forest", f"{forest.selection_ap:.3f}", f"{forest.cost_per_transaction:.2f}"]
    ], [310, 100, 101], size=8.8)
    y = p.paragraph(y, "All candidate economics use a common sigmoid mapping for comparability. These optimized development costs demonstrate a ranking reversal; they are not independent test savings. The AP-selected model was retained rather than switched after examining test results.", size=9.1)
    higher = sensitivity[sensitivity.cost_scenario.eq("higher_block_friction")].iloc[0]
    p.paragraph(y, f"Sensitivity matters: doubling legitimate-block friction selects review {higher.review_threshold:.2%} and block {higher.block_threshold:.0%}, eliminating blocks on policy-selection rows. Under doubled fraud loss, development optimization flags all rows. Increasing recall can therefore create unacceptable friction or review demand.", size=9.1)
    pages.append(p)

    # 10. Importance with stability and five real local examples.
    p = Page(10, "Explainability and model behavior", "Every flagged score includes a local explanation; explanations describe model associations.", source("feature_importance.csv; feature_stability_summary.csv; local_explanations.csv; dependence.csv"))
    y = p.paragraph(155, "Selection permutation importance ranks prior chargebacks, new-device status, merchant category, hourly velocity and channel highest. Shuffling these inputs reduces average precision; this measures model reliance in that sample, not a causal effect on fraud.", size=9.5)
    y = p.table(y, ["Predictor", "AP drop", "Positive folds"], [
        [row.feature.replace("_", " "), f"{row.ap_drop_mean:.3f}", f"{int(stability.set_index('feature').loc[row.feature, 'positive_folds'])} / 3"] for row in importance.head(5).itertuples()
    ], [270, 120, 121], size=8.6, row_min=27)
    y = p.paragraph(y, "Merchant category is positive in only one of three training-fold importance estimates despite its high selection rank. Merchant risk, new-device status, IP risk and channel are positive in all three. Rankings are sample-sensitive; correlated signals and small folds can produce negative importances.", size=9.2)
    y = p.heading(y, "Five flagged transactions: largest positive displayed term")
    rows = []
    for row in local.head(5).itertuples():
        matches = re.findall(r"((?:num|cat)__[^;=]+)=([+-]\d+\.\d+)", row.explanation)
        positives = sorted([(name, float(value)) for name, value in matches if float(value)>0], key=lambda pair: -pair[1])
        term = positives[0] if positives else ("No positive term displayed", 0)
        name = term[0].replace("num__", "").replace("cat__", "").replace("_", " ")
        rows.append([f"{row.transaction_id} / {row.fraud_probability:.2%} / {row.decision.title()}", f"{name} ({term[1]:+.3f} logit)"])
    y = p.table(y, ["Transaction / score / action", "Largest positive displayed contribution"], rows, [205, 306], size=8.1, row_min=27)
    p.callout(y, "Interpretation limits", "Displayed terms are base-logit contributions relative to encoded/scaled zero, not calibrated-probability contributions. A negative risk-score term can mean a below-reference value, not that risk prevents fraud. Dependence plots are descriptive. Customer-facing decline reasons require review.")
    pages.append(p)

    # 11. Stress results and missing-input/financial-exposure distinction.
    p = Page(11, "Robustness and operational risk", "A stable ranking score does not guarantee stable workload or financial outcomes.", source("robustness.csv; candidate_cv_summary.csv; ../docs/robustness_protocol.md"))
    names = {"base":"Unchanged test rows", "amount_plus_50pct":"Input amounts +50%", "velocity_plus_2":"Hourly count +2", "missing_amount_10pct":"10% newly masked amounts", "unseen_channel":"Unseen channel", "ecommerce_channel_mix":"All channels set to e-commerce", "extreme_top_amounts":"Largest 1% amounts x10", "fraud_prevalence_weight_2x":"Fraud-weighted resample"}
    y = p.table(156, ["Controlled scenario", "AP", "Reviews", "Capacity use", "Cost / txn"], [
        [names[row.scenario], f"{row.average_precision:.3f}", str(int(row.review_count)), f"{row.review_capacity_utilization:.2f}x", f"{row.cost_per_transaction:.2f}"] for row in stress.itertuples()
    ], [225, 55, 65, 85, 81], size=8.4, row_min=31)
    y = p.callout(y, "Missingness creates a concrete queue risk", "Masking 10% of input amounts raises reviews from 79 to 126: 1.40 times the base review allowance. Increasing hourly velocity also breaches capacity. Production integration needs a staffed overflow policy and an alert; silent automatic approval of the overflow would change the evaluated policy.", color=AMBER)
    y = p.heading(y, "What these experiments mean")
    y = p.paragraph(y, "Perturbed fields affect scoring and operational routing, while original transaction amounts remain the financial exposure. Amount inflation is an input-corruption probe, not a forecast of higher actual losses. Newly missing amounts correctly invoke the mandatory-review fallback.", size=9.5)
    y = p.paragraph(y, "Except for the prevalence resample, scenarios reuse the same 600 labels. The resample weights fraud rows twice as heavily and intentionally repeats rows. Reassigning channel or amount does not create realistic counterfactual labels. These results are exploratory stress evidence, not future-period performance estimates.", size=9.5)
    p.paragraph(y, "Likely degradation sources include fraud adaptation, upstream risk-score changes, device/channel mix shifts, delayed labels, source outages and changed analyst success. Monitor input quality, score distributions, mature-label calibration and actual customer outcomes together.", size=9.5)
    pages.append(p)

    # 12. Actionable governance, release gates and provenance.
    p = Page(12, "Governance and next steps", "Progress through evidence-based release gates, with owners and a defined fallback.", source("../docs/deployment_and_monitoring.md; environment.json; ../README.md"))
    y = p.table(156, ["Monitoring signal", "Provisional trigger", "Proposed owner / response"], [
        ["Schema / missing fields", ">1% schema failures; critical missingness +5 pp", "Data engineering: investigate feed; suspend model blocking"],
        ["Feature / score drift", "PSI >0.25; unknown categories >5%", "ML engineering: inspect reference and recent windows"],
        ["Queue and decisions", ">100% staffed capacity; decision rates change >50%", "Fraud operations: activate approved overflow rules"],
        ["Mature-label quality", "Precision/recall down >20%; Brier up >20%", "Model risk: review labels, calibration and policy"],
        ["Loss / inference service", "Cost +20%; errors >1% or latency >5x", "Operations / platform: investigate; restore approved bundle"]
    ], [132, 184, 195], size=8.2, row_min=30)
    y = p.heading(y, "Recommended release gates")
    y = p.paragraph(y, "1. Data foundation: collect event/label times, customer keys, real loss and review outcomes; prove authorization-time feature availability and label maturity.\n2. Independent validation: use later-period holdouts, time-valid folds and a fresh test; reassess calibration, fairness, friction costs and threshold sensitivity.\n3. Shadow operation: log predictions without model-driven declines; measure analyst demand, customer impact and real interception success.\n4. Controlled release: obtain model-risk and operations approval; version the model, calibrator, feature contract and policy together; rehearse rollback.", size=9.1)
    y = p.callout(y, "Retrain, review or disable", "Investigate drift before retraining. Retrain/recalibrate only with checked, mature labels and a fresh holdout. On severe data failures or verified degradation, disable model-driven blocking and use approved platform fallback rules with review up to staffed capacity. Thresholds and owners here are proposals, not deployed controls.")
    y = p.heading(y, "Delivery and reproducibility")
    p.text(42, y, f"The repository includes the executed notebook, processed export, base pipeline, calibrated decision engine, cost grid, model card, this report and README. Supporting CSVs preserve audit and decision evidence. Python {environment['python']}; seeds 42-45; raw-data SHA-256 begins {environment['dataset_sha256'][:16]}. Rebuild this report with python scripts/build_business_report.py; rebuild the experiment with python scripts/run_analysis.py.\n\nThe practical achievement is a transparent and reproducible fraud decision prototype. The next investment should improve trustworthy validation data and operational evidence before live payment decisions.", 8.8)
    pages.append(p)

    path = root / "business_fraud_strategy.pdf"
    # Draw the exact PDF figures as PNG previews for visual layout review.
    preview_dir = reports / "business_report_preview"
    preview_dir.mkdir(exist_ok=True)
    with PdfPages(path, metadata={"Title": "Fraud Risk Intelligence Pipeline - Final Business Report",
                                 "Author": "Fraud Risk Intelligence Project",
                                 "Subject": "Business strategy, validation evidence and operational readiness",
                                 "Keywords": "fraud, calibration, cost, model governance"}) as pdf:
        for page in pages:
            page.fig.savefig(preview_dir / f"page_{page.number:02d}.png", dpi=100)
            pdf.savefig(page.fig)
            plt.close(page.fig)
    return path
