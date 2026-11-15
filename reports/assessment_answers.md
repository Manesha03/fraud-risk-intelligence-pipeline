# Evidence-based assessment



All model/cost comparisons below use development data. They are exploratory optimized costs, not unbiased savings estimates.



## Predictors and stability

             feature  ap_drop_mean  ap_drop_sd
   prior_chargebacks      0.057747    0.007382
          new_device      0.050912    0.006316
   merchant_category      0.049698    0.010903
transactions_last_1h      0.042090    0.019408
             channel      0.027903    0.015989



Fold estimates are in selected_baseline_importance_folds.csv. Negative permutation importance means the shuffled feature improved AP in that sample; correlated inputs and small samples can make rankings unstable. No causal claim follows.



Measured baseline stability (3 stratified folds):

                            mean     std     min     max  positive_folds
feature                                                                 
merchant_risk_score       0.0256  0.0166  0.0070  0.0386               3
new_device                0.0226  0.0223  0.0085  0.0483               3
billing_address_mismatch  0.0192  0.0219 -0.0009  0.0426               2
ip_risk_score             0.0131  0.0084  0.0078  0.0228               3
transactions_last_1h      0.0102  0.0097 -0.0006  0.0183               2
prior_chargebacks         0.0097  0.0086 -0.0001  0.0162               2
channel                   0.0094  0.0066  0.0021  0.0151               3
customer_tier             0.0043  0.0034  0.0010  0.0077               3

A feature positive in all three folds has more consistent direction in this experiment; overlapping uncertainty, only three folds, and unknown chronology prevent a claim of long-term stability.

## Model economics

          cost_scenario                      model  selection_ap  cost_per_transaction
                   base  logistic_plain_engineered      0.261445             12.520462
  higher_block_friction  logistic_plain_engineered      0.261445             14.631573
higher_fraud_block_cost  logistic_plain_engineered      0.261445             12.662684
      higher_fraud_loss forest_weighted_engineered      0.199165             16.833049
    higher_missing_loss  logistic_plain_engineered      0.261445             12.520462
     higher_review_cost  logistic_plain_engineered      0.261445             12.947129
 higher_review_friction  logistic_plain_engineered      0.261445             12.880462
   lower_review_success  logistic_plain_engineered      0.261445             13.091102
tighter_review_capacity  logistic_plain_engineered      0.261445             13.159849



Highest selection ROC-AUC candidate: logistic_plain_engineered. Compare its cost below with each scenario winner:

          cost_scenario  cost_per_transaction
                   base             12.520462
      higher_fraud_loss             19.883858
  higher_block_friction             14.631573
   lower_review_success             13.091102
     higher_review_cost             12.947129
 higher_review_friction             12.880462
higher_fraud_block_cost             12.662684
    higher_missing_loss             12.520462
tighter_review_capacity             13.159849

Measured examples where lower AP has lower optimized development cost:

         scenario             lower_ap_model  lower_ap  lower_ap_cost     higher_ap_model  higher_ap  higher_ap_cost  cost_advantage
higher_fraud_loss forest_weighted_engineered  0.199165      16.833049 logistic_tuned_base   0.252572       21.075271        4.242222
higher_fraud_loss forest_weighted_engineered  0.199165      16.833049 logistic_plain_base   0.289115       21.008818        4.175769
higher_fraud_loss       forest_weighted_base  0.183012      17.136267 logistic_tuned_base   0.252572       21.075271        3.939004
higher_fraud_loss    forest_plain_engineered  0.220756      17.143004 logistic_tuned_base   0.252572       21.075271        3.932267
higher_fraud_loss       forest_weighted_base  0.183012      17.136267 logistic_plain_base   0.289115       21.008818        3.872551



## Calibration

Calibration maps scores to empirical fraud frequencies on separate fit rows. A score of 0.10 should imply approximately 10 frauds per 100 comparable transactions only if validated calibration persists. Small calibration partitions and distribution shifts weaken this interpretation. See raw/sigmoid/isotonic Brier scores and calibration_methods.png.



## Threshold sensitivity

          cost_scenario  review_threshold  block_threshold  review_count  cost_per_transaction
                   base          0.124193         0.132910            33             12.766027
      higher_fraud_loss          0.000000         0.107485            26             21.008818
  higher_block_friction          0.138239         1.000000            31             14.854169
   lower_review_success          0.124193         0.132910            33             13.338453
     higher_review_cost          0.124193         0.132910            33             13.206027
 higher_review_friction          0.124193         0.132910            33             13.139360
higher_fraud_block_cost          0.124193         0.132910            33             12.961582
    higher_missing_loss          0.124193         0.132910            33             12.766027
tighter_review_capacity          0.132910         0.138239            18             13.656827



## Recall and review workload

Review-only threshold sweep (no model-driven blocking):

 review_threshold   recall  review_count  review_capacity_utilization
         0.000000 1.000000           225                     6.666667
         0.108360 0.777778           191                     5.659259
         0.112780 0.740741           146                     4.325926
         0.118811 0.629630            99                     2.933333
         0.129674 0.407407            52                     1.540741
         0.217985 0.000000             5                     0.148148

Higher recall requires more flagged transactions in this sweep; missing-amount review supplies a fixed workload floor. Capacity violations remain visible.



## Interactions

                                 interaction  eligible_cells  max_minus_min_rate
new_device x distance_from_usual_location_km               6            0.161287
    merchant_risk_score x transaction_amount               9            0.102182
              authentication_level x channel              11            0.192068

These are observed cell contrasts, not isolated interaction coefficients or causal effects. Counts and labeled heatmaps are supplied.



## Drift priorities

Prioritize the leading predictors above, transaction amount, velocity, authentication missingness, merchant/IP risk scores, channel mix, and output probabilities. Monitor point-in-time feature lineage and mature-label calibration. Upstream score changes, new merchants/devices, fraud adaptation, label delays, and changed review success can degrade performance.



## Counterintuitive explanations

Measured base-model dependence endpoints on development rows:

                                   first     last      min      max
feature                                                            
distance_from_usual_location_km  0.11882  0.13828  0.11882  0.13828
merchant_risk_score              0.09220  0.18055  0.09220  0.18055
transaction_amount               0.12397  0.12956  0.12397  0.12956
transactions_last_1h             0.08549  0.18472  0.08549  0.18472

A decreasing or flat amount/risk response can reflect weak signals, correlated fields, regularization, or sampling noise. These responses are model behavior, not evidence that increasing an amount or upstream risk score prevents fraud. Signed local terms are relative to encoded/scaled zero, not to an individual customer's normal behavior. A missing-input flag can be predictive of collection practices rather than fraud.



## Validation boundaries

The final model remains the average-precision-selected candidate; the economics table is a development comparison, not a retrospective switch based on the test. Every cost-table candidate uses the same sigmoid method for comparability, whereas the chosen candidate also has an isotonic comparison. Method choice and threshold optimization share the policy-selection sample, so optimized development cost is optimistic. CV scores for tuned candidates reuse training-selected hyperparameters and are descriptive, not nested-CV estimates. The final test had already been inspected before this revision; repeated reporting is regression evidence, not a new untouched holdout.