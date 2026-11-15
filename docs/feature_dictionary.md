# Feature Dictionary

The source CSV has no event timestamp or customer identifier. The first thirteen features below use the current record only; the added population-reference features use statistics learned exclusively within the relevant training fit. "Available at authorization" is an assumption requiring upstream verification, especially for precomputed history and risk scores. No target-derived encoding or future transaction is used by `features()`.

| Feature | Definition | Required source fields | History / cold start | Authorization-time caveat |
| --- | --- | --- | --- | --- |
| `log_amount` | `log1p(max(amount, 0))` | `transaction_amount` | None; missing stays missing | Amount must be known before approval; missing triggers review. |
| `amount_risk` | Amount times merchant risk | `transaction_amount`, `merchant_risk_score` | Merchant score is upstream state | Score version and effective time must be recorded. |
| `device_distance` | New-device flag times distance | `new_device`, `distance_from_usual_location_km` | Upstream device and location history | Define how both were known before authorization. |
| `distance_risk` | Distance times merchant risk | Distance, merchant risk score | Two upstream histories | Both need point-in-time lineage. |
| `velocity_ratio` | `(last_1h + 1) / (last_24h + 1)` | `transactions_last_1h`, `transactions_last_24h` | Precomputed counts; no history is reconstructed here | Counts must exclude the current and later transactions. |
| `velocity_amount` | Last-hour count times amount | `transactions_last_1h`, `transaction_amount` | Upstream count | Count must be strictly prior to scoring. |
| `new_account` | Account age below 30 days | `account_age_days` | Current account metadata | Age must be measured at authorization. |
| `young_device` | Device age below 30 days | `device_age_days` | Current device metadata | Age must be measured at authorization. |
| `night_transaction` | Hour in 00:00-05:59 | `hour` | None | Timezone and clock convention are undocumented. |
| `weekend` | Day of week 5 or 6 | `day_of_week` | None | Dataset weekday coding must be confirmed. |
| `auth_channel` | Authentication level plus channel | `authentication_level`, `channel` | None | Authentication outcome must be available before decision. |
| `login_speed` | `1 / (1 + max(login_minutes, 0))` | `login_to_transaction_minutes` | Current-session data | Login timing must precede the transaction. |
| `chargeback_amount` | Prior chargebacks times amount | `prior_chargebacks`, `transaction_amount` | Upstream chargeback history | Only matured chargebacks known before authorization qualify. |

`SimpleImputer` learns numeric medians and categorical modes from the training partition only. Missing amount is allowed for model scoring, but the decision policy routes an otherwise approved record to `REVIEW`. Its hypothetical fraud-loss amount comes from `missing_amount_fraud_loss` in `configs/cost.json`. A cold-start customer receives only the upstream values supplied in the row; this project does not synthesize customer history.

## Added normalization and deviation features

| Feature | Definition | Availability and fitting |
| --- | --- | --- |
| `velocity_burst` | `24 * (last_1h + 1) / (last_24h + 24)` | Smoothed hourly count relative to an average hour. Uses supplied strictly prior counts; not reconstructed customer history. |
| `combined_risk` | Merchant risk times IP risk | Both upstream scores must be available before authorization. |
| `amount_to_merchant_median` | Amount divided by training merchant-category median, denominator at least 1 | `ReferenceFeatures` learns medians inside every training/CV pipeline. Unseen categories use the training global median. |
| `amount_merchant_deviation` | `(amount - reference median) / reference median` | Population behavioral deviation, not deviation from the individual customer's usual amount. Shares the same fitted reference. |
| `distance_channel_deviation` | Distance minus training channel median distance | Population location deviation; unseen channels use the training global median. |

There are 18 derived features. The two amount-reference features are intentionally related; their usefulness is evaluated with the entire engineered feature set and their correlation can destabilize individual coefficients. In `features()` the three learned-reference columns begin as missing placeholders; the fitted `ReferenceFeatures` step fills them before imputation. The exported processed CSV uses references learned from the training partition. Do not use that precomputed CSV as input to a new cross-validation experiment: start from the raw CSV so references are relearned within folds.

No personal average transaction amount, customer spending z-score, event recency, or newly reconstructed rolling-window history is claimed. Those require customer IDs and trustworthy timestamps that are absent. Age comparisons and time flags are current-record indicators; upstream history and timezone assumptions still require verification.
