# Deployment and Monitoring Plan

This is a proposed plan, not a deployed monitoring service. All thresholds below are provisional and require fraud-operations and model-risk approval before use.

## Before integration

1. Obtain event timestamps, label timestamps, customer identifiers, feature computation timestamps, and documented timezone rules.
2. Prove that prior chargebacks, velocity counts, merchant/IP risk scores, and location history were available at authorization time.
3. Run chronological shadow evaluation with label-maturity and any needed embargo; validate real review success, customer friction, and loss amounts.
4. Review fairness and customer-impact outcomes with a suitable, lawfully collected dataset. The supplied CSV does not support this analysis.
5. Freeze a signed model artifact, calibrator, policy configuration, source revision, training-data version, and approval record together.

## Proposed monitoring

| Metric | Frequency | Provisional warning | Provisional critical | Owner |
| --- | --- | --- | --- | --- |
| Input schema failure rate | Hourly | >0.1% | >1% | Data engineering |
| Missing critical field rate | Hourly | +2 percentage points vs reference | +5 points | Data engineering |
| Unknown-category rate | Daily | >2% | >5% | ML engineering |
| Feature drift (PSI for numeric; category shares) | Daily | PSI >0.1 | PSI >0.25 | ML engineering |
| Score distribution drift | Daily | PSI >0.1 | PSI >0.25 | ML engineering |
| Approve/review/block rates | Hourly | >20% relative change | >50% relative change | Fraud operations |
| Review queue and SLA breach | Hourly | >80% of staffed capacity | >100% of staffed capacity | Fraud operations |
| Mature-label fraud prevalence | Weekly | >20% relative change | >50% relative change | Fraud analytics |
| Precision and recall after label maturity | Weekly | >10% relative drop | >20% relative drop | Model risk |
| Brier score and risk-band calibration | Weekly | >10% relative increase | >20% relative increase | Model risk |
| Estimated realized loss and cost per transaction | Daily | >10% vs approved baseline | >20% | Fraud operations |
| Inference errors and p95 latency | Hourly | >0.1% errors or 2x latency | >1% errors or 5x latency | Platform engineering |

PSI thresholds are screening signals, not proof of model degradation. Investigate schema changes, upstream outages, mix shifts, and delayed labels before retraining. Keep a separate reference period and a recent comparison window; do not calculate drift from the same batch being scored.

## Response and governance

- Warning: inspect source feeds, score distributions, case mix, and mature labels; document an owner and resolution date.
- Critical data failure or inference error: disable model-driven blocking, route uncertain cases to staffed review up to capacity, and use the platform's approved fallback authorization rules. This repository does not implement that integration.
- Verified performance or calibration degradation: initiate manual model-risk review; retrain only after data and label checks. Recalibrate and retune thresholds on later development periods, followed by a fresh chronological holdout.
- Rollback: restore the last approved model, calibrator, policy, and feature contract as one versioned unit. Retain immutable decision logs containing model and policy versions, score, decision, reason code, and feature-quality flags, without raw card numbers or credentials.
- Temporary disablement: require fraud-operations and platform approval for sustained schema failure, severe latency/error rates, or a material unexplained rise in legitimate blocks.

The current offline thresholds and cost assumptions do not authorize automated declines in production.
