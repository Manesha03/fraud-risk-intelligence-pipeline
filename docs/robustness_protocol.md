# Robustness Protocol

All scenarios reuse the frozen model, calibration mapping, and thresholds. Except for the prevalence scenario, they retain the same 600 final-test labels and transaction count; changes to input fields are controlled stress probes, not observations of real fraud adaptation. Reported cost uses original transaction amounts as the financial exposure while scoring uses the perturbed fields.

Operational decisions also use the perturbed input amount: newly missing amounts must trigger the same review fallback as batch inference. `policy_stats(..., exposure_amount=original_amount)` separates this decision input from the original financial exposure. Amount inflation is therefore an input-corruption probe with fixed exposure, not a forecast of increased monetary losses. Missingness can push review workload over capacity; the simulation reports that breach rather than silently approving overflow.

| Scenario | Exact perturbation |
| --- | --- |
| `base` | Unchanged final-test rows. |
| `amount_plus_50pct` | Multiply each nonmissing input amount by 1.5 for scoring. |
| `velocity_plus_2` | Add 2 to each `transactions_last_1h` input. |
| `missing_amount_10pct` | Set 10% of test input amounts to missing, sampled without replacement using seed 42. |
| `unseen_channel` | Replace all channel values with `Novel channel`. |
| `ecommerce_channel_mix` | Replace all channel values with `E-commerce`. |
| `extreme_top_amounts` | Multiply the largest 1% of nonmissing test input amounts by 10. |
| `fraud_prevalence_weight_2x` | Draw 600 test rows with replacement, weighting fraud rows twice as heavily as legitimate rows, using seed 42. Repeated rows are intentional; this is a prevalence-mix check, not covariate or concept drift. |

For each scenario, `reports/robustness.csv` records discrimination, Brier score, cost, decision counts, review utilization, and failed predictions. The test labels are reused, so the stress results are exploratory and should not guide a new model or threshold choice. There is no temporal holdout, adversarial simulation, or claim that these perturbations represent actual future transactions.
