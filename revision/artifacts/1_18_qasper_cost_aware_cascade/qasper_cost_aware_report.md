# QASPER Cost-Aware Verification Cascade

Thresholds were selected only on the deterministic calibration split. The disjoint 150-question held-out set was evaluated once with frozen policies. Answer quality is normalized QASPER Answer F1.

Verifier input tokens are offline estimates based on total input characters divided by four. They are not measured API usage, latency, or monetary cost.

## Held-out results

| Risk budget | Policy | Feasible | Coverage | Answer F1 | Risk | Verifier call rate | Estimated tokens | Savings vs always verify |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.45 | No Gate | False | 0.6400 | 0.4362 | 0.5638 | 0.0000 | 0 | 1.0000 |
| 0.45 | Cheap Only | False | 0.1600 | 0.5236 | 0.4764 | 0.0000 | 0 | 1.0000 |
| 0.45 | Always Verify | True | 0.0800 | 0.5898 | 0.4102 | 0.6400 | 56777 | 0.0000 |
| 0.45 | Cost-Aware Cascade | False | 0.1533 | 0.5046 | 0.4954 | 0.0867 | 6811 | 0.8800 |
| 0.50 | No Gate | False | 0.6400 | 0.4362 | 0.5638 | 0.0000 | 0 | 1.0000 |
| 0.50 | Cheap Only | True | 0.1600 | 0.5236 | 0.4764 | 0.0000 | 0 | 1.0000 |
| 0.50 | Always Verify | True | 0.0800 | 0.5898 | 0.4102 | 0.6400 | 56777 | 0.0000 |
| 0.50 | Cost-Aware Cascade | True | 0.1533 | 0.5046 | 0.4954 | 0.0867 | 6811 | 0.8800 |
| 0.55 | No Gate | False | 0.6400 | 0.4362 | 0.5638 | 0.0000 | 0 | 1.0000 |
| 0.55 | Cheap Only | True | 0.1600 | 0.5236 | 0.4764 | 0.0000 | 0 | 1.0000 |
| 0.55 | Always Verify | True | 0.0800 | 0.5898 | 0.4102 | 0.6400 | 56777 | 0.0000 |
| 0.55 | Cost-Aware Cascade | True | 0.1533 | 0.5046 | 0.4954 | 0.0867 | 6811 | 0.8800 |
| 0.60 | No Gate | True | 0.6400 | 0.4362 | 0.5638 | 0.0000 | 0 | 1.0000 |
| 0.60 | Cheap Only | True | 0.2267 | 0.5400 | 0.4600 | 0.0000 | 0 | 1.0000 |
| 0.60 | Always Verify | True | 0.2267 | 0.5400 | 0.4600 | 0.6400 | 56777 | 0.0000 |
| 0.60 | Cost-Aware Cascade | True | 0.2267 | 0.5400 | 0.4600 | 0.0000 | 0 | 1.0000 |

## Policy-selection boundary

The held-out labels were not used to choose thresholds. A held-out row can therefore violate its nominal calibration risk budget; the `feasible` column reports the observed held-out outcome rather than retroactively changing the policy.

## Frozen thresholds

|   risk_budget | policy             |   cheap_low |   cheap_high |   verified_threshold | calibration_feasible   |
|--------------:|:-------------------|------------:|-------------:|---------------------:|:-----------------------|
|          0.45 | Cheap Only         |    0.55119  |   nan        |           nan        | False                  |
|          0.45 | Always Verify      |  nan        |   nan        |             0.615958 | False                  |
|          0.45 | Cost-Aware Cascade |    0.55119  |     0.594414 |             0.573892 | False                  |
|          0.5  | Cheap Only         |    0.55119  |   nan        |           nan        | False                  |
|          0.5  | Always Verify      |  nan        |   nan        |             0.615958 | False                  |
|          0.5  | Cost-Aware Cascade |    0.55119  |     0.594414 |             0.573892 | False                  |
|          0.55 | Cheap Only         |    0.55119  |   nan        |           nan        | False                  |
|          0.55 | Always Verify      |  nan        |   nan        |             0.615958 | False                  |
|          0.55 | Cost-Aware Cascade |    0.55119  |     0.594414 |             0.573892 | True                   |
|          0.6  | Cheap Only         |    0.523161 |   nan        |           nan        | True                   |
|          0.6  | Always Verify      |  nan        |   nan        |             0.53392  | True                   |
|          0.6  | Cost-Aware Cascade |    0.523161 |     0.523161 |             0.162365 | True                   |
