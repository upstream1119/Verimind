# QASPER Cheap-Gate Feasibility

The cheap model is trained on the main 300-question set and evaluated once on the disjoint 150-question held-out set. It does not use gold answers, gold evidence, LLM-judge scores, or the existing trust score as input features.

| Score | Spearman with Answer F1 | Observed AURC | Top-25% Answer F1 | Top-50% Answer F1 |
|---|---:|---:|---:|---:|
| cheap_score | 0.4137 | 0.2915 | 0.5840 | 0.5818 |
| similarity_score | -0.2099 | 0.3932 | 0.4017 | 0.3889 |
| faithfulness_score | 0.0464 | 0.3320 | 0.5526 | 0.4532 |
| legacy_trust_score | -0.1046 | 0.3840 | 0.3786 | 0.4150 |

## Interpretation boundary

This is a feasibility test on existing outputs, not a final deployment policy. A cost-aware cascade should be constructed only if the cheap score generalizes on the held-out set.
