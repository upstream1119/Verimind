# QASPER Selective-Risk Reanalysis

Source strategy: `VeriMind_No_Gating`

The analysis replays trust thresholds on fixed generated answers. Answer quality uses the official QASPER-style token F1; it does not use the LLM judge that produced the gate score.

| Split | N | No-gate coverage | No-gate Answer F1 | Threshold-5 coverage | Threshold-5 Answer F1 | Observed AURC |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 100 | 0.5000 | 0.4380 | 0.4200 | 0.3860 | 0.3197 |
| test | 200 | 0.4950 | 0.4026 | 0.4450 | 0.4037 | 0.2970 |

## Interpretation boundary

Answer F1 is a dataset-native utility measure, not a complete factual-support label. Human and independent-judge audits remain separate evidence sources.
