# Current Artifacts Index

Last updated: 2026-09-27

## Canonical Artifacts (Use These)

### 1.1 SciFact 外部验证
- Dataset: `revision/artifacts/1_1_scifact/dataset/scifact_bench.json`
- Results: `revision/artifacts/1_1_scifact/results/scifact_experiment_results.csv`

### 1.2 十策略基线与一致性
- Main smoke: `revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3_smoke.csv`
- Main full: `revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv`（3000 行）
- Cross-judge smoke: `revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results_smoke.csv`
- Cross-judge full: `revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results.csv`（9000 行）
- Agreement smoke csv/txt:
  - `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement_smoke.csv`
  - `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement_smoke.txt`
- Agreement full csv/txt:
  - `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.csv`
  - `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.txt`

### 1.3 跨架构 E2-E5
- Dry run: `revision/artifacts/1_3_cross_arch/cross_arch_dry_run.csv`
- E2: `revision/artifacts/1_3_cross_arch/cross_arch_E2.csv`
- E3: `revision/artifacts/1_3_cross_arch/cross_arch_E3.csv`
- E4: `revision/artifacts/1_3_cross_arch/cross_arch_E4.csv`
- E5: `revision/artifacts/1_3_cross_arch/cross_arch_E5.csv`
- Summary: `revision/artifacts/1_3_cross_arch/cross_arch_E1_E5_summary.csv`
- Fleiss csv/txt:
  - `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.txt`

### 2.x 补充统计
- BERTScore full: `revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv`
- BERTScore summary: `revision/artifacts/2_1_bertscore/table7_bertscore_summary.csv`
- Cohen's h csv/txt:
  - `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.csv`
  - `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.txt`
- Threshold curve csv/txt/svg:
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.csv`
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_summary.txt`
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.svg`

### 3.1 标注一致性
- `revision/artifacts/3_1_annotation/annotation_audit_summary.csv`
- `revision/artifacts/3_1_annotation/annotation_audit_summary.txt`
- `revision/annotation_process_3_1.md`

### 4.x 写回与引用
- `revision/references/related_work_2024_2025.bib`
- `revision/paper_revision_evidence.md`
- `revision/final_writeback_packet.md`

## Legacy (Do Not Use as Current Input)
- `revision/artifacts/legacy/experiment_results_sci_v3.legacy_pre_10_strategy.csv`
  - Historical 8-strategy output; protocol differs from current 1.2 standard.

## 5.x Held-Out Selective Policy (Computing Manuscript)

- Selective-risk analysis: `revision/artifacts/1_16_qasper_selective_risk/`
- Deployment-time quality model: `revision/artifacts/1_17_qasper_cheap_gate/`
- Cost-aware verification cascade: `revision/artifacts/1_18_qasper_cost_aware_cascade/`
- Current manuscript source: `revision/submission/Computing_2026/source/`

## 6.x Supplemental Run (2026-09-27)

- Full run and integrity metadata: `revision/artifacts/supplement_20260927_full/`
- Human-readable result report: `revision/artifacts/supplement_20260927_full/RESULTS.md`
- Final diagnostics: `revision/artifacts/final_resume_diagnostics_20260927/`
- Gate analyses: `revision/artifacts/four_gate_analyses_20260927/`
- Retrieval audit: `revision/artifacts/retrieval_audit_20260927/`
- Route-score audit: `revision/artifacts/route_score_audit_20260927/`
- Independent latency run: `revision/artifacts/latency_20260927_full/`
