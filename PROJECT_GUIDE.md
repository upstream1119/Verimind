# VeriMind Project Guide

## 1. Research Question

The project asks when a scientific RAG system should release an answer, request additional verification, or abstain. The current paper frames this as risk-constrained selective prediction with an explicit quality--coverage--verification-cost operating point.

The current primary outcome is normalized QASPER Answer F1 on a disjoint held-out split. LLM-judge faithfulness, stress-test hallucination labels, and author audits are supporting diagnostics rather than interchangeable ground truth.

## 2. Recommended Reading Order

For a new collaborator or an AI code assistant:

1. `README.md`
2. `revision/submission/Computing_2026/source/main.tex`
3. `revision/index/current_artifacts.md`
4. `revision/artifacts/1_16_qasper_selective_risk/qasper_selective_risk_report.md`
5. `revision/artifacts/1_17_qasper_cheap_gate/cheap_gate_report.md`
6. `revision/artifacts/1_18_qasper_cost_aware_cascade/qasper_cost_aware_report.md`
7. `revision/artifacts/supplement_20260927_full/RESULTS.md`
8. The implementation modules listed below.

## 3. Architecture and Code Entry Points

### Interactive system

- `rag_app_final.py`: Streamlit application.
- `build_index.py`: local multi-granularity index construction.
- `rag_adaptive_qwen.py`: intent-aware retrieval and answer generation.
- `rag_trust_md_qwen.py`: legacy trust-gating implementation.

### Controlled stress test

- `run_sci_experiment_main.py`: principal runner.
- `revision/cross_eval/`: cross-judge evaluation.
- `revision/cross_arch/`: heterogeneous generator--judge experiments.

### QASPER

- `revision/qasper/prepare_qasper.py`: deterministic public-data preparation.
- `revision/qasper/build_qasper_index.py`: local index construction.
- `revision/qasper/run_qasper_experiment.py`: six-strategy public benchmark.
- `revision/qasper/gate_v2.py`: tri-judge Gate V2 policy.
- `revision/analysis/analyze_qasper_selective_risk.py`: risk--coverage analysis.
- `revision/analysis/analyze_qasper_cheap_gate.py`: deployment-time quality score.
- `revision/analysis/analyze_qasper_cost_aware_cascade.py`: frozen three-way cascade.

### SciFact and supplemental evaluation

- `revision/scifact/`: SciFact preparation and evaluation.
- `revision/experiments/supplement_20260927.py`: paired supplemental experiment.
- `revision/experiments/latency_20260927.py`: independent latency experiment.
- `revision/experiments/four_gate_analyses_20260927.py`: gate diagnostics.

## 4. Canonical Evidence

Use `revision/index/current_artifacts.md` as the artifact registry. The most recent full supplemental run is `revision/artifacts/supplement_20260927_full/`; it records configuration, hashes, raw answers, retrieval outputs, metrics, retry diagnostics, and narrative results.

The Computing manuscript's held-out selective-policy evidence is under:

```text
revision/artifacts/1_16_qasper_selective_risk/
revision/artifacts/1_17_qasper_cheap_gate/
revision/artifacts/1_18_qasper_cost_aware_cascade/
```

Historical experiments remain useful for provenance but must not silently replace the current protocol.

## 5. Local-Only Data

The following are intentionally ignored and must be regenerated or downloaded:

- `revision/qasper/raw/`
- `revision/qasper/chroma_db_*/`
- `revision/scifact/raw/`
- `revision/scifact/chroma_db_*/`
- root `chroma_db_*/`
- `.env`

These paths contain public dataset caches, local indexes, or secrets rather than authored source code.

## 6. Reproducibility Boundaries

- API model aliases do not freeze provider-side model weights.
- QASPER figure/table evidence that is absent from the text corpus remains a retrieval limitation.
- SciFact supplemental samples include documented train/dev provenance and are not represented as an official hidden-test evaluation.
- Estimated verifier tokens are offline estimates, not billed-token measurements.
- Author audits apply only to the frozen outputs shown in their corresponding artifact directories.

## 7. Safe Modification Workflow

1. Preserve original output files.
2. Add a new dated artifact directory for a new run.
3. Record command, models, temperature, seed, dataset split, and output hashes.
4. Run offline tests before any paid API experiment.
5. Update `revision/index/current_artifacts.md` only after artifacts pass integrity checks.
6. Update manuscript numbers only from canonical result files.

## 8. Credential Safety

All provider keys must come from environment variables. Required names currently include:

```text
DASHSCOPE_API_KEY
DEEPSEEK_API_KEY
ZHIPUAI_API_KEY
```

Do not place credentials in source files, notebooks, logs, screenshots, or committed `.env` files.
