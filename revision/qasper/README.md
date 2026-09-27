# QASPER public-benchmark experiment

This directory contains the public scientific-paper QA experiment added for the
VeriMind journal resubmission.

## Design

- Dataset: QASPER validation split (CC BY 4.0)
- Deterministic subset: 300 questions across 182 papers, targeting 20%
  unanswerable questions
- Generator: GLM-4.5-Air
- Judge: DeepSeek (`deepseek-chat`)
- Main systems: Vanilla RAG, HyDE, VeriMind
- Core ablations: no gating, no routing, no structure
- Primary metrics:
  - raw hallucination rate before the gate
  - passed hallucination rate among accepted answers
  - answer coverage
  - evidence recall
  - ROUGE-L F1

## Commands

```powershell
D:\anaconda\envs\rag_domestic_env\python.exe revision/qasper/prepare_qasper.py
D:\anaconda\envs\rag_domestic_env\python.exe revision/qasper/build_qasper_index.py
D:\anaconda\envs\rag_domestic_env\python.exe revision/qasper/run_qasper_experiment.py --limit 5
D:\anaconda\envs\rag_domestic_env\python.exe revision/qasper/run_qasper_experiment.py
```

The runner is append-only and resumes from completed `(item_id, strategy)`
pairs.

Index construction writes every 100 nodes and retries transient API failures,
so it can resume from the existing Chroma collection count.

## Robustness validation

Re-evaluate only the 802 accepted answers. Existing DeepSeek scores are reused;
Qwen-Plus and GLM-4.5-Air are called incrementally, with checkpointing after
each successful score.

```powershell
D:\anaconda\envs\rag_domestic_env\python.exe revision/qasper/run_qasper_cross_judge.py --limit 5
D:\anaconda\envs\rag_domestic_env\python.exe revision/qasper/run_qasper_cross_judge.py
D:\anaconda\envs\rag_domestic_env\python.exe revision/qasper/analyze_qasper_cross_judge.py
```

Prepare the deterministic 50-case author blind audit:

```powershell
D:\anaconda\envs\rag_domestic_env\python.exe revision/qasper/prepare_qasper_author_audit.py
```

Fill `author_label` in the blind CSV before opening the key. Allowed values are
`supported`, `unsupported`, and `uncertain`. Then run:

```powershell
D:\anaconda\envs\rag_domestic_env\python.exe revision/qasper/analyze_qasper_author_audit.py
```

The 50-case audit is challenge-enriched for judge validation and must not be
used to estimate the population hallucination rate.
