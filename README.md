# VeriMind: Reliable Scientific RAG

VeriMind is a research codebase for evidence-aware retrieval-augmented generation (RAG) and selective answer control in scientific question answering. The current work studies a risk-constrained policy that can answer directly, invoke an optional verifier, or abstain while reporting the quality--coverage--cost trade-off.

The repository contains the implementation, public-benchmark experiments, audit artifacts, analysis scripts, and the current Computing manuscript. It intentionally excludes API credentials, downloaded dataset caches, local vector databases, and third-party paper PDFs.

## Start Here

1. Read [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) for the architecture, directory map, evidence boundaries, and recommended reading order.
2. Read [`revision/submission/Computing_2026/source/main.tex`](revision/submission/Computing_2026/source/main.tex) for the current paper formulation.
3. Read [`revision/index/current_artifacts.md`](revision/index/current_artifacts.md) for canonical experiment outputs.
4. Read [`revision/artifacts/supplement_20260927_full/RESULTS.md`](revision/artifacts/supplement_20260927_full/RESULTS.md) for the latest supplemental evaluation.

## Main Components

```text
rag_app_final.py                  Interactive RAG application
run_sci_experiment_main.py       Controlled stress-test experiment
revision/
  qasper/                        QASPER preparation, retrieval, evaluation, audits
  scifact/                       SciFact preparation and evaluation
  analysis/                      Statistical and selective-risk analyses
  experiments/                   2026-09-27 supplemental experiments
  artifacts/                     Raw and processed experiment evidence
  plot_scripts/                  Reproducible paper-figure scripts
  submission/Computing_2026/     Current manuscript and submission source
```

## Environment

Python 3.10 is recommended. Install the project dependencies in an isolated environment:

```powershell
python -m pip install -r requirements.txt
```

Configure provider credentials in a local `.env` file or the process environment. Never commit credentials.

```env
DASHSCOPE_API_KEY=
DEEPSEEK_API_KEY=
ZHIPUAI_API_KEY=
```

The code uses provider model aliases. Results may vary if a provider changes the model behind an alias.

## Reproducing the Public-Benchmark Pipeline

QASPER:

```powershell
python revision/qasper/prepare_qasper.py
python revision/qasper/build_qasper_index.py
python revision/qasper/run_qasper_experiment.py
python revision/qasper/analyze_qasper.py
```

SciFact:

```powershell
python revision/scifact/download_scifact.py
python revision/scifact/build_scifact_index.py
python revision/scifact/run_scifact_experiment.py
```

The preparation and indexing steps recreate ignored local dataset and vector-index directories. API-based runs incur provider costs.

## Offline Checks

```powershell
python -m pytest revision/analysis revision/experiments revision/qasper -q
python revision/analysis/recompute_publication_metrics.py
```

## Manuscript

The current Computing source is under `revision/submission/Computing_2026/source/`. Compile `main.tex` with a standard `pdflatex -> bibtex -> pdflatex -> pdflatex` workflow or `latexmk -pdf main.tex`.

## Evidence Rules

- Treat generated CSV/JSON/JSONL files as research evidence; do not edit values manually.
- Preserve raw outputs before deriving tables or figures.
- Distinguish LLM-judge faithfulness labels from human-supported claims and reference-based answer quality.
- Report coverage and accepted-answer denominators together with selective risk.
- Do not treat development-set diagnostics or stress tests as held-out public-benchmark evidence.

## Data and License Notes

QASPER and SciFact remain governed by their original licenses. The repository does not redistribute local dataset caches, provider credentials, or third-party paper corpora. No repository-wide software license has yet been declared; contact the author before redistribution or reuse beyond research inspection.
