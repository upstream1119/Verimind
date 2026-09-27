# 3.1 数据标注流程补充说明

## 目标
- 为论文补齐人工标注流程与一致性（IAA）证据。
- 产出可追溯、可复核的量化结果，支撑方法学严谨性描述。

## 数据来源
- `csv/human_validation_50_samples_scored.csv`
  - 用于 `LLM Judge` 与 `Human Judge` 的一致性核验（50 条）。
- `csv/relabeled_audit_sheet.csv`
  - 复标审计主表（含 `human_label`）。
- `csv/relabeled_audit_sheet2.csv`
  - 第二轮复标标签（`label2`）。

## 审计规模与分层
- 复标审计总样本：`600`
- 系统覆盖：`VeriMind_Full / Vanilla_RAG / Self_RAG / HyDE`，每系统 `150` 条。
- 分层覆盖：
  - `TRUSTED = 256`
  - `WARNING = 48`
  - `REJECTED = 296`

## 一致性结果（3.1 正式产物）
- 产物文件：
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.csv`
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.txt`
- 关键数值：
  - `LLM vs Human (n=50)`：`agreement=0.9600`，`kappa=0.9151`
  - `Annotator A vs B overall (n=600)`：`agreement=0.9033`，`kappa=0.7767`
  - `TRUSTED`：`agreement=0.9844`，`kappa=0.9686`
  - `WARNING`：`agreement=1.0000`，`kappa=1.0000`
  - `REJECTED`：`agreement=0.8176`，`kappa=0.0000`

## 写作边界（必须保留）
- 可写“整体一致性达到较高水平（overall kappa≈0.78）”。
- `REJECTED` 分层需说明：该层正例占比极低，Kappa 对边际分布敏感，出现“agreement 较高但 kappa 偏低”的统计现象。

## 复现命令
```powershell
$env:PYTHONIOENCODING='utf-8'
D:\anaconda\envs\rag_domestic_env\python.exe revision/analysis/annotation_audit_3_1.py
```

