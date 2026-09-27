# 实验结果记录

> 本文档汇总所有实验的关键结果，供论文写作直接引用。
> 进度跟踪见 `revision/progress.md`，任务规划见 `revision/task_plan.md`。

---

## 实验索引

| 编号 | 名称 | 状态 | 数据集 | 生成器 | 评判器 | 完成日期 |
|---|---|---|---|---|---|---|
| E0 | 主实验（VeriMind-Bench，10策略） | 已完成 | VeriMind-Bench (300) | Qwen-Plus | Qwen-Plus | 已有 |
| SF | SciFact 外部基准验证 | 已完成 | SciFact (500) | GLM-4.5-Air | DeepSeek-V3.2 | 2026-04-13 |
| E2 | 跨架构：GLM 生成 + Qwen 评判 | 已完成 | VeriMind-Bench (300) | GLM-4.5-Air | Qwen-Plus | 2026-04-15 |
| E3 | 跨架构：Qwen 生成 + DeepSeek 评判 | 已完成 | VeriMind-Bench (300) | Qwen-Plus | DeepSeek-V3.2 | 2026-04-15 |
| E4 | 跨架构：DeepSeek 生成 + GLM 评判 | 已完成 | VeriMind-Bench (300) | DeepSeek-V3.2 | GLM-4.5-Air | 2026-04-15 |
| E5 | 最强模型验证（分层100） | 已完成 | VeriMind-Bench (100) | GLM-4.7 | DeepSeek-V3.2 | 2026-04-15 |
| CA | 跨架构总收口（E1-E5） | 已完成 | VeriMind-Bench | 多组合 | 多评判位统计 | 2026-04-15 |
| B1 | 主结果 BERTScore 补齐 | 已完成 | VeriMind-Bench (300) | - | - | 2026-04-15 |
| B2 | Cohen's h 效应量 | 已完成 | VeriMind-Bench (300) | - | - | 2026-04-15 |
| B3 | 阈值敏感性分析 | 已完成 | VeriMind-Bench (300) | - | - | 2026-04-15 |
| A1 | 标注流程与一致性审计（3.1） | 已完成 | 人工复标集 (600) | - | 双标注一致性 | 2026-04-15 |
| CE | 三评判器一致性统计（full） | 已完成 | VeriMind-Bench (300) | - | Qwen/GLM/DeepSeek | 2026-04-14 |

---

## E0：主实验（VeriMind-Bench，10策略）

- 数据文件：`revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv`
- 规模：`3000` 行（`300` 题 × `10` 策略）

### 策略级汇总（按主表均值）

| 策略 | 幻觉率 | faith_score | f1_score |
|---|---:|---:|---:|
| Vanilla_RAG | 0.380000 | 6.186667 | 0.222392 |
| HyDE | 0.353333 | 6.476667 | 0.229042 |
| Self-RAG | 0.040000 | 9.620000 | 0.145594 |
| Long-Context_LLM | 0.600000 | 4.000000 | 0.217790 |
| VeriMind_No_Routing | 0.000000 | 10.000000 | 0.138766 |
| VeriMind_No_Gating | 0.386667 | 6.160000 | 0.223768 |
| Ablation_no_structure | 0.000000 | 10.000000 | 0.146625 |
| VeriMind_Full | 0.000000 | 10.000000 | 0.133966 |
| CRAG_Local | 0.036667 | 9.633333 | 0.152785 |
| CoT_RAG | 0.033333 | 9.666667 | 0.098778 |

---

## SF：SciFact 外部基准验证（500 条）

- 数据文件：`revision/artifacts/1_1_scifact/results/scifact_experiment_results.csv`
- 完成日期：2026-04-13

| 系统 | 幻觉率 | Coverage | ROUGE-L F1 | BERTScore F1 |
|---|---:|---:|---:|---:|
| VeriMind_Full | **0.0000** | 0.8100 | 0.0359 | 0.8104 |
| Vanilla_RAG | 0.3960 | 0.8160 | 0.0325 | 0.8051 |
| HyDE | 0.5920 | 0.8240 | 0.1151 | 0.8232 |

---

## CE：三评判器一致性统计（full）

### 输入
- `revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results.csv`
- 输入规模：`9000` 行（`3000` 基础样本 × `3` 评判器）

### 输出
- `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.csv`
- `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.txt`

### 校验
- Triplet 覆盖：`3000/3000 = 1.0000`
- 缺失样本：`0`

### 统计结果
- Cohen's Kappa (qwen-glm)：`0.559403`
- Cohen's Kappa (qwen-deepseek)：`0.546824`
- Cohen's Kappa (glm-deepseek)：`0.700133`
- Fleiss' Kappa (qwen-glm-deepseek)：`0.601333`

### 论文可用一句话
- 在 full 3000 个三评判样本上，Fleiss' Kappa 为 0.6013，表明跨评判器一致性达到可接受水平；两两 Cohen's Kappa 显示一致性处于中等到较高区间，支持“多评判器结论稳定但仍有差异”的保守表述。

---

## E2-E5（已完成）

### E2：GLM 生成 + Qwen 评判（已完成）
- 输出文件：`revision/artifacts/1_3_cross_arch/cross_arch_E2.csv`
- 输入规模：`300` 题 × `3` 策略
- 最终规模：`900` 行（完整）
- 关键结果：
  - Vanilla_RAG：faith=`5.4533`，hallucination_rate=`0.4600`，f1=`0.2584`
  - HyDE：faith=`5.9433`，hallucination_rate=`0.4133`，f1=`0.2680`
  - VeriMind_Full：faith=`9.9700`，hallucination_rate=`0.0033`，f1=`0.1054`
- 过程记录：
  - 首轮出现 `need to escape, but no escapechar set`，导致 `893/900`。
  - 修复 `escapechar` 后补跑，最终 `900/900`。

### E3：Qwen 生成 + DeepSeek 评判（已完成）
- 输出文件：`revision/artifacts/1_3_cross_arch/cross_arch_E3.csv`
- 输入规模：`300` 题 × `3` 策略
- 最终规模：`900` 行（完整）
- 关键结果：
  - Vanilla_RAG：faith=`2.6200`，hallucination_rate=`0.7267`，f1=`0.2251`
  - HyDE：faith=`2.8133`，hallucination_rate=`0.7067`，f1=`0.2330`
  - VeriMind_Full：faith=`9.9667`，hallucination_rate=`0.0000`，f1=`0.1064`
- 论文可用一句话：
  - 在评判器切换至 DeepSeek 后，基线幻觉率显著升高，而 VeriMind 仍保持近 0 幻觉率，显示跨评判器下的相对稳定优势。

### E4：DeepSeek 生成 + GLM 评判（已完成）
- 输出文件：`revision/artifacts/1_3_cross_arch/cross_arch_E4.csv`
- 输入规模：`300` 题 × `3` 策略
- 最终规模：`900` 行（完整）
- 关键结果：
  - Vanilla_RAG：faith=`3.3000`，hallucination_rate=`0.6700`，f1=`0.2545`
  - HyDE：faith=`3.3333`，hallucination_rate=`0.6667`，f1=`0.2757`
  - VeriMind_Full：faith=`10.0000`，hallucination_rate=`0.0000`，f1=`0.1132`
- 论文可用一句话：
  - 在 `DeepSeek -> GLM` 组合下，基线策略幻觉率仍维持高位，而 VeriMind 继续保持近 0 幻觉率，进一步支持跨架构稳健性结论。

### E5：GLM-4.7 生成 + DeepSeek 评判（已完成）
- 输出文件：`revision/artifacts/1_3_cross_arch/cross_arch_E5.csv`
- 输入规模：`100` 题 × `3` 策略
- 最终规模：`300` 行（完整）
- 关键结果：
  - `Vanilla_RAG`：faith=`2.2400`，hallucination_rate=`0.7600`，f1=`0.2877`
  - `HyDE`：faith=`1.9400`，hallucination_rate=`0.7700`，f1=`0.3118`
  - `VeriMind_Full`：faith=`9.9700`，hallucination_rate=`0.0000`，f1=`0.1222`
- 论文可用一句话：
  - 在 GLM-4.7 强生成器条件下，VeriMind 依然维持 0 幻觉率，进一步补强跨架构稳健性证据；但其更保守的回答模式仍需以“安全-充分性权衡”进行解释。

### CA：跨架构总收口（E1-E5，已完成）
- 汇总文件：
  - `revision/artifacts/1_3_cross_arch/cross_arch_E1_E5_summary.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.txt`
- 统计口径：
  - E1/E2/E3/E4 为 300 题 × 3 策略；
  - E5 为 100 题 × 3 策略；
  - Fleiss 采用 common100（E5 公共题集）与二值幻觉标签（`hallucination_rate >= 0.5`）。
- 关键结果：
  - overall Fleiss' Kappa（common100）=`0.521101`
  - `Vanilla_RAG`=`0.270416`，`HyDE`=`0.329063`，`VeriMind_Full`=`-0.002004`
  - 幻觉率区间（common100）：
    - Vanilla_RAG：`0.38 -> 0.76`
    - HyDE：`0.40 -> 0.77`
    - VeriMind_Full：`0.00 -> 0.01`
- 论文可用一句话：
  - E1-E5 汇总显示，VeriMind 在跨架构组合下保持近零幻觉且波动最小，而基线策略对生成器/评判器切换更敏感，支持“稳定控幻觉优势”但不宣称“全面指标最优”。

### B1：主结果 BERTScore 补齐（已完成）
- 输出文件：
  - `revision/artifacts/2_1_bertscore/experiment_results_sci_v3_with_bertscore.csv`
  - `revision/artifacts/2_1_bertscore/table7_bertscore_summary.csv`
- 输入规模：
  - 主表 `3000` 行（`300` 题 × `10` 策略）
- 完整性校验：
  - `bert_score_f1` 全覆盖（无缺失、无 `-1.0`）
- Table 7 关键值：
  - `Vanilla_RAG`：`0.795088`
  - `HyDE`：`0.796373`
  - `VeriMind_Full`：`0.789337`
- 论文可用一句话：
  - BERTScore 指标已在主表范围内补齐，结果显示各策略语义相似度处于相近区间，VeriMind 的核心优势仍主要体现在低幻觉与高可信度维度。

### B2：Cohen's h 效应量（已完成）
- 输出文件：
  - `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.csv`
  - `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.txt`
- 输入规模：
  - 主表 `3000` 行（`300` 题 × `10` 策略）
- 方法：
  - 锚定 `VeriMind_Full`，对每个其他策略计算 `hallucination_rate` 的 Cohen's h。
- 关键结果：
  - `Long-Context_LLM`: `h=1.772154`（large）
  - `Vanilla_RAG`: `h=1.328430`（large）
  - `HyDE`: `h=1.273085`（large）
  - 效应量分布：`large=4`，`small=3`，`negligible=2`
- 论文可用一句话：
  - Cohen's h 显示 VeriMind 相对主要基线在幻觉率上达到 large 级别效应量，说明其优势不仅统计显著且具有较强实践意义。

### B3：阈值敏感性分析（已完成）
- 输出文件：
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.csv`
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_summary.txt`
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.svg`
- 方法：
  - 以 `VeriMind_Full(t=5)` 观测结果为锚点，结合 `VeriMind_No_Gating` 配对样本进行阈值变动仿真。
- 关键结果：
  - `t=5.0`：`hal=0.000000`，`f1=0.133966`，`accept_rate=0.966667`
  - `t=0.0`：`hal=0.030000`，`f1=0.144875`
  - `t=8.0`：`hal=0.000000`，`f1=0.117097`，`accept_rate=0.710000`
- 论文可用一句话：
  - 阈值敏感性结果表明，`4.0~5.0` 区间可在维持零幻觉的前提下保持较高回答充分性，是当前系统的稳健部署区间。

### A1：标注流程与一致性审计（3.1，已完成）
- 输出文件：
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.csv`
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.txt`
  - `revision/annotation_process_3_1.md`
- 输入：
  - `csv/human_validation_50_samples_scored.csv`
  - `csv/relabeled_audit_sheet.csv`
  - `csv/relabeled_audit_sheet2.csv`
- 规模：
  - 复标总样本 `600`（4 系统均衡，每系统 `150`）
  - 分层：`TRUSTED=256`，`WARNING=48`，`REJECTED=296`
- 一致性结果：
  - `LLM vs Human (n=50)`：`agreement=0.9600`，`kappa=0.9151`
  - `Annotator A vs B overall (n=600)`：`agreement=0.9033`，`kappa=0.7767`
  - 分层：`TRUSTED kappa=0.9686`，`WARNING kappa=1.0000`，`REJECTED kappa=0.0000`
- 论文可用一句话：
  - 我们在 600 条分层复标样本上得到 `Cohen's Kappa=0.7767` 的整体标注一致性，表明人工标注流程具有较高可重复性；对 `REJECTED` 分层的 kappa 偏低现象，需结合其高度不均衡标签分布进行解释。

---

*最后更新：2026-04-15*
