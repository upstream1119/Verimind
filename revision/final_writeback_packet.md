# 终稿写回包（4.4 执行说明）

> 状态：确认论文如何修改，等待执行  
> 用途：将“证据文件 -> 论文段落 -> 表格/图 -> 写作口径”一次性整理，供 Overleaf 直接落地。  
> 说明：本文件定义“怎么改”，不直接修改论文正文。

---

## 1. 写回顺序（严格）

1. 外部验证（SciFact）
2. 10 策略统一主实验
3. 三评判器 cross-judge 与一致性统计
4. 跨架构 E2-E5 与汇总（E1-E5）
5. BERTScore / Cohen's h / 阈值敏感性
6. 标注一致性、AI 使用声明、Related Work
7. 全文口径统一与终稿润色

---

## 2. Overleaf 逐章节如何修改

### 2.1 摘要（Abstract）
- 增加三条关键信息：
  - 外部验证：SciFact 已完成且支持核心结论。
  - 对等比较：10 策略统一协议完成。
  - 跨评判器一致性：Fleiss' Kappa = `0.601333`（可接受一致性）。
- 避免使用“全面最优/普遍最优”表述，统一为“在当前评测范围内”。

### 2.2 引言（Introduction）
- 在动机段增加“RAG 幻觉风险 + 多评判器偏置风险”。
- 在贡献段按以下顺序改写：
  1. 外部验证补齐（SciFact）
  2. 10 策略统一协议
  3. 三评判器一致性 + 跨架构稳健性（E1-E5）
  4. 补充统计（BERTScore、Cohen's h、阈值敏感性）

### 2.3 方法（Method）
- 保留 VeriMind 的核心机制描述。
- 新增一段“风险优先目标”：
  - 目标是降低幻觉风险（risk-aware / trust-gated），
  - 不是无条件追求所有指标同时最优。

### 2.4 实验设置（Experimental Setup）
- 明确数据与规模：
  - VeriMind-Bench：300 题
  - 10 策略主表：3000 行
  - cross-judge：9000 行（3 评判器）
  - 跨架构：E2/E3/E4 各 900 行，E5 为 300 行
- 明确模型口径与声明：
  - Qwen-Plus / GLM-4.5-Air / GLM-4.7（E5）/ DeepSeek-V3.2（deepseek-chat）
  - 不写 ChatGPT-5.0、GLM-4.6V 为本实验模型

### 2.5 结果章节（Results）

#### A. 外部验证（SciFact）
- 引用证据：`revision/artifacts/1_1_scifact/results/scifact_experiment_results.csv`
- 写回要点：
  - VeriMind 在 SciFact 上显著低幻觉
  - Coverage 接近，差异主要来自门控机制

#### B. 主实验（10 策略）
- 引用证据：`revision/artifacts/1_2_baseline/1_2_1_main/experiment_results_sci_v3.csv`
- 写回要点：
  - 统一协议下 10 策略公平比较成立
  - 消除“基线不对等”质疑

#### C. 三评判器一致性
- 引用证据：
  - `revision/artifacts/1_2_baseline/1_2_2_cross_judge/cross_judge_results.csv`
  - `revision/artifacts/1_2_baseline/1_2_3_agreement/cross_judge_agreement.txt`
- 写回要点：
  - Triplet coverage = `3000/3000`
  - Fleiss' Kappa = `0.601333`
  - 两两 Cohen's Kappa 同时报告
- 口径约束：
  - 写“可接受一致性”，不写“高度一致”。

#### D. 跨架构结果（E2-E5）
- 引用证据：
  - `revision/artifacts/1_3_cross_arch/cross_arch_E2.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_E3.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_E4.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_E5.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_E1_E5_summary.csv`
  - `revision/artifacts/1_3_cross_arch/cross_arch_fleiss_E1_E5.txt`
- 写回要点：
  - VeriMind 在 E1-E5 下维持近零幻觉区间
  - 基线对生成器/评判器切换更敏感
  - 明确“保守性-充分性”权衡

#### E. 补充统计（2.x）
- BERTScore：
  - `revision/artifacts/2_1_bertscore/table7_bertscore_summary.csv`
- Cohen's h：
  - `revision/artifacts/2_2_cohen_h/cohen_h_vs_verimind.txt`
- 阈值敏感性：
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_summary.txt`
  - `revision/artifacts/2_3_threshold/threshold_sensitivity_curve.svg`
- 写回要点：
  - BERTScore 作为补充指标，不替代幻觉率主结论
  - 主要基线对 VeriMind 的幻觉率差异达到 large 效应量
  - 阈值 `4.0~5.0` 为稳健工作区间（零幻觉优先）

### 2.6 标注一致性与 AI 使用声明
- 标注一致性证据：
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.txt`
  - `revision/annotation_process_3_1.md`
- AI 声明与表述降调依据：
  - `revision/🛠️ 核心战役：全面修改方案与执行计划.md`
- 写回要点：
  - 标注流程可复核，整体一致性可接受
  - AI 工具用途边界清晰
  - 结论为经验性证据，不做不当外推

### 2.7 Related Work 与参考文献
- 统一引用库：
  - `revision/references/related_work_2024_2025.bib`
- 写回要点：
  - 将 2024/2025 条目合并入 Overleaf `.bib`
  - 引言与 Related Work 的 citation key 保持一致

---

## 3. 图表与证据路径映射（写回时直接查）

- 资产索引总入口：`revision/index/current_artifacts.md`
- 论文证据总表：`revision/paper_revision_evidence.md`
- 实验汇总日志：`revision/experiment_log.md`

---

## 4. 写作口径约束（必须遵守）

- 必须使用“在当前评测范围内”限定语。
- 一致性表述使用“可接受一致性/中等至较高一致性”。
- 对 VeriMind 统一采用“双句式”：
  - 句 1：低幻觉稳定优势；
  - 句 2：保守性与充分性权衡。

---

## 5. 执行检查单（4.4 启动后逐项勾选）

- [ ] 外部验证段落与表格已写回正文
- [ ] 10 策略主实验与对比表已写回正文
- [ ] cross-judge 与一致性统计已写回正文
- [ ] E2-E5 跨架构结果与讨论已写回正文
- [ ] BERTScore / Cohen's h / 阈值曲线已写回正文
- [ ] 标注流程、AI 声明、Related Work 已写回正文
- [ ] 全文术语、图表编号、引用统一校对完成

---

## 6. 启动条件

- 当你说“开始 4.4”时，按本文件顺序逐段执行写回。
- 每写回一个小节，立即回填：
  - `revision/task_plan.md`
  - `revision/progress.md`
  - `revision/findings.md`
