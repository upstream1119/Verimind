# SciFact supported 标签来源与一致性审计

## 必须修复

- 当前 `supported` 不宜作为真实证据支持的 ground truth：450 条候选中，**20 条**原始 evaluator 输出的 `supported=true` 与其文字理由中的明确“证据不足”信号相冲突；**28 条**同时出现 `supported=true`、`evidence_sufficient=false`。前者包含在后者之中。关键词仅用于定位待复核记录，不生成修正标签。
- 以 Judge 分数 ≥5 作为*比较代理阈值*，与 evaluator `supported` 布尔值不一致 **160 条**；这些只是两个自动判断器的分歧，不是经人工确认的错判数。
- Judge=0 共 **166 条**，其中 **145 条**被 evaluator 标为 supported；TrustScore [0.30,0.40) 子集中 Judge=0 为 **145 条**、其中 supported=true 为 **127 条**。这两组分母不同，不要混写。

## 应修复

- 对冲突记录安排独立盲审并建立明确的 supported 定义；此前 30 条内容审计是模型执行的初步判读，不能外推为 450 条的人工真值。
- 将 `supported`、`evidence_sufficient`、`correct` 三个字段分别审核，避免用 gold verdict 的正确性替代检索证据支持性。

## 可选

- 对 Judge 0–10 分数做带有独立标注集的校准；不要用这 450 条自动标签直接训练后再评价。

## 证据缺口与代码溯源

1. 当前运行使用 `paired-v1`：GLM 生成、DeepSeek 给出 0–10 的 `faith_score`、Qwen 独立返回包含 `supported` 的 JSON。生成和评分写入同一条 `answers.jsonl` 记录。源头在 `revision/experiments/supplement_20260927.py` 的 `generate_and_evaluate`；`parse_evaluation` 只解析 JSON 并检查布尔类型，不从文字理由映射 `supported`。
2. `offline_gate_diagnostics.read_rows` 直接读取 `answer["evaluation"]["supported"]`；`aggregate`/`case_metrics.csv` 也直接用该字段。未发现当前路径存在 label merge、旧标签覆盖或理由关键词映射。
3. 450/450 原始 evaluator JSON 与保存解析对象一致；450/450 原始 judge 整数与保存分数一致；450/450 UID、dataset 和检索上下文 SHA-256 对齐。全部记录来自同一 `answers.jsonl`，时间戳日期均为 2026-09-27。**未发现确定性的 parser、ID 对齐或版本混用 bug**；这不能排除模型自身的输出矛盾或上游未记录的差异。
4. 样本来自 SciFact train（249 条）和 dev（201 条）的候选回答，不能称为独立 held-out test。当前产物中 `ID_ALIGNMENT_ERROR`、`PARSER_ERROR` 检出数为 0；未发现版本混用证据，但不能证明上游不存在未记录的版本差异。
5. `BOOLEAN_REASON_CONFLICT` 的文字模式是保守检索信号；`PARTIAL_SUPPORT_AMBIGUITY_CANDIDATE` 也只是待审标记。CSV 的 `derived_supported_boolean` 是从**同一原始 JSON 重新解析**，不是 corrected label；`corrected_supported_label` 留空。

## 建议下一步与指标边界

- 可继续作为开发集描述统计使用：检索 Recall@K、与 gold verdict 确定性比较的 label accuracy、覆盖率/拒答率、原始分数分布；写清数据切分和指标定义。
- 暂停把 SciFact 的 `supported`/`unsupported`、Faithfulness、Unsupported AURC 和以它们构成的 safe AUROC 写成论文或简历的真实事实性/幻觉率结论，直至独立复核或重新标注。
- 逐例证据见 `scifact_label_trace.csv`，冲突子集见 `scifact_conflicts.csv`。冲突类型允许重叠，不能相加当作独立问题数。
