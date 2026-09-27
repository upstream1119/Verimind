# VeriMind 最终离线诊断与简历可引用边界

本轮只读取既有 QASPER/SciFact 实验产物及本地索引；没有调用 LLM/API、生成答案或重新检索。新结果均保存在本目录。QASPER 的 300 题属于开发实验；另列的 150 题来自独立 held-out 产物，两组数据不可混称。

## 1. Router

**核心数字。** QASPER 全量 300 题中，237 题有可评估 gold evidence；文本可检索子集为 251 题，其中 188 题可评估。全量 mean Router Regret 为 **0.0546 (@3)** / **0.0782 (@5)**，median 均为 0。@5 有 **26/237 (10.97%)** 题出现正 regret；Router 达到最优为 **211/237 (89.03%)**，但三粒度完全相同就有 **183** 题，Router *唯一最佳且选对*为 **0**。@3 相应为最优 217/237、正 regret 20 题、唯一最佳且选对 1 题。

文本子集 @5 mean regret **0.0919**，正 regret **23/188**，Router 最优 **165/188**，三粒度全相同 **141**。全量 @5 的唯一最佳粒度是 128（22 题）；512、1024 独占最佳均为 0。最优粒度出现并列时每个粒度都计入对应的“包含并列最优”次数，不强行分配。

**问题分布。** Oracle 相对固定 512 的 @5 收益只发生在 **22/237** 题；其中收益最大的 10 题贡献 **61.73%** 总增益。Router 正 regret 的 26 题中，前 10 题贡献 **53.96%** 总 regret。当前 `route_query` 是关键词规则 Router，**不是 LLM Router**。没有可靠的独立 factual/reasoning 问题类型标签；现有 DETAIL/CONCEPT/CONTEXT 是 Router 规则类别，不能用来证明题目本体类型。按规则类别，@5 平均 regret：DETAIL 17 题为 **0.1373**，CONCEPT 210 题为 **0.0771**，CONTEXT 10 题为 **0**；CONCEPT 因样本多贡献主要绝对 regret。

**可安全下的结论：** 在这批开发问题里当前 Router 的检索 Recall 略低于固定 512，Oracle 上限的改善集中于少数可获益题。**不可下的结论：** “Router 准确率 89%”“动态多粒度提高召回”“DETAIL 问题天然最难”。当前 Router 不宜作为简历或论文的核心性能贡献；可以作为实现与消融对象。

## 2. Multi-granularity Retrieval

RRF 使用 `k_rrf=60`。主结果只将同一论文中**重建后的原始 parent 段落文本完全相同**的 chunk 合并，以论文 ID 和完整段落哈希作为键；同一路由重复 chunk 只取其最高排名。没有语义相似去重，也没有使用 gold evidence 排序。对照列“仅精确 chunk ID”不做跨粒度合并。并列分数按预定粒度顺序 128→512→1024 处理；另试遍六种固定粒度顺序，两个子集的汇总 Recall@3/5 均未变化，详见 `fusion_tie_sensitivity.csv`。

| 范围 | Recall | 128 | Fixed 512 | 1024 | Router | RRF 段落去重 | RRF 仅 chunk ID | Oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | @3 | 0.388 | 0.429 | 0.427 | 0.421 | 0.417 | 0.282 | 0.476 |
| all | @5 | 0.502 | 0.503 | 0.501 | 0.493 | 0.503 | 0.409 | 0.572 |
| text_retrievable | @3 | 0.437 | 0.479 | 0.479 | 0.473 | 0.463 | 0.323 | 0.533 |
| text_retrievable | @5 | 0.568 | 0.564 | 0.564 | 0.555 | 0.564 | 0.461 | 0.647 |

全量可评估题上，段落去重 RRF @3 为 **0.417**，低于固定 512 的 **0.429** 和 Router 的 **0.421**；@5 为 **0.503**，与固定 512 **逐题相同**，比 Router **0.493** 略高，距 Oracle **0.572** 仍差 **0.068**。文本子集 @5 同样与固定 512 相同（**0.564**），距 Oracle **0.082**。仅 chunk ID 去重时全量 @3/@5 降为 **0.282/0.409**，说明跨粒度重复证据处理对融合结果影响很大。

**最终判断：** 三粒度 Oracle 显示问题特异性的检索空间，但当前**关键词规则 Router**与这个无模型 RRF 都未超过固定 512。Fusion 未接近 Oracle，不能据此推断未来学习式/LLM Router 一定无用；只能说**当前规则 Router 没有证明必要性或性能收益**。Oracle 的收益主要来自少数在 128 下更容易召回 gold 的题，后续若研究路由应针对这些题预先学习或识别，而非事后选优。

固定 512 逐题重算与既有 Oracle/冻结结果完全一致；输入 300 题一致；Fusion 排序不读 gold。`fusion_retrieval.jsonl` 包含逐题两种 RRF 排序与命中信息，`fusion_per_question.csv`、`fusion_summary.csv/json` 给出复算值。

## 3. SciFact Label Reliability

450 条 SciFact+NEI 候选来自 train 249 / dev 201。`supported` 来源是 Qwen 原始 JSON；解析器没有从 reason 生成布尔值。原始 JSON、保存解析对象、judge 原始整数、样本 ID 与检索上下文哈希 **均 450/450 对齐**；**未发现确定性的解析、merge、ID 或版本混用 bug**。但有 **20 条**布尔值与明确“证据不足”的文字理由冲突，**28 条** `supported=true` 且 `evidence_sufficient=false`，以及 **160 条** Judge≥5 的代理判断与 evaluator 布尔值冲突（类型可重叠）。Judge=0 共 **166** 条，其中 **145** 条被 evaluator 标为 supported；TrustScore [0.30,0.40) 的 Judge=0 子集为 **145** 条，其中 **127** 条标 supported。主要问题是**自动 evaluator 输出内部不一致与不同判断器的定义/判断分歧**；现有证据不能把冲突全部归因于 Judge，也不能生成 corrected label。

**暂停引用**以 SciFact 自动 `supported` 为真值的 faithfulness、unsupported rate、Unsupported AURC、safe AUROC，以及“真实幻觉率下降”。可继续引用明确标注为开发集描述统计的检索 Recall、gold verdict 的确定性准确率、coverage/拒答比例和原始分数分布。详见 `scifact_label_audit_report.md` 和两份逐例 CSV。

## 4. Resume-safe Metrics

下列数字来自与上述 300 题开发分析**分开的 150 题 QASPER held-out**，已从 `1_17_qasper_cheap_gate/cheap_gate_ranking_comparison.csv` 和 `1_18_qasper_cost_aware_cascade/qasper_cost_aware_summary.csv` 复核：

| 可引用数字 | Dataset / N | Metric 与评价范围 | 必要限制 |
|---|---|---|---|
| Cheap score observed AURC **0.2915**；legacy TrustScore **0.3840** | QASPER held-out 150 题中的 **96 条自然回答** | 按归一化 Answer F1 排序，observed candidate-answer coverage 范围 | 不等于事实支持性或生产环境性能 |
| Cheap score 与 Answer F1 的 Spearman **0.4137** | 同上，96 条候选 | 候选答案质量排序 | 不表示 Answer F1 本身达到 0.4137 |
| Frozen cost-aware cascade 接受 **23/150（15.33%）**，已答平均 Answer F1 **0.5046** | QASPER held-out 150 题 | 预先在 calibration 选定的 ε=0.55 工作点 | 不宣称优于所有 Always Verify 工作点 |
| Verifier 调用 **13/150**；估计输入 tokens **6,811 vs 56,777（约少 88%）** | QASPER held-out 150 题；Always Verify 对 96 条自然回答调用 | 离线字符数/4 的 verifier 输入量估计 | 非真实 API 账单、测得延迟或端到端成本 |

不把本轮开发集 Oracle 写成实际系统性能；不把 17/30 初步审计外推；不把 SciFact 自动 unsupported 率称为真实幻觉率；不把开发题称为 held-out。

## 5. Recommended Resume Wording

- 构建 VeriMind 科研文献问答原型，打通 Chroma 检索、证据追踪、答案生成与可拒答评估链路，并为 QASPER 开发集建立逐题可复算的检索与消融产物。
- 设计不依赖 LLM Judge 输出的轻量答案质量评分；在独立 QASPER held-out 150 题中的 96 条候选回答上，按 Answer F1 排序的 observed AURC 为 **0.2915**，优于旧版 TrustScore 的 **0.3840**。
- 实现校准后冻结阈值的按需验证级联；在 QASPER held-out 150 题上回答 **23 题**、调用 verifier **13 次**，估计 verifier 输入量较对全部 96 条自然回答验证减少约 **88%**（离线 token 估计）。
- 建立逐样本审计与回归校验：对 300 道 QASPER 开发题完成三粒度 Recall/Oracle/RRF 对照，并追踪 450 条 SciFact 候选的原始评分、解析值和上下文哈希，识别自动标签内部不一致。

## 审计优先级与复现

**必须修复：** SciFact 自动支持性标签的独立复核；简历和论文停止把它当真实幻觉率。**应修复：** 若继续研究多粒度，设计独立训练/校准的路由器并与固定 512、无模型融合公平比较。**可选：** 在更大 held-out 上复核检索与工作点。**证据缺口：** 独立 SciFact 人工真值、Router 的 held-out 收益、生产调用成本。**下一检查：** 先复核 28 条内部布尔冲突，再确定标签修订协议。

复现顺序：`python -B run_router.py`、`python -B run_fusion.py`、`python -B run_tie_sensitivity.py`、`python -B run_scifact.py`、`python -B run_report.py`。`run_fusion.py` 仅读本地 Chroma 索引；无模型或网络调用。输入哈希、运行时间和所有汇总见 `summary.json`。
