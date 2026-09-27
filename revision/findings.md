# 发现与决策

## 1. 研究结论（已确认）

### 1.1 SciFact 外部验证（1.1.3）
- VeriMind 在 SciFact 上实现了显著更低的幻觉率。
- 与 Vanilla_RAG、HyDE 对比差异显著，支撑“外部基准可复现优势”。
- 论文价值：可直接支撑“并非仅在自建数据上有效”。

### 1.2 十策略统一主表（1.2.1）
- 10 个系统已全部进入同一实验管线（含 `CRAG_Local` 与 `CoT_RAG`）。
- full 主表已落地 3000 行（300 题 × 10 策略）。
- 论文价值：可回应“基线不对等”审稿质疑。

### 1.3 三评判器全量矩阵（1.2.2）
- `cross_judge_results.csv` 已完成 9000 行（3000 × 3 judge）。
- judge 覆盖完整：`qwen/glm/deepseek`。
- 论文价值：可回应“单评判器偏置”风险。

### 1.4 一致性统计（1.2.3）
- full 正式一致性输出已生成（非 smoke）。
- 关键结果：
  - `Triplet coverage = 3000/3000 = 1.0000`
  - `Cohen's Kappa: qwen-glm=0.559403, qwen-deepseek=0.546824, glm-deepseek=0.700133`
  - `Fleiss' Kappa: qwen-glm-deepseek=0.601333`
- 论文价值：可正式写入“跨评判器一致性达到可接受区间（Fleiss' Kappa > 0.6）”。

### 1.5 跨架构基础设施（1.3.1）
- 已交付统一客户端抽象层，三厂商调用协议统一，错误与重试口径统一。
- 已交付统一跨架构入口，后续 E2-E5 仅需切换 `--generator/--judge`，不再新增分支脚本。
- `dry_run` 可在 0 API 消耗下完成结构验收，降低实验前风险。

### 1.6 跨架构 E2 结果（1.3.2）
- 已完成 `GLM-4.5-Air -> Qwen-Plus` 组合，覆盖 `300题 × 3策略 = 900` 行。
- VeriMind_Full 在 E2 中保持极低幻觉率：
  - `hallucination_rate = 0.0033`（8 次 gated_reject）
- 对比同批次基线：
  - `Vanilla_RAG hal = 0.4600`
  - `HyDE hal = 0.4133`
- 现象：
  - VeriMind 的拒答/保守输出比例较高（`Information not found`/`REJECTED` 较多），对应 f1 偏低。
- 论文价值：
  - 可支撑“跨生成器-评判器组合下，控幻觉优势仍保留”，同时需诚实呈现“充分性-保守性”权衡。

### 1.7 跨架构 E3 结果（1.3.3）
- 已完成 `Qwen-Plus -> DeepSeek` 组合，覆盖 `300题 × 3策略 = 900` 行。
- 基线策略在 E3 下幻觉率显著升高：
  - `Vanilla_RAG hallucination_rate = 0.7267`
  - `HyDE hallucination_rate = 0.7067`
- VeriMind_Full 仍保持近 0 幻觉率：
  - `hallucination_rate = 0.0000`（33 次 gated_reject）
- 论文价值：
  - 可强化“评判器切换后，VeriMind 控幻觉优势仍稳定”的跨架构证据；
  - 同时必须明确“保守性增强（拒答增多）”这一 trade-off。

### 1.8 跨架构 E4 结果（1.3.4）
- 已完成 `DeepSeek -> GLM` 组合，覆盖 `300题 × 3策略 = 900` 行。
- 基线策略在 E4 下幻觉率仍处高位：
  - `Vanilla_RAG hallucination_rate = 0.6700`
  - `HyDE hallucination_rate = 0.6667`
- VeriMind_Full 保持近 0 幻觉率：
  - `hallucination_rate = 0.0000`（29 次 gated_reject）
- 论文价值：
  - 可进一步强化“跨生成器与跨评判器组合下的控幻觉稳定优势”；
  - 仍需在正文中保留“保守性与充分性权衡”的边界陈述。

### 1.9 跨架构 E5 结果（1.3.5）
- 已完成 `GLM-4.7 -> DeepSeek` 组合，覆盖 `100题 × 3策略 = 300` 行。
- 基线策略在 E5 下幻觉率继续维持高位：
  - `Vanilla_RAG hallucination_rate = 0.7600`
  - `HyDE hallucination_rate = 0.7700`
- VeriMind_Full 保持 0 幻觉率：
  - `hallucination_rate = 0.0000`（faith=9.9700，gated_reject=7）
- 论文价值：
  - 补上“强生成器（GLM-4.7）条件”的跨架构证据闭环；
  - 继续支持“保守性换低幻觉”的讨论框架。

### 1.10 跨架构汇总结果（1.3.6）
- 已完成 E1-E5 汇总产物落盘（summary + Fleiss csv/txt）。
- 关键统计（common100，二值标签 `hallucination_rate>=0.5`）：
  - overall Fleiss' Kappa = `0.521101`
  - `Vanilla_RAG` = `0.270416`
  - `HyDE` = `0.329063`
  - `VeriMind_Full` = `-0.002004`
- 解释边界：
  - `VeriMind_Full` 的 kappa 接近 0 并非“结果不稳定”，而是因为其在 E1-E5 中几乎恒为非幻觉（低方差），kappa 对近常量标签不敏感。
- 稳定性结论：
  - 基线在不同组合下波动显著（hal 区间约 `0.38~0.77`），VeriMind 维持 `0.00~0.01` 的近零区间。

### 2.1 主结果 BERTScore 补齐（已完成）
- 已对主表 `3000` 行补齐 `bert_score_f1`，无缺失值。
- 策略级 BERTScore F1 均值（Table 7 素材）：
  - `Vanilla_RAG = 0.795088`
  - `HyDE = 0.796373`
  - `VeriMind_Full = 0.789337`
- 观察：
  - 各策略 BERTScore 区间较集中（约 `0.783~0.796`），说明该指标对当前任务区分度有限；
  - 论文中应将 BERTScore 作为“语义相似度补充指标”，主结论仍以幻觉率/faith 为核心。

### 2.2 Cohen's h 效应量（已完成）
- 已基于主表（3000 行）计算“各基线 vs VeriMind_Full”的幻觉率效应量。
- 结果分布：
  - `large`: 4 组（Long-Context_LLM / VeriMind_No_Gating / Vanilla_RAG / HyDE）
  - `small`: 3 组（Self-RAG / CRAG_Local / CoT_RAG）
  - `negligible`: 2 组（Ablation_no_structure / VeriMind_No_Routing）
- 关键值：
  - `Vanilla_RAG`: `h=1.328430`（large）
  - `HyDE`: `h=1.273085`（large）
  - `Long-Context_LLM`: `h=1.772154`（large）
- 论文价值：
  - 可将“低幻觉优势”从均值差提升为“具有大效应量”的统计论证。

### 2.3 阈值敏感性（已完成）
- 已产出阈值曲线数据与曲线图（CSV/TXT/SVG）。
- 关键观察：
  - `t=5.0`（当前）下：`hal=0.000000`，`f1=0.133966`，`accept_rate=0.966667`
  - 降到 `t=0.0`：`f1` 提升到 `0.144875`，但 `hal` 上升到 `0.030000`
  - 升到 `t=8.0`：`hal` 维持 `0`，但 `f1` 降到 `0.117097`，接受率降到 `0.710000`
- 决策建议：
  - 论文中建议阈值工作区间描述为 `4.0~5.0`，强调“零幻觉优先”的部署偏好。

### 3.1 数据标注流程与一致性（已完成）
- 已补齐标注流程证据与一致性统计产物：
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.csv`
  - `revision/artifacts/3_1_annotation/annotation_audit_summary.txt`
  - `revision/annotation_process_3_1.md`
- 审计规模：
  - 复标样本 `600` 条，覆盖 `VeriMind_Full / Vanilla_RAG / Self_RAG / HyDE`（每系统 `150`）。
  - 分层分布：`TRUSTED=256`，`WARNING=48`，`REJECTED=296`。
- 一致性结果：
  - `LLM vs Human (n=50)`：`agreement=0.9600`，`kappa=0.9151`
  - `Annotator A vs B overall (n=600)`：`agreement=0.9033`，`kappa=0.7767`
  - 分层：`TRUSTED kappa=0.9686`，`WARNING kappa=1.0000`，`REJECTED kappa=0.0000`
- 解释边界：
  - `REJECTED` 分层为高度不均衡标签分布，kappa 对边际分布敏感，需在正文单独说明。

### 3.2 AI 使用声明修订（已完成）
- 已修订声明口径并与实际实验执行对齐：
  - 实验模型为 `Qwen-Plus / GLM-4.5-Air / GLM-4.7(E5) / DeepSeek-V3.2(deepseek-chat)`；
  - 明确不使用 `ChatGPT-5.0` 与 `GLM-4.6V` 作为本实验模型。
- 已删除残留 `GLM-4.6V` 预算项，避免“已弃用但仍计费”的叙述冲突。
- 论文价值：
  - 可降低“模型版本陈述不一致”审稿风险，提升方法与复现实验描述的一致性。

### 3.3 过度宣称降调（已完成）
- 已完成强断言措辞降调，统一为“经验验证 + 评测范围限定”写法。
- 关键修订包括：
  - “建立铁证” -> “形成经验性证据链”
  - “证明最强模型纳入验证” -> “验证强模型条件下结论趋势一致性”
  - “条件完全一致” -> “在统一协议下尽量对齐”
  - 0.00% 结论补充“在当前评测范围内”限定
- 论文价值：
  - 降低“过度宣称/外推过强”审稿风险，提高论证稳健性。

### 3.4 Related Work 更新（已完成）
- 已将 Related Work 从“文献列举”升级为“主线化对齐”：
  - 检索自适应（Adaptive-RAG）
  - 检索纠错/选择性回答（Self-RAG, CRAG）
  - 评测与一致性（RAGChecker 等）
- 写作定位明确为“与既有方向互补”，避免“首创/绝对领先”风险。
- 论文价值：
  - 可降低“文献覆盖不足”与“贡献定位过强”的审稿风险。

### 4.1 参考文献核对（BibTeX，已完成）
- 已新增 `revision/references/related_work_2024_2025.bib` 作为可复用参考文献总表。
- 已核对并写入 2024 四个核心条目：
  - Adaptive-RAG / Self-RAG / CRAG / RAGChecker
- 决策：
  - 2025 文献改为单独 `4.2` 二次核对任务，避免一次性批量补录带来的错引风险。

### 4.2 2025 参考文献二次核对（已完成）
- 已补齐 2025 BibTeX 条目并写入 `revision/references/related_work_2024_2025.bib`。
- 当前新增覆盖：
  - FaithfulRAG（ACL 2025）
  - Source Reliability for RAG（EMNLP 2025）
  - Unified RAG Evaluation（NAACL 2025）
- 价值：
  - Related Work 引用库从“方向级”升级为“可直接引用级”，降低投稿前参考文献遗漏风险。

### 4.3 论文写回包与证据索引收口（已完成）
- 已新增 `revision/final_writeback_packet.md`，将当前实验证据整理为“正文可直接写回”的统一入口。
- 已更新 `revision/index/current_artifacts.md`，移除过期状态（如“partial/待重跑”）并对齐当前真实产物。
- 已在 `revision/paper_revision_evidence.md` 补充 4.3 收口说明，形成“实验产物 -> 写作草稿 -> 正文落地”的闭环映射。
- 价值：
  - 阶段四从“证据分散”转为“写回可执行”，可显著降低终稿阶段遗漏与错引风险。

---

## 2. 工程与流程决策（已生效）
- 决策：默认前台可见输出，后台仅按明确要求启用。
- 决策：每完成一个子任务，先同步三文档再做下一步。
- 决策：实验事实必须同步沉淀到 `revision/paper_revision_evidence.md`。
- 决策：将“实验完成 -> 三文档同步 -> 论文证据追加”写入项目规则。

---

## 3. 关键风险与现状
- 风险 A：一致性已达可接受水平，但 `qwen-glm` 与 `qwen-deepseek` 仅中等一致性；正文应避免写成“高度一致”。
- 风险 B：4.3 已完成材料收口，但论文正文尚未落地；若不尽快执行 4.4，证据与正文仍可能短暂脱节。
- 风险 C：统计口径需持续保持“保守表述”，尤其是跨评判器一致性与跨架构 kappa 的解释边界。

---

## 4. 对论文修改的直接帮助（可复制到正文草稿）

### 可直接使用的事实句
- 我们已将全部 10 个系统纳入统一实验管线，并在 300 个问题上生成 3000 条可回溯结果。
- 对每个 `(question, strategy)` 样本，我们均收集了来自 Qwen、GLM 与 DeepSeek 的评判结果，共形成 9000 条 cross-judge 记录。
- 基于 full 9000 行 cross-judge 结果，三评判器 Fleiss' Kappa 为 `0.601333`，说明跨评判器一致性达到可接受水平。
- 在 600 条复标审计样本上，标注员间整体一致性达到 `kappa=0.7767`，可支撑“人工标注质量可接受且流程可追溯”的方法学说明。
- 实验 AI 使用声明已与真实模型配置对齐，不再存在“占位版本名/错误模型名”歧义。

### 可直接使用的“审稿质疑回应点”
- 针对“外部可验证性不足”：已补 SciFact 外部实验。
- 针对“基线比较不对等”：已将 CRAG_Local/CoT_RAG 纳入统一协议。
- 针对“评判器自验证偏置”：已完成三厂商评判矩阵采样与落盘。

---

## 5. 当前下一步（严格顺序）
1. 当前状态：确认论文如何修改，等待执行（4.4 待在 Overleaf 落地时启动）。

---

## 6. 备注
- 本文件已从乱码状态恢复为中文可读版。
- 旧版乱码文件已保留在 `revision/backup/text_recovery/`，可随时对照。

*最后更新：2026-04-15*

## 2026-06-22：JIIS 重投实验与指标审计

- 完成 QASPER 公共基准：300 个问题、6 个策略、共 1800 条唯一结果。
- VeriMind Full 在 QASPER 上为 `1/133` passed hallucinations，回答覆盖率
  `44.33%`；Vanilla RAG 为 `22/147`，HyDE 为 `13/143`。
- 去除 trust gate 后升至 `19/149`，说明 gating 是主要安全贡献；routing
  的影响更偏向覆盖率与效用。
- 发现并修正两项旧口径：
  - SciFact 的约 `0.81` 是 retrieval hit rate，不是 answer coverage。
  - 旧主表中的所谓 ROUGE-L 是 token-overlap F1，现已重算 true ROUGE-L。
- 不再把 answer-to-context BERTScore 写成 answer-to-reference 质量指标。
- 旧阈值实验的 gate-pass rate 不等于最终 answer coverage，因此删除
  `34.6%` 等覆盖比较结论。
- 结论边界：当前证据支持“低风险的选择性回答策略”，不支持“无条件全面
  优于基线”或“普遍零幻觉”。

*最后更新：2026-06-22*

## 2026-06-22：QASPER 多裁判结果审计

- 观察事实：802 个已放行答案均具备 DeepSeek、Qwen、GLM 三个标签。
- 原始一致率较高（`89.65%`），但 Fleiss' Kappa 为中等水平
  （`0.5041`）；这是 accepted-answer 子集标签高度不均衡时的合理现象。
- 论文应写“moderate chance-corrected agreement”，不能写“highly
  consistent”或把任一 LLM judge 当作 ground truth。
- 多数票仍保持核心排序：VeriMind 明显低于 Vanilla、HyDE 和 No Gating。
- GLM 对 unsupported 的判定整体更严格，说明具体事件标签存在评判器敏感性。
- 旧 600 条所谓独立双标注实际缺乏独立人工标注证据，已从正文清理。
- 当前 50 条作者盲审是挑战富集样本，只能用于评判标签一致性，不能用于估计
  总体幻觉率；完成前不得写入论文结果。

*最后更新：2026-06-22*


## 2026-09-27：实验口径与新协议

- SciFact 500 行含重复 claim 和文档展开，只有 454 个不同 claim 文本；主表同时报告 unique 与 legacy-row weighting。
- QASPER Recall 按人工标注 evidence 段落映射到索引父段落；空 gold 不计为零分。SciFact 按原始 claim 全部 gold 文档统计，重复命中不重复加分。MRR 截断于 top-5。
- 五策略：fixed_512、multi（等同 no_gate）、full、no_routing、no_judge。no_judge 使用归一化检索分数独立门控，固定阈值 5，不用测试标签调参。
- GLM-4.5-Air 生成，DeepSeek gate，Qwen-Plus 独立评估。相同上下文复用同一答案；自然拒答单列。
- 延迟按实测检索、生成、gate 阶段累计，不含独立评估；这是配对阶段耗时，不宣称独立服务压测。
- SciFact 新协议为 SUPPORT/CONTRADICT 科学声明验证，旧 QA 自由回答结果另列，不混用。


### 2026-09-27：全分支 smoke 已验证，准备全量

- `supplement_20260927_smoke_routes/validation.json`：12 输入、24 策略答案、60 policy 行，complete=true；三数据组均覆盖 DETAIL/CONCEPT/CONTEXT。
- `python -B -m unittest revision.experiments.test_supplement_20260927`：14 tests passed。
- 全量范围：QASPER 300 + SciFact 去重 454 + 单列 NEI 60 = 814 输入；每输入两种检索策略，五种决策策略。
- SciFact correctness 主指标使用原始 gold label 的确定性匹配，Qwen correctness 作为辅助，避免评判器忽略标签差异。
- 新输出：`revision/artifacts/supplement_20260927_full/`。正文与简历的全量量化结论仍待完成后验证。


### 2026-09-27：QASPER 全量阶段完成与语料限制

- 本次 full 产物已包含 300 个 QASPER 输入的 600 条策略答案；全任务仍在运行，SciFact/NEI 尚未完成。
- 237 个问题具有非空 evidence；49 个问题共 63 条 gold 证据标为 `FLOAT SELECTED`，全部不在现有 qasper_corpus 文本段落中。
- SQLite 只读源段落重建与 corpus 交叉核对表明，当前未映射证据数恰好等于 63 条图表证据，不是文本段落映射程序遗漏。
- 这些 gold 保留在标准召回分母；报告必须注明当前是文本索引，不能把图表缺失解释为路由模块单独导致的失败。
- 正文候选说明：QASPER 检索评估使用完整标注证据分母，而现有索引只覆盖文本段落，因此图表证据的缺失构成语料覆盖上限。


### 2026-09-27：独立计时入口与回归验证

- 新增 `revision/experiments/latency_20260927.py`、`test_latency_20260927.py`；计划 QASPER/SciFact 各 50 个固定分层样本 × 3 策略，策略顺序固定随机化。
- baseline/multi 在线路径不调用 judge；Full 仅计入在线 gate，不计入 Qwen 离线评估。每次独立调用，不复用答案。
- 计时 smoke：2 输入、6 次实际调用，全部完成。
- 新增行为测试 16 项通过；现有 selective-risk、cheap-gate、cost-aware-cascade、gate-v2 回归测试 26 项通过，共 42 项。
- 证据：`revision/artifacts/supplement_20260927_full/test_verification.json`、`revision/artifacts/latency_20260927_smoke/validation.json`。
- 全量主任务仍在运行；已发现两条评判器非 JSON 回复，保留失败记录，结束后只补齐缺失条目，不用默认分数填充。


### 2026-09-27：主实验全量完成，独立性能试验进行中

- `supplement_20260927_full/validation.json`：814 输入、814 检索行、1628 策略答案、4070 policy 行，complete=true。
- 首轮 3 个输入未完成（2 条评估 JSON 截断/格式异常、1 条 TLS 中断）；保留 errors.jsonl，补跑后无缺失输入。
- 最后一条诊断确认 Qwen 的长说明被 300-token 上限截断；仅该条将输出上限增至 1024，原模型/提示/温度不变。详见 protocol_exception.json、retry_diagnostics.jsonl。
- 主结论草稿：动态路由未改善当前基准检索召回；QASPER gate 在独立评判口径下未降低已回答无支持比例；SciFact 风险降低伴随覆盖率下降。完整定量报告等待独立计时完成后一次交付。
- 独立计时作业：`latency_20260927_full`，100 个固定输入 × 3 策略，原始 timings.jsonl 单独保存。


## 2026-09-27：十项补充实验全部完成并验证

- [x] 全量：QASPER 300 + SciFact 454 去重 claim + NEI 60 = 814 输入；1628 策略答案、4070 决策行完整，无未解决缺失。900 份成功独立生成记录复用为两检索策略结果；失败重试另存日志。
- [x] 独立计时：两数据集各 50 输入 × 3 策略，共 300 次实际调用，无失败；baseline/multi 不调用在线 judge。
- [x] 42 项测试通过（16 新测试 + 26 回归）；报告审计通过唯一键、上下文哈希、源文件哈希、代码哈希、gate重放、summary分母。
- 交付：`revision/artifacts/supplement_20260927_full/TEN_ITEM_RESULTS.md`、`RESULTS.md`、`summary.csv`、`paired_gate_audit.csv`；性能：`revision/artifacts/latency_20260927_full/summary.csv`。
- 检索固定→动态：QASPER R@3 42.92%→42.11%、R@5 50.33%→49.35%；SciFact R@3 85.18%→84.52%、R@5 89.74%→89.44%。当前无多粒度召回提升证据。
- 无gate→Full，Qwen无支持率：QASPER 9.72%→9.92%，Coverage 48.00%→43.67%；SciFact 6.92%→4.33%，Coverage 92.29%→61.01%。不写成普遍降低真实幻觉。
- Full 拒答：QASPER TP=54/60、FP=115，P/R/F1=31.95%/90.00%/47.16%；SciFact 454+60混合集 TP=53/60、FP=177，P/R/F1=23.04%/88.33%/36.55%。
- 独立平均/P95秒：QASPER baseline 1.264/2.026，Full 1.598/2.955；SciFact baseline 1.330/1.638，Full 1.960/2.782。均为每策略50样本客户端测量，不是生产SLA。
- 边界：QASPER 63条图表gold不在文本索引；SciFact主样本dev188/train266，并非纯官方test；Faithfulness与无支持率是独立Qwen自动判定，未宣称新答案经人工认证。
- 一个评估JSON截断案例只增加输出上限300→1024，其模型/提示/温度不变，protocol_exception.json记录；所有原始失败与补跑记录保留。
- 论文候选表述：在固定输入与配对答案的补充评估中，动态粒度路由未提高检索召回；信任门控在SciFact降低了自动判定的已答无支持率，但减少了回答覆盖，在QASPER未观察到该条件风险改善。结果说明当前门控仍存在较高误拒，需同时报告风险、覆盖率与延迟。未据此作统计显著性结论。
- 未修改论文正文、应用代码或依赖；未覆盖历史实验原始结果。


## 2026-09-27：Gate 离线诊断完成

- 证据：`revision/artifacts/gate_diagnostics_20260927/REPORT.md`、`scores.csv`、`threshold_sweep.csv`、`gate_activation.csv`、`debug50.csv`、`provenance.json`。
- 814条冻结 multi 原始答案，QASPER 300；SciFact 454+NEI 60。50条分层debug；无API调用、无重新生成。
- retrieval gate阈值0.5：QASPER实际新增拦截0（4条低分已主动拒答），SciFact+NEI新增拦截0；与no-gate决策差异均0。极端/边界断言与P10-P90 coverage单调性通过，支持阈值未触发而不是已证实实现bug。
- QASPER：retrieval预测answerable AUROC 0.592；judge预测supported 0.500；Trust预测safe 0.532。SciFact+NEI分别为0.768/0.634/0.757。对应AUPRC与正例率详见scores.csv。
- 0.1–0.9 Trust阈值表已保存。QASPER无稳定风险收益；SciFact+NEI在0.3→0.4时coverage 84.4%→56.2%，表明校准不平滑。当前不依据测试集选择新阈值。
- 必须保留口径：judge/Trust AUROC只在候选答案上计算，主动拒答默认score=10排除；支持标签来自独立Qwen，safe还要求答案正确；AUPRC受高正例率影响。
- 论文候选写法：当前retrieval-only阈值未产生额外拒答，门控消融与无门控重合；离线分数诊断表明QASPER的现有judge/Trust信号区分力弱，而SciFact的融合分数具有一定排序信息但存在显著的coverage跳变。后续应在独立校准集调参并在留出集验证。
